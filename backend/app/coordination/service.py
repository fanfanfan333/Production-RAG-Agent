"""Coordinator service: bounded fan-out, persistence, cancellation, reconciliation."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from app.coordination.forgeflow_client import (
    ForgeFlowClient,
    ForgeFlowRejected,
    ForgeFlowUnavailable,
    forgeflow_state,
)
from app.coordination.planner import Plan, SubTask, plan_request
from app.coordination.web_research import (
    ControlledWebResearch,
    ResearchUnavailable,
    UnsafeUrl,
)
from app.db.user_models import User
from app.utils.logging import get_logger

logger = get_logger(__name__)

TERMINAL_STATUSES = frozenset(
    {"completed", "partial_failure", "failed", "stopped", "unavailable", "interrupted"}
)
ACTIVE_STATUSES = frozenset({"queued", "running", "awaiting_approval"})
_CODE_APPROVAL_STATUSES = frozenset({"awaiting_approval", "pending_approval"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CoordinationConflict(RuntimeError):
    """The task is in a state that cannot accept this operation."""


class CoordinationService:
    def __init__(
        self,
        repository,
        *,
        settings=None,
        forgeflow_factory: Callable[..., ForgeFlowClient] = ForgeFlowClient,
        research_factory: Callable[..., ControlledWebResearch] = ControlledWebResearch,
        document_runner: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self._forgeflow_factory = forgeflow_factory
        self._research_factory = research_factory
        self._document_runner = document_runner
        # One ceiling for the service, not one per request: several tasks may be
        # admitted concurrently and must still not exceed the global fan-out.
        self._semaphore = asyncio.Semaphore(
            max(1, int(self._settings().COORDINATION_MAX_CONCURRENCY))
        )

    def _settings(self):
        if self.settings is not None:
            return self.settings
        from app.config import get_settings

        return get_settings()

    async def create(
        self,
        *,
        user: User,
        query: str,
        idempotency_key: str,
        kind: str = "auto",
        collection_id: uuid.UUID | None = None,
    ) -> tuple[dict, bool]:
        key = (idempotency_key or "").strip()
        if not 8 <= len(key) <= 128:
            raise ValueError("幂等键长度必须为 8–128 字符")
        settings = self._settings()
        plan = plan_request(
            query,
            kind=kind,
            document_timeout=float(settings.COORDINATION_DOCUMENT_TIMEOUT_SECONDS),
            research_timeout=float(settings.COORDINATION_RESEARCH_TIMEOUT_SECONDS),
            code_timeout=float(settings.COORDINATION_CODE_TIMEOUT_SECONDS),
        )
        task, created = await self.repository.create_or_get(
            tenant_id=str(user.tenant_id),
            user_id=user.id,
            idempotency_key=key,
            request_text=" ".join(query.split()),
            requested_kind=kind,
            plan={
                "reason": plan.reason,
                "subtasks": [asdict(item) for item in plan.subtasks],
                "collection_id": str(collection_id) if collection_id else None,
            },
        )
        if not created and (
            task["request_text"] != " ".join(query.split())
            or task["requested_kind"] != kind
            or (task.get("plan") or {}).get("collection_id") != (str(collection_id) if collection_id else None)
        ):
            raise CoordinationConflict("幂等键已用于不同的任务请求")
        if created:
            await self.repository.append_event(
                uuid.UUID(task["id"]),
                "task.accepted",
                {
                    "status": "queued",
                    "subtasks": list(plan.kinds),
                    "reason": plan.reason,
                },
            )
        return task, created

    async def get_owned(self, task_id: uuid.UUID, user: User) -> dict:
        task = await self.repository.get(task_id)
        if (
            task is None
            or task["tenant_id"] != str(user.tenant_id)
            or task["user_id"] != str(user.id)
        ):
            raise LookupError("task not found")
        return task

    async def list_owned(self, user: User, limit: int = 30) -> list[dict]:
        return await self.repository.list_for_owner(
            str(user.tenant_id), user.id, limit=limit
        )

    async def events(self, task_id: uuid.UUID, user: User, after_seq: int = 0) -> list[dict]:
        await self.get_owned(task_id, user)
        return await self.repository.list_events(task_id, after_seq=after_seq)

    def _forgeflow(self, token: str) -> ForgeFlowClient:
        settings = self._settings()
        return self._forgeflow_factory(
            base_url=settings.FORGEFLOW_BASE_URL,
            token=token,
            timeout_seconds=float(settings.COORDINATION_STATUS_RECONCILE_TIMEOUT_SECONDS),
        )

    async def link_forgeflow(self, user: User, *, token: str) -> dict:
        from sqlalchemy.exc import IntegrityError
        identity = await self._forgeflow(token).identity()
        if (not isinstance(identity, dict) or set(identity) != {"tenant_id", "user_id"}
                or any(not isinstance(value, str) or not value.strip() or len(value) > 128 for value in identity.values())):
            raise ForgeFlowUnavailable("ForgeFlow 身份响应无效")
        try:
            await self.repository.set_link(str(user.tenant_id), user.id, identity)
        except IntegrityError as exc:
            raise CoordinationConflict("该 ForgeFlow 账号已关联其他用户") from exc
        return identity

    async def unlink_forgeflow(self, user: User) -> None:
        await self.repository.remove_link(str(user.tenant_id), user.id)

    def _research(self) -> ControlledWebResearch:
        settings = self._settings()
        return self._research_factory(
            search_endpoint=settings.WEB_SEARCH_ENDPOINT,
            search_api_key=settings.WEB_SEARCH_API_KEY,
            timeout_seconds=float(settings.WEB_RESEARCH_TIMEOUT_SECONDS),
        )

    async def run(
        self,
        task_id: uuid.UUID,
        *,
        user: User,
        forgeflow_token: str = "",
        user_scope=None,
    ) -> dict:
        """Execute a queued task. Safe to call only once from the API worker."""
        task = await self.get_owned(task_id, user)
        if not await self.repository.claim_start(task_id):
            return await self.get_owned(task_id, user)
        saved_subtasks = task["plan"]["subtasks"]
        legacy_dependencies = tuple(item["kind"] for item in saved_subtasks if item["kind"] != "code")
        plan = Plan(
            subtasks=tuple(
                SubTask(**{**item, "depends_on": tuple(item.get("depends_on", legacy_dependencies if item["kind"] == "code" else ()))})
                for item in saved_subtasks
            ),
            reason=str(task["plan"].get("reason") or ""),
        )
        async def one(subtask, handoff_results=None) -> dict:
            async with self._semaphore:
                fresh = await self.repository.get(task_id)
                if fresh and fresh.get("stop_requested"):
                    return {
                        "kind": subtask.kind,
                        "status": "stopped",
                        "error": "任务已请求停止",
                        "duration_ms": 0,
                    }
                await self.repository.append_event(
                    task_id,
                    "subtask.started",
                    {"kind": subtask.kind, "objective": subtask.objective},
                )
                started = asyncio.get_running_loop().time()
                try:
                    if subtask.kind == "code" and task.get("requested_kind") == "code":
                        inferred = plan_request(subtask.objective, kind="auto")
                        if not inferred.default_document and any(kind != "code" for kind in inferred.kinds):
                            raise RuntimeError("请求还需要文档或联网任务，单独代码模式无法传递前置结果")
                    async with asyncio.timeout(subtask.timeout_seconds):
                        payload = await self._execute(
                            subtask,
                            user=user,
                            task_id=task_id,
                            collection_id=self._collection_id(task),
                            forgeflow_token=forgeflow_token,
                            user_scope=user_scope,
                            handoff_results=handoff_results,
                        )
                    result = {
                        "kind": subtask.kind,
                        "status": payload.pop("status", "completed"),
                        "duration_ms": round(
                            (asyncio.get_running_loop().time() - started) * 1000, 1
                        ),
                        **payload,
                    }
                except TimeoutError:
                    result = {
                        "kind": subtask.kind,
                        "status": "timeout",
                        "error": f"{subtask.kind} 超过 {subtask.timeout_seconds:g}s 未完成",
                        "duration_ms": round(
                            (asyncio.get_running_loop().time() - started) * 1000, 1
                        ),
                    }
                except (
                    ForgeFlowUnavailable,
                    ForgeFlowRejected,
                    ResearchUnavailable,
                    UnsafeUrl,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    result = {
                        "kind": subtask.kind,
                        "status": "unavailable"
                        if isinstance(exc, (ForgeFlowUnavailable, ResearchUnavailable))
                        else "failed",
                        "error": str(exc)[:1000],
                        "duration_ms": round(
                            (asyncio.get_running_loop().time() - started) * 1000, 1
                        ),
                    }
                except Exception as exc:  # noqa: BLE001 - persisted, never swallowed
                    logger.exception("coordination subtask failed: %s", exc)
                    result = {
                        "kind": subtask.kind,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"[:1000],
                        "duration_ms": round(
                            (asyncio.get_running_loop().time() - started) * 1000, 1
                        ),
                    }
                if subtask.kind == "code" and result["status"] in {"timeout", "unavailable", "failed"}:
                    current = await self.repository.get(task_id)
                    if current and current.get("stop_requested") and not current.get("execution_id") and (current.get("plan") or {}).get("forgeflow_identity"):
                        try:
                            client = self._forgeflow(forgeflow_token)
                            await self._verify_forgeflow_identity(current, client)
                            remote = await client.get_request(str(task_id))
                            await self.repository.update(task_id, fields={"execution_id": remote.run_id})
                            await client.abort(remote.run_id)
                            current = await self.repository.get(task_id)
                        except (ForgeFlowUnavailable, ForgeFlowRejected) as exc:
                            await self.repository.append_event(task_id, "code.abort_failed", {"error": str(exc)})
                    if current and current.get("execution_id"):
                        result["status"] = "running"
                        result["execution_id"] = current["execution_id"]
                await self.repository.append_event(
                    task_id, "subtask.finished", result
                )
                return result

        if any(item.kind == "code" for item in plan.subtasks) and any(item.kind != "code" for item in plan.subtasks):
            prerequisites = [item for item in plan.subtasks if item.kind != "code"]
            code_tasks = [item for item in plan.subtasks if item.kind == "code"]
            results = list(await asyncio.gather(*(one(item) for item in prerequisites)))
            dependency_status = {item["kind"]: item.get("status") for item in results}
            if code_tasks and (any(not code.depends_on for code in code_tasks) or any(dependency_status.get(dep) != "completed" for code in code_tasks for dep in code.depends_on)):
                results.extend({"kind": "code", "status": "failed", "error": "前置文档或联网任务未完成，代码任务未执行", "duration_ms": 0} for _ in code_tasks)
            elif code_tasks and prerequisites:
                if any(item.kind == "document" for item in prerequisites):
                    results.extend({"kind": "code", "status": "unavailable", "error": "文档权限尚未在传递时复核，代码任务未执行", "duration_ms": 0} for _ in code_tasks)
                else:
                    evidence = []
                    for result in results:
                        for page in result.get("pages", [])[:3]:
                            content = str(page.get("text") or "")
                            source = str(page.get("url") or "")
                            bounded = content.encode("utf-8")[:9000].decode("utf-8", errors="ignore")
                            if source and bounded.strip():
                                evidence.append({"kind": "research", "source": source, "content": bounded,
                                                 "truncated": bool(page.get("truncated")) or bounded != content})
                    if (not evidence or any(len(item["source"].encode("utf-8")) > 512 for item in evidence)
                            or len(json.dumps(evidence, ensure_ascii=False).encode("utf-8")) > 32768):
                        results.extend({"kind": "code", "status": "failed", "error": "前置联网结果为空，代码任务未执行", "duration_ms": 0} for _ in code_tasks)
                    else:
                        results.extend(await asyncio.gather(*(one(item, evidence) for item in code_tasks)))
            else:
                results.extend(await asyncio.gather(*(one(item) for item in code_tasks)))
        else:
            results = list(await asyncio.gather(*(one(item) for item in plan.subtasks)))
        final_status = self._aggregate_status(results, task_id, await self.repository.get(task_id))
        return await self.repository.update(
            task_id,
            fields={
                "status": final_status,
                "result": {"subtasks": results, "summary": self._summary(results)},
                "finished_at": _now() if final_status in TERMINAL_STATUSES else None,
            },
            event={
                "type": "task.finished" if final_status in TERMINAL_STATUSES else "task.pending",
                "data": {"status": final_status, "summary": self._summary(results)},
            },
        )

    def _collection_id(self, task: dict) -> uuid.UUID | None:
        raw = (task.get("plan") or {}).get("collection_id")
        try:
            return uuid.UUID(str(raw)) if raw else None
        except (ValueError, TypeError):
            return None

    async def _execute(
        self,
        subtask,
        *,
        user: User,
        task_id: uuid.UUID,
        collection_id: uuid.UUID | None,
        forgeflow_token: str,
        user_scope=None,
        handoff_results=None,
    ) -> dict:
        if subtask.kind == "document":
            runner = self._document_runner
            if runner is None:
                from app.coordination.document_agent import answer_document_question

                runner = answer_document_question
            answer = await runner(
                query=subtask.objective,
                user=user,
                collection_id=collection_id,
                user_scope=user_scope,
            )
            conversation_id = getattr(answer, "conversation_id", None)
            if conversation_id:
                await self.repository.update(
                    task_id, fields={"conversation_id": uuid.UUID(str(conversation_id))}
                )
            return {
                "status": "completed",
                "answer": getattr(answer, "answer", ""),
                "sources": list(getattr(answer, "sources", []) or []),
                "intent": getattr(answer, "intent", "knowledge_qa"),
            }
        if subtask.kind == "research":
            return await self._execute_research(subtask.objective)
        if subtask.kind == "code":
            return await self._execute_code(
                subtask.objective,
                task_id=task_id,
                token=forgeflow_token,
                timeout_seconds=subtask.timeout_seconds,
                handoff_results=handoff_results,
            )
        raise ValueError(f"unknown subtask kind: {subtask.kind}")

    async def _execute_code(
        self, objective: str, *, task_id: uuid.UUID, token: str, timeout_seconds: float,
        handoff_results=None,
    ) -> dict:
        client = self._forgeflow(token)
        identity = await client.identity()
        task = await self.repository.get(task_id)
        plan = dict(task.get("plan") or {})
        plan["forgeflow_identity"] = identity
        if handoff_results:
            plan["requires_identity_link"] = True
        if handoff_results:
            async with self.repository.locked_link(task["tenant_id"], uuid.UUID(task["user_id"])) as linked:
                if not linked or linked != identity:
                    raise ForgeFlowUnavailable("ForgeFlow 账号未关联或凭证与关联身份不一致")
                await self.repository.update(task_id, fields={"plan": plan})
                run = await client.create_task(intent=objective, request_key=str(task_id),
                                               context={"handoff_results": handoff_results})
        else:
            await self.repository.update(task_id, fields={"plan": plan})
            run = await client.create_task(intent=objective, request_key=str(task_id))
        await self.repository.update(
            task_id,
            fields={"execution_id": run.run_id},
            event={
                "type": "code.accepted",
                "data": {"run_id": run.run_id, "status": run.status,
                         "sources": [{"kind": item["kind"], "source": item["source"], "truncated": item["truncated"]} for item in (handoff_results or [])]},
            },
        )
        current = await self.repository.get(task_id)
        if current and current.get("stop_requested"):
            await client.abort(run.run_id)
        deadline = asyncio.get_running_loop().time() + max(1.0, timeout_seconds - 5)
        latest = run
        while (
            latest.status.lower() in {"queued", "running", "pending"}
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(1.0)
            latest = await client.get_run(run.run_id)
        state = forgeflow_state(latest)
        payload = {
            "status": state,
            "execution_id": latest.run_id,
            "forgeflow_status": latest.status,
            "outcome": latest.outcome,
            "codeplane": latest.codeplane,
            "artifacts": latest.artifacts,
            "errors": latest.errors,
        }
        if state == "running":
            payload["status"] = "running"
            payload["error"] = "代码任务在时限内未到终态；执行仍在 ForgeFlow 侧，可稍后刷新"
        elif state in {"failed", "rejected", "aborted", "interrupted"}:
            payload["error"] = "; ".join(latest.errors) or f"ForgeFlow 状态: {latest.status}"
        return payload

    async def _execute_research(self, objective: str) -> dict:
        research = self._research()
        results, provider = await research.search(objective)
        pages: list[dict] = []
        failures: list[dict] = []
        for result in results[:3]:
            try:
                page = await research.read_url(result.url)
                if not page.text.strip():
                    raise ResearchUnavailable("网页未返回可用正文")
                pages.append(
                    {
                        "url": page.url,
                        "title": page.title or result.title,
                        "text": page.text,
                        "truncated": page.truncated,
                    }
                )
            except (ResearchUnavailable, UnsafeUrl) as exc:
                failures.append({"url": result.url, "error": str(exc)[:300]})
        return {
            "status": ("partial_failure" if failures else "completed") if pages else "failed",
            "provider": provider,
            "sources": [asdict(item) for item in results],
            "pages": pages,
            "read_failures": failures,
        }

    def _aggregate_status(
        self, results: list[dict], task_id: uuid.UUID, task: dict | None
    ) -> str:
        statuses = [item.get("status") for item in results]
        if statuses and all(status == "completed" for status in statuses):
            return "completed"
        if "awaiting_approval" in statuses:
            return "awaiting_approval"
        if "running" in statuses:
            return "running"
        if "partial_failure" in statuses:
            return "partial_failure"
        if any(status == "completed" for status in statuses):
            if all(status == "stopped" for status in statuses):
                return "stopped"
            return "partial_failure"
        if statuses and all(status == "unavailable" for status in statuses):
            return "unavailable"
        if "stopped" in statuses:
            return "stopped"
        return "failed"

    @staticmethod
    def _summary(results: list[dict]) -> str:
        completed = [item["kind"] for item in results if item.get("status") == "completed"]
        partial = [item["kind"] for item in results if item.get("status") == "partial_failure"]
        pending = [
            item["kind"]
            for item in results
            if item.get("status") in {"awaiting_approval", "timeout"}
        ]
        failed = [
            item["kind"]
            for item in results
            if item.get("status") not in {"completed", "partial_failure", "awaiting_approval", "timeout"}
        ]
        parts = []
        if completed:
            parts.append("已完成：" + "、".join(completed))
        if partial:
            parts.append("部分完成：" + "、".join(partial))
        if pending:
            parts.append("待处理：" + "、".join(pending))
        if failed:
            parts.append("未完成：" + "、".join(failed))
        return "；".join(parts) if parts else "没有可汇总的子任务"

    async def stop(self, task_id: uuid.UUID, user: User, *, token: str = "") -> dict:
        task = await self.get_owned(task_id, user)
        recoverable_code = bool((task.get("plan") or {}).get("forgeflow_identity")) and not task.get("execution_id")
        uncertain_code = recoverable_code and task["status"] in {"interrupted", "unavailable", "failed", "partial_failure"}
        if task["status"] in TERMINAL_STATUSES and not uncertain_code and not (
            task["status"] == "interrupted" and task.get("execution_id")
        ):
            raise CoordinationConflict("任务已结束，无法停止")
        await self.repository.update(
            task_id,
            fields={"stop_requested": True},
            event={"type": "task.stop_requested", "data": {}},
        )
        abort_detail = ""
        if task.get("execution_id") or recoverable_code:
            try:
                client = self._forgeflow(token)
                await self._verify_forgeflow_identity(task, client)
                execution_id = task.get("execution_id")
                if not execution_id:
                    execution_id = (await client.get_request(str(task_id))).run_id
                    await self.repository.update(task_id, fields={"execution_id": execution_id})
                abort_detail = await client.abort(str(execution_id))
            except (ForgeFlowUnavailable, ForgeFlowRejected) as exc:
                abort_detail = str(exc)
                await self.repository.append_event(
                    task_id, "code.abort_failed", {"error": abort_detail}
                )
        # Do not claim "stopped" before the worker actually observes the flag.
        current = await self.repository.get(task_id) or task
        current["abort_detail"] = abort_detail
        return current

    async def refresh(self, task_id: uuid.UUID, user: User, *, token: str = "") -> dict:
        """Reconcile a code task with ForgeFlow, including after a restart."""
        task = await self.get_owned(task_id, user)
        if not token:
            task["reconcile_error"] = "未提供 ForgeFlow 凭证，无法刷新真实执行状态"
            return task
        try:
            client = self._forgeflow(token)
            await self._verify_forgeflow_identity(task, client)
            run = (await client.get_run(str(task["execution_id"]))) if task.get("execution_id") else (await client.get_request(str(task_id)))
        except (ForgeFlowUnavailable, ForgeFlowRejected) as exc:
            task["reconcile_error"] = str(exc)
            return task
        state = forgeflow_state(run)
        current = dict(task.get("result") or {})
        subtasks = list(current.get("subtasks") or [])
        if not subtasks:
            if task["status"] != "interrupted":
                task["reconcile_error"] = "任务仍在执行，等待子任务状态写入"
                return task
            subtasks = [
                {"kind": item["kind"], "status": "interrupted", "error": "服务重启时子任务结果未保存"}
                for item in (task.get("plan") or {}).get("subtasks", [])
            ]
        for item in subtasks:
            if item.get("kind") == "code":
                item.update(
                    {
                        "status": state,
                        "execution_id": run.run_id,
                        "forgeflow_status": run.status,
                        "codeplane": run.codeplane,
                        "artifacts": run.artifacts,
                        "errors": run.errors,
                    }
                )
        current["subtasks"] = subtasks
        current["summary"] = self._summary(subtasks)
        fields: dict[str, Any] = {"result": current, "execution_id": run.run_id}
        if state in {"completed", "failed", "rejected", "aborted", "interrupted"}:
            if state == "aborted":
                for item in subtasks:
                    if item.get("kind") == "code":
                        item["status"] = "stopped"
            fields["status"] = self._aggregate_status(subtasks, task_id, task)
            fields["finished_at"] = _now()
        elif state == "awaiting_approval":
            fields["status"] = "awaiting_approval"
            fields["finished_at"] = None
        elif state == "running":
            fields["status"] = "running"
            fields["finished_at"] = None
        return await self.repository.update(
            task_id,
            fields=fields,
            event={"type": "code.reconciled", "data": {"status": state, "run_id": run.run_id}},
        )

    async def decide_code(
        self,
        task_id: uuid.UUID,
        user: User,
        *,
        decision: str,
        note: str,
        token: str,
    ) -> dict:
        task = await self.get_owned(task_id, user)
        if task["status"] != "awaiting_approval" or not task.get("execution_id"):
            raise CoordinationConflict("任务当前不在待审批状态")
        client = self._forgeflow(token)
        await self._verify_forgeflow_identity(task, client)
        if decision == "approve":
            data = await client.approve(str(task["execution_id"]), note)
        elif decision == "reject":
            data = await client.reject(str(task["execution_id"]), note)
        else:
            raise ValueError("decision must be approve or reject")
        await self.repository.append_event(
            task_id,
            f"code.{decision}",
            {"run_id": task["execution_id"], "result_status": data.get("status", "")},
        )
        return await self.repository.update(
            task_id,
            fields={
                "status": "running" if decision == "approve" else "stopped",
                "result": {**(task.get("result") or {}), "approval": data},
            },
        )

    async def _verify_forgeflow_identity(self, task: dict, client: ForgeFlowClient) -> None:
        expected = (task.get("plan") or {}).get("forgeflow_identity")
        if not expected or await client.identity() != expected:
            raise ForgeFlowUnavailable("ForgeFlow 身份与任务绑定的身份不一致")
        if (task.get("plan") or {}).get("requires_identity_link"):
            linked = await self.repository.get_link(task["tenant_id"], uuid.UUID(task["user_id"]))
            if linked != expected:
                raise ForgeFlowUnavailable("ForgeFlow 账号关联已撤销或变更")
