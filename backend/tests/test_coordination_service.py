"""Coordinator lifecycle tests with injected fakes (no DB, container or network)."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from app.coordination.forgeflow_client import ForgeRun  # noqa: E402
from app.coordination.repository import MemoryTaskRepository  # noqa: E402
from app.coordination.service import CoordinationService  # noqa: E402


def _user(name: str = "alice") -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), tenant_id="tenant-a", username=name)


def _settings(**overrides):
    values = {
        "COORDINATION_DOCUMENT_TIMEOUT_SECONDS": 2.0,
        "COORDINATION_RESEARCH_TIMEOUT_SECONDS": 2.0,
        "COORDINATION_CODE_TIMEOUT_SECONDS": 2.0,
        "COORDINATION_MAX_CONCURRENCY": 2,
        "COORDINATION_STATUS_RECONCILE_TIMEOUT_SECONDS": 1.0,
        "FORGEFLOW_BASE_URL": "https://forgeflow.example.com",
        "WEB_SEARCH_ENDPOINT": "https://search.example.com",
        "WEB_SEARCH_API_KEY": "test-key",
        "WEB_RESEARCH_TIMEOUT_SECONDS": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeForge:
    def __init__(self, *, status="awaiting_approval", fail=False):
        self.status = status
        self.fail = fail
        self.created = 0
        self.aborted = 0
        self.approved = 0
        self.identity_value = {"tenant_id": "ff-tenant", "user_id": "ff-user"}

    async def identity(self):
        return self.identity_value

    async def create_task(self, intent, workflow_type="generic", context=None, **kwargs):
        if self.fail:
            raise RuntimeError("should not be used")
        self.created += 1
        self.submitted_context = context
        return ForgeRun(
            run_id="ff-run-1",
            status="queued",
            outcome="pending",
            intent=intent,
            session_id="ff-session-1",
            parent_run_id="",
            artifacts=[],
            codeplane={"workspace": {"workspace_id": "ws-1"}},
            errors=[],
        )

    async def get_run(self, run_id):
        return ForgeRun(
            run_id=run_id,
            status=self.status,
            outcome="pending",
            intent="code",
            session_id="ff-session-1",
            parent_run_id="",
            artifacts=[{"id": "diff", "format": "diff", "content": "+ ok"}],
            codeplane={"workspace": {"workspace_id": "ws-1"}, "tests": {"passed": 3}},
            errors=[],
        )

    async def abort(self, run_id):
        self.aborted += 1
        return "aborted"

    async def approve(self, run_id, note=""):
        self.approved += 1
        return {"run_id": run_id, "status": "running"}

    async def reject(self, run_id, note=""):
        return {"run_id": run_id, "status": "rejected"}


class _FakeResearch:
    def __init__(self, *, unavailable=False):
        self.unavailable = unavailable

    async def search(self, query):
        if self.unavailable:
            from app.coordination.web_research import ResearchUnavailable

            raise ResearchUnavailable("未配置联网搜索服务")
        from app.coordination.web_research import SearchResult

        return [
            SearchResult(title="Official", url="https://example.com/doc", snippet="real")
        ], "real-provider"

    async def read_url(self, url):
        return SimpleNamespace(
            url=url, title="Official", text="real page body", truncated=False
        )


def _service(*, forge=None, research=None, doc_delay=0.0, active=None):
    async def doc_runner(**kwargs):
        if active is not None:
            active["current"] += 1
            active["max"] = max(active["max"], active["current"])
        await asyncio.sleep(doc_delay)
        if active is not None:
            active["current"] -= 1
        return SimpleNamespace(
            answer=f"answer: {kwargs['query']}",
            sources=[{"n": 1, "filename": "doc.pdf"}],
            conversation_id=str(uuid.uuid4()),
            intent="knowledge_qa",
        )

    return CoordinationService(
        MemoryTaskRepository(),
        settings=_settings(),
        forgeflow_factory=lambda **kwargs: forge or _FakeForge(),
        research_factory=lambda **kwargs: research or _FakeResearch(),
        document_runner=doc_runner,
    )


def test_idempotency_key_returns_the_same_task_without_duplicate_events():
    async def case():
        service = _service()
        user = _user()
        first, created = await service.create(
            user=user, query="总结这份文档", idempotency_key="retry-key-1"
        )
        second, created_again = await service.create(
            user=user, query="总结这份文档", idempotency_key="retry-key-1"
        )
        assert created is True and created_again is False
        assert first["id"] == second["id"]
        assert len(await service.events(uuid.UUID(first["id"]), user)) == 1

    asyncio.run(case())


def test_reusing_idempotency_key_with_different_request_is_conflict():
    from app.coordination.service import CoordinationConflict

    async def case():
        service = _service()
        user = _user()
        await service.create(user=user, query="总结文档 A", idempotency_key="same-key-1")
        try:
            await service.create(user=user, query="总结文档 B", idempotency_key="same-key-1")
        except CoordinationConflict:
            return
        raise AssertionError("a different request cannot reuse an idempotency key")

    asyncio.run(case())


def test_user_can_link_and_revoke_verified_forgeflow_identity():
    async def case():
        forge = _FakeForge()
        service = _service(forge=forge)
        user = _user()
        identity = await service.link_forgeflow(user, token="personal-token")
        assert identity == forge.identity_value
        assert await service.repository.get_link(str(user.tenant_id), user.id) == identity
        assert "personal-token" not in str(service.repository.links)
        await service.unlink_forgeflow(user)
        assert await service.repository.get_link(str(user.tenant_id), user.id) is None

    asyncio.run(case())


def test_forgeflow_link_rejects_identity_already_linked_to_another_user():
    import pytest
    from app.coordination.service import CoordinationConflict

    async def case():
        service = _service(forge=_FakeForge())
        await service.link_forgeflow(_user(), token="personal-token")
        with pytest.raises(CoordinationConflict):
            await service.link_forgeflow(_user("other"), token="personal-token")

    asyncio.run(case())


def test_run_rejects_another_user_before_dispatch():
    import pytest

    async def case():
        forge = _FakeForge()
        service = _service(forge=forge)
        task, _ = await service.create(user=_user(), query="改代码", idempotency_key="owner-run-key")
        with pytest.raises(LookupError):
            await service.run(uuid.UUID(task["id"]), user=_user("bob"), forgeflow_token="token")
        assert forge.created == 0
        assert (await service.repository.get(uuid.UUID(task["id"])))["status"] == "queued"

    asyncio.run(case())


def test_concurrent_run_claims_execute_document_only_once():
    async def case():
        calls = 0
        service = _service()
        user = _user()
        task, _ = await service.create(user=user, query="总结文档", idempotency_key="claim-run-key")
        original_get_owned = service.get_owned
        arrived = 0
        ready = asyncio.Event()
        async def simultaneous_get(task_id, owner):
            nonlocal arrived
            result = await original_get_owned(task_id, owner)
            arrived += 1
            if arrived == 2:
                ready.set()
            await ready.wait()
            return result
        async def runner(**kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(answer="ok", sources=[], conversation_id=None)
        service.get_owned = simultaneous_get
        service._document_runner = runner
        await asyncio.gather(*(service.run(uuid.UUID(task["id"]), user=user) for _ in range(2)))
        assert calls == 1

    asyncio.run(case())


def test_refresh_after_restart_restores_remote_status_without_results():
    async def case():
        forge = _FakeForge(status="awaiting_approval")
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="改代码", idempotency_key="restart-refresh-1")
        task_id = uuid.UUID(task["id"])
        plan = {**task["plan"], "forgeflow_identity": forge.identity_value}
        await service.repository.update(task_id, fields={"status": "interrupted", "execution_id": "ff-run-1", "result": {}, "plan": plan})
        result = await service.refresh(task_id, user, token="token")
        assert result["status"] == "awaiting_approval"
        assert result["result"]["subtasks"][0]["execution_id"] == "ff-run-1"

    asyncio.run(case())


def test_another_user_in_the_same_tenant_cannot_read_or_stop():
    async def case():
        service = _service()
        owner, other = _user("owner"), _user("other")
        task, _ = await service.create(
            user=owner, query="总结这份文档", idempotency_key="owner-key-1"
        )
        task_id = uuid.UUID(task["id"])
        for coro in (
            service.get_owned(task_id, other),
            service.events(task_id, other),
            service.stop(task_id, other),
        ):
            try:
                await coro
            except LookupError:
                pass
            else:
                raise AssertionError("same-tenant cross-user access must be hidden")
        assert await service.list_owned(other) == []

    asyncio.run(case())


def test_fanout_is_bounded_and_document_runs_stay_concurrent():
    async def case():
        active = {"current": 0, "max": 0}
        service = _service(active=active, doc_delay=0.05)
        user = _user()
        task, _ = await service.create(
            user=user,
            query="查文档说明，联网搜索最新规范，然后改代码并运行测试",
            idempotency_key="fanout-key-1",
        )
        final = await service.run(
            uuid.UUID(task["id"]), user=user, forgeflow_token="user-token"
        )
        assert {item["kind"] for item in final["result"]["subtasks"]} == {
            "document",
            "research",
            "code",
        }
        assert final["status"] == "partial_failure"
        assert next(item for item in final["result"]["subtasks"] if item["kind"] == "code")["status"] == "unavailable"

    asyncio.run(case())


def test_code_does_not_run_if_required_research_is_unavailable():
    async def case():
        forge = _FakeForge()
        service = _service(forge=forge, research=_FakeResearch(unavailable=True))
        user = _user()
        task, _ = await service.create(
            user=user, query="先联网搜索最新规范，然后改代码并运行测试", idempotency_key="dependency-key-1"
        )
        result = await service.run(uuid.UUID(task["id"]), user=user, forgeflow_token="token")
        assert forge.created == 0
        assert result["status"] != "completed"

    asyncio.run(case())


def test_dependent_code_requires_linked_forgeflow_identity():
    async def case():
        forge = _FakeForge()
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(
            user=user, query="先联网搜索最新规范，然后改代码并运行测试", idempotency_key="handoff-key-1"
        )
        result = await service.run(uuid.UUID(task["id"]), user=user, forgeflow_token="token")
        assert forge.created == 0
        assert result["status"] != "completed"
        assert "关联" in result["result"]["subtasks"][-1]["error"]

    asyncio.run(case())


def test_mixed_research_and_code_transfers_source_after_linking():
    async def case():
        forge = _FakeForge()
        service = _service(forge=forge)
        user = _user()
        await service.link_forgeflow(user, token="token")
        task, _ = await service.create(user=user, query="联网搜索然后改代码", idempotency_key="mixed-handoff-key")
        result = await service.run(uuid.UUID(task["id"]), user=user, forgeflow_token="token")
        assert forge.created == 1
        evidence = forge.submitted_context["handoff_results"]
        assert evidence[0]["source"] == "https://example.com/doc"
        assert evidence[0]["content"] == "real page body"
        assert result["status"] != "completed"

    asyncio.run(case())


def test_mixed_handoff_rejects_revoked_or_changed_identity():
    async def case():
        for revoke in (True, False):
            forge = _FakeForge()
            service = _service(forge=forge)
            user = _user()
            await service.link_forgeflow(user, token="token")
            if revoke:
                await service.unlink_forgeflow(user)
            else:
                forge.identity_value = {"tenant_id": "ff-tenant", "user_id": "different"}
            task, _ = await service.create(user=user, query="联网搜索然后改代码", idempotency_key="revoke-link-1")
            result = await service.run(uuid.UUID(task["id"]), user=user, forgeflow_token="token")
            assert forge.created == 0
            assert result["status"] != "completed"

    asyncio.run(case())


def test_mixed_plan_with_empty_dependencies_never_dispatches_code():
    async def case():
        forge = _FakeForge()
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="联网搜索然后改代码", idempotency_key="empty-dependency-key")
        plan = task["plan"]
        plan["subtasks"][-1]["depends_on"] = []
        await service.repository.update(uuid.UUID(task["id"]), fields={"plan": plan})
        result = await service.run(uuid.UUID(task["id"]), user=user, forgeflow_token="token")
        assert forge.created == 0
        assert result["status"] != "completed"

    asyncio.run(case())


def test_explicit_code_mode_cannot_silently_skip_requested_research():
    async def case():
        forge = _FakeForge()
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="先联网搜索规范然后改代码", kind="code", idempotency_key="explicit-dependent-1")
        result = await service.run(uuid.UUID(task["id"]), user=user, forgeflow_token="token")
        assert forge.created == 0
        assert result["status"] != "completed"

    asyncio.run(case())


def test_explicit_code_mode_accepts_a_request_without_router_keywords(monkeypatch):
    from dataclasses import replace
    from app.coordination import service as service_module
    original = service_module.plan_request

    def changed_description(*args, **kwargs):
        plan = original(*args, **kwargs)
        return replace(plan, reason="modified description") if plan.kinds == ("document",) else plan

    monkeypatch.setattr(service_module, "plan_request", changed_description)
    async def case():
        forge = _FakeForge()
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="优化登录页", kind="code", idempotency_key="explicit-plain-code")
        result = await service.run(uuid.UUID(task["id"]), user=user, forgeflow_token="token")
        assert forge.created == 1
        assert result["status"] == "awaiting_approval"

    asyncio.run(case())


def test_concurrency_ceiling_applies_across_tasks():
    async def case():
        active = {"current": 0, "max": 0}
        service = _service(active=active, doc_delay=0.05)
        first_user, second_user = _user("one"), _user("two")
        first, _ = await service.create(
            user=first_user, query="总结文档 A", idempotency_key="concurrent-key-1"
        )
        second, _ = await service.create(
            user=second_user, query="总结文档 B", idempotency_key="concurrent-key-2"
        )
        await asyncio.gather(
            service.run(uuid.UUID(first["id"]), user=first_user),
            service.run(uuid.UUID(second["id"]), user=second_user),
        )
        assert active["max"] == 2

    asyncio.run(case())


def test_three_tasks_share_the_concurrency_limit():
    async def case():
        active = {"current": 0, "max": 0}
        service = _service(active=active, doc_delay=0.1)
        user = _user()
        tasks = [await service.create(user=user, query="总结文档", idempotency_key=f"limit-key-{i}") for i in range(3)]
        await asyncio.gather(*(service.run(uuid.UUID(task[0]["id"]), user=user) for task in tasks))
        assert active["max"] <= 2

    asyncio.run(case())


def test_missing_search_provider_is_an_honest_unavailable_result():
    async def case():
        service = _service(research=_FakeResearch(unavailable=True))
        user = _user()
        task, _ = await service.create(
            user=user, query="联网搜索最新规范", idempotency_key="research-key-1"
        )
        final = await service.run(uuid.UUID(task["id"]), user=user)
        assert final["status"] == "unavailable"
        assert "未配置" in final["result"]["subtasks"][0]["error"]
        assert "sources" not in final["result"]["subtasks"][0]

    asyncio.run(case())


def test_research_reports_failure_when_no_page_can_be_read():
    class UnreadableResearch(_FakeResearch):
        async def read_url(self, url):
            from app.coordination.web_research import ResearchUnavailable
            raise ResearchUnavailable("page unavailable")

    async def case():
        service = _service(research=UnreadableResearch())
        user = _user()
        task, _ = await service.create(user=user, query="联网搜索最新规范", idempotency_key="unreadable-page-1")
        result = await service.run(uuid.UUID(task["id"]), user=user)
        assert result["status"] != "completed"
        assert result["result"]["subtasks"][0]["read_failures"]

    asyncio.run(case())


def test_research_rejects_empty_page_content():
    class EmptyResearch(_FakeResearch):
        async def read_url(self, url):
            return SimpleNamespace(url=url, title="Empty", text="  ", truncated=False)

    async def case():
        service = _service(research=EmptyResearch())
        user = _user()
        task, _ = await service.create(user=user, query="联网搜索最新规范", idempotency_key="empty-page-key")
        result = await service.run(uuid.UUID(task["id"]), user=user)
        assert result["status"] == "failed"
        assert result["result"]["subtasks"][0]["read_failures"]

    asyncio.run(case())


def test_research_reports_partial_failure_when_one_page_fails():
    class PartialResearch(_FakeResearch):
        async def search(self, query):
            from app.coordination.web_research import SearchResult
            return [SearchResult(title="one", url="https://example.com/one", snippet=""),
                    SearchResult(title="two", url="https://example.com/two", snippet="")], "provider"

        async def read_url(self, url):
            if url.endswith("two"):
                from app.coordination.web_research import ResearchUnavailable
                raise ResearchUnavailable("unavailable")
            return await super().read_url(url)

    async def case():
        service = _service(research=PartialResearch())
        user = _user()
        task, _ = await service.create(user=user, query="联网搜索最新规范", idempotency_key="partial-page-key")
        result = await service.run(uuid.UUID(task["id"]), user=user)
        assert result["status"] == "partial_failure"
        assert result["result"]["subtasks"][0]["read_failures"]
        assert "部分完成" in result["result"]["summary"]

    asyncio.run(case())


def test_code_task_records_execution_id_and_does_not_claim_completion_at_approval():
    async def case():
        forge = _FakeForge(status="awaiting_approval")
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(
            user=user, query="修复代码并运行测试", idempotency_key="code-key-1"
        )
        final = await service.run(
            uuid.UUID(task["id"]), user=user, forgeflow_token="user-token"
        )
        assert final["execution_id"] == "ff-run-1"
        assert final["status"] == "awaiting_approval"
        assert final["result"]["subtasks"][0]["status"] == "awaiting_approval"
        assert final["result"]["subtasks"][0]["artifacts"]

    asyncio.run(case())


def test_stop_is_recorded_but_not_reported_as_finished_before_worker_observes_it():
    async def case():
        forge = _FakeForge()
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(
            user=user, query="修复代码并运行测试", idempotency_key="stop-key-1"
        )
        task_id = uuid.UUID(task["id"])
        await service.run(task_id, user=user, forgeflow_token="user-token")
        stopped = await service.stop(task_id, user, token="user-token")
        assert stopped["stop_requested"] is True
        assert stopped["status"] == "awaiting_approval"
        assert forge.aborted == 1

    asyncio.run(case())


def test_restart_marks_unfinished_tasks_interrupted():
    async def case():
        repo = MemoryTaskRepository()
        service = CoordinationService(
            repo,
            settings=_settings(),
            forgeflow_factory=lambda **kwargs: _FakeForge(),
            research_factory=lambda **kwargs: _FakeResearch(),
        )
        user = _user()
        task, _ = await service.create(
            user=user, query="总结这份文档", idempotency_key="restart-key-1"
        )
        assert await repo.recover_interrupted() == 1
        recovered = await service.get_owned(uuid.UUID(task["id"]), user)
        assert recovered["status"] == "interrupted"
        assert "未自动重放" in recovered["error"]

    asyncio.run(case())


def test_running_remote_task_remains_refreshable_and_stoppable():
    async def case():
        forge = _FakeForge(status="running")
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="修复代码并运行测试", idempotency_key="running-key-1")
        task_id = uuid.UUID(task["id"])
        result = await service.run(task_id, user=user, forgeflow_token="user-token")
        assert result["status"] == "running"
        assert result["finished_at"] is None
        await service.stop(task_id, user, token="user-token")
        assert forge.aborted == 1

    asyncio.run(case())


def test_refresh_keeps_failure_of_other_subtasks():
    async def case():
        forge = _FakeForge(status="awaiting_approval")
        service = _service(forge=forge, research=_FakeResearch(unavailable=True))
        user = _user()
        task, _ = await service.create(user=user, query="联网搜索然后改代码", idempotency_key="refresh-key-1")
        task_id = uuid.UUID(task["id"])
        plan = {**task["plan"], "forgeflow_identity": forge.identity_value}
        await service.repository.update(task_id, fields={"status": "awaiting_approval", "execution_id": "ff-run-1", "plan": plan,
            "result": {"subtasks": [{"kind": "research", "status": "unavailable"}, {"kind": "code", "status": "awaiting_approval"}]}})
        forge.status = "completed"
        result = await service.refresh(task_id, user, token="token")
        assert result["status"] == "partial_failure"

    asyncio.run(case())


def test_remote_poll_failure_stays_reconcilable():
    class BrokenPoll(_FakeForge):
        async def get_run(self, run_id):
            from app.coordination.forgeflow_client import ForgeFlowUnavailable
            raise ForgeFlowUnavailable("connection lost")

    async def case():
        service = _service(forge=BrokenPoll())
        user = _user()
        task, _ = await service.create(user=user, query="改代码", idempotency_key="poll-error-1")
        task_id = uuid.UUID(task["id"])
        result = await service.run(task_id, user=user, forgeflow_token="token")
        assert result["status"] == "running"
        assert result["execution_id"] == "ff-run-1"
        assert "connection lost" in result["result"]["subtasks"][0]["error"]

    asyncio.run(case())


def test_lost_remote_submission_response_can_be_recovered_by_request_key():
    class LostResponse(_FakeForge):
        async def create_task(self, intent, **kwargs):
            from app.coordination.forgeflow_client import ForgeFlowUnavailable
            self.request_key = kwargs["request_key"]
            raise ForgeFlowUnavailable("response lost")

        async def get_request(self, request_key):
            assert request_key == self.request_key
            return await self.get_run("ff-run-1")

    async def case():
        forge = LostResponse(status="awaiting_approval")
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="改代码", idempotency_key="lost-response-1")
        task_id = uuid.UUID(task["id"])
        await service.run(task_id, user=user, forgeflow_token="token")
        recovered = await service.refresh(task_id, user, token="token")
        assert recovered["execution_id"] == "ff-run-1"
        assert recovered["status"] == "awaiting_approval"

    asyncio.run(case())


def test_recovery_rejects_different_forgeflow_identity():
    async def case():
        forge = _FakeForge()
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="改代码", idempotency_key="identity-key-1")
        task_id = uuid.UUID(task["id"])
        await service.run(task_id, user=user, forgeflow_token="token-a")
        forge.identity_value = {"tenant_id": "ff-tenant", "user_id": "other-user"}
        result = await service.refresh(task_id, user, token="token-b")
        assert "身份" in result["reconcile_error"]

    asyncio.run(case())


def test_stop_finds_remote_run_after_lost_submission_response():
    class LostResponse(_FakeForge):
        async def create_task(self, intent, **kwargs):
            from app.coordination.forgeflow_client import ForgeFlowUnavailable
            raise ForgeFlowUnavailable("response lost")

        async def get_request(self, request_key):
            return await self.get_run("ff-run-1")

    async def case():
        forge = LostResponse(status="running")
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="改代码", idempotency_key="lost-stop-key-1")
        task_id = uuid.UUID(task["id"])
        await service.run(task_id, user=user, forgeflow_token="token")
        await service.stop(task_id, user, token="token")
        assert forge.aborted == 1

    asyncio.run(case())


def test_stop_during_remote_submission_tries_to_abort_known_request():
    class PendingResponse(_FakeForge):
        def __init__(self):
            super().__init__(status="running")
            self.accepted = asyncio.Event()
            self.release = asyncio.Event()

        async def create_task(self, intent, **kwargs):
            from app.coordination.forgeflow_client import ForgeFlowUnavailable
            self.accepted.set()
            await self.release.wait()
            raise ForgeFlowUnavailable("response lost")

        async def get_request(self, request_key):
            return await self.get_run("ff-run-1")

    async def case():
        forge = PendingResponse()
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="改代码", idempotency_key="pending-stop-key")
        task_id = uuid.UUID(task["id"])
        worker = asyncio.create_task(service.run(task_id, user=user, forgeflow_token="token"))
        await forge.accepted.wait()
        await service.stop(task_id, user, token="token")
        forge.release.set()
        await worker
        assert forge.aborted >= 1

    asyncio.run(case())


def test_stop_retries_lookup_after_submission_response_is_lost():
    class LateAcceptance(_FakeForge):
        def __init__(self):
            super().__init__(status="running")
            self.waiting = asyncio.Event()
            self.release = asyncio.Event()
            self.lookups = 0

        async def create_task(self, intent, **kwargs):
            from app.coordination.forgeflow_client import ForgeFlowUnavailable
            self.waiting.set()
            await self.release.wait()
            raise ForgeFlowUnavailable("response lost")

        async def get_request(self, request_key):
            from app.coordination.forgeflow_client import ForgeFlowRejected
            self.lookups += 1
            if self.lookups == 1:
                raise ForgeFlowRejected(404, "not accepted yet")
            return await self.get_run("ff-run-1")

    async def case():
        forge = LateAcceptance()
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="改代码", idempotency_key="late-stop-key")
        task_id = uuid.UUID(task["id"])
        worker = asyncio.create_task(service.run(task_id, user=user, forgeflow_token="token"))
        await forge.waiting.wait()
        await service.stop(task_id, user, token="token")
        forge.release.set()
        await worker
        assert forge.aborted == 1

    asyncio.run(case())


def test_interrupted_task_with_execution_id_can_still_be_stopped():
    async def case():
        forge = _FakeForge()
        service = _service(forge=forge)
        user = _user()
        task, _ = await service.create(user=user, query="改代码", idempotency_key="restart-code-1")
        task_id = uuid.UUID(task["id"])
        await service.run(task_id, user=user, forgeflow_token="token")
        await service.repository.update(task_id, fields={"status": "interrupted"})
        await service.stop(task_id, user, token="token")
        assert forge.aborted == 1

    asyncio.run(case())
