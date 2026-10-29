"""Persistence boundary for coordination tasks.

The protocol keeps the service testable without PostgreSQL. The production
implementation is the SQLAlchemy repository; the in-memory one is used only by
tests and is never selected by the application.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _task_dict(row) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "tenant_id": row.tenant_id,
        "user_id": str(row.user_id),
        "idempotency_key": row.idempotency_key,
        "request_text": row.request_text,
        "requested_kind": row.requested_kind,
        "status": row.status,
        "plan": dict(row.plan or {}),
        "result": dict(row.result or {}),
        "error": row.error,
        "execution_id": row.execution_id,
        "conversation_id": str(row.conversation_id) if row.conversation_id else None,
        "stop_requested": bool(row.stop_requested),
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        "finished_at": _iso(row.finished_at),
    }


class TaskRepository(Protocol):
    async def get_link(self, tenant_id: str, user_id: uuid.UUID) -> dict | None: ...
    async def set_link(self, tenant_id: str, user_id: uuid.UUID, identity: dict) -> None: ...
    async def remove_link(self, tenant_id: str, user_id: uuid.UUID) -> None: ...
    async def create_or_get(
        self,
        *,
        tenant_id: str,
        user_id: uuid.UUID,
        idempotency_key: str,
        request_text: str,
        requested_kind: str,
        plan: dict,
    ) -> tuple[dict, bool]: ...

    async def get(self, task_id: uuid.UUID) -> dict | None: ...

    async def claim_start(self, task_id: uuid.UUID) -> bool: ...

    async def list_for_owner(
        self, tenant_id: str, user_id: uuid.UUID, limit: int = 30
    ) -> list[dict]: ...

    async def update(
        self, task_id: uuid.UUID, *, fields: dict[str, Any], event: dict | None = None
    ) -> dict: ...

    async def append_event(
        self, task_id: uuid.UUID, event_type: str, data: dict
    ) -> dict: ...

    async def list_events(
        self, task_id: uuid.UUID, after_seq: int = 0
    ) -> list[dict]: ...

    async def recover_interrupted(self) -> int: ...


class SqlTaskRepository:
    def __init__(self) -> None:
        # Imported lazily so the planner/web research unit tests stay pure.
        from app.db.coordination_models import CoordinationTask

        self._model = CoordinationTask

    async def get_link(self, tenant_id: str, user_id: uuid.UUID) -> dict | None:
        from app.db.coordination_models import ForgeFlowIdentityLink
        from app.db.postgres import get_db_session
        async with get_db_session() as session:
            row = await session.get(ForgeFlowIdentityLink, (tenant_id, user_id))
            return {"tenant_id": row.forgeflow_tenant_id, "user_id": row.forgeflow_user_id} if row else None

    @asynccontextmanager
    async def locked_link(self, tenant_id: str, user_id: uuid.UUID):
        from app.db.coordination_models import ForgeFlowIdentityLink
        from app.db.postgres import get_db_session
        async with get_db_session() as session:
            row = await session.get(ForgeFlowIdentityLink, (tenant_id, user_id), with_for_update=True)
            yield {"tenant_id": row.forgeflow_tenant_id, "user_id": row.forgeflow_user_id} if row else None

    async def set_link(self, tenant_id: str, user_id: uuid.UUID, identity: dict) -> None:
        from app.db.coordination_models import ForgeFlowIdentityLink
        from app.db.postgres import get_db_session
        async with get_db_session() as session:
            stmt = pg_insert(ForgeFlowIdentityLink).values(
                tenant_id=tenant_id, user_id=user_id,
                forgeflow_tenant_id=identity["tenant_id"], forgeflow_user_id=identity["user_id"])
            stmt = stmt.on_conflict_do_update(index_elements=["tenant_id", "user_id"], set_={
                "forgeflow_tenant_id": identity["tenant_id"], "forgeflow_user_id": identity["user_id"]})
            await session.execute(stmt)

    async def remove_link(self, tenant_id: str, user_id: uuid.UUID) -> None:
        from app.db.coordination_models import ForgeFlowIdentityLink
        from app.db.postgres import get_db_session
        async with get_db_session() as session:
            await session.execute(delete(ForgeFlowIdentityLink).where(
                ForgeFlowIdentityLink.tenant_id == tenant_id, ForgeFlowIdentityLink.user_id == user_id))

    async def create_or_get(self, **kwargs) -> tuple[dict, bool]:
        from app.db.coordination_models import CoordinationTask
        from app.db.postgres import get_db_session

        async with get_db_session() as session:
            stmt = (
                pg_insert(CoordinationTask)
                .values(
                    id=uuid.uuid4(),
                    tenant_id=kwargs["tenant_id"],
                    user_id=kwargs["user_id"],
                    idempotency_key=kwargs["idempotency_key"],
                    request_text=kwargs["request_text"],
                    requested_kind=kwargs["requested_kind"],
                    status="queued",
                    plan=kwargs["plan"],
                    result={},
                )
                .on_conflict_do_nothing(
                    constraint="uq_coordination_tasks_owner_idempotency"
                )
                .returning(CoordinationTask.id)
            )
            created_id = (await session.execute(stmt)).scalar_one_or_none()
            if created_id is not None:
                row = await session.get(CoordinationTask, created_id)
                return _task_dict(row), True
            existing = (
                await session.execute(
                    select(CoordinationTask).where(
                        CoordinationTask.tenant_id == kwargs["tenant_id"],
                        CoordinationTask.user_id == kwargs["user_id"],
                        CoordinationTask.idempotency_key == kwargs["idempotency_key"],
                    )
                )
            ).scalar_one()
            return _task_dict(existing), False

    async def get(self, task_id: uuid.UUID) -> dict | None:
        from app.db.coordination_models import CoordinationTask
        from app.db.postgres import get_db_session

        async with get_db_session() as session:
            row = await session.get(CoordinationTask, task_id)
            return _task_dict(row) if row else None

    async def claim_start(self, task_id: uuid.UUID) -> bool:
        from app.db.coordination_models import CoordinationTask, CoordinationTaskEvent
        from app.db.postgres import get_db_session

        async with get_db_session() as session:
            row = await session.get(CoordinationTask, task_id, with_for_update=True)
            if row is None or row.status != "queued":
                return False
            now = datetime.now(timezone.utc)
            row.status = "running"
            row.updated_at = now
            seq = (await session.execute(
                select(func.coalesce(func.max(CoordinationTaskEvent.seq), 0)).where(
                    CoordinationTaskEvent.task_id == task_id
                )
            )).scalar_one() + 1
            session.add(CoordinationTaskEvent(task_id=task_id, seq=seq,
                        event_type="task.started", data={"started_at": now.isoformat()}))
            await session.flush()
            return True

    async def list_for_owner(
        self, tenant_id: str, user_id: uuid.UUID, limit: int = 30
    ) -> list[dict]:
        from app.db.coordination_models import CoordinationTask
        from app.db.postgres import get_db_session

        async with get_db_session() as session:
            rows = (
                await session.execute(
                    select(CoordinationTask)
                    .where(
                        CoordinationTask.tenant_id == tenant_id,
                        CoordinationTask.user_id == user_id,
                    )
                    .order_by(CoordinationTask.created_at.desc())
                    .limit(max(1, min(100, limit)))
                )
            ).scalars()
            return [_task_dict(row) for row in rows]

    async def update(
        self, task_id: uuid.UUID, *, fields: dict[str, Any], event: dict | None = None
    ) -> dict:
        from app.db.coordination_models import CoordinationTask, CoordinationTaskEvent
        from app.db.postgres import get_db_session

        async with get_db_session() as session:
            row = await session.get(CoordinationTask, task_id, with_for_update=True)
            if row is None:
                raise LookupError("task not found")
            allowed = {
                "plan",
                "status",
                "result",
                "error",
                "execution_id",
                "conversation_id",
                "stop_requested",
                "finished_at",
            }
            for key, value in fields.items():
                if key in allowed:
                    setattr(row, key, value)
            row.updated_at = datetime.now(timezone.utc)
            if event:
                seq = (
                    await session.execute(
                        select(func.coalesce(func.max(CoordinationTaskEvent.seq), 0)).where(
                            CoordinationTaskEvent.task_id == task_id
                        )
                    )
                ).scalar_one() + 1
                session.add(
                    CoordinationTaskEvent(
                        task_id=task_id,
                        seq=seq,
                        event_type=str(event["type"]),
                        data=dict(event.get("data") or {}),
                    )
                )
            await session.flush()
            return _task_dict(row)

    async def append_event(
        self, task_id: uuid.UUID, event_type: str, data: dict
    ) -> dict:
        from app.db.coordination_models import CoordinationTask, CoordinationTaskEvent
        from app.db.postgres import get_db_session

        async with get_db_session() as session:
            row = await session.get(CoordinationTask, task_id, with_for_update=True)
            if row is None:
                raise LookupError("task not found")
            seq = (
                await session.execute(
                    select(func.coalesce(func.max(CoordinationTaskEvent.seq), 0)).where(
                        CoordinationTaskEvent.task_id == task_id
                    )
                )
            ).scalar_one() + 1
            event = CoordinationTaskEvent(
                task_id=task_id, seq=seq, event_type=event_type, data=data
            )
            session.add(event)
            await session.flush()
            return {
                "seq": seq,
                "type": event_type,
                "data": data,
                "created_at": _iso(event.created_at),
            }

    async def list_events(
        self, task_id: uuid.UUID, after_seq: int = 0
    ) -> list[dict]:
        from app.db.coordination_models import CoordinationTaskEvent
        from app.db.postgres import get_db_session

        async with get_db_session() as session:
            rows = (
                await session.execute(
                    select(CoordinationTaskEvent)
                    .where(
                        CoordinationTaskEvent.task_id == task_id,
                        CoordinationTaskEvent.seq > max(0, after_seq),
                    )
                    .order_by(CoordinationTaskEvent.seq)
                )
            ).scalars()
            return [
                {
                    "seq": row.seq,
                    "type": row.event_type,
                    "data": dict(row.data or {}),
                    "created_at": _iso(row.created_at),
                }
                for row in rows
            ]

    async def recover_interrupted(self) -> int:
        """Mark queued/running tasks interrupted after a process restart.

        An in-process worker cannot survive the restart. Re-running it blindly
        could duplicate a container or a write, so the honest outcome is
        ``interrupted`` until the owner explicitly retries with a new key.
        """
        from app.db.coordination_models import CoordinationTask
        from app.db.postgres import get_db_session

        async with get_db_session() as session:
            rows = (
                await session.execute(
                    select(CoordinationTask).where(
                        CoordinationTask.status.in_(("queued", "running"))
                    )
                )
            ).scalars()
            count = 0
            now = datetime.now(timezone.utc)
            for row in rows:
                row.status = "interrupted"
                row.error = "服务重启时任务尚未完成；未自动重放，请核对后重新提交"
                row.finished_at = now
                row.updated_at = now
                count += 1
            return count


class MemoryTaskRepository:
    """Test-only repository with the same contract (not used in production)."""

    def __init__(self) -> None:
        self.tasks: dict[uuid.UUID, dict] = {}
        self.events: dict[uuid.UUID, list[dict]] = {}
        self.links: dict[tuple[str, uuid.UUID], dict] = {}
        self._link_lock = asyncio.Lock()

    @asynccontextmanager
    async def locked_link(self, tenant_id: str, user_id: uuid.UUID):
        async with self._link_lock:
            yield await self.get_link(tenant_id, user_id)

    async def get_link(self, tenant_id: str, user_id: uuid.UUID) -> dict | None:
        link = self.links.get((tenant_id, user_id))
        return dict(link) if link else None

    async def set_link(self, tenant_id: str, user_id: uuid.UUID, identity: dict) -> None:
        from sqlalchemy.exc import IntegrityError
        async with self._link_lock:
            key = (tenant_id, user_id)
            if any(other != key and value == identity for other, value in self.links.items()):
                raise IntegrityError("duplicate identity", {}, Exception("duplicate"))
            self.links[key] = dict(identity)

    async def remove_link(self, tenant_id: str, user_id: uuid.UUID) -> None:
        async with self._link_lock:
            self.links.pop((tenant_id, user_id), None)

    async def create_or_get(self, **kwargs) -> tuple[dict, bool]:
        for task in self.tasks.values():
            if (
                task["tenant_id"] == kwargs["tenant_id"]
                and task["user_id"] == str(kwargs["user_id"])
                and task["idempotency_key"] == kwargs["idempotency_key"]
            ):
                return dict(task), False
        now = datetime.now(timezone.utc).isoformat()
        task_id = uuid.uuid4()
        task = {
            "id": str(task_id),
            "tenant_id": kwargs["tenant_id"],
            "user_id": str(kwargs["user_id"]),
            "idempotency_key": kwargs["idempotency_key"],
            "request_text": kwargs["request_text"],
            "requested_kind": kwargs["requested_kind"],
            "status": "queued",
            "plan": kwargs["plan"],
            "result": {},
            "error": None,
            "execution_id": None,
            "conversation_id": None,
            "stop_requested": False,
            "created_at": now,
            "updated_at": now,
            "finished_at": None,
        }
        self.tasks[task_id] = task
        self.events[task_id] = []
        return dict(task), True

    async def get(self, task_id: uuid.UUID) -> dict | None:
        task = self.tasks.get(task_id)
        return dict(task) if task else None

    async def claim_start(self, task_id: uuid.UUID) -> bool:
        task = self.tasks.get(task_id)
        if task is None or task["status"] != "queued":
            return False
        task["status"] = "running"
        task["updated_at"] = datetime.now(timezone.utc).isoformat()
        await self.append_event(task_id, "task.started", {"started_at": task["updated_at"]})
        return True

    async def list_for_owner(
        self, tenant_id: str, user_id: uuid.UUID, limit: int = 30
    ) -> list[dict]:
        rows = [
            task
            for task in self.tasks.values()
            if task["tenant_id"] == tenant_id and task["user_id"] == str(user_id)
        ]
        rows.sort(key=lambda item: item["created_at"], reverse=True)
        return [dict(row) for row in rows[:limit]]

    async def update(
        self, task_id: uuid.UUID, *, fields: dict[str, Any], event: dict | None = None
    ) -> dict:
        task = self.tasks[task_id]
        for key in (
            "plan",
            "status",
            "result",
            "error",
            "execution_id",
            "conversation_id",
            "stop_requested",
            "finished_at",
        ):
            if key in fields:
                task[key] = fields[key]
        task["updated_at"] = datetime.now(timezone.utc).isoformat()
        if event:
            await self.append_event(task_id, str(event["type"]), dict(event.get("data") or {}))
        return dict(task)

    async def append_event(
        self, task_id: uuid.UUID, event_type: str, data: dict
    ) -> dict:
        items = self.events.setdefault(task_id, [])
        event = {
            "seq": len(items) + 1,
            "type": event_type,
            "data": data,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        items.append(event)
        return dict(event)

    async def list_events(
        self, task_id: uuid.UUID, after_seq: int = 0
    ) -> list[dict]:
        return [dict(item) for item in self.events.get(task_id, []) if item["seq"] > after_seq]

    async def recover_interrupted(self) -> int:
        count = 0
        for task in self.tasks.values():
            if task["status"] in {"queued", "running"}:
                task["status"] = "interrupted"
                task["error"] = "服务重启时任务尚未完成；未自动重放，请核对后重新提交"
                task["finished_at"] = datetime.now(timezone.utc).isoformat()
                count += 1
        return count
