"""Concurrency checks require a disposable PostgreSQL database."""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.coordination.repository import SqlTaskRepository
from app.db.coordination_models import CoordinationTask, CoordinationTaskEvent


@pytest.mark.skipif(not os.getenv("IDEMPOTENCY_TEST_PG_DSN"), reason="disposable PostgreSQL not configured")
@pytest.mark.asyncio
async def test_concurrent_events_keep_unique_order(monkeypatch):
    engine = create_async_engine(os.environ["IDEMPOTENCY_TEST_PG_DSN"])
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def session_scope():
        async with factory() as session:
            async with session.begin():
                yield session

    monkeypatch.setattr("app.db.postgres.get_db_session", session_scope)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda sync: CoordinationTask.__table__.create(sync, checkfirst=True))
            await conn.run_sync(lambda sync: CoordinationTaskEvent.__table__.create(sync, checkfirst=True))
        repo = SqlTaskRepository()
        task, _ = await repo.create_or_get(tenant_id="tenant", user_id=uuid.uuid4(),
                                            idempotency_key="test-key", request_text="x",
                                            requested_kind="code", plan={})
        task_id = uuid.UUID(task["id"])
        await asyncio.gather(*(repo.append_event(task_id, "test", {"n": i}) for i in range(12)))
        events = await repo.list_events(task_id)
        assert [item["seq"] for item in events] == list(range(1, 13))
    finally:
        await engine.dispose()


@pytest.mark.skipif(not os.getenv("IDEMPOTENCY_TEST_PG_DSN"), reason="disposable PostgreSQL not configured")
@pytest.mark.asyncio
async def test_coordinator_lease_excludes_another_process():
    from app.coordination.lease import acquire_coordinator_lease, release_coordinator_lease

    engine = create_async_engine(os.environ["IDEMPOTENCY_TEST_PG_DSN"])
    try:
        lease = await acquire_coordinator_lease(engine)
        with pytest.raises(RuntimeError, match="协调器"):
            await acquire_coordinator_lease(engine)
        await release_coordinator_lease(lease)
        replacement = await acquire_coordinator_lease(engine)
        await release_coordinator_lease(replacement)
    finally:
        await engine.dispose()


@pytest.mark.skipif(not os.getenv("IDEMPOTENCY_TEST_PG_DSN"), reason="disposable PostgreSQL not configured")
@pytest.mark.asyncio
async def test_concurrent_start_claim_has_single_winner(monkeypatch):
    engine = create_async_engine(os.environ["IDEMPOTENCY_TEST_PG_DSN"])
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def session_scope():
        async with factory() as session:
            async with session.begin():
                yield session

    monkeypatch.setattr("app.db.postgres.get_db_session", session_scope)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda sync: CoordinationTask.__table__.create(sync, checkfirst=True))
            await conn.run_sync(lambda sync: CoordinationTaskEvent.__table__.create(sync, checkfirst=True))
        repo = SqlTaskRepository()
        task, _ = await repo.create_or_get(tenant_id="tenant", user_id=uuid.uuid4(),
                                            idempotency_key="claim-key", request_text="x",
                                            requested_kind="document", plan={})
        task_id = uuid.UUID(task["id"])
        assert sum(await asyncio.gather(*(repo.claim_start(task_id) for _ in range(12)))) == 1
        assert (await repo.get(task_id))["status"] == "running"
        assert [event["type"] for event in await repo.list_events(task_id)] == ["task.started"]
    finally:
        await engine.dispose()


@pytest.mark.skipif(not os.getenv("IDEMPOTENCY_TEST_PG_DSN"), reason="disposable PostgreSQL not configured")
@pytest.mark.asyncio
async def test_link_revocation_waits_for_inflight_handoff(monkeypatch):
    from app.db.coordination_models import ForgeFlowIdentityLink
    engine = create_async_engine(os.environ["IDEMPOTENCY_TEST_PG_DSN"])
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def session_scope():
        async with factory() as session:
            async with session.begin():
                yield session

    monkeypatch.setattr("app.db.postgres.get_db_session", session_scope)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda sync: ForgeFlowIdentityLink.__table__.create(sync, checkfirst=True))
        repo = SqlTaskRepository()
        owner = uuid.uuid4()
        identity = {"tenant_id": "ff-t", "user_id": str(owner)}
        await repo.set_link("rag-t", owner, identity)
        async with repo.locked_link("rag-t", owner) as linked:
            assert linked == identity
            revocation = asyncio.create_task(repo.remove_link("rag-t", owner))
            await asyncio.sleep(0.1)
            assert not revocation.done()
        await asyncio.wait_for(revocation, timeout=2)
        assert await repo.get_link("rag-t", owner) is None
    finally:
        await engine.dispose()
