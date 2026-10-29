"""Opt-in PostgreSQL concurrency check; requires a disposable verify database."""
import asyncio
import os
import uuid

import pytest


@pytest.mark.skipif(os.getenv("RUN_AGENT_MEMORY_DB_TESTS") != "1", reason="requires disposable PostgreSQL")
def test_concurrent_identical_memories_do_not_create_duplicate_rows():
    from sqlalchemy import delete, select
    from app.agent_runtime.memory_store import save_memory
    from app.config import get_settings
    from app.db.agent_memory_models import AgentMemory
    from app.db.postgres import get_db_session, get_engine

    assert get_settings().POSTGRES_DB.endswith("_verify")
    tenant, user = "probe-" + uuid.uuid4().hex, uuid.uuid4()

    async def run():
        async with get_engine().begin() as connection:
            await connection.run_sync(AgentMemory.__table__.create, checkfirst=True)
        start, ready = asyncio.Event(), asyncio.Event()
        waiting = 0
        async def writer():
            nonlocal waiting
            async with get_db_session() as session:
                await session.connection()
                waiting += 1
                if waiting == 12:
                    ready.set()
                await start.wait()
                row = await save_memory(session, tenant_id=tenant, user_id=user,
                                        kind="note", content="金额使用万元")
                return row.id
        tasks = [asyncio.create_task(writer()) for _ in range(12)]
        try:
            await asyncio.wait_for(ready.wait(), timeout=10)
            start.set()
            ids = await asyncio.gather(*tasks)
            async with get_db_session() as session:
                rows = list((await session.execute(select(AgentMemory).where(
                    AgentMemory.tenant_id == tenant, AgentMemory.user_id == user,
                    AgentMemory.revoked_at.is_(None)))).scalars())
                assert len(rows) == 1
                assert len(set(ids)) == 1
                assert rows[0].version == 12
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            async with get_db_session() as session:
                await session.execute(delete(AgentMemory).where(AgentMemory.tenant_id == tenant))
    asyncio.run(run())
