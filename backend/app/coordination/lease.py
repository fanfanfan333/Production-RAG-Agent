"""One in-process coordinator per database; session lock survives commits."""

from sqlalchemy import text

_LEASE_KEY = 701362907828


async def acquire_coordinator_lease(engine):
    connection = await engine.connect()
    try:
        acquired = (await connection.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": _LEASE_KEY})).scalar_one()
        await connection.commit()
        if not acquired:
            raise RuntimeError("另一个协调器已连接此数据库，请使用单个协调器进程")
        return connection
    except BaseException:
        await connection.close()
        raise


async def release_coordinator_lease(connection) -> None:
    try:
        await connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _LEASE_KEY})
        await connection.commit()
    finally:
        await connection.close()
