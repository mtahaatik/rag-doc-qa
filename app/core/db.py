"""
Async Postgres connection pool. Uses psycopg3 + pgvector's async helper to
register the vector type so we can pass numpy arrays / lists directly.
"""
import logging

from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

log = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None


async def _configure_connection(conn) -> None:
    # Register the vector type on every new connection so SELECTs return
    # numpy arrays and INSERTs accept lists/arrays directly.
    await register_vector_async(conn)


async def init_pool() -> None:
    global _pool
    if _pool is not None:
        return
    _pool = AsyncConnectionPool(
        conninfo=settings.db_dsn,
        min_size=1,
        max_size=10,
        configure=_configure_connection,
        open=False,
    )
    await _pool.open()
    log.info("db pool opened")


async def close_pool() -> None:
    global _pool
    if _pool is None:
        return
    await _pool.close()
    _pool = None


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("db pool not initialized; call init_pool() first")
    return _pool
