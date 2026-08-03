# Async SQLAlchemy session factory.
# Used as a FastAPI dependency via get_db().
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import settings
from src.db.models import Base

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Create (once) the async engine and expose it to callers.

    The engine is made lazily so imports of this module never touch the
    database or require DATABASE_URL to be present at import time.
    """
    global _engine, _sessionmaker
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        _sessionmaker = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _engine


def get_session() -> AsyncSession:
    """Return a fresh AsyncSession bound to the current engine."""
    return get_engine_sessionmaker()()


def get_engine_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Lazily build (or return) the singleton session factory."""
    get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yield a session that is always closed/rolled back."""
    session = get_session()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """Create all tables. Intended for tests/dev; use Alembic in production."""
    from src.db.models import Chunk, Document  # noqa: F401  (register on Base)

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Close all pooled connections (for app shutdown)."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None