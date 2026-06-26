import asyncpg
from config import settings

class DatabasePool:
    _pool: asyncpg.Pool = None

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(dsn=settings.DATABASE_URL)
        return cls._pool

    @classmethod
    async def close(cls):
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None
