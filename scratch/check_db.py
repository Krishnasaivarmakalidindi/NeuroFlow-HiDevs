import asyncio
import os
import sys

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from db.pool import DatabasePool

async def check():
    pool = await DatabasePool.get_pool()
    async with pool.acquire() as conn:
        tables = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        print("Tables:", [t['table_name'] for t in tables])
        
        pipelines = await conn.fetch("SELECT * FROM pipelines;")
        print("Pipelines:", pipelines)
        
        runs = await conn.fetch("SELECT * FROM pipeline_runs;")
        print("Pipeline runs:", runs)

if __name__ == "__main__":
    asyncio.run(check())
