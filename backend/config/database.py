"""
Database Configuration — asyncpg connection pool
Switch DB_MODE=test (local) or DB_MODE=real (production) in .env
"""
 
import os
import logging
import asyncpg
from typing import Optional
 
logger = logging.getLogger(__name__)
 
DB_MODE = os.environ.get("DB_MODE", "test")
 
TEST_DB = {
    "host":     os.environ.get("TEST_DB_HOST",     "localhost"),
    "port":     int(os.environ.get("TEST_DB_PORT", "5432")),
    "database": os.environ.get("TEST_DB_NAME",     "bank_ai_test"),
    "user":     os.environ.get("TEST_DB_USER",     "postgres"),
    "password": os.environ.get("TEST_DB_PASSWORD", "postgres"),
}
 
REAL_DB = {
    "host":     os.environ.get("REAL_DB_HOST",     "192.168.68.101"),
    "port":     int(os.environ.get("REAL_DB_PORT", "5432")),
    "database": os.environ.get("REAL_DB_NAME",     "db_123"),
    "user":     os.environ.get("REAL_DB_USER",     "readonly_ai"),
    "password": os.environ.get("REAL_DB_PASSWORD", "readonly_ai"),
}
 
DB_CONFIG = TEST_DB if DB_MODE == "test" else REAL_DB
 
_pool: Optional[asyncpg.Pool] = None
 
 
async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            **DB_CONFIG,
            min_size=2,
            max_size=20,
            command_timeout=30,
        )
        logger.info(f"✓ DB pool ready [{DB_MODE.upper()}] → {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    return _pool
 
 
async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("DB pool closed")
 
 
async def ping_db() -> bool:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"DB ping failed: {e}")
        return False