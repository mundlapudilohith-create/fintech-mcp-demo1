"""
agent/user_storage.py — Bank AI Assistant
Production PostgreSQL storage layer.
 
Tables:
    conversations   every user/assistant message turn
    sessions        per-session metadata (company, gstin, intent list)
    user_profiles   persistent per-user knowledge (auto pre-warms new sessions)
    intent_logs     ML prediction analytics + correction feedback loop
 
Install: pip install asyncpg
"""
 
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
 
logger = logging.getLogger(__name__)
 
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not installed — run: pip install asyncpg")
 
 
# ── Schema ─────────────────────────────────────────────────────────────
# All CREATE statements are idempotent (IF NOT EXISTS).
 
SCHEMA_SQL = """
-- conversations table managed by backend
 
-- sessions table managed by backend (port 4000)
 
-- user_profiles table managed by backend
 
CREATE TABLE IF NOT EXISTS intent_logs (
    id              BIGSERIAL       PRIMARY KEY,
    user_id         VARCHAR(128)    NOT NULL,
    session_id      VARCHAR(256)    NOT NULL,
    message         TEXT            NOT NULL,
    intent          VARCHAR(128)    NOT NULL,
    confidence      FLOAT           NOT NULL DEFAULT 0,
    is_multi_intent BOOLEAN         NOT NULL DEFAULT FALSE,
    all_intents     TEXT[]          NOT NULL DEFAULT '{}',
    entities        JSONB           NOT NULL DEFAULT '{}',
    was_corrected   BOOLEAN         NOT NULL DEFAULT FALSE,
    correct_intent  VARCHAR(128),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ilog_intent  ON intent_logs(intent);
CREATE INDEX IF NOT EXISTS idx_ilog_user    ON intent_logs(user_id);
 
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id      VARCHAR(256)    PRIMARY KEY,
    memory_data     JSONB           NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_sess_active ON agent_sessions(last_active_at DESC);
CREATE INDEX IF NOT EXISTS idx_ilog_created ON intent_logs(created_at DESC);
"""
 
 
class UserStorage:
    """
    Async PostgreSQL storage via asyncpg connection pool.
 
    Usage:
        storage = UserStorage(host=..., database=..., user=..., password=...)
        await storage.connect()          # creates pool + schema
        await storage.save_conversation({...})
        await storage.close()
 
    Or as async context manager:
        async with UserStorage(...) as storage:
            ...
    """
 
    def __init__(
        self,
        dsn:             Optional[str] = None,
        host:            str  = "localhost",
        port:            int  = 5432,
        database:        str  = "bankdb",
        user:            str  = "postgres",
        password:        str  = "",
        min_connections: int  = 2,
        max_connections: int  = 10,
        statement_timeout_ms: int = 10_000,   # 10 s per query
    ):
        # Handle Unix socket paths (e.g. /tmp for PostgreSQL socket dir)
        if dsn:
            self.dsn = dsn
        elif host and host.startswith("/"):
            # Unix socket — asyncpg uses host as socket directory
            self.dsn = None
            self.host = host
            self.port = port
            self.database = database
            self.user = user
            self.password = password
        else:
            self.dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            self.host = host
            self.port = port
            self.database = database
            self.user = user
            self.password = password
        self.min_connections      = min_connections
        self.max_connections      = max_connections
        self.statement_timeout_ms = statement_timeout_ms
        # Store connection params for Unix socket fallback
        self.host     = host
        self.port     = port
        self.database = database
        self.user     = user
        self.password = password
        self._pool: Optional[asyncpg.Pool] = None
        logger.info(f"UserStorage configured → {host}:{port}/{database}")
 
    # ── Lifecycle ──────────────────────────────────────────────────────
 
    async def connect(self) -> None:
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg is required — run: pip install asyncpg")
        if self.dsn:
            self._pool = await asyncpg.create_pool(
                dsn      = self.dsn,
                min_size = self.min_connections,
                max_size = self.max_connections,
                command_timeout = 30,
            )
        else:
            # Unix socket connection
            self._pool = await asyncpg.create_pool(
                host     = self.host,
                port     = self.port,
                database = self.database,
                user     = self.user,
                password = self.password,
                min_size = self.min_connections,
                max_size = self.max_connections,
                command_timeout = 30,
            )
        await self._ensure_schema()
        logger.info("✓ UserStorage pool ready")
 
    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("UserStorage pool closed")
 
    async def __aenter__(self):
        await self.connect()
        return self
 
    async def __aexit__(self, *_):
        await self.close()
 
    def _require_pool(self) -> None:
        if not self._pool:
            raise RuntimeError("UserStorage not connected — call await connect() first")
 
    async def _ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("✓ DB schema verified")
 
    # ── Conversations ──────────────────────────────────────────────────
 
    async def save_conversation(self, data) -> Optional[int]:
        """Stubbed — conversations table schema managed by backend."""
        return None

    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> List[Dict]:
        """Return full conversation history for a session, oldest first."""
        self._require_pool()
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT role, content, intent, entities,
                           tool_name, processing_time, created_at
                    FROM conversations
                    WHERE session_id = $1
                    ORDER BY created_at ASC
                    LIMIT $2
                    """,
                    session_id, limit,
                )
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[Storage] get_conversation_history error: {e}")
            return []
 
    # ── Sessions ───────────────────────────────────────────────────────
 
    async def _upsert_session(self, conn, data: Dict) -> None:
        """Stubbed — backend manages sessions table."""
        pass

    
    async def end_session(self, session_id: str) -> None:
        """Stubbed — backend manages sessions table."""
        pass

    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Stubbed — backend manages sessions table."""
        return None

    
    async def save_session_memory(self, session_id: str, memory_data: Dict[str, Any]) -> None:
        """
        Persist full agent memory dict to PostgreSQL.
        Called after every verify-otp and after every chat turn.
        This ensures memory survives server restarts.
        """
        self._require_pool()
        try:
            # Filter out non-serializable values
            safe = {k: v for k, v in memory_data.items()
                    if k not in ("conversation_history", "last_accessed", "created_at")}
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent_sessions (session_id, memory_data, last_active_at)
                    VALUES ($1, $2::jsonb, NOW())
                    ON CONFLICT (session_id) DO UPDATE SET
                        memory_data    = $2::jsonb,
                        last_active_at = NOW()
                    """,
                    session_id, json.dumps(safe),
                )
        except Exception as e:
            logger.error(f"[Storage] save_session_memory error: {e}")
 
    async def load_session_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load agent memory from PostgreSQL.
        Called at the start of every chat request to restore memory after restarts.
        """
        self._require_pool()
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT memory_data FROM agent_sessions WHERE session_id = $1",
                    session_id,
                )
            if row and row["memory_data"]:
                data = row["memory_data"]
                return dict(data) if isinstance(data, dict) else json.loads(data)
            return None
        except Exception as e:
            logger.error(f"[Storage] load_session_memory error: {e}")
            return None
 
    async def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Stubbed — backend manages sessions table."""
        return []

    
    async def _upsert_user_profile(
        self, conn: asyncpg.Connection, data: Dict
    ) -> None:
        """
        Insert or update user profile.
        Stores company_id / gstin so next session can pre-warm memory
        without the user re-stating them.
        """
        await conn.execute(
            """
            INSERT INTO user_profiles
                (user_id, company_id, company_name, gstin,
                 total_messages, last_seen_at)
            VALUES ($1, $2, $3, $4, 1, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                total_messages = user_profiles.total_messages + 1,
                last_seen_at   = NOW(),
                updated_at     = NOW(),
                company_id     = COALESCE($2, user_profiles.company_id),
                company_name   = COALESCE($3, user_profiles.company_name),
                gstin          = COALESCE($4, user_profiles.gstin)
            """,
            data["user_id"],
            data.get("company_id"), data.get("company_name"), data.get("gstin"),
        )
 
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """
        Load persistent user profile.
        Call at session start to pre-warm agent memory — user won't
        need to re-state company_id / gstin every session.
        """
        self._require_pool()
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM user_profiles WHERE user_id = $1", user_id
                )
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"[Storage] get_user_profile error: {e}")
            return None
 
    async def update_user_profile(
        self, user_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update specific user profile fields. Only whitelisted columns accepted."""
        self._require_pool()
        allowed = {"company_id", "company_name", "gstin", "pan", "account_number", "preferences"}
        updates = {k: v for k, v in updates.items() if k in allowed}
        if not updates:
            return True
        try:
            set_clauses = []
            values      = [user_id]
            for i, (col, val) in enumerate(updates.items(), start=2):
                if col == "preferences":
                    set_clauses.append(f"{col} = ${i}::jsonb")
                    values.append(json.dumps(val))
                else:
                    set_clauses.append(f"{col} = ${i}")
                    values.append(val)
            set_clauses.append("updated_at = NOW()")
            sql = f"UPDATE user_profiles SET {', '.join(set_clauses)} WHERE user_id = $1"
            async with self._pool.acquire() as conn:
                await conn.execute(sql, *values)
            return True
        except Exception as e:
            logger.error(f"[Storage] update_user_profile error: {e}")
            return False
 
    # ── Intent Analytics ───────────────────────────────────────────────
 
    async def log_intent(self, data: Dict[str, Any]) -> None:
        """Log one ML prediction. Used for analytics and retraining."""
        self._require_pool()
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO intent_logs
                        (user_id, session_id, message, intent, confidence,
                         is_multi_intent, all_intents, entities)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb)
                    """,
                    data["user_id"], data["session_id"],
                    data["message"], data["intent"],
                    float(data.get("confidence") or 0),
                    bool(data.get("is_multi_intent")),
                    list(data.get("all_intents") or []),
                    json.dumps(data.get("entities") or {}),
                )
        except Exception as e:
            logger.error(f"[Storage] log_intent error: {e}")
 
    async def mark_intent_correction(
        self, log_id: int, correct_intent: str
    ) -> None:
        """
        Mark a prediction as wrong and record the correct intent.
        Feeds the retraining pipeline.
        """
        self._require_pool()
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE intent_logs
                    SET was_corrected = TRUE, correct_intent = $2
                    WHERE id = $1
                    """,
                    log_id, correct_intent,
                )
        except Exception as e:
            logger.error(f"[Storage] mark_intent_correction error: {e}")
 
    async def get_intent_stats(self, days: int = 7) -> List[Dict]:
        """Intent frequency + avg confidence for the past N days."""
        self._require_pool()
        try:
            since = datetime.now() - timedelta(days=days)
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        intent,
                        COUNT(*)                                AS count,
                        ROUND(AVG(confidence)::numeric, 3)     AS avg_confidence,
                        ROUND(
                            SUM(was_corrected::int)::numeric
                            / NULLIF(COUNT(*), 0), 3
                        )                                       AS correction_rate
                    FROM intent_logs
                    WHERE created_at >= $1
                    GROUP BY intent
                    ORDER BY count DESC
                    """,
                    since,
                )
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[Storage] get_intent_stats error: {e}")
            return []
 
    # ── Health + Stats ─────────────────────────────────────────────────
 
    async def health_check(self) -> bool:
        """Ping DB — True if healthy."""
        if not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
 
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Row counts for all tables — used in /health endpoint."""
        if not self._pool:
            return {"status": "disconnected"}
        try:
            async with self._pool.acquire() as conn:
                stats: Dict[str, Any] = {"status": "connected"}
                for table in ("conversations", "sessions", "user_profiles", "intent_logs"):
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = int(count)
            return stats
        except Exception as e:
            logger.error(f"[Storage] get_storage_stats error: {e}")
            return {"status": "error", "detail": str(e)}
 
 
# ── Factory ────────────────────────────────────────────────────────────
 
def create_user_storage(
    dsn:      Optional[str] = None,
    host:     str = "localhost",
    port:     int = 5432,
    database: str = "bankdb",
    user:     str = "postgres",
    password: str = "",
) -> UserStorage:
    """Factory — call `await storage.connect()` before use."""
    return UserStorage(
        dsn=dsn, host=host, port=port,
        database=database, user=user, password=password,
    )