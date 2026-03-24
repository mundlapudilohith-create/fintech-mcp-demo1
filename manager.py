"""
manager.py — Bank AI Assistant
Production-grade AgentManager.

Wires ConversationAgent on top of the existing LocalMLService.
Handles startup, shutdown, health, and all config via env vars.

Component map:
    manager.py
        ├── client/llm_service.py          ← ML + MCP routing  (UNCHANGED)
        │     ├── ml_intent_classifier.py
        │     └── mcp_client.py
        ├── agent/conversation_agent.py    ← memory + validation layer
        ├── agent/agent_memory.py          ← 60-min session memory
        └── agent/user_storage.py          ← PostgreSQL persistence

Usage in client/main.py:
    from manager import agent_manager

    # In lifespan startup:
    await agent_manager.initialize()

    # In /api/chat:
    result = await agent_manager.process(message, session_id, user_id)

    # In lifespan shutdown:
    await agent_manager.shutdown()
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from client.llm_service import claude_service        # LocalMLService — UNCHANGED
from agent.conversation_agent import ConversationAgent
from agent.user_storage import create_user_storage

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Singleton manager — owns the ConversationAgent lifecycle.
    All configuration is read from environment variables at
    initialize() time so .env is already loaded by then.
    """

    def __init__(self) -> None:
        self.agent:    Optional[ConversationAgent] = None
        self._storage = None
        self._ready   = False

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Connect storage, build agent. Call once in FastAPI lifespan
        AFTER MCP clients have connected.
        """
        logger.info("=" * 55)
        logger.info("AgentManager: initializing")
        logger.info("=" * 55)

        storage      = await self._init_postgres()
        redis_client = self._init_redis()
        memory_ttl   = int(os.getenv("AGENT_MEMORY_TTL_MINUTES", "60"))

        self.agent = ConversationAgent(
            llm_service  = claude_service,   # LocalMLService — untouched
            user_storage = storage,
            memory_ttl   = memory_ttl,
            redis_client = redis_client,
        )

        self._ready = True
        logger.info("✓ AgentManager ready")
        logger.info("=" * 55)

    async def shutdown(self) -> None:
        """Graceful shutdown — close DB pool."""
        self._ready = False
        if self._storage:
            await self._storage.close()
            logger.info("✓ PostgreSQL pool closed")

    def is_ready(self) -> bool:
        return self._ready and self.agent is not None

    # ── Main API ───────────────────────────────────────────────────────

    async def process(
        self,
        message:    str,
        session_id: str,
        user_id:    str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Process one user message end-to-end.
        Drop-in replacement for claude_service.process_query().

        Returns everything llm_service returns, plus:
            session_id, user_id, context_used,
            processing_time, memory_snapshot
        """
        self._assert_ready()

        # Pre-warm memory from user profile on first turn of a new session
        await self._prewarm_session(session_id, user_id)

        return await self.agent.process(
            message    = message,
            session_id = session_id,
            user_id    = user_id,
        )

    # ── Session helpers ────────────────────────────────────────────────

    def get_history(self, session_id: str) -> List[Dict]:
        """In-memory conversation history for a session."""
        return self.agent.get_conversation_history(session_id) if self.agent else []

    def get_context(self, session_id: str) -> Dict:
        """Known session context (company_id, gstin, intent_chain, etc.)."""
        return self.agent.get_memory_snapshot(session_id) if self.agent else {}

    def clear_session(self, session_id: str) -> None:
        """Clear in-memory session (call on logout)."""
        if self.agent:
            self.agent.clear_memory(session_id)
        # Also mark session ended in DB
        if self._storage:
            asyncio.create_task(self._storage.end_session(session_id))

    # ── Analytics ──────────────────────────────────────────────────────

    async def storage_stats(self) -> Dict:
        """Row counts per table — for /health endpoint."""
        if self._storage:
            return await self._storage.get_storage_stats()
        return {"status": "disabled"}

    async def intent_stats(self, days: int = 7) -> List[Dict]:
        """Intent frequency + confidence for the past N days."""
        if self._storage:
            return await self._storage.get_intent_stats(days=days)
        return []

    async def db_health(self) -> bool:
        """True if PostgreSQL pool is alive."""
        if self._storage:
            return await self._storage.health_check()
        return True   # no storage configured = not unhealthy

    # ── User profile pre-warming ───────────────────────────────────────

    async def _prewarm_session(self, session_id: str, user_id: str) -> None:
        """
        On the very first turn of a new session, load the user's
        persistent profile from PostgreSQL and inject company_id /
        gstin into memory — so the user doesn't have to re-state them.
        """
        if not self._storage or not self.agent:
            return

        try:
            mem = self.agent.memory.get(session_id)

            # Only prewarm if this is a fresh session (no company_id yet)
            if mem.get("company_id") or mem.get("gstin"):
                return

            profile = await self._storage.get_user_profile(user_id)
            if not profile:
                return

            updates = {}
            for key in ("company_id", "company_name", "gstin", "pan", "account_number"):
                if profile.get(key):
                    updates[key] = profile[key]

            if updates:
                self.agent.memory.bulk_update(session_id, updates)
                logger.info(
                    f"[Manager] 🔄 Pre-warmed session={session_id} "
                    f"user={user_id} keys={list(updates.keys())}"
                )
        except Exception as e:
            # Non-fatal — session just starts without pre-warmed context
            logger.warning(f"[Manager] Pre-warm failed (non-fatal): {e}")

    # ── Internal setup ─────────────────────────────────────────────────

    async def _init_postgres(self) -> Optional[Any]:
        """
        Connect to PostgreSQL using environment variables.
        Returns None (with a warning) if unavailable — app still works.

        Env vars (all optional — fall back to defaults):
            DB_HOST        default: localhost
            DB_PORT        default: 5432
            DB_NAME        default: bankdb
            DB_USER        default: postgres
            DB_PASSWORD    default: (empty)
        """
        # Skip if explicitly disabled
        if os.getenv("POSTGRES_ENABLED", "true").lower() == "false":
            logger.info("PostgreSQL disabled via POSTGRES_ENABLED=false")
            return None

        try:
            storage = create_user_storage(
                host     = os.getenv("DB_HOST",     "localhost"),
                port     = int(os.getenv("DB_PORT", "5432")),
                database = os.getenv("DB_NAME",     "bankdb"),
                user     = os.getenv("DB_USER",     "postgres"),
                password = os.getenv("DB_PASSWORD", ""),
            )
            await storage.connect()
            self._storage = storage
            logger.info("✓ PostgreSQL connected")
            return storage

        except Exception as e:
            logger.warning(
                f"⚠️  PostgreSQL unavailable ({e}) — "
                "running without persistence. Set DB_HOST / DB_PASSWORD in .env"
            )
            return None

    def _init_redis(self) -> Optional[Any]:
        """
        Connect to Redis for distributed session memory.
        Falls back to in-process memory if unavailable.

        Env vars:
            USE_REDIS      default: false
            REDIS_HOST     default: localhost
            REDIS_PORT     default: 6379
            REDIS_DB       default: 0
            REDIS_PASSWORD default: (empty)
        """
        if os.getenv("USE_REDIS", "false").lower() != "true":
            return None

        try:
            import redis as redis_lib
            client = redis_lib.Redis(
                host     = os.getenv("REDIS_HOST",     "localhost"),
                port     = int(os.getenv("REDIS_PORT", "6379")),
                db       = int(os.getenv("REDIS_DB",   "0")),
                password = os.getenv("REDIS_PASSWORD") or None,
                decode_responses = True,
                socket_timeout   = 3,
            )
            client.ping()
            logger.info("✓ Redis connected (distributed memory)")
            return client
        except Exception as e:
            logger.warning(
                f"⚠️  Redis unavailable ({e}) — "
                "using in-process memory (not suitable for multi-worker)"
            )
            return None

    def _assert_ready(self) -> None:
        if not self._ready or not self.agent:
            raise RuntimeError(
                "AgentManager is not ready. "
                "Ensure await agent_manager.initialize() is called in FastAPI lifespan."
            )


# ── Singleton ──────────────────────────────────────────────────────────
# Import and use this everywhere:
#   from manager import agent_manager
agent_manager = AgentManager()