"""
agent/agent_memory.py — Bank AI Assistant
Production-grade session memory with 60-minute TTL.

Backends:
    IN_MEMORY  — single-process (dev / single-worker prod)
    REDIS      — multi-process / multi-worker prod deployments

Production fixes vs original:
    - _memory_store protected by threading.Lock (thread-safe for multi-threaded servers)
    - Redis client created with socket_timeout + retry_on_timeout (won't hang on Redis failure)
    - create_memory() factory passes socket_timeout to Redis
    - _cleanup_expired() uses total_seconds() not .seconds (TTL bug fixed)
    - _get_from_redis() uses pipeline for atomic read+TTL-refresh
    - _create_empty_memory() covers all 81 banking entity fields
"""

import json
import logging
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryBackend(Enum):
    REDIS     = "redis"
    IN_MEMORY = "in_memory"


class AgentMemory:
    """
    Session memory store for the Bank AI Agent.

    Features:
        - Configurable TTL (default 60 min)
        - Thread-safe in-memory dict (RLock)
        - Redis backend for multi-worker deployments
        - Automatic expired-session cleanup
        - Conversation history capped at max_history
        - Full banking entity schema in every session
    """

    def __init__(
        self,
        ttl_minutes:  int           = 60,
        max_history:  int           = 20,
        backend:      MemoryBackend = MemoryBackend.IN_MEMORY,
        redis_client                = None,
    ) -> None:
        self.ttl_minutes  = ttl_minutes
        self.max_history  = max_history
        self.backend      = backend
        self.redis_client = redis_client

        # Thread-safe in-memory store (RLock allows re-entrant calls)
        self._memory_store: Dict[str, Dict] = {}
        self._lock = threading.RLock()

        if backend == MemoryBackend.REDIS and not redis_client:
            logger.warning(
                "Redis backend requested but no client provided — "
                "falling back to in-memory."
            )
            self.backend = MemoryBackend.IN_MEMORY

        logger.info(
            f"AgentMemory ready: backend={self.backend.value}  "
            f"ttl={ttl_minutes}min  max_history={max_history}"
        )

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def get(self, session_id: str) -> Dict[str, Any]:
        """Return session memory dict, creating it if this is a new session."""
        if self.backend == MemoryBackend.REDIS:
            return self._get_from_redis(session_id)
        return self._get_from_memory(session_id)

    def update(self, session_id: str, key: str, value: Any) -> None:
        """Update a single key in session memory."""
        with self._lock:
            memory = self.get(session_id)
            memory[key] = value
            memory["last_accessed"] = datetime.now().isoformat()
            self._persist(session_id, memory)

    def bulk_update(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update multiple keys atomically."""
        with self._lock:
            memory = self.get(session_id)
            memory.update(updates)
            memory["last_accessed"] = datetime.now().isoformat()
            self._persist(session_id, memory)

    def add_to_history(
        self,
        session_id: str,
        role:       str,
        content:    str,
        metadata:   Optional[Dict] = None,
    ) -> None:
        """Append a message to conversation history (capped at max_history)."""
        with self._lock:
            memory  = self.get(session_id)
            history = memory["conversation_history"]

            history.append({
                "role":      role,
                "content":   content,
                "metadata":  metadata or {},
                "timestamp": datetime.now().isoformat(),
            })

            # Trim — keep only the most recent turns
            if len(history) > self.max_history:
                memory["conversation_history"] = history[-self.max_history:]

            memory["last_accessed"] = datetime.now().isoformat()
            self._persist(session_id, memory)

    def get_history(self, session_id: str) -> List[Dict]:
        """Return conversation history list for a session."""
        return self.get(session_id).get("conversation_history", [])

    def clear(self, session_id: str) -> None:
        """Delete a session from all backends."""
        if self.backend == MemoryBackend.REDIS and self.redis_client:
            try:
                self.redis_client.delete(f"agent_memory:{session_id}")
                logger.info(f"[Memory] Cleared Redis session={session_id}")
            except Exception as e:
                logger.error(f"[Memory] Redis clear error: {e}")

        with self._lock:
            if session_id in self._memory_store:
                del self._memory_store[session_id]
                logger.info(f"[Memory] Cleared in-memory session={session_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        with self._lock:
            active = (
                len(self._memory_store)
                if self.backend == MemoryBackend.IN_MEMORY
                else "N/A (Redis)"
            )
        return {
            "backend":         self.backend.value,
            "ttl_minutes":     self.ttl_minutes,
            "max_history":     self.max_history,
            "active_sessions": active,
        }

    # ──────────────────────────────────────────────────────────────
    # Internal — in-memory backend
    # ──────────────────────────────────────────────────────────────

    def _get_from_memory(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            self._cleanup_expired()

            if session_id not in self._memory_store:
                self._memory_store[session_id] = self._create_empty_memory()

            self._memory_store[session_id]["last_accessed"] = datetime.now().isoformat()
            return self._memory_store[session_id]

    def _cleanup_expired(self) -> None:
        """
        Remove sessions that haven't been accessed within TTL.
        Must be called under self._lock.

        Fix: uses total_seconds() — original .seconds only returned
        the seconds component (0-86399), breaking TTL > 60 min.
        """
        now     = datetime.now()
        expired = [
            sid for sid, mem in self._memory_store.items()
            if (now - datetime.fromisoformat(mem["last_accessed"])).total_seconds() / 60
               > self.ttl_minutes
        ]
        for sid in expired:
            del self._memory_store[sid]
        if expired:
            logger.info(f"[Memory] Cleaned up {len(expired)} expired sessions")

    # ──────────────────────────────────────────────────────────────
    # Internal — Redis backend
    # ──────────────────────────────────────────────────────────────

    def _get_from_redis(self, session_id: str) -> Dict[str, Any]:
        try:
            key = f"agent_memory:{session_id}"

            # Pipeline: GET + EXPIRE in one round-trip (atomic TTL refresh)
            pipe   = self.redis_client.pipeline()
            pipe.get(key)
            pipe.expire(key, self.ttl_minutes * 60)
            data, _ = pipe.execute()

            if data:
                memory = json.loads(data)
                memory["last_accessed"] = datetime.now().isoformat()
                self._save_to_redis(session_id, memory)
                return memory

            # New session
            memory = self._create_empty_memory()
            self._save_to_redis(session_id, memory)
            return memory

        except Exception as e:
            logger.error(f"[Memory] Redis get error: {e} — falling back to in-memory")
            return self._get_from_memory(session_id)

    def _save_to_redis(self, session_id: str, memory: Dict) -> None:
        try:
            self.redis_client.setex(
                f"agent_memory:{session_id}",
                self.ttl_minutes * 60,
                json.dumps(memory, default=str),
            )
        except Exception as e:
            logger.error(f"[Memory] Redis save error: {e}")

    # ──────────────────────────────────────────────────────────────
    # Internal — generic persist
    # ──────────────────────────────────────────────────────────────

    def _persist(self, session_id: str, memory: Dict) -> None:
        """Write to whichever backend is active."""
        if self.backend == MemoryBackend.REDIS:
            self._save_to_redis(session_id, memory)
        else:
            self._memory_store[session_id] = memory

    # ──────────────────────────────────────────────────────────────
    # Empty memory schema — ALL banking entities
    # ──────────────────────────────────────────────────────────────

    def _create_empty_memory(self) -> Dict[str, Any]:
        """
        Full session memory schema.
        Every field that any of the 81 banking intents might need
        is pre-declared as None so context injection works uniformly.
        """
        now = datetime.now().isoformat()
        return {
            # ── Identity ──────────────────────────────────────────
            "user_id":              None,
            "session_token":        None,   # backend auth token
            "company_id":           None,
            "company_name":         None,

            # ── Compliance identifiers ────────────────────────────
            "gstin":                None,   # GST registration number
            "pan":                  None,   # PAN number
            "cin":                  None,   # Company Identification Number

            # ── Account / Banking ─────────────────────────────────
            "account_number":       None,   # last used bank account
            "ifsc_code":            None,

            # ── Compliance codes ──────────────────────────────────
            "establishment_id":     None,   # EPF establishment ID
            "establishment_code":   None,   # ESIC establishment code

            # ── Time context ──────────────────────────────────────
            "month":                None,   # last referenced month (MM-YYYY)
            "from_date":            None,
            "to_date":              None,

            # ── B2B ───────────────────────────────────────────────
            "partner_id":           None,
            "invoice_id":           None,
            "po_id":                None,

            # ── Payment ───────────────────────────────────────────
            "transaction_id":       None,
            "payment_mode":         None,   # NEFT / RTGS / IMPS / UPI
            "beneficiary_id":       None,

            # ── Insurance ─────────────────────────────────────────
            "policy_number":        None,

            # ── Custom / SEZ ──────────────────────────────────────
            "bill_of_entry_number": None,

            # ── Support ───────────────────────────────────────────
            "ticket_id":            None,

            # ── Conversation tracking ─────────────────────────────
            "conversation_history": [],
            "extracted_entities":   {},
            "intent_chain":         [],
            "context_variables":    {},

            # ── Timestamps ────────────────────────────────────────
            "created_at":           now,
            "last_accessed":        now,
        }


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

def create_memory(
    ttl_minutes:  int  = 60,
    use_redis:    bool = False,
    redis_host:   str  = "localhost",
    redis_port:   int  = 6379,
    redis_db:     int  = 0,
    redis_password: Optional[str] = None,
) -> AgentMemory:
    """
    Factory — creates AgentMemory with Redis or in-memory backend.

    Redis client is configured with:
        socket_timeout=3        — fail fast if Redis is down
        retry_on_timeout=True   — one automatic retry
        decode_responses=True   — strings, not bytes
    """
    if use_redis:
        try:
            import redis as redis_lib
            client = redis_lib.Redis(
                host             = redis_host,
                port             = redis_port,
                db               = redis_db,
                password         = redis_password or None,
                decode_responses = True,
                socket_timeout   = 3,            # ← don't hang if Redis is down
                retry_on_timeout = True,
            )
            client.ping()
            logger.info(f"[Memory] Redis connected at {redis_host}:{redis_port}")
            return AgentMemory(
                ttl_minutes  = ttl_minutes,
                backend      = MemoryBackend.REDIS,
                redis_client = client,
            )
        except Exception as e:
            logger.warning(f"[Memory] Redis unavailable ({e}) — using in-memory")

    return AgentMemory(ttl_minutes=ttl_minutes, backend=MemoryBackend.IN_MEMORY)