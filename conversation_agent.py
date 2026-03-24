"""
agent/conversation_agent.py — Bank AI Assistant
Production-grade orchestration layer wrapping LocalMLService.
 
Adds on top of llm_service (which stays untouched):
  - Input validation + sanitisation
  - 60-min session memory with context enrichment
  - Background PostgreSQL persistence (with asyncio safety)
  - Structured error responses
  - Processing time tracking
"""
 
import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
 
from agent.agent_memory import AgentMemory, MemoryBackend, create_memory
 
logger = logging.getLogger(__name__)
 
# ── Constants ──────────────────────────────────────────────────────────
MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_TURNS  = 10
MAX_INTENT_CHAIN   = 20
 
# ── Public tools — no login required ──────────────────────────────────
PUBLIC_TOOLS = {
    # GST Calculator
    "calculate_gst", "reverse_calculate_gst", "gst_breakdown",
    "compare_gst_rates", "validate_gstin",
    # Info / Guides
    "get_company_onboarding_guide", "get_vendor_onboarding_guide",
    "get_bank_onboarding_guide", "get_faq", "get_contact_details",
    "chat_with_support", "raise_support_ticket",
}
 
# ── Private tools — login required (need account/gstin/company context) ──
PRIVATE_TOOLS = {
    # Account & Balance
    "get_account_balance", "get_account_summary", "get_account_details",
    "get_linked_accounts", "set_default_account",
    # Transactions
    "get_transaction_history", "get_transaction_details",
    "search_transactions", "get_pending_transactions",
    "download_transaction_report",
    # Payments
    "initiate_payment", "get_payment_status", "cancel_payment",
    "retry_payment", "get_payment_receipt", "validate_beneficiary",
    "upload_bulk_payment", "validate_payment_file",
    # GST private
    "fetch_gst_dues", "pay_gst", "create_gst_challan",
    "get_gst_payment_history",
    # EPF
    "fetch_epf_dues", "pay_epf", "get_epf_payment_history",
    # ESIC
    "fetch_esic_dues", "pay_esic", "get_esic_payment_history",
    # Payroll
    "fetch_payroll_summary", "process_payroll", "get_payroll_history",
    # Tax
    "fetch_tax_dues", "pay_direct_tax", "pay_state_tax",
    "pay_bulk_tax", "get_tax_payment_history",
    # Insurance
    "fetch_insurance_dues", "pay_insurance_premium",
    "get_insurance_payment_history",
    # Custom Duty
    "pay_custom_duty", "track_custom_duty_payment", "get_custom_duty_history",
    # Bank Statement
    "fetch_bank_statement", "download_bank_statement",
    # Dashboard & Analytics
    "get_dashboard_summary", "get_spending_analytics",
    "get_cashflow_summary", "get_monthly_report",
    "get_vendor_payment_summary",
    # Company
    "get_company_profile", "update_company_details",
    "get_gst_profile", "get_authorized_signatories", "manage_user_roles",
    # Dues & Reminders
    "get_upcoming_dues", "get_overdue_payments",
    "set_payment_reminder", "get_reminder_list", "delete_reminder",
    # B2B
    "onboard_business_partner", "send_invoice", "get_received_invoices",
    "acknowledge_payment", "create_proforma_invoice",
    "create_cd_note", "create_purchase_order",
}
 
 
def _is_logged_in(mem: dict) -> bool:
    """User is logged in if session memory has a valid session_token from backend."""
    token = mem.get("session_token", "")
    # Only check session_token — user_id in chat is always 'anonymous' by default
    return bool(token and not token.startswith("BKAI-"))
 
 
def _requires_login(intents: list) -> bool:
    """Check if any detected intent maps to a private tool."""
    # intent name == tool name in most cases (or check intent_routing)
    for intent in intents:
        if intent in PRIVATE_TOOLS:
            return True
    return False
 
 
 
 
class ConversationAgent:
    """
    Production orchestration layer on top of LocalMLService.
 
    Pipeline per request:
        1. Validate + sanitise input
        2. Load 60-min session memory
        3. Build enriched history (inject known entities as system context)
        4. Call llm_service.process_query()  ← ALL ML/MCP logic here, unchanged
        5. Extract new entities from tool results
        6. Update session memory
        7. Fire-and-forget PostgreSQL persistence (safe background task)
        8. Return enriched result dict
    """
 
    def __init__(
        self,
        llm_service,
        user_storage=None,
        memory_ttl: int = 60,
        redis_client=None,
    ):
        """
        Args:
            llm_service:   LocalMLService instance (client/llm_service.py)
            user_storage:  UserStorage instance — None means no persistence
            memory_ttl:    Session TTL in minutes
            redis_client:  Optional Redis client for distributed deployments
        """
        self.llm_service  = llm_service
        self.user_storage = user_storage
 
        self.memory = (
            AgentMemory(
                ttl_minutes=memory_ttl,
                backend=MemoryBackend.REDIS,
                redis_client=redis_client,
            )
            if redis_client
            else create_memory(ttl_minutes=memory_ttl, use_redis=False)
        )
 
        logger.info(
            "ConversationAgent ready — "
            f"memory_ttl={memory_ttl}min  "
            f"storage={'PostgreSQL' if user_storage else 'disabled'}"
        )
 
    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
 
    async def process(
        self,
        message: str,
        session_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Process one user turn end-to-end.
 
        Returns llm_service result dict extended with:
            session_id, user_id, context_used,
            processing_time, memory_snapshot
        """
        start = datetime.now()
 
        # ── 1. Validate input ─────────────────────────────────────────
        error = self._validate(message, session_id, user_id)
        if error:
            return self._error_response(error, session_id, user_id)
 
        message = message.strip()
 
        try:
            # ── 2. Load session memory ────────────────────────────────
            mem = self.memory.get(session_id)
 
            # ── 2b. Restore from PostgreSQL if in-memory is empty ────
            # This handles server restarts — memory survives in DB
            logger.info(f"[Agent] user_storage={self.user_storage is not None} session_token={bool(mem.get('session_token'))}")
            if self.user_storage and not mem.get("session_token"):
                try:
                    db_mem = await self.user_storage.load_session_memory(session_id)
                    if db_mem:
                        self.memory.bulk_update(session_id, db_mem)
                        mem = self.memory.get(session_id)
                        logger.info(f"[Agent] ✓ Memory restored from PostgreSQL: session={session_id}")
                except Exception as e:
                    logger.warning(f"[Agent] Memory restore failed (non-fatal): {e}")
 
            mem["user_id"] = user_id
 
            # ── 3. Build enriched history ─────────────────────────────
            enriched_history = self._build_history(session_id, mem)
            # context_used = we injected something the user didn't re-state
            context_used = bool(
                mem.get("company_id") or mem.get("gstin") or mem.get("account_number")
            )
 
            # ── 3b. Quick intent peek — check login requirement ───────
            # Run a lightweight intent check first to see if this query
            # needs private data. If yes and user not logged in → block.
            try:
                quick_intents = self.llm_service.intent_classifier.predict_intents(message)
            except Exception:
                quick_intents = []
 
            if _requires_login(quick_intents) and not _is_logged_in(mem):
                elapsed = (datetime.now() - start).total_seconds()
                logger.info(
                    f"[Agent] Login required — session={session_id} "
                    f"intents={quick_intents}"
                )
                return {
                    "success":          False,
                    "intents_detected": quick_intents,
                    "is_multi_intent":  len(quick_intents) > 1,
                    "response": (
                        "🔒 This information is private and requires login.\n\n"
                        "Please login first:\n"
                        "1. Send your mobile number → **POST /api/send-otp**\n"
                        "2. Verify OTP → **POST /api/verify-otp**\n"
                        "3. Use the session_id in your chat requests\n\n"
                        "Public features available without login: "
                        "GST calculator, guides, FAQ."
                    ),
                    "tool_calls":       [],
                    "requires_login":   True,
                    "session_id":       session_id,
                    "user_id":          user_id,
                    "context_used":     False,
                    "processing_time":  round(elapsed, 3),
                    "memory_snapshot":  self._safe_snapshot(mem),
                }
 
            # ── 4. Call llm_service ───────────────────────────────────
            result = await self.llm_service.process_query(
                user_message         = message,
                conversation_history = enriched_history,
                session_token        = mem.get("session_token"),  # ← inject user's backend token
            )
 
            # ── 5. Extract new entities ───────────────────────────────
            new_entities = self._extract_entities(result)
 
            # ── 6. Update session memory ──────────────────────────────
            self._update_memory(session_id, result, new_entities)
 
            # ── 6b. Record turn in history ────────────────────────────
            self.memory.add_to_history(
                session_id, "user", message,
                {"intents": result.get("intents_detected", [])},
            )
            self.memory.add_to_history(
                session_id, "assistant", result.get("response", ""),
                {"tool_calls": [t.get("tool") for t in result.get("tool_calls", [])]},
            )
 
            # ── 7. Background persistence (safe — won't crash request) ─
            if self.user_storage:
                self._schedule_persist(user_id, session_id, message, result, mem)
                # Save full memory to PostgreSQL so it survives restarts
                asyncio.create_task(
                    self._save_memory_to_db(session_id, mem)
                )
 
            # ── 8. Return ─────────────────────────────────────────────
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(
                f"[Agent] ✓ session={session_id} user={user_id} "
                f"intents={result.get('intents_detected')} "
                f"time={elapsed:.2f}s"
            )
 
            return {
                **result,
                "session_id":      session_id,
                "user_id":         user_id,
                "context_used":    context_used,
                "processing_time": round(elapsed, 3),
                "memory_snapshot": self._safe_snapshot(mem),
            }
 
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            logger.error(
                f"[Agent] process() error — session={session_id} "
                f"user={user_id} error={e}",
                exc_info=True,
            )
            return self._error_response(str(e), session_id, user_id, elapsed)
 
    # ──────────────────────────────────────────────────────────────────
    # Input validation
    # ──────────────────────────────────────────────────────────────────
 
    def _validate(self, message: str, session_id: str, user_id: str) -> Optional[str]:
        if not message or not message.strip():
            return "Message cannot be empty."
        if len(message) > MAX_MESSAGE_LENGTH:
            return f"Message too long ({len(message)} chars). Max {MAX_MESSAGE_LENGTH}."
        if not session_id or not session_id.strip():
            return "session_id is required."
        if not user_id or not user_id.strip():
            return "user_id is required."
        # Basic injection guard — strip control characters
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", message):
            return "Message contains invalid characters."
        return None
 
    # ──────────────────────────────────────────────────────────────────
    # Context / history
    # ──────────────────────────────────────────────────────────────────
 
    def _build_history(
        self,
        session_id: str,
        mem: Dict,
    ) -> Optional[List[Dict[str, str]]]:
        """
        Build conversation_history for llm_service.
        Prepends a system message with known session context so the
        ML classifier doesn't need to re-ask for company_id / gstin.
        """
        raw       = self.memory.get_history(session_id)
        formatted = [
            {"role": h["role"], "content": h["content"]}
            for h in raw[-MAX_HISTORY_TURNS:]
        ]
 
        known = []
        for key, label in (
            ("company_id",     "company_id"),
            ("company_name",   "company"),
            ("gstin",          "gstin"),
            ("account_number", "account"),
            ("pan",            "pan"),
        ):
            if mem.get(key):
                known.append(f"{label}={mem[key]}")
 
        if known:
            formatted.insert(0, {
                "role":    "system",
                "content": f"[Session context: {', '.join(known)}]",
            })
 
        return formatted or None
 
    # ──────────────────────────────────────────────────────────────────
    # Entity extraction from tool results
    # ──────────────────────────────────────────────────────────────────
 
    # Maps tool name → list of (result_key, memory_key) to extract
    _ENTITY_MAP: Dict[str, List[tuple]] = {
        "get_company_profile":    [("company_name", "company_name"), ("gstin", "gstin"), ("pan", "pan")],
        "update_company_details": [("company_name", "company_name")],
        "get_gst_profile":        [],   # handled separately (nested list)
        "get_account_balance":    [("account_number", "account_number")],
        "get_account_details":    [("account_number", "account_number")],
        "fetch_bank_statement":   [("account_number", "account_number")],
        "get_account_summary":    [],   # handled separately (list of accounts)
    }
 
    def _extract_entities(self, result: Dict) -> Dict[str, Any]:
        """Scan tool results for entities worth persisting into session memory."""
        entities: Dict[str, Any] = {}
 
        for call in result.get("tool_calls", []):
            if not call.get("success"):
                continue
            data = call.get("result", {})
            if not isinstance(data, dict):
                continue
            tool = call.get("tool", "")
 
            # Standard field extraction
            if tool in self._ENTITY_MAP:
                for result_key, memory_key in self._ENTITY_MAP[tool]:
                    val = data.get(result_key)
                    if val and memory_key not in entities:
                        entities[memory_key] = val
 
            # Nested: get_gst_profile returns a list of gst_numbers
            if tool == "get_gst_profile":
                gst_list = data.get("gst_numbers", [])
                if gst_list and isinstance(gst_list[0], dict):
                    gstin = gst_list[0].get("gstin")
                    if gstin and "gstin" not in entities:
                        entities["gstin"] = gstin
 
            # Nested: get_account_summary returns list of accounts
            if tool == "get_account_summary":
                accounts = data.get("accounts", [])
                if accounts and isinstance(accounts[0], dict):
                    acc_num = accounts[0].get("account_number")
                    if acc_num and "account_number" not in entities:
                        entities["account_number"] = acc_num
 
        return entities
 
    # ──────────────────────────────────────────────────────────────────
    # Memory update
    # ──────────────────────────────────────────────────────────────────
 
    def _update_memory(
        self,
        session_id: str,
        result: Dict,
        new_entities: Dict,
    ) -> None:
        """Persist new entities and update intent chain in session memory."""
        updates = {k: v for k, v in new_entities.items() if v}
        if updates:
            for k, v in updates.items():
                logger.info(f"[Agent] 💾 session={session_id} stored {k}={v}")
            self.memory.bulk_update(session_id, updates)
 
        # Append intents to chain, bounded at MAX_INTENT_CHAIN
        mem   = self.memory.get(session_id)
        chain = mem.get("intent_chain", [])
        for intent in result.get("intents_detected", []):
            chain.append({"intent": intent, "ts": datetime.now().isoformat()})
        self.memory.update(session_id, "intent_chain", chain[-MAX_INTENT_CHAIN:])
 
    # ──────────────────────────────────────────────────────────────────
    # PostgreSQL persistence — safe background scheduling
    # ──────────────────────────────────────────────────────────────────
 
    def _schedule_persist(
        self,
        user_id: str,
        session_id: str,
        message: str,
        result: Dict,
        mem: Dict,
    ) -> None:
        """
        Schedule persistence as a background task.
        Guards against event loop edge cases (e.g. during shutdown).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running() and not loop.is_closed():
                loop.create_task(
                    self._persist(user_id, session_id, message, result, mem)
                )
        except RuntimeError:
            # No running event loop — skip silently (shouldn't happen in FastAPI)
            logger.debug("[Agent] No event loop for persistence task — skipping")
 
    async def _persist(
        self,
        user_id: str,
        session_id: str,
        message: str,
        result: Dict,
        mem: Dict,
    ) -> None:
        """Save conversation + intent logs to PostgreSQL."""
        try:
            intents    = result.get("intents_detected", [])
            tool_calls = result.get("tool_calls", [])
            entities   = result.get("debug_info", {}).get("entities_extracted", {})
 
            await self.user_storage.save_conversation({
                "user_id":            user_id,
                "session_id":         session_id,
                "user_message":       message,
                "assistant_response": result.get("response", ""),
                "intent":             intents[0] if intents else "unknown",
                "entities":           entities,
                "confidence":         result.get("confidence"),        # real value when available
                "tool_name":          tool_calls[0]["tool"] if tool_calls else None,
                "context_used":       bool(mem.get("company_id") or mem.get("gstin")),
                "company_id":         mem.get("company_id"),
                "company_name":       mem.get("company_name"),
                "gstin":              mem.get("gstin"),
                "processing_time":    result.get("processing_time"),
                "timestamp":          datetime.now().isoformat(),
            })
 
            # Log every intent separately for analytics + retraining
            for intent in intents:
                await self.user_storage.log_intent({
                    "user_id":         user_id,
                    "session_id":      session_id,
                    "message":         message,
                    "intent":          intent,
                    "confidence":      result.get("confidence", 0.0) or 0.0,
                    "is_multi_intent": len(intents) > 1,
                    "all_intents":     intents,
                    "entities":        entities,
                })
 
        except Exception as e:
            # Never let persistence failures surface to the user
            logger.error(f"[Agent] Persistence error (non-fatal): {e}", exc_info=True)
 
    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
 
    def _safe_snapshot(self, mem: Dict) -> Dict:
        """Return safe subset of memory for API response."""
        return {
            "company_id":   mem.get("company_id"),
            "company_name": mem.get("company_name"),
            "gstin":        mem.get("gstin"),
            "account_number": mem.get("account_number"),
        }
 
    def _error_response(
        self,
        error: str,
        session_id: str,
        user_id: str,
        processing_time: float = 0.0,
    ) -> Dict[str, Any]:
        logger.warning(f"[Agent] Error response — session={session_id}: {error}")
        return {
            "success":          False,
            "intents_detected": [],
            "is_multi_intent":  False,
            "response":         "I couldn't process your request. Please try again.",
            "tool_calls":       [],
            "llm_provider":     "local_ml",
            "error":            error,
            "session_id":       session_id,
            "user_id":          user_id,
            "context_used":     False,
            "processing_time":  round(processing_time, 3),
            "memory_snapshot":  {},
        }
 
    # ──────────────────────────────────────────────────────────────────
    # Public helpers (called by AgentManager)
    # ──────────────────────────────────────────────────────────────────
 
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        return self.memory.get_history(session_id)
 
    def get_memory_snapshot(self, session_id: str) -> Dict:
        mem = self.memory.get(session_id)
        return {
            "company_id":     mem.get("company_id"),
            "company_name":   mem.get("company_name"),
            "gstin":          mem.get("gstin"),
            "pan":            mem.get("pan"),
            "account_number": mem.get("account_number"),
            "intent_chain":   mem.get("intent_chain", []),
            "has_session_token": bool(mem.get("session_token")),  # show presence not value
        }
 
    def clear_memory(self, session_id: str) -> None:
        self.memory.clear(session_id)
        logger.info(f"[Agent] 🗑️ Cleared session={session_id}")
 
    async def _save_memory_to_db(self, session_id: str, mem: Dict) -> None:
        """Save current memory to PostgreSQL in background (fire-and-forget)."""
        try:
            if self.user_storage:
                await self.user_storage.save_session_memory(session_id, mem)
        except Exception as e:
            logger.warning(f"[Agent] Memory save to DB failed (non-fatal): {e}")