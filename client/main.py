"""
client/main.py — Bank AI Assistant  v3.2.0
 
Changes from v3.1.0:
  - /api/send-otp   → calls backend port 4000 to send OTP
  - /api/verify-otp → verifies OTP, seeds agent memory with real DB data
  - /api/logout     → clears agent memory + invalidates backend session
  - session_token   → stored in agent memory, injected into MCP tool calls
"""
from dotenv import load_dotenv
load_dotenv()
 
import asyncio
import importlib.util
import logging
import os
import uuid
import httpx
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
 
from config.config import settings
from client.mcp_client import bank_client_manager, gst_client_manager, info_client_manager
from manager import agent_manager
 
logging.basicConfig(
    level  = getattr(logging, settings.log_level.upper()),
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
 
# ── Optional query logger ──────────────────────────────────────────────
_qlog_spec = importlib.util.find_spec("query_logger")
if _qlog_spec is not None:
    from query_logger import metrics_router, query_logger
    _query_logging = True
else:
    metrics_router = None
    query_logger   = None
    _query_logging = False
 
# ── Config ─────────────────────────────────────────────────────────────
REQUEST_TIMEOUT_SECS = int(os.getenv("REQUEST_TIMEOUT_SECS", "30"))
BACKEND_URL          = os.getenv("BACKEND_URL", "http://localhost:4000")
 
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: List[str] = (
    ["*"] if _raw_origins == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)
 
logger.info(f"Backend URL  : {BACKEND_URL}")
 
 
# ══════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting Bank AI Assistant v3.2.0")
    logger.info(f"LLM Provider : {settings.llm_provider}")
    logger.info(f"Backend URL  : {BACKEND_URL}")
    logger.info("=" * 60)
 
    for name, mgr in (
        ("Bank MCP Server ", bank_client_manager),
        ("GST Calculator  ", gst_client_manager),
        ("Onboarding Info ", info_client_manager),
    ):
        try:
            client = await mgr.get_client()
            logger.info(f"✓ {name}: {len(client.available_tools)} tools")
        except Exception as e:
            logger.error(f"✗ {name}: {e}")
 
    await agent_manager.initialize()
 
    if _query_logging:
        logger.info("✓ Query logging enabled → logs/queries.jsonl")
 
    logger.info("=" * 60)
    logger.info("Bank AI Assistant ready!")
    logger.info("=" * 60)
 
    yield
 
    logger.info("Shutting down...")
    await agent_manager.shutdown()
    await bank_client_manager.close()
    await gst_client_manager.close()
    await info_client_manager.close()
    logger.info("Goodbye!")
 
 
# ══════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "Bank AI Assistant",
    description = (
        "AI-powered banking assistant.\n\n"
        "## Auth Flow\n"
        "1. `POST /api/send-otp` — send OTP to mobile\n"
        "2. `POST /api/verify-otp` — verify OTP → get session_id\n"
        "3. `POST /api/chat` with session_id — full banking AI\n\n"
        "## Public (no login needed)\n"
        "GST calculator, guides, FAQ\n\n"
        "## Private (login required)\n"
        "Balance, transactions, GST dues, EPF, ESIC, payroll"
    ),
    version     = "3.2.0",
    lifespan    = lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins      = ALLOWED_ORIGINS,
    allow_credentials  = True,
    allow_methods      = ["*"],
    allow_headers      = ["*"],
)
 
if metrics_router:
    app.include_router(metrics_router)
 
 
# ══════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [
            {"message": "show my account balance",   "session_id": "sess-abc-123"},
            {"message": "calculate GST for 10000 at 18%", "session_id": "public-test"},
        ]}
    )
    message:    str           = Field(...,           description="User's natural language banking query")
    session_id: Optional[str] = Field(default=None, description="Session ID from /api/verify-otp")
    user_id:    Optional[str] = Field(default=None, description="User ID (auto-set after login)")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None)
 
 
class ChatResponse(BaseModel):
    success:          bool
    intents_detected: List[str]
    is_multi_intent:  bool
    response:         str
    tool_calls:       List[Dict[str, Any]]
    llm_provider:     str            = "local_ml"
    session_id:       Optional[str] = None
    context_used:     Optional[bool] = None
    memory_snapshot:  Optional[Dict] = None
    requires_login:   Optional[bool] = None
    error:            Optional[str]  = None
 
 
class SendOtpRequest(BaseModel):
    mobile: str = Field(..., description="10-digit mobile number")
 
 
class VerifyOtpRequest(BaseModel):
    mobile: str = Field(..., description="10-digit mobile number")
    otp:    str = Field(..., description="6-digit OTP")
 
 
class LoginResponse(BaseModel):
    success:        bool
    session_id:     str
    session_token:  str
    user_id:        str
    company_name:   Optional[str] = None
    gstin:          Optional[str] = None
    account_number: Optional[str] = None
    message:        str
 
 
# ══════════════════════════════════════════════════════════════════════
# STANDARD ENDPOINTS
# ══════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {
        "message":      "Bank AI Assistant",
        "version":      "3.2.0",
        "llm_provider": settings.llm_provider,
        "backend_url":  BACKEND_URL,
        "docs":         "/docs",
        "health":       "/health",
    }
 
 
@app.get("/health")
async def health():
    server_status: Dict[str, Any] = {}
    total_tools = 0
 
    for name, mgr in (
        ("bank", bank_client_manager),
        ("gst",  gst_client_manager),
        ("info", info_client_manager),
    ):
        try:
            client = await mgr.get_client()
            server_status[name] = {"connected": True, "tools": len(client.available_tools)}
            total_tools += len(client.available_tools)
        except Exception as e:
            server_status[name] = {"connected": False, "error": str(e)}
 
    backend_ok = False
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{BACKEND_URL}/health", timeout=5)
            backend_ok = r.status_code == 200
    except Exception:
        pass
 
    return {
        "status":        "healthy",
        "version":       "3.2.0",
        "llm_provider":  settings.llm_provider,
        "total_tools":   total_tools,
        "agent_ready":   agent_manager.is_ready(),
        "db_healthy":    await agent_manager.db_health(),
        "backend_ok":    backend_ok,
        "backend_url":   BACKEND_URL,
        "query_logging": _query_logging,
        "cors_origins":  ALLOWED_ORIGINS,
        "servers":       server_status,
        "storage":       await agent_manager.storage_stats(),
    }
 
 
# ══════════════════════════════════════════════════════════════════════
# MAIN CHAT ENDPOINT
# ══════════════════════════════════════════════════════════════════════
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = (request.session_id or str(uuid.uuid4())).strip()
    user_id    = (request.user_id    or "anonymous").strip()
 
    logger.info(f"[Chat] session={session_id} user={user_id} msg={request.message[:80]!r}")
 
    try:
        result = await asyncio.wait_for(
            agent_manager.process(
                message    = request.message,
                session_id = session_id,
                user_id    = user_id,
            ),
            timeout = REQUEST_TIMEOUT_SECS,
        )
 
        processing_ms = int(result.get("processing_time", 0) * 1000)
 
        if _query_logging:
            query_logger.log_query(
                query      = request.message,
                intents    = result["intents_detected"],
                tools      = [t["tool"] for t in result.get("tool_calls", [])],
                latency_ms = processing_ms,
                success    = True,
            )
 
        return ChatResponse(
            success          = True,
            intents_detected = result["intents_detected"],
            is_multi_intent  = result["is_multi_intent"],
            response         = result["response"],
            tool_calls       = result["tool_calls"],
            llm_provider     = settings.llm_provider,
            session_id       = session_id,
            context_used     = result.get("context_used"),
            memory_snapshot  = result.get("memory_snapshot"),
            requires_login   = result.get("requires_login", False),
        )
 
    except asyncio.TimeoutError:
        logger.error(f"[Chat] Timeout after {REQUEST_TIMEOUT_SECS}s — session={session_id}")
        return ChatResponse(
            success=False, intents_detected=[], is_multi_intent=False,
            response="Request timed out. Please try again.",
            tool_calls=[], session_id=session_id, error="timeout",
        )
 
    except Exception as e:
        logger.error(f"[Chat] Error session={session_id}: {e}", exc_info=True)
        return ChatResponse(
            success=False, intents_detected=[], is_multi_intent=False,
            response="I encountered an error. Please try again.",
            tool_calls=[], session_id=session_id, error=str(e),
        )
 
 
# ══════════════════════════════════════════════════════════════════════
# SESSION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/session/{session_id}/context")
async def get_session_context(session_id: str):
    if not agent_manager.is_ready():
        raise HTTPException(status_code=503, detail="Agent not ready")
    return {"session_id": session_id, "context": agent_manager.get_context(session_id)}
 
 
@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    if not agent_manager.is_ready():
        raise HTTPException(status_code=503, detail="Agent not ready")
    return {"session_id": session_id, "history": agent_manager.get_history(session_id)}
 
 
@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    if not agent_manager.is_ready():
        raise HTTPException(status_code=503, detail="Agent not ready")
    agent_manager.clear_session(session_id)
    return {"session_id": session_id, "cleared": True}
 
 
# ══════════════════════════════════════════════════════════════════════
# OTP AUTH — connected to backend (port 4000)
# ══════════════════════════════════════════════════════════════════════
@app.post("/api/send-otp")
async def send_otp(request: SendOtpRequest):
    """Step 1 — Send OTP to mobile via backend."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{BACKEND_URL}/auth/send-otp",
                json={"mobile": request.mobile},
                timeout=10,
            )
            return resp.json()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Backend not reachable at {BACKEND_URL}. Start: uvicorn main:app --port 4000"
            )
 
 
@app.post("/api/verify-otp", response_model=LoginResponse)
async def verify_otp(request: VerifyOtpRequest):
    """
    Step 2 — Verify OTP → get session_id.
    Fetches real company/account data from backend and seeds agent memory.
    Use session_id in all /api/chat calls.
    """
    async with httpx.AsyncClient() as client:
        try:
            # 1. Verify OTP
            resp = await client.post(
                f"{BACKEND_URL}/auth/verify-otp",
                json={"mobile": request.mobile, "otp": request.otp},
                timeout=10,
            )
            data = resp.json()
 
            if not data.get("success"):
                raise HTTPException(
                    status_code=400,
                    detail=data.get("detail", "OTP verification failed")
                )
 
            bd            = data["data"]
            session_token = bd["session_token"]
            user_id       = bd["user_id"]
            session_id    = str(uuid.uuid4())
 
            # 2. Fetch real user context
            headers      = {"Authorization": f"Bearer {session_token}"}
            company_resp = await client.get(f"{BACKEND_URL}/user/company", headers=headers, timeout=10)
            balance_resp = await client.get(f"{BACKEND_URL}/user/balance", headers=headers, timeout=10)
 
            company = company_resp.json().get("data", {}) if company_resp.status_code == 200 else {}
            balance = balance_resp.json().get("data", {}) if balance_resp.status_code == 200 else {}
 
            # 3. Pre-seed agent memory
            memory_data = {
                "user_id":        user_id,
                "company_id":     company.get("company_id"),
                "company_name":   company.get("company_name"),
                "gstin":          company.get("gstin"),
                "account_number": balance.get("account_number_masked"),
                "session_token":  session_token,
            }
 
            if agent_manager.is_ready():
                try:
                    agent_manager.agent.memory.bulk_update(session_id, memory_data)
                    # Save to PostgreSQL immediately so it survives restarts
                    await agent_manager.agent._save_memory_to_db(session_id, memory_data)
                    logger.info(f"[OTP] ✓ session={session_id} user={user_id} memory seeded + persisted")
                except Exception as e:
                    logger.warning(f"[OTP] Memory seed failed (non-fatal): {e}")
 
            return LoginResponse(
                success        = True,
                session_id     = session_id,
                session_token  = session_token,
                user_id        = user_id,
                company_name   = company.get("company_name"),
                gstin          = company.get("gstin"),
                account_number = balance.get("account_number_masked"),
                message        = f"Welcome {company.get('company_name', '')}! Session ready.",
            )
 
        except HTTPException:
            raise
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Backend not reachable at {BACKEND_URL}"
            )
 
 
@app.post("/api/logout")
async def logout(session_id: str, session_token: str):
    """Logout — clears agent memory + invalidates backend session."""
    if agent_manager.is_ready():
        agent_manager.clear_session(session_id)
 
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{BACKEND_URL}/auth/logout",
                headers={"Authorization": f"Bearer {session_token}"},
                timeout=10,
            )
        except Exception:
            pass
 
    return {"session_id": session_id, "cleared": True, "message": "Logged out successfully"}
 
 
@app.post("/api/login")
async def login_legacy():
    """Legacy endpoint — use /api/send-otp + /api/verify-otp instead."""
    return {
        "message":   "Please use the new OTP flow",
        "step_1":    "POST /api/send-otp   {mobile: '9999999999'}",
        "step_2":    "POST /api/verify-otp {mobile: '9999999999', otp: '<otp>'}",
        "then_chat": "POST /api/chat       {message: '...', session_id: '<from step 2>'}",
    }
 
 
# ══════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/analytics/intents")
async def intent_analytics(days: int = 7):
    return {"days": days, "stats": await agent_manager.intent_stats(days=days)}
 
 
# ══════════════════════════════════════════════════════════════════════
# API INFO
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/info")
async def info():
    return {
        "api_version":  "3.2.0",
        "llm_provider": settings.llm_provider,
        "platform":     "Bank AI Assistant",
        "backend_url":  BACKEND_URL,
        "auth_flow": {
            "step_1": "POST /api/send-otp   {mobile}",
            "step_2": "POST /api/verify-otp {mobile, otp} → session_id",
            "step_3": "POST /api/chat       {message, session_id}",
        },
        "public_tools":  ["calculate_gst", "validate_gstin", "get_company_onboarding_guide", "get_faq"],
        "private_tools": ["get_account_balance", "get_transaction_history", "fetch_gst_dues", "..."],
        "features": {
            "otp_auth":             True,
            "real_db_data":         True,
            "session_memory":       True,
            "ml_intent":            True,
            "mcp_integration":      True,
            "multi_intent":         True,
            "encryption":           True,
            "public_private_split": True,
            "query_logging":        _query_logging,
        }
    }
 
 
@app.get("/api/mcp/tools")
async def list_all_tools():
    all_tools: Dict[str, Any] = {}
    total = 0
    for name, mgr in (
        ("bank", bank_client_manager),
        ("gst",  gst_client_manager),
        ("info", info_client_manager),
    ):
        try:
            client = await mgr.get_client()
            all_tools[name] = client.available_tools
            total += len(client.available_tools)
        except Exception as e:
            all_tools[name] = {"error": str(e)}
    return {"total_tools": total, "servers": all_tools}
 
 
# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "client.main:app",
        host   = settings.host,
        port   = settings.port,
        reload = settings.debug,
    )