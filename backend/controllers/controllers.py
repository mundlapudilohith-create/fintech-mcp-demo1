"""
Controllers — All API Routes
==============================
 
PUBLIC  (no login needed):
  POST /auth/send-otp          → request OTP for mobile
  POST /auth/verify-otp        → verify OTP → get session_token
  POST /public/gst/calculate   → GST calculator
  GET  /public/info/{topic}    → onboarding guides, FAQ
  GET  /health                 → server health
 
PROTECTED  (requires: Authorization: Bearer <session_token>):
  POST /auth/logout
  GET  /user/profile
  PUT  /user/profile
  POST /user/company
  GET  /user/company
  POST /user/accounts
  GET  /user/accounts
  GET  /user/balance
  GET  /user/transactions
  POST /user/transactions
  GET  /user/transactions/{id}
  GET  /user/gst/dues
  POST /user/gst/challan
  GET  /user/reminders
  POST /user/reminders
  DELETE /user/reminders/{id}
  POST /seed                   → insert test data (test mode only)
"""
 
import os
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
 
from models.schemas import (
    SendOtpRequest, VerifyOtpRequest, LogoutRequest,
    UpdateProfileRequest, CompanyCreate, BankAccountCreate,
    TransactionCreate, GstChallanRequest, GstCalcRequest,
    ReminderCreate,
)
from services.services import (
    OtpService, UserService, SessionService, CompanyService,
    BankAccountService, TransactionService, GstService,
    ReminderService, AuditService, SeedService,
)
from config.database import ping_db
 
from fastapi import Header
 
# ─────────────────────────────────────────────
# Auth helpers (inline — no separate module needed)
# ─────────────────────────────────────────────
async def verify_session(
    authorization: Optional[str] = Header(None, description="Bearer <session_token>")
) -> dict:
    """Require valid session token. Returns session dict or raises 401/403."""
    if not authorization:
        raise HTTPException(status_code=401, detail={
            "error": "Authentication required",
            "message": "Login first: POST /auth/send-otp → POST /auth/verify-otp",
        })
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"error": "Use: Authorization: Bearer <token>"})
    token = authorization.replace("Bearer ", "").strip()
    session = await SessionService.validate(token)
    if not session:
        raise HTTPException(status_code=403, detail={
            "error": "Session expired or invalid",
            "message": "Please login again via POST /auth/send-otp",
        })
    return dict(session)
 
 
async def optional_session(
    authorization: Optional[str] = Header(None)
) -> Optional[dict]:
    """Public route helper — returns session if token provided, else None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        token = authorization.replace("Bearer ", "").strip()
        return await SessionService.validate(token)
    except Exception:
        return None
 
 
 
logger = logging.getLogger(__name__)
TEST_MODE = os.environ.get("TEST_MODE", "true").lower() == "true"
 
 
# ═══════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════
health_router = APIRouter(tags=["Health"])
 
@health_router.get("/health")
async def health():
    db_ok = await ping_db()
    return {
        "status":   "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
        "test_mode": TEST_MODE,
    }
 
 
# ═══════════════════════════════════════════════════════════
# AUTH ROUTES  (public)
# ═══════════════════════════════════════════════════════════
auth_router = APIRouter(prefix="/auth", tags=["Auth (Public)"])
 
@auth_router.post("/send-otp")
async def send_otp(body: SendOtpRequest, request: Request):
    """
    Step 1 of login.
    Sends OTP to mobile number.
    In TEST_MODE the OTP is returned in the response.
    In production it is sent via SMS only.
    """
    try:
        result = await OtpService.send_otp(body.mobile)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"send_otp error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@auth_router.post("/verify-otp")
async def verify_otp(body: VerifyOtpRequest, request: Request):
    """
    Step 2 of login.
    Verifies OTP → returns session_token.
    Use this token as: Authorization: Bearer <session_token>
    in all protected requests.
    """
    try:
        result = await OtpService.verify_otp(body.mobile, body.otp)
        await AuditService.log(
            user_id=result["user_id"],
            action="LOGIN",
            ip=request.client.host
        )
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"verify_otp error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@auth_router.post("/logout")
async def logout(session=Depends(verify_session)):
    """Invalidate current session token."""
    await SessionService.invalidate(session["session_token"])
    return {"success": True, "message": "Logged out successfully"}
 
 
# ═══════════════════════════════════════════════════════════
# PUBLIC ROUTES  (no login — GST calc, info, guides)
# ═══════════════════════════════════════════════════════════
public_router = APIRouter(prefix="/public", tags=["Public (No Login Required)"])
 
@public_router.post("/gst/calculate")
async def gst_calculate(body: GstCalcRequest):
    """
    GST Calculator — fully public, no login needed.
    Works for anyone: user, accountant, business owner.
    """
    result = GstService.calculate(body.amount, body.gst_rate, body.inclusive)
    return {"success": True, "data": result}
 
 
@public_router.get("/info/{topic}")
async def get_info(topic: str):
    """
    Public information — onboarding guides, FAQ.
    No login required.
    """
    topics = {
        "company-onboarding": {
            "title": "Company Onboarding Guide",
            "steps": [
                "Register your company with MCA",
                "Apply for PAN and TAN",
                "Register for GST if turnover > ₹20L",
                "Open a current account",
                "Register for EPF if employees > 20",
                "Register for ESIC if employees > 10",
            ]
        },
        "gst-guide": {
            "title": "GST Filing Guide",
            "steps": [
                "File GSTR-1 by 11th of next month",
                "File GSTR-3B by 20th of next month",
                "Pay GST dues before filing",
                "Use CPIN generated from GST portal",
            ]
        },
        "epf-guide": {
            "title": "EPF Contribution Guide",
            "info": "Employee: 12% of basic salary. Employer: 12% (EPF 3.67% + EPS 8.33%). Due by 15th of next month."
        },
        "esic-guide": {
            "title": "ESIC Contribution Guide",
            "info": "Employee: 0.75% of wages. Employer: 3.25% of wages. Due by 15th of next month."
        },
        "bank-guide": {
            "title": "Banking Setup Guide",
            "steps": [
                "Choose between current/savings account",
                "Submit KYC documents (PAN, Aadhaar, address proof)",
                "Minimum balance requirements vary by bank",
                "Set up internet banking for payments",
            ]
        },
    }
 
    if topic not in topics:
        raise HTTPException(
            status_code=404,
            detail=f"Topic '{topic}' not found. Available: {list(topics.keys())}"
        )
    return {"success": True, "data": topics[topic]}
 
 
@public_router.get("/faq")
async def get_faq():
    """Common FAQ — public, no login needed."""
    return {
        "success": True,
        "data": [
            {"q": "What is GSTIN?", "a": "15-digit GST identification number assigned by GST Council."},
            {"q": "When is EPF due?", "a": "By the 15th of every month for the previous month's wages."},
            {"q": "What is RTGS minimum?", "a": "RTGS minimum transfer amount is ₹2 lakhs."},
            {"q": "NEFT vs IMPS?", "a": "NEFT: batch settlement, works 24x7. IMPS: real-time, works 24x7."},
        ]
    }
 
 
# ═══════════════════════════════════════════════════════════
# PROTECTED ROUTES  (require: Authorization: Bearer <token>)
# ═══════════════════════════════════════════════════════════
user_router = APIRouter(prefix="/user", tags=["Protected (Login Required)"])
 
# ── Profile ──────────────────────────────────────────────
@user_router.get("/profile")
async def get_profile(session=Depends(verify_session)):
    """Get logged-in user's profile."""
    user = await UserService.get_by_id(session["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    company = await CompanyService.get_by_user(session["user_id"])
    return {
        "success": True,
        "data": {
            **user,
            "has_company": company is not None,
            "company": company,
        }
    }
 
 
@user_router.put("/profile")
async def update_profile(body: UpdateProfileRequest, session=Depends(verify_session)):
    """Update name or email."""
    await UserService.update_profile(session["user_id"], name=body.name, email=body.email)
    return {"success": True, "message": "Profile updated"}
 
 
# ── Company ──────────────────────────────────────────────
@user_router.post("/company")
async def create_company(body: CompanyCreate, session=Depends(verify_session)):
    """Register company for logged-in user."""
    try:
        company = await CompanyService.create(session["user_id"], body.model_dump())
        await AuditService.log(session["user_id"], "COMPANY_CREATED", "companies", company["id"])
        # Refresh session memory with new company data
        mem = dict(session.get("memory_data") or {})
        mem.update({"company_id": company["id"], "company_name": company["company_name"], "gstin": company["gstin"]})
        await SessionService.update_memory(session["session_token"], mem)
        return {"success": True, "data": company}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
 
 
@user_router.get("/company")
async def get_company(session=Depends(verify_session)):
    """Get company info for logged-in user."""
    company = await CompanyService.get_by_user(session["user_id"])
    if not company:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "No company registered",
                "message": "Please register your company first using POST /user/company",
            }
        )
    return {"success": True, "data": company}
 
 
# ── Bank Accounts ─────────────────────────────────────────
@user_router.post("/accounts")
async def add_account(body: BankAccountCreate, session=Depends(verify_session)):
    """Link a bank account to company."""
    company = await CompanyService.get_by_user(session["user_id"])
    if not company:
        raise HTTPException(status_code=400, detail="Register your company first")
    account = await BankAccountService.create(company["id"], session["user_id"], body.model_dump())
    await AuditService.log(session["user_id"], "ACCOUNT_ADDED", "bank_accounts", account["id"])
    return {"success": True, "data": account}
 
 
@user_router.get("/accounts")
async def get_accounts(session=Depends(verify_session)):
    """Get all linked bank accounts."""
    company = await CompanyService.get_by_user(session["user_id"])
    if not company:
        raise HTTPException(status_code=404, detail="No company found")
    accounts = await BankAccountService.get_all(company["id"])
    return {"success": True, "data": accounts}
 
 
@user_router.get("/balance")
async def get_balance(session=Depends(verify_session)):
    """Get default account balance."""
    company = await CompanyService.get_by_user(session["user_id"])
    if not company:
        raise HTTPException(status_code=404, detail="No company found")
    account = await BankAccountService.get_default(company["id"])
    if not account:
        raise HTTPException(status_code=404, detail="No bank account linked")
    return {
        "success": True,
        "data": {
            "account_number_masked": account["account_number_masked"],
            "bank_name":             account["bank_name"],
            "available_balance":     account["available_balance"],
            "currency":              "INR",
        }
    }
 
 
# ── Transactions ──────────────────────────────────────────
@user_router.get("/transactions")
async def get_transactions(
    limit: int = 50,
    status: str = "ALL",
    session=Depends(verify_session)
):
    """Get transaction history for logged-in user's company."""
    company = await CompanyService.get_by_user(session["user_id"])
    if not company:
        raise HTTPException(status_code=404, detail="No company found")
    txns = await TransactionService.get_list(company["id"], limit, status)
    return {"success": True, "data": txns, "total": len(txns)}
 
 
@user_router.post("/transactions")
async def create_transaction(body: TransactionCreate, session=Depends(verify_session)):
    """Initiate a payment."""
    company = await CompanyService.get_by_user(session["user_id"])
    if not company:
        raise HTTPException(status_code=400, detail="Register your company first")
    txn = await TransactionService.create(company["id"], session["user_id"], body.model_dump())
    await AuditService.log(session["user_id"], "PAYMENT_INITIATED", "transactions", txn["transaction_id"])
    return {"success": True, "data": txn}
 
 
@user_router.get("/transactions/{txn_id}")
async def get_transaction(txn_id: str, session=Depends(verify_session)):
    """Get details of a specific transaction."""
    txn = await TransactionService.get_by_id(txn_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return {"success": True, "data": txn}
 
 
# ── GST ───────────────────────────────────────────────────
@user_router.get("/gst/dues")
async def get_gst_dues(session=Depends(verify_session)):
    """Get pending GST dues for logged-in company."""
    company = await CompanyService.get_by_user(session["user_id"])
    if not company:
        raise HTTPException(status_code=404, detail="No company found")
    dues = await GstService.get_dues(company["id"])
    return {"success": True, "data": dues}
 
 
@user_router.post("/gst/challan")
async def create_gst_challan(body: GstChallanRequest, session=Depends(verify_session)):
    """Create GST challan for payment."""
    company = await CompanyService.get_by_user(session["user_id"])
    if not company:
        raise HTTPException(status_code=400, detail="No company found")
    result = await GstService.create_challan(company["id"], body.gstin, body.model_dump())
    return {"success": True, "data": result}
 
 
# ── Reminders ─────────────────────────────────────────────
@user_router.get("/reminders")
async def get_reminders(session=Depends(verify_session)):
    company = await CompanyService.get_by_user(session["user_id"])
    if not company:
        raise HTTPException(status_code=404, detail="No company found")
    reminders = await ReminderService.get_list(company["id"])
    return {"success": True, "data": reminders}
 
 
@user_router.post("/reminders")
async def create_reminder(body: ReminderCreate, session=Depends(verify_session)):
    company = await CompanyService.get_by_user(session["user_id"])
    if not company:
        raise HTTPException(status_code=400, detail="No company found")
    result = await ReminderService.create(company["id"], session["user_id"], body.model_dump())
    return {"success": True, "data": result}
 
 
@user_router.delete("/reminders/{reminder_id}")
async def delete_reminder(reminder_id: str, session=Depends(verify_session)):
    await ReminderService.delete(reminder_id)
    return {"success": True, "message": "Reminder deleted"}
 
 
# ── Session info (for AI agent) ───────────────────────────
@user_router.get("/session-context")
async def get_session_context(session=Depends(verify_session)):
    """Returns memory_data for AI agent to pre-warm context."""
    return {
        "success": True,
        "data": {
            "session_token": session["session_token"],
            "user_id":       session["user_id"],
            "memory_data":   session.get("memory_data", {}),
            "expires_at":    session["expires_at"],
        }
    }
 
 
# ═══════════════════════════════════════════════════════════
# SEED  (test only)
# ═══════════════════════════════════════════════════════════
seed_router = APIRouter(tags=["Seed (Test Only)"])
 
@seed_router.post("/seed")
async def seed(request: Request):
    """
    Insert test users with encrypted data.
    Only works when TEST_MODE=true.
    """
    if not TEST_MODE:
        raise HTTPException(status_code=403, detail="Seed endpoint disabled in production")
    results = await SeedService.seed()
    return {
        "success": True,
        "message": "Test data seeded",
        "results": results,
        "test_mobiles": ["9999999999", "8888888888"],
        "how_to_test": "POST /auth/send-otp with mobile → POST /auth/verify-otp with OTP"
    }