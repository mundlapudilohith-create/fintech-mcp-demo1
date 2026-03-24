"""
Auth Middleware
===============
- verify_session: requires valid session token (protected routes)
- optional_session: extracts session if present but doesn't block
"""
 
import logging
from fastapi import Header, HTTPException, Request
from typing import Optional
from services.services import SessionService
 
logger = logging.getLogger(__name__)
 
 
async def verify_session(
    authorization: Optional[str] = Header(None, description="Bearer <session_token>")
) -> dict:
    """
    Dependency for PROTECTED routes.
    Requires: Authorization: Bearer <session_token>
    Returns session dict with user_id, company_id, memory_data.
    Raises 401 if missing, 403 if invalid/expired.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Authentication required",
                "message": "Please login with your mobile number to access this data.",
                "how_to_login": "POST /auth/send-otp with your mobile → then POST /auth/verify-otp with the OTP received",
            }
        )
 
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid authorization format", "message": "Use: Authorization: Bearer <session_token>"}
        )
 
    token = authorization.replace("Bearer ", "").strip()
    session = await SessionService.validate(token)
 
    if not session:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Session expired or invalid",
                "message": "Your session has expired. Please login again.",
                "how_to_login": "POST /auth/send-otp with your mobile number",
            }
        )
 
    return dict(session)
 
 
async def optional_session(
    authorization: Optional[str] = Header(None)
) -> Optional[dict]:
    """
    Dependency for PUBLIC routes that optionally use session.
    Returns session dict if valid token provided, else None.
    Never raises — always proceeds.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        token = authorization.replace("Bearer ", "").strip()
        return await SessionService.validate(token)
    except Exception:
        return None