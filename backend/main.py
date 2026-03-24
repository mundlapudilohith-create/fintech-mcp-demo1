"""
Bank AI Backend — Main App (v2)
================================
Mobile OTP authentication + PostgreSQL + AES-256 Encryption
 
Run:
    uvicorn main:app --host 0.0.0.0 --port 4000 --reload
 
Docs:
    http://localhost:4000/docs
"""
 
import os
import logging
from contextlib import asynccontextmanager
 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
 
from config.database import get_pool, close_pool
from controllers.controllers import (
    health_router, auth_router, public_router,
    user_router, seed_router,
)
 
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
 
TEST_MODE = os.environ.get("TEST_MODE", "true").lower() == "true"
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Bank AI Backend...")
    await get_pool()
    logger.info("✓ Database pool ready")
    yield
    await close_pool()
    logger.info("Shutdown complete")
 
 
app = FastAPI(
    title="Bank AI Backend",
    description="""
## Authentication Flow
 
All user data is identified by **mobile number only**.
 
### Step 1 — Request OTP (Public)
```
POST /auth/send-otp
Body: {"mobile": "9999999999"}
```
 
### Step 2 — Verify OTP → Get session token (Public)
```
POST /auth/verify-otp
Body: {"mobile": "9999999999", "otp": "123456"}
→ Returns: {"session_token": "abc123..."}
```
 
### Step 3 — Use session token for all protected routes
```
Header: Authorization: Bearer <session_token>
```
 
---
 
## Route Types
 
| Type | Routes | Login needed? |
|------|--------|--------------|
| **Public** | `/auth/*`, `/public/*`, `/health` | ❌ No |
| **Protected** | `/user/*` | ✅ Yes (Bearer token) |
 
---
 
## Test Mode
When `TEST_MODE=true`, OTP is returned in the `/auth/send-otp` response for easy testing.
In production, OTP is sent via SMS only.
""",
    version="2.0.0",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Swagger Authorize button ──────────────────────────────────
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Add Bearer token security scheme
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "session_token",
            "description": "Paste your session_token from POST /auth/verify-otp",
        }
    }
    # Apply to all paths
    for path in schema.get("paths", {}).values():
        for method in path.values():
            method.setdefault("security", [{"BearerAuth": []}])
    app.openapi_schema = schema
    return schema
 
app.openapi = custom_openapi
 
# Register all routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(public_router)
app.include_router(user_router)
app.include_router(seed_router)
 
 
@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Bank AI Backend",
        "version": "2.0.0",
        "auth": "Mobile OTP",
        "encryption": "AES-256-GCM",
        "test_mode": TEST_MODE,
        "docs": "http://localhost:4000/docs",
        "quick_start": {
            "1_seed": "POST /seed  (inserts test users)",
            "2_otp":  "POST /auth/send-otp  {mobile: '9999999999'}",
            "3_login":"POST /auth/verify-otp {mobile: '9999999999', otp: '<from step 2>'}",
            "4_use":  "Add header: Authorization: Bearer <session_token from step 3>",
        }
    }
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4000, reload=True)