"""
Services — Business Logic Layer
All DB operations, OTP management, session handling.
"""
 
import os
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
 
from config.database import get_pool
from utils.encryption import encrypt, decrypt, hash_value, mask, generate_otp
 
logger = logging.getLogger(__name__)
 
TEST_MODE = os.environ.get("TEST_MODE", "true").lower() == "true"
OTP_TTL_SECONDS = int(os.environ.get("OTP_TTL_SECONDS", "600"))   # 10 minutes
SESSION_TTL_HOURS = int(os.environ.get("SESSION_TTL_HOURS", "1"))  # 1 hour
 
 
def _uid(prefix="ID") -> str:
    return f"{prefix}{int(time.time() * 1000)}"
 
 
# ═══════════════════════════════════════════════════════════
# OTP SERVICE
# ═══════════════════════════════════════════════════════════
 
class OtpService:
 
    @staticmethod
    async def send_otp(mobile: str) -> dict:
        """
        Generate OTP for mobile number.
        - Invalidates any existing active OTP for this mobile
        - Stores SHA-256 hash of OTP (never plain OTP)
        - In TEST_MODE: returns OTP in response for easy testing
        - In production: sends via SMS provider
        """
        pool = await get_pool()
        mobile_hash = hash_value(mobile)
        otp = generate_otp(6)
        otp_hash = hash_value(otp)
        expires_at = datetime.utcnow() + timedelta(seconds=OTP_TTL_SECONDS)
 
        async with pool.acquire() as conn:
            # Invalidate all previous OTPs for this mobile
            await conn.execute(
                "UPDATE otps SET used = TRUE WHERE mobile_hash = $1 AND used = FALSE",
                mobile_hash
            )
            # Insert new OTP
            await conn.execute(
                """
                INSERT INTO otps (mobile_hash, otp_hash, purpose, expires_at)
                VALUES ($1, $2, 'LOGIN', $3)
                """,
                mobile_hash, otp_hash, expires_at
            )
 
        # In production: integrate SMS provider here
        # e.g. fast2sms, MSG91, Twilio
        # await sms_provider.send(mobile, f"Your Bank AI OTP is {otp}. Valid for 10 minutes.")
 
        logger.info(f"OTP generated for mobile_hash={mobile_hash[:8]}...")
 
        result = {
            "message": "OTP sent successfully",
            "expires_in_seconds": OTP_TTL_SECONDS,
        }
 
        # Return OTP only in test mode
        if TEST_MODE:
            result["otp_for_testing"] = otp
            result["note"] = "OTP returned because TEST_MODE=true. Remove in production."
 
        return result
 
    @staticmethod
    async def verify_otp(mobile: str, otp: str) -> dict:
        """
        Verify OTP. Returns session token on success.
        - Checks OTP hash match
        - Checks expiry
        - Checks max attempts
        - Marks OTP as used on success
        - Auto-creates user if first login
        """
        pool = await get_pool()
        mobile_hash = hash_value(mobile)
        otp_hash    = hash_value(otp)
 
        async with pool.acquire() as conn:
            # Get latest active OTP
            row = await conn.fetchrow(
                """
                SELECT id, otp_hash, attempts, max_attempts, expires_at, used
                FROM otps
                WHERE mobile_hash = $1 AND used = FALSE
                ORDER BY created_at DESC LIMIT 1
                """,
                mobile_hash
            )
 
            if not row:
                raise ValueError("No active OTP found. Please request a new OTP.")
 
            if row["used"]:
                raise ValueError("OTP already used. Please request a new OTP.")
 
            if datetime.utcnow() > row["expires_at"].replace(tzinfo=None):
                raise ValueError("OTP has expired. Please request a new OTP.")
 
            if row["attempts"] >= row["max_attempts"]:
                raise ValueError("Too many failed attempts. Please request a new OTP.")
 
            # Increment attempt counter
            await conn.execute(
                "UPDATE otps SET attempts = attempts + 1 WHERE id = $1",
                row["id"]
            )
 
            if row["otp_hash"] != otp_hash:
                remaining = row["max_attempts"] - row["attempts"] - 1
                raise ValueError(f"Incorrect OTP. {remaining} attempts remaining.")
 
            # ✓ OTP correct — mark as used
            await conn.execute(
                "UPDATE otps SET used = TRUE WHERE id = $1",
                row["id"]
            )
 
        # Get or create user
        user = await UserService.get_by_mobile_hash(mobile_hash)
        is_new = False
        if not user:
            user = await UserService.create(mobile, mobile_hash)
            is_new = True
 
        # Create session
        session = await SessionService.create(user["id"], mobile_hash)
 
        logger.info(f"OTP verified. user_id={user['id']} new={is_new}")
 
        return {
            "session_token": session["session_token"],
            "user_id": user["id"],
            "is_new_user": is_new,
            "mobile_masked": mask(mobile),
            "expires_at": session["expires_at"],
            "message": "Login successful",
        }
 
 
# ═══════════════════════════════════════════════════════════
# USER SERVICE
# ═══════════════════════════════════════════════════════════
 
class UserService:
 
    @staticmethod
    async def create(mobile: str, mobile_hash: str) -> dict:
        """Create new user. Encrypts mobile before storing."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (mobile, mobile_hash)
                VALUES ($1, $2)
                RETURNING id, kyc_status, created_at
                """,
                encrypt(mobile), mobile_hash
            )
            return {"id": str(row["id"]), "kyc_status": row["kyc_status"], "created_at": row["created_at"]}
 
    @staticmethod
    async def get_by_mobile_hash(mobile_hash: str) -> Optional[dict]:
        """Lookup user by mobile hash (fast indexed search without decryption)."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, name, mobile, email, kyc_status, created_at FROM users WHERE mobile_hash = $1",
                mobile_hash
            )
            if not row:
                return None
            mobile_plain = decrypt(row["mobile"]) if row["mobile"] else ""
            return {
                "id":            str(row["id"]),
                "name":          row["name"],
                "mobile":        mobile_plain,
                "mobile_masked": mask(mobile_plain),
                "email":         decrypt(row["email"]) if row["email"] else None,
                "kyc_status":    row["kyc_status"],
                "created_at":    row["created_at"],
            }
 
    @staticmethod
    async def get_by_id(user_id: str) -> Optional[dict]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, name, mobile, email, kyc_status, created_at FROM users WHERE id = $1",
                user_id
            )
            if not row:
                return None
            mobile_plain = decrypt(row["mobile"]) if row["mobile"] else ""
            return {
                "id":            str(row["id"]),
                "name":          row["name"],
                "mobile_masked": mask(mobile_plain),
                "email":         decrypt(row["email"]) if row["email"] else None,
                "kyc_status":    row["kyc_status"],
                "created_at":    row["created_at"],
            }
 
    @staticmethod
    async def update_profile(user_id: str, name: str = None, email: str = None):
        pool = await get_pool()
        async with pool.acquire() as conn:
            if name:
                await conn.execute("UPDATE users SET name=$1, updated_at=NOW() WHERE id=$2", name, user_id)
            if email:
                await conn.execute(
                    "UPDATE users SET email=$1, updated_at=NOW() WHERE id=$2",
                    encrypt(email), user_id
                )
 
 
# ═══════════════════════════════════════════════════════════
# SESSION SERVICE
# ═══════════════════════════════════════════════════════════
 
class SessionService:
 
    @staticmethod
    async def create(user_id: str, mobile_hash: str) -> dict:
        """Create a new session token after successful OTP verification."""
        pool = await get_pool()
        token      = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)
 
        # Pre-load company/account data into memory_data for AI agent
        company = await CompanyService.get_by_user(user_id)
        account = await BankAccountService.get_default(company["id"]) if company else None
 
        memory_data = {
            "user_id":        user_id,
            "company_id":     company["id"]        if company else None,
            "company_name":   company["company_name"] if company else None,
            "gstin":          company["gstin"]      if company else None,
            "account_number": account["account_number"] if account else None,
        }
 
        async with pool.acquire() as conn:
            # Invalidate old sessions for this user
            await conn.execute(
                "UPDATE sessions SET is_active=FALSE WHERE mobile_hash=$1",
                mobile_hash
            )
            await conn.execute(
                """
                INSERT INTO sessions (session_token, user_id, mobile_hash, company_id, memory_data, expires_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                """,
                token, user_id, mobile_hash,
                company["id"] if company else None,
                json.dumps(memory_data), expires_at
            )
 
        return {"session_token": token, "expires_at": expires_at}
 
    @staticmethod
    async def validate(token: str) -> Optional[dict]:
        """
        Validate session token.
        Returns session data if valid, None if expired/invalid.
        Also refreshes last_active_at.
        """
        if not token:
            return None
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT s.*, u.name, u.kyc_status
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.session_token = $1
                  AND s.is_active = TRUE
                  AND s.expires_at > NOW()
                """,
                token
            )
            if not row:
                return None
            await conn.execute(
                "UPDATE sessions SET last_active_at=NOW() WHERE session_token=$1",
                token
            )
            return dict(row)
 
    @staticmethod
    async def invalidate(token: str):
        """Logout — invalidate session."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET is_active=FALSE WHERE session_token=$1",
                token
            )
 
    @staticmethod
    async def update_memory(token: str, memory_data: dict):
        """Update AI agent memory data in session."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET memory_data=$1::jsonb, last_active_at=NOW() WHERE session_token=$2",
                json.dumps(memory_data), token
            )
 
 
# ═══════════════════════════════════════════════════════════
# COMPANY SERVICE
# ═══════════════════════════════════════════════════════════
 
class CompanyService:
 
    @staticmethod
    async def create(user_id: str, data: dict) -> dict:
        pool = await get_pool()
        gstin = data.get("gstin", "")
        pan   = data.get("pan", "")
 
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO companies
                    (user_id, company_name, gstin, gstin_hash, pan, pan_hash,
                     cin, company_type, industry, address, state, pincode)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                RETURNING id, company_name, kyc_status
                """,
                user_id,
                data["company_name"],
                encrypt(gstin) if gstin else None,
                hash_value(gstin) if gstin else None,
                encrypt(pan) if pan else None,
                hash_value(pan) if pan else None,
                encrypt(data.get("cin","")) if data.get("cin") else None,
                data.get("company_type","PRIVATE"),
                data.get("industry"),
                data.get("address"),
                data.get("state"),
                data.get("pincode"),
            )
            return {
                "id":           str(row["id"]),
                "company_name": row["company_name"],
                "gstin_masked": mask(gstin, 6) if gstin else None,
                "pan_masked":   mask(pan) if pan else None,
                "kyc_status":   row["kyc_status"],
                "gstin":        gstin,
            }
 
    @staticmethod
    async def get_by_user(user_id: str) -> Optional[dict]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM companies WHERE user_id=$1 AND is_active=TRUE ORDER BY created_at LIMIT 1",
                user_id
            )
            if not row:
                return None
            gstin_plain = decrypt(row["gstin"]) if row["gstin"] else ""
            pan_plain   = decrypt(row["pan"]) if row["pan"] else ""
            return {
                "id":           str(row["id"]),
                "company_name": row["company_name"],
                "gstin":        gstin_plain,
                "gstin_masked": mask(gstin_plain, 6),
                "pan_masked":   mask(pan_plain),
                "kyc_status":   row["kyc_status"],
                "state":        row["state"],
            }
 
    @staticmethod
    async def get_by_id(company_id: str) -> Optional[dict]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM companies WHERE id=$1", company_id)
            if not row:
                return None
            gstin_plain = decrypt(row["gstin"]) if row["gstin"] else ""
            return {
                "id":           str(row["id"]),
                "company_name": row["company_name"],
                "gstin":        gstin_plain,
                "gstin_masked": mask(gstin_plain, 6),
                "kyc_status":   row["kyc_status"],
                "state":        row["state"],
            }
 
 
# ═══════════════════════════════════════════════════════════
# BANK ACCOUNT SERVICE
# ═══════════════════════════════════════════════════════════
 
class BankAccountService:
 
    @staticmethod
    async def create(company_id: str, user_id: str, data: dict) -> dict:
        pool = await get_pool()
        acct = data["account_number"]
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO bank_accounts
                    (company_id, user_id, account_number, account_hash,
                     ifsc_code, bank_name, branch, account_type, available_balance, total_balance)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$9)
                RETURNING id, bank_name, account_type, available_balance
                """,
                company_id, user_id,
                encrypt(acct), hash_value(acct),
                encrypt(data["ifsc_code"]),
                data["bank_name"], data.get("branch"),
                data.get("account_type","CURRENT"),
                float(data.get("balance", 0)),
            )
            return {
                "id":                     str(row["id"]),
                "account_number_masked":  mask(acct),
                "bank_name":              row["bank_name"],
                "account_type":           row["account_type"],
                "available_balance":      float(row["available_balance"]),
            }
 
    @staticmethod
    async def get_default(company_id: str) -> Optional[dict]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM bank_accounts
                WHERE company_id=$1 AND is_active=TRUE
                ORDER BY is_default DESC, created_at ASC LIMIT 1
                """,
                company_id
            )
            if not row:
                return None
            acct_plain = decrypt(row["account_number"]) if row["account_number"] else ""
            ifsc_plain = decrypt(row["ifsc_code"])      if row["ifsc_code"] else ""
            return {
                "id":                    str(row["id"]),
                "account_number":        acct_plain,
                "account_number_masked": mask(acct_plain),
                "ifsc_code":             ifsc_plain,
                "bank_name":             row["bank_name"],
                "available_balance":     float(row["available_balance"]),
                "is_default":            row["is_default"],
            }
 
    @staticmethod
    async def get_all(company_id: str) -> List[dict]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM bank_accounts WHERE company_id=$1 AND is_active=TRUE",
                company_id
            )
            result = []
            for row in rows:
                acct = decrypt(row["account_number"]) if row["account_number"] else ""
                result.append({
                    "id":                    str(row["id"]),
                    "account_number_masked": mask(acct),
                    "bank_name":             row["bank_name"],
                    "account_type":          row["account_type"],
                    "available_balance":     float(row["available_balance"]),
                    "is_default":            row["is_default"],
                })
            return result
 
 
# ═══════════════════════════════════════════════════════════
# TRANSACTION SERVICE
# ═══════════════════════════════════════════════════════════
 
class TransactionService:
 
    @staticmethod
    async def create(company_id: str, user_id: str, data: dict) -> dict:
        pool = await get_pool()
        txn_id = _uid("TXN")
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO transactions
                    (transaction_id, company_id, user_id, beneficiary_account,
                     beneficiary_name, beneficiary_ifsc, amount, payment_mode, remarks, status)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'PENDING')
                RETURNING transaction_id, amount, payment_mode, status, created_at
                """,
                txn_id, company_id, user_id,
                encrypt(data["beneficiary_account"]),
                data["beneficiary_name"],
                data["beneficiary_ifsc"],
                data["amount"], data["payment_mode"],
                data.get("remarks",""),
            )
            return {
                "transaction_id": row["transaction_id"],
                "amount":         float(row["amount"]),
                "payment_mode":   row["payment_mode"],
                "beneficiary_name": data["beneficiary_name"],
                "status":         row["status"],
                "created_at":     row["created_at"],
            }
 
    @staticmethod
    async def get_list(company_id: str, limit: int = 50, status: str = "ALL") -> List[dict]:
        pool = await get_pool()
        query  = "SELECT * FROM transactions WHERE company_id=$1"
        params = [company_id]
        if status != "ALL":
            query += " AND status=$2"
            params.append(status)
        query += f" ORDER BY created_at DESC LIMIT ${len(params)+1}"
        params.append(limit)
 
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [{
                "transaction_id":  r["transaction_id"],
                "amount":          float(r["amount"]),
                "payment_mode":    r["payment_mode"],
                "beneficiary_name":r["beneficiary_name"],
                "txn_type":        r["txn_type"],
                "status":          r["status"],
                "created_at":      r["created_at"],
            } for r in rows]
 
    @staticmethod
    async def get_by_id(txn_id: str) -> Optional[dict]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM transactions WHERE transaction_id=$1", txn_id
            )
            if not row:
                return None
            bene_acct = decrypt(row["beneficiary_account"]) if row["beneficiary_account"] else ""
            return {
                "transaction_id":           row["transaction_id"],
                "amount":                   float(row["amount"]),
                "payment_mode":             row["payment_mode"],
                "beneficiary_name":         row["beneficiary_name"],
                "beneficiary_account_masked": mask(bene_acct),
                "status":                   row["status"],
                "utr_number":               decrypt(row["utr_number"]) if row["utr_number"] else None,
                "created_at":               row["created_at"],
            }
 
 
# ═══════════════════════════════════════════════════════════
# GST SERVICE
# ═══════════════════════════════════════════════════════════
 
class GstService:
 
    @staticmethod
    async def get_dues(company_id: str) -> List[dict]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM gst_records WHERE company_id=$1 AND status='PENDING' ORDER BY due_date",
                company_id
            )
            return [{
                "return_type":  r["return_type"],
                "period":       r["period"],
                "total_amount": float(r["total_amount"]),
                "due_date":     str(r["due_date"]) if r["due_date"] else None,
                "status":       r["status"],
            } for r in rows]
 
    @staticmethod
    async def create_challan(company_id: str, gstin: str, data: dict) -> dict:
        pool = await get_pool()
        total = sum([data.get(k, 0) for k in ["igst","cgst","sgst","cess"]])
        cpin  = _uid("CPIN")
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO gst_records
                    (company_id, gstin, return_type, period, igst, cgst, sgst, cess, total_amount, cpin, status)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'PENDING')
                """,
                company_id, encrypt(gstin),
                data.get("return_type","GSTR3B"), data.get("return_period"),
                data.get("igst",0), data.get("cgst",0), data.get("sgst",0), data.get("cess",0),
                total, cpin,
            )
        return {"cpin": cpin, "total_amount": total, "status": "CREATED"}
 
    @staticmethod
    def calculate(amount: float, rate: float, inclusive: bool) -> dict:
        """Pure calculation — no DB needed. Used by public endpoint."""
        r = rate / 100
        if inclusive:
            base = round(amount / (1 + r), 2)
            gst  = round(amount - base, 2)
        else:
            base = amount
            gst  = round(amount * r, 2)
        half = round(gst / 2, 2)
        return {
            "base_amount":      base,
            "gst_rate_percent": rate,
            "total_gst":        gst,
            "cgst":             half,
            "sgst":             half,
            "igst":             gst,
            "total_amount":     round(base + gst, 2),
            "inclusive":        inclusive,
        }
 
 
# ═══════════════════════════════════════════════════════════
# REMINDER SERVICE
# ═══════════════════════════════════════════════════════════
 
class ReminderService:
 
    @staticmethod
    async def create(company_id: str, user_id: str, data: dict) -> dict:
        pool = await get_pool()
        rid  = _uid("REM")
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO reminders
                    (reminder_id, company_id, user_id, title, payment_type, amount, due_date, notify_days_before)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                """,
                rid, company_id, user_id,
                data["title"], data.get("payment_type"),
                data.get("amount",0), data["due_date"],
                data.get("notify_days_before",3),
            )
        return {"reminder_id": rid, "status": "SET"}
 
    @staticmethod
    async def get_list(company_id: str) -> List[dict]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM reminders WHERE company_id=$1 AND is_active=TRUE ORDER BY due_date",
                company_id
            )
            return [{
                "reminder_id":       r["reminder_id"],
                "title":             r["title"],
                "due_date":          str(r["due_date"]),
                "amount":            float(r["amount"]),
                "notify_days_before":r["notify_days_before"],
            } for r in rows]
 
    @staticmethod
    async def delete(reminder_id: str):
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE reminders SET is_active=FALSE WHERE reminder_id=$1",
                reminder_id
            )
 
 
# ═══════════════════════════════════════════════════════════
# AUDIT SERVICE
# ═══════════════════════════════════════════════════════════
 
class AuditService:
 
    @staticmethod
    async def log(user_id: str, action: str, entity: str = None,
                  entity_id: str = None, ip: str = None, meta: dict = None):
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO audit_logs (user_id, action, entity, entity_id, ip_address, metadata) VALUES ($1,$2,$3,$4,$5,$6)",
                    user_id, action, entity, entity_id, ip, meta or {}
                )
        except Exception as e:
            logger.error(f"Audit log failed: {e}")
 
 
# ═══════════════════════════════════════════════════════════
# SEED SERVICE  (test data only)
# ═══════════════════════════════════════════════════════════
 
class SeedService:
 
    @staticmethod
    async def seed() -> List[dict]:
        results = []
 
        for mobile, name, gstin, company_name, acct, ifsc, bank, balance in [
            ("9999999999","Rahul Sharma","27AAAPD1234F1ZK","Acme Pvt Ltd",  "1234567890","HDFC0001234","HDFC Bank",650000),
            ("8888888888","Priya Singh", "29AABCB1234C1ZK","Beta Enterprises","9876543210","SBIN0001234","SBI",      980000),
        ]:
            mobile_hash = hash_value(mobile)
 
            # Check if user exists
            existing = await UserService.get_by_mobile_hash(mobile_hash)
            if existing:
                results.append({"mobile": mask(mobile), "status": "already_exists"})
                continue
 
            user = await UserService.create(mobile, mobile_hash)
            await UserService.update_profile(user["id"], name=name)
 
            company = await CompanyService.create(user["id"], {
                "company_name": company_name,
                "gstin": gstin,
                "state": "Maharashtra" if "27" in gstin else "Karnataka",
            })
 
            await BankAccountService.create(company["id"], user["id"], {
                "account_number": acct,
                "ifsc_code": ifsc,
                "bank_name": bank,
                "account_type": "CURRENT",
                "balance": balance,
            })
 
            results.append({"mobile": mask(mobile), "name": name, "company": company_name, "status": "created"})
 
        return results