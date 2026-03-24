"""
Encryption & Decryption — AES-256-GCM
All sensitive DB fields go through here before storage and after reading.
"""
 
import os, base64, hashlib, secrets, logging
from typing import Optional
 
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
 
logger = logging.getLogger(__name__)
 
_RAW_KEY = os.environ.get("ENCRYPTION_KEY", "")
if not _RAW_KEY:
    _RAW_KEY = secrets.token_hex(32)
    logger.warning(f"\n  ⚠️  ENCRYPTION_KEY not set — auto-generated (data lost on restart):\n  ENCRYPTION_KEY={_RAW_KEY}\n  Add to .env to persist!\n")
 
try:
    _KEY = bytes.fromhex(_RAW_KEY)
    assert len(_KEY) == 32, f"Key must be 32 bytes, got {len(_KEY)}"
except Exception as e:
    raise EnvironmentError(f"Invalid ENCRYPTION_KEY: {e}")
 
 
def encrypt(value: str) -> str:
    """Encrypt with AES-256-GCM → base64url string stored in DB."""
    if not value:
        return ""
    nonce = os.urandom(12)
    ct    = AESGCM(_KEY).encrypt(nonce, value.encode(), None)
    return base64.urlsafe_b64encode(nonce + ct).decode()
 
 
def decrypt(ciphertext: str) -> str:
    """Decrypt base64url AES-256-GCM ciphertext → original string."""
    if not ciphertext:
        return ""
    try:
        raw   = base64.urlsafe_b64decode(ciphertext.encode())
        nonce = raw[:12]
        ct    = raw[12:]
        return AESGCM(_KEY).decrypt(nonce, ct, None).decode()
    except Exception:
        raise ValueError("Decryption failed — wrong key or corrupted data")
 
 
def hash_value(value: str) -> str:
    """SHA-256 of value (uppercase stripped). Used for indexed DB lookups."""
    if not value:
        return ""
    return hashlib.sha256(value.strip().upper().encode()).hexdigest()
 
 
def mask(value: str, show_last: int = 4) -> str:
    """Show only last N chars: 9999999999 → XXXXXX9999"""
    if not value or len(value) <= show_last:
        return value
    return "X" * (len(value) - show_last) + value[-show_last:]
 
 
def generate_otp(length: int = 6) -> str:
    """Cryptographically secure numeric OTP."""
    return "".join([str(secrets.randbelow(10)) for _ in range(length)])