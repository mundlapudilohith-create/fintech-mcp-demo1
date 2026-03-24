"""
GSTIN API Connector
Tries real GST verification APIs with local fallback.

Priority chain:
  1. GST Suvidha Provider API (if API key configured)
  2. MasterGST API (if API key configured)  
  3. Local regex validation (always available, no API key needed)

Setup:
  Add to your .env file:
    GSTIN_API_KEY=your_key_here
    GSTIN_API_PROVIDER=gst_suvidha   # or mastergst
"""

import re
import os
import logging
import httpx
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# ── State code lookup ──────────────────────────────────────────────────────────
STATE_CODES: Dict[str, str] = {
    "01": "Jammu & Kashmir",    "02": "Himachal Pradesh",   "03": "Punjab",
    "04": "Chandigarh",         "05": "Uttarakhand",        "06": "Haryana",
    "07": "Delhi",              "08": "Rajasthan",          "09": "Uttar Pradesh",
    "10": "Bihar",              "11": "Sikkim",             "12": "Arunachal Pradesh",
    "13": "Nagaland",           "14": "Manipur",            "15": "Mizoram",
    "16": "Tripura",            "17": "Meghalaya",          "18": "Assam",
    "19": "West Bengal",        "20": "Jharkhand",          "21": "Odisha",
    "22": "Chhattisgarh",       "23": "Madhya Pradesh",     "24": "Gujarat",
    "25": "Daman & Diu",        "26": "Dadra & Nagar Haveli", "27": "Maharashtra",
    "28": "Andhra Pradesh",     "29": "Karnataka",          "30": "Goa",
    "31": "Lakshadweep",        "32": "Kerala",             "33": "Tamil Nadu",
    "34": "Puducherry",         "35": "Andaman & Nicobar",  "36": "Telangana",
    "37": "Andhra Pradesh (New)", "38": "Ladakh",           "97": "Other Territory",
    "99": "Centre Jurisdiction",
}

GSTIN_REGEX = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}$"


class GSTINValidator:

    def __init__(self):
        self.api_key      = os.getenv("GSTIN_API_KEY", "")
        self.api_provider = os.getenv("GSTIN_API_PROVIDER", "local").lower()
        self.timeout      = 8  # seconds

        if self.api_key:
            logger.info(f"✓ GSTIN API configured: provider={self.api_provider}")
        else:
            logger.info("ℹ  No GSTIN_API_KEY found — using local validation only")

    # ── Public entry point ─────────────────────────────────────────────────────

    async def validate(self, gstin: str) -> Dict[str, Any]:
        """
        Validate a GSTIN. Returns a standardised response dict regardless
        of which provider (or local fallback) handled the request.
        """
        gstin = gstin.strip().upper()

        # Always run local format check first — reject obviously bad GSTINs
        # before wasting an API call
        local = self._local_validate(gstin)
        if not local["valid"]:
            return local

        # Try real API if configured
        if self.api_key:
            try:
                if self.api_provider == "mastergst":
                    return await self._mastergst(gstin)
                else:
                    return await self._gst_suvidha(gstin)
            except httpx.TimeoutException:
                logger.warning(f"GSTIN API timeout for {gstin} — falling back to local")
            except httpx.HTTPStatusError as e:
                logger.warning(f"GSTIN API HTTP {e.response.status_code} — falling back to local")
            except Exception as e:
                logger.warning(f"GSTIN API error: {e} — falling back to local")

        # Local fallback
        return local

    # ── Provider: GST Suvidha (sandbox + production) ──────────────────────────

    async def _gst_suvidha(self, gstin: str) -> Dict[str, Any]:
        """
        GST Suvidha Provider API
        Docs: https://developer.gstsuvidha.com/

        Free sandbox: register at gstsuvidha.com → get test API key
        Production  : paid plan required
        """
        url = "https://api.gstsuvidha.com/taxpayer/v1/search"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"gstin": gstin}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Map provider response → standard format
        taxpayer = data.get("data", {})
        return {
            "valid": True,
            "gstin": gstin,
            "source": "gst_suvidha_api",
            "trade_name":    taxpayer.get("tradeNam", ""),
            "legal_name":    taxpayer.get("lgnm", ""),
            "status":        taxpayer.get("sts", ""),
            "registration_date": taxpayer.get("rgdt", ""),
            "business_type": taxpayer.get("ctb", ""),
            "state":         taxpayer.get("pradr", {}).get("addr", {}).get("stcd", ""),
            "components":    self._parse_components(gstin),
            "checked_at":    datetime.utcnow().isoformat() + "Z",
        }

    # ── Provider: MasterGST ───────────────────────────────────────────────────

    async def _mastergst(self, gstin: str) -> Dict[str, Any]:
        """
        MasterGST API
        Docs : https://mastergst.com/gst-api/
        Plans: Free tier available (limited calls/month)

        Set env: GSTIN_API_PROVIDER=mastergst
        """
        url = f"https://api.mastergst.com/taxpayerapi/v0.3/authenticate/gstin/{gstin}"
        headers = {
            "ip_address": "127.0.0.1",  # Required by MasterGST
            "client_id":  self.api_key,
            "client_secret": os.getenv("GSTIN_API_SECRET", ""),
            "username":   os.getenv("GSTIN_API_USERNAME", ""),
            "password":   os.getenv("GSTIN_API_PASSWORD", ""),
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        taxpayer = data.get("taxpayerInfo", {})
        return {
            "valid": True,
            "gstin": gstin,
            "source": "mastergst_api",
            "trade_name":    taxpayer.get("tradeNam", ""),
            "legal_name":    taxpayer.get("lgnm", ""),
            "status":        taxpayer.get("sts", ""),
            "registration_date": taxpayer.get("rgdt", ""),
            "business_type": taxpayer.get("ctb", ""),
            "state":         STATE_CODES.get(gstin[:2], "Unknown"),
            "components":    self._parse_components(gstin),
            "checked_at":    datetime.utcnow().isoformat() + "Z",
        }

    # ── Local validation (regex + structure) ──────────────────────────────────

    def _local_validate(self, gstin: str) -> Dict[str, Any]:
        """
        Offline structural validation — no API call needed.
        Checks: format, state code, embedded PAN structure.
        """
        if not re.match(GSTIN_REGEX, gstin):
            return {
                "valid": False,
                "gstin": gstin,
                "source": "local",
                "error": self._format_error(gstin),
                "checked_at": datetime.utcnow().isoformat() + "Z",
            }

        components = self._parse_components(gstin)
        state_code = components["state_code"]
        state_name = STATE_CODES.get(state_code, "Unknown State")

        return {
            "valid": True,
            "gstin": gstin,
            "source": "local",
            "state": state_name,
            "components": components,
            "note": "Structural validation only — register an API key for live business data",
            "checked_at": datetime.utcnow().isoformat() + "Z",
        }

    def _parse_components(self, gstin: str) -> Dict[str, str]:
        return {
            "state_code":     gstin[0:2],
            "pan_number":     gstin[2:12],
            "entity_number":  gstin[12],
            "default_letter": gstin[13],   # always Z
            "checksum":       gstin[14],
        }

    def _format_error(self, gstin: str) -> str:
        if len(gstin) != 15:
            return f"Invalid length: expected 15 chars, got {len(gstin)}"
        if not re.match(r"^[0-9]{2}", gstin):
            return "First 2 characters must be digits (state code)"
        if not re.match(r"^[0-9]{2}[A-Z]{5}", gstin):
            return "Characters 3-7 must be uppercase letters (PAN prefix)"
        if not re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}", gstin):
            return "Characters 8-11 must be digits (PAN digits)"
        if gstin[13] != "Z":
            return "Character 14 must be 'Z'"
        return "Invalid GSTIN format"


# ── Singleton ─────────────────────────────────────────────────────────────────
gstin_validator = GSTINValidator()


# ── Convenience wrapper for sync callers ──────────────────────────────────────
async def validate_gstin(gstin: str) -> Dict[str, Any]:
    return await gstin_validator.validate(gstin)