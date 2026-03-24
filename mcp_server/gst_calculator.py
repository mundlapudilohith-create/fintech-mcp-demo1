"""
GST Calculator - Business Logic
Primary API: http://localhost:3000/gst-calculation
Fallback: local calculation if API is unreachable
"""
from typing import Dict, List, Any, Optional
import httpx
import logging

logger = logging.getLogger(__name__)

# ── Your local GST API ─────────────────────────────────────────────────────────
GST_API_BASE    = "http://localhost:3000"
GST_CALC_URL    = f"{GST_API_BASE}/gst-calculation"
API_TIMEOUT     = 10.0 


class GSTCalculator:
    """GST calculation — calls http://localhost:3000/gst-calculation, local fallback"""

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        # Allow override via constructor; default to localhost:3000
        self.api_url        = api_url or GST_CALC_URL
        self.api_key        = api_key
        self.use_local_api  = True   # always try localhost:3000 first

        # Optional: load gstin_validator.py if present in workspace
        # To enable real API validation: add gstin_validator.py to your project root
        # and set GSTIN_API_KEY in your .env file
        import importlib
        _mod = importlib.util.find_spec("gstin_validator")
        if _mod is not None:
            from gstin_validator import gstin_validator as _v
            self._gstin_validator = _v
            logger.info("✓ Real GSTIN validator loaded")
        else:
            self._gstin_validator = None
            logger.info("ℹ  gstin_validator.py not in workspace — using local regex only")

    # ── GST Calculation ────────────────────────────────────────────────────────

    async def calculate_gst(self, base_amount: float, gst_rate: float) -> Dict[str, Any]:
        """
        Call http://localhost:3000/gst-calculation.
        Falls back to local math if the API is unreachable or returns an error.
        """
        try:
            return await self._call_gst_api(base_amount, gst_rate)
        except Exception as e:
            logger.warning(f"localhost:3000/gst-calculation unreachable ({e}) — using local calc")
            return self._calculate_locally(base_amount, gst_rate)

    async def _call_gst_api(self, base_amount: float, gst_rate: float) -> Dict[str, Any]:
        """
        POST http://localhost:3000/gst-calculation
        Request body : { "base_amount": 10000, "gst_rate": 18 }
        Expected resp: { "base_amount": 10000, "gst_rate": 18,
                         "gst_amount": 1800, "total_amount": 11800 }
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "base_amount": base_amount,
            "gst_rate":    gst_rate
        }

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        logger.info(f"localhost:3000/gst-calculation responded: {data}")

        # Normalise response — ensure all expected keys exist
        return {
            "base_amount":  float(data.get("base_amount",  base_amount)),
            "gst_rate":     float(data.get("gst_rate",     gst_rate)),
            "gst_amount":   float(data.get("gst_amount",   0)),
            "total_amount": float(data.get("total_amount", 0)),
            "source":       "localhost_api"
        }

    def _calculate_locally(self, base_amount: float, gst_rate: float) -> Dict[str, Any]:
        if base_amount < 0 or gst_rate < 0:
            raise ValueError("Amount and rate must be positive")
        gst_amount   = (base_amount * gst_rate) / 100
        total_amount = base_amount + gst_amount
        return {
            "base_amount":  round(base_amount, 2),
            "gst_rate":     gst_rate,
            "gst_amount":   round(gst_amount, 2),
            "total_amount": round(total_amount, 2),
            "source":       "local"
        }

    # ── Reverse GST (local — no separate API endpoint) ─────────────────────────

    def reverse_calculate_gst(self, total_amount: float, gst_rate: float) -> Dict[str, Any]:
        if total_amount < 0 or gst_rate < 0:
            raise ValueError("Amount and rate must be positive")
        base_amount = total_amount / (1 + gst_rate / 100)
        gst_amount  = total_amount - base_amount
        return {
            "total_amount": round(total_amount, 2),
            "gst_rate":     gst_rate,
            "base_amount":  round(base_amount, 2),
            "gst_amount":   round(gst_amount, 2)
        }

    # ── GST Breakdown (uses API result for base calculation) ───────────────────

    async def get_gst_breakdown_async(
        self, base_amount: float, gst_rate: float, is_intra_state: bool = True
    ) -> Dict[str, Any]:
        """Async breakdown — gets base calc from localhost:3000 then applies split"""
        calculation = await self.calculate_gst(base_amount, gst_rate)
        return self._apply_breakdown(calculation, gst_rate, is_intra_state)

    def get_gst_breakdown(
        self, base_amount: float, gst_rate: float, is_intra_state: bool = True
    ) -> Dict[str, Any]:
        """Sync breakdown — uses local calc (for backward compat)"""
        calculation = self._calculate_locally(base_amount, gst_rate)
        return self._apply_breakdown(calculation, gst_rate, is_intra_state)

    def _apply_breakdown(
        self, calculation: Dict, gst_rate: float, is_intra_state: bool
    ) -> Dict[str, Any]:
        if is_intra_state:
            cgst = calculation["gst_amount"] / 2
            breakdown = {
                "type":      "Intra-State",
                "cgst":      round(cgst, 2),
                "sgst":      round(cgst, 2),
                "igst":      0,
                "cgst_rate": gst_rate / 2,
                "sgst_rate": gst_rate / 2,
                "igst_rate": 0
            }
        else:
            breakdown = {
                "type":      "Inter-State",
                "cgst":      0,
                "sgst":      0,
                "igst":      round(calculation["gst_amount"], 2),
                "cgst_rate": 0,
                "sgst_rate": 0,
                "igst_rate": gst_rate
            }
        return {**calculation, "breakdown": breakdown}

    # ── Compare Rates (calls API for each rate) ────────────────────────────────

    async def compare_gst_rates_async(
        self, base_amount: float, rates: List[float]
    ) -> Dict[str, Any]:
        """Async compare — fetches each rate from localhost:3000"""
        if not rates:
            raise ValueError("Rates list cannot be empty")
        import asyncio
        results = await asyncio.gather(
            *[self.calculate_gst(base_amount, r) for r in rates],
            return_exceptions=False
        )
        comparisons = [{"rate": r, **res} for r, res in zip(rates, results)]
        comparisons.sort(key=lambda x: x["rate"])
        lowest = comparisons[0]["total_amount"]
        for c in comparisons:
            c["difference_from_lowest"] = round(c["total_amount"] - lowest, 2)
        return {
            "base_amount":    base_amount,
            "comparisons":    comparisons,
            "lowest_rate":    comparisons[0]["rate"],
            "highest_rate":   comparisons[-1]["rate"],
            "max_difference": round(comparisons[-1]["total_amount"] - comparisons[0]["total_amount"], 2)
        }

    def compare_gst_rates(self, base_amount: float, rates: List[float]) -> Dict[str, Any]:
        """Sync compare — local calc (for backward compat)"""
        if not rates:
            raise ValueError("Rates list cannot be empty")
        comparisons = [{"rate": r, **self._calculate_locally(base_amount, r)} for r in rates]
        comparisons.sort(key=lambda x: x["rate"])
        lowest = comparisons[0]["total_amount"]
        for c in comparisons:
            c["difference_from_lowest"] = round(c["total_amount"] - lowest, 2)
        return {
            "base_amount":    base_amount,
            "comparisons":    comparisons,
            "lowest_rate":    comparisons[0]["rate"],
            "highest_rate":   comparisons[-1]["rate"],
            "max_difference": round(comparisons[-1]["total_amount"] - comparisons[0]["total_amount"], 2)
        }

    # ── GSTIN Validation ───────────────────────────────────────────────────────

    async def validate_gstin_async(self, gstin: str) -> Dict[str, Any]:
        """Async — uses gstin_validator.py (real API) if available, local fallback"""
        if self._gstin_validator:
            return await self._gstin_validator.validate(gstin)
        return self.validate_gstin(gstin)

    def validate_gstin(self, gstin: str) -> Dict[str, Any]:
        """Sync local regex validation"""
        import re
        if not gstin or not isinstance(gstin, str):
            return {"valid": False, "error": "GSTIN must be a non-empty string"}
        pattern = re.compile(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}$')
        if not pattern.match(gstin.strip().upper()):
            return {
                "valid": False,
                "error": "Invalid GSTIN format",
                "expected_format": "2-digit state + 5 letters + 4 digits + 1 letter + 1 alphanumeric + Z + 1 alphanumeric"
            }
        gstin = gstin.strip().upper()
        return {
            "valid": True,
            "gstin": gstin,
            "components": {
                "state_code":     gstin[:2],
                "pan_number":     gstin[2:12],
                "entity_number":  gstin[12],
                "default_letter": gstin[13],
                "checksum":       gstin[14]
            }
        }