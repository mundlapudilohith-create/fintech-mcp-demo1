"""
MCP Server using FastMCP
Implements GST calculation tools with:
  - Real GSTIN API validation (via gstin_validator.py)
  - Production query logging (via query_logger.py)
"""
from fastmcp import FastMCP
from mcp_server.gst_calculator import GSTCalculator
from typing import List
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Init ───────────────────────────────────────────────────────────────────────
mcp        = FastMCP("GST Calculator")
calculator = GSTCalculator()

# Query logger — writes to logs/queries.jsonl + logs/app.log
# Fixed — linter safe
import importlib.util
_mod = importlib.util.find_spec("query_logger")
if _mod is not None:
    from query_logger import query_logger
    _logging_enabled = True
else:
    query_logger     = None
    _logging_enabled = False


def _log(tool_name: str, intents: list, latency_ms: float, success: bool, error: str = None):
    """Helper to fire-and-forget a log entry"""
    if _logging_enabled:
        try:
            query_logger.log_query(
                query      = tool_name,
                intents    = intents,
                tools      = [tool_name],
                latency_ms = latency_ms,
                success    = success,
                error      = error,
            )
        except Exception as e:
            logger.warning(f"Logging failed: {e}")


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
async def calculate_gst(base_amount: float, gst_rate: float) -> dict:
    """
    Calculate GST amount and total from base amount.

    Args:
        base_amount: The base amount before GST (e.g., 10000)
        gst_rate: The GST rate percentage (e.g., 18 for 18%)

    Returns:
        Dictionary with base_amount, gst_amount, total_amount, and gst_rate

    Example:
        calculate_gst(10000, 18)
        Returns: {"base_amount": 10000, "gst_amount": 1800, "total_amount": 11800, "gst_rate": 18}
    """
    logger.info(f"Calculating GST: base={base_amount}, rate={gst_rate}")
    t0 = time.time()
    try:
        result = await calculator.calculate_gst(base_amount, gst_rate)
        _log("calculate_gst", ["calculate_gst"], (time.time()-t0)*1000, True)
        logger.info(f"GST calculated: {result}")
        return result
    except Exception as e:
        _log("calculate_gst", ["calculate_gst"], (time.time()-t0)*1000, False, str(e))
        logger.error(f"Error calculating GST: {e}")
        raise


@mcp.tool()
def reverse_calculate_gst(total_amount: float, gst_rate: float) -> dict:
    """
    Calculate base amount from total amount (reverse calculation).

    Args:
        total_amount: The total amount including GST (e.g., 11800)
        gst_rate: The GST rate percentage (e.g., 18 for 18%)

    Returns:
        Dictionary with total_amount, base_amount, gst_amount, and gst_rate

    Example:
        reverse_calculate_gst(11800, 18)
        Returns: {"total_amount": 11800, "base_amount": 10000, "gst_amount": 1800, "gst_rate": 18}
    """
    logger.info(f"Reverse GST: total={total_amount}, rate={gst_rate}")
    t0 = time.time()
    try:
        result = calculator.reverse_calculate_gst(total_amount, gst_rate)
        _log("reverse_calculate_gst", ["reverse_gst"], (time.time()-t0)*1000, True)
        return result
    except Exception as e:
        _log("reverse_calculate_gst", ["reverse_gst"], (time.time()-t0)*1000, False, str(e))
        logger.error(f"Error in reverse calculation: {e}")
        raise


@mcp.tool()
async def gst_breakdown(base_amount: float, gst_rate: float, is_intra_state: bool = True) -> dict:
    """
    Get detailed GST breakdown showing CGST, SGST, or IGST.

    Args:
        base_amount: The base amount before GST
        gst_rate: The GST rate percentage
        is_intra_state: True for intra-state (CGST+SGST), False for inter-state (IGST)

    Returns:
        Dictionary with breakdown of CGST/SGST/IGST

    Example:
        gst_breakdown(10000, 18, True)
        Returns: {"base_amount": 10000, ..., "breakdown": {"cgst": 900, "sgst": 900, "igst": 0}}
    """
    logger.info(f"GST breakdown: base={base_amount}, rate={gst_rate}, intra={is_intra_state}")
    t0 = time.time()
    try:
        result = await calculator.get_gst_breakdown_async(base_amount, gst_rate, is_intra_state)
        _log("gst_breakdown", ["gst_breakdown"], (time.time()-t0)*1000, True)
        return result
    except Exception as e:
        _log("gst_breakdown", ["gst_breakdown"], (time.time()-t0)*1000, False, str(e))
        logger.error(f"Error in breakdown: {e}")
        raise


@mcp.tool()
async def compare_gst_rates(base_amount: float, rates: List[float]) -> dict:
    """
    Compare the same base amount with different GST rates.

    Args:
        base_amount: The base amount to compare
        rates: List of GST rates (e.g., [5, 12, 18, 28])

    Returns:
        Dictionary with comparisons for each rate and max difference

    Example:
        compare_gst_rates(10000, [5, 12, 18])
    """
    logger.info(f"Comparing GST rates: base={base_amount}, rates={rates}")
    t0 = time.time()
    try:
        result = await calculator.compare_gst_rates_async(base_amount, rates)
        _log("compare_gst_rates", ["compare_rates"], (time.time()-t0)*1000, True)
        return result
    except Exception as e:
        _log("compare_gst_rates", ["compare_rates"], (time.time()-t0)*1000, False, str(e))
        logger.error(f"Error in rate comparison: {e}")
        raise


@mcp.tool()
async def validate_gstin(gstin: str) -> dict:
    """
    Validate GSTIN — uses real GST API if GSTIN_API_KEY is set, local fallback otherwise.

    Args:
        gstin: 15-character GSTIN (e.g., "29ABCDE1234F1Z5")

    Returns:
        Validation result. With API key: includes trade_name, legal_name, status,
        registration_date, business_type. Without API key: structural check only.

    Example:
        validate_gstin("29ABCDE1234F1Z5")
        Returns: {"valid": True, "gstin": "...", "state": "Karnataka", "components": {...}}

    Setup for live data:
        Add to .env:  GSTIN_API_KEY=your_key
                      GSTIN_API_PROVIDER=gst_suvidha   # or mastergst
    """
    logger.info(f"Validating GSTIN: {gstin}")
    t0 = time.time()
    try:
        # Uses real API if configured, local regex fallback otherwise
        result = await calculator.validate_gstin_async(gstin)
        _log("validate_gstin", ["validate_gstin"], (time.time()-t0)*1000, True)
        logger.info(f"GSTIN valid={result.get('valid')} source={result.get('source','local')}")
        return result
    except Exception as e:
        _log("validate_gstin", ["validate_gstin"], (time.time()-t0)*1000, False, str(e))
        logger.error(f"Error validating GSTIN: {e}")
        raise


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting GST Calculator MCP Server...")
    mcp.run()