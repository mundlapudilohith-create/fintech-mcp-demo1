"""
Local ML Service — Bank AI Assistant
Uses custom domain-specific ML + NLP model for intent detection.
100% on-premise — no data leaves your infrastructure.
"""
from typing import List, Dict, Any, Optional
import json
import logging
 
from ml_intent_classifier import intent_classifier
from client.mcp_client import bank_client_manager, gst_client_manager, info_client_manager
from config.config import settings          # FIX 4: module-level import, not per-call
 
logger = logging.getLogger(__name__)
 
 
class LocalMLService:
    """
    Local ML-based service for intent detection and tool calling.
    Routes tool calls across Bank / GST / Info MCP servers.
    """
 
    def __init__(self):
        self.intent_classifier = intent_classifier
        logger.info("✓ Local ML Service initialized (NO external LLM)")
 
    async def process_query(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_token: Optional[str] = None,   # ← user's backend session token
    ) -> Dict[str, Any]:
        """
        Process user query using LOCAL ML model.
        No data sent to external APIs — all processing on-premise.
        session_token: if provided, injected as api_key for private MCP tools.
                       Falls back to settings.bank_api_key for public tools.
        """
        logger.info(f"Processing query locally: {user_message[:100]}...")
 
        # STEP 1: ML-based intent + entity detection
        analysis         = self.intent_classifier.process_query(user_message)
        intents_detected = analysis.get("intents_detected", [])
        tool_calls_specs = analysis.get("tool_calls", [])
 
        # Inject api_key centrally:
        # - Private tools → use session_token (authenticates against backend DB)
        # - Public tools  → use bank_api_key (static, no user context needed)
        INFO_SERVER_TOOLS = {
            # Onboarding info tools — no api_key param
            "get_company_onboarding_guide", "get_company_required_documents",
            "get_validation_formats", "get_onboarding_faq",
            "get_bank_onboarding_guide", "get_vendor_onboarding_guide",
            # GST calculator tools — no api_key param
            "calculate_gst", "reverse_calculate_gst", "gst_breakdown",
            "compare_gst_rates", "validate_gstin",
        }
 
        # Determine which key to inject
        auth_key = session_token if session_token else settings.bank_api_key
 
        for spec in tool_calls_specs:
            tool_nm = spec.get("tool_name", "")
            if tool_nm in INFO_SERVER_TOOLS:
                continue   # info/gst-calc server has no api_key param — skip
            params = spec.get("parameters", {})
            if "api_key" not in params:
                params["api_key"] = auth_key
                spec["parameters"] = params
 
        logger.info(f"✓ Intents detected : {intents_detected}")
        logger.info(f"✓ Tool calls       : {len(tool_calls_specs)}")
 
        # STEP 2: Get all 3 MCP clients and build routing map
        bank_client = await bank_client_manager.get_client()
        gst_client  = await gst_client_manager.get_client()
        info_client = await info_client_manager.get_client()
 
        tool_client_map = {}
        for tool in bank_client.available_tools:
            tool_client_map[tool["name"]] = bank_client
        for tool in gst_client.available_tools:
            tool_client_map[tool["name"]] = gst_client
        for tool in info_client.available_tools:
            tool_client_map[tool["name"]] = info_client
 
        # STEP 3: Execute tool calls via MCP
        mcp_results = []
 
        for idx, tool_spec in enumerate(tool_calls_specs, 1):
            tool_name       = tool_spec["tool_name"]
            tool_parameters = tool_spec["parameters"]
 
            logger.info(f"[{idx}/{len(tool_calls_specs)}] Executing: {tool_name}")
            logger.info(f"  Parameters: {tool_parameters}")
 
            client = tool_client_map.get(tool_name)
            if not client:
                logger.error(f"No client for tool: {tool_name}")
                mcp_results.append({
                    "tool":    tool_name,
                    "input":   tool_parameters,
                    "error":   f"Tool '{tool_name}' not found in any MCP server (Bank / GST / Info)",
                    "success": False,
                })
                continue
 
            try:
                result = await client.call_tool(tool_name, tool_parameters)
 
                if result.get("success") and result.get("result"):
                    try:
                        parsed = (
                            json.loads(result["result"])
                            if isinstance(result["result"], str)
                            else result["result"]
                        )
                    except json.JSONDecodeError:
                        parsed = result["result"]
 
                    mcp_results.append({
                        "tool":    tool_name,
                        "input":   tool_parameters,
                        "result":  parsed,
                        "success": True,
                    })
                    logger.info(f"[{idx}/{len(tool_calls_specs)}] ✓ Success")
 
                else:
                    error_msg = result.get("error", "Tool returned no result")
                    mcp_results.append({
                        "tool":    tool_name,
                        "input":   tool_parameters,
                        "error":   error_msg,
                        "success": False,
                    })
                    logger.error(f"[{idx}/{len(tool_calls_specs)}] ✗ Failed: {error_msg}")
 
            except Exception as e:
                logger.error(f"[{idx}/{len(tool_calls_specs)}] Exception: {e}")
                mcp_results.append({
                    "tool":    tool_name,
                    "input":   tool_parameters,
                    "error":   str(e),
                    "success": False,
                })
 
        # STEP 4: Generate response (template-based, NO LLM)
        response_text = self._generate_response(mcp_results, intents_detected, user_message)
 
        return {
            "success":          True,
            "intents_detected": intents_detected,
            "is_multi_intent":  len(intents_detected) > 1,
            "tool_calls":       mcp_results,
            "response":         response_text,
            "stop_reason":      "complete",
            "ml_model":         "local_domain_specific",
            "debug_info": {
                "total_tools_called": len(mcp_results),
                "successful_tools":   len([r for r in mcp_results if r.get("success")]),
                "intents":            intents_detected,
                "entities_extracted": analysis.get("entities", {}),
            },
        }
 
    # ─────────────────────────────────────────────────────────
    # RESPONSE TEMPLATES
    # ─────────────────────────────────────────────────────────
 
    def _generate_response(
        self,
        mcp_results: List[Dict],
        intents: List[str],
        user_query: str,
    ) -> str:
        """Generate natural language response using templates (NO LLM).
 
        Design rule: every tool that produces a list builds the whole
        block as a single "\n".join(lines) string before appending to
        response_parts, so that the final "\n\n".join(response_parts)
        only inserts blank lines *between* tool blocks, not between
        individual bullet points.   (FIX 3)
        """
        if not mcp_results:
            return "I couldn't find relevant information for your query. Please try rephrasing."
 
        response_parts = []
 
        for result in mcp_results:
            tool_name = result.get("tool", "")
 
            if not result.get("success"):
                response_parts.append(f"❌ Error in {tool_name}: {result.get('error', 'Unknown error')}")
                continue
 
            data = result.get("result", {})
 
            # FIX 2: guard against non-dict MCP responses
            if not isinstance(data, dict):
                response_parts.append(f"⚠️ {tool_name} returned unexpected response: {data}")
                continue
 
            # ── CORE PAYMENT ──────────────────────────────────────────
            if tool_name == "initiate_payment":
                response_parts.append(
                    f"**Payment Initiated ✅**\n"
                    f"• Transaction ID : {data.get('transaction_id', '')}\n"
                    f"• Amount         : ₹{data.get('amount', 0):,.2f}\n"
                    f"• Mode           : {data.get('payment_mode', '')}\n"
                    f"• Status         : {data.get('status', '')}"
                )
 
            elif tool_name == "get_payment_status":
                response_parts.append(
                    f"**Payment Status**\n"
                    f"• Transaction ID : {data.get('transaction_id', '')}\n"
                    f"• Status         : {data.get('status', '')}\n"
                    f"• UTR Number     : {data.get('utr_number', '')}"
                )
 
            elif tool_name == "cancel_payment":
                response_parts.append(
                    f"**Payment Cancelled ✅**\n"
                    f"• Transaction ID : {data.get('transaction_id', '')}\n"
                    f"• Reason         : {data.get('reason', '')}"
                )
 
            elif tool_name == "retry_payment":
                response_parts.append(
                    f"**Payment Retry Initiated**\n"
                    f"• Original TXN : {data.get('original_transaction_id', '')}\n"
                    f"• New TXN ID   : {data.get('new_transaction_id', '')}\n"
                    f"• Status       : {data.get('status', '')}"
                )
 
            elif tool_name == "get_payment_receipt":
                response_parts.append(
                    f"**Payment Receipt**\n"
                    f"• Transaction ID : {data.get('transaction_id', '')}\n"
                    f"• Format         : {data.get('format', '')}\n"
                    f"• [Download Receipt]({data.get('download_url', '')})"
                )
 
            elif tool_name == "validate_beneficiary":
                valid  = data.get("valid", False)
                symbol = "✅" if valid else "❌"
                response_parts.append(
                    f"**Beneficiary Validation {symbol}**\n"
                    f"• Account Holder : {data.get('account_holder_name', '')}\n"
                    f"• Bank           : {data.get('bank', '')}\n"
                    f"• Valid          : {valid}"
                )
 
            # ── UPLOAD PAYMENT ────────────────────────────────────────
            elif tool_name == "upload_bulk_payment":
                response_parts.append(
                    f"**Bulk Payment Upload ✅**\n"
                    f"• Upload ID       : {data.get('upload_id', '')}\n"
                    f"• Total Records   : {data.get('total_records', 0)}\n"
                    f"• Valid Records   : {data.get('valid_records', 0)}\n"
                    f"• Invalid Records : {data.get('invalid_records', 0)}\n"
                    f"• Total Amount    : ₹{data.get('total_amount', 0):,.2f}\n"
                    f"• Status          : {data.get('status', '')}"
                )
 
            elif tool_name == "validate_payment_file":
                response_parts.append(
                    f"**File Validation: {data.get('validation_status', '')}**\n"
                    f"• Errors   : {len(data.get('errors', []))}\n"
                    f"• Warnings : {len(data.get('warnings', []))}"
                )
 
            # ── B2B ───────────────────────────────────────────────────
            elif tool_name == "onboard_business_partner":
                response_parts.append(
                    f"**Partner Onboarded ✅**\n"
                    f"• Partner ID : {data.get('partner_id', '')}\n"
                    f"• Company    : {data.get('company_name', '')}\n"
                    f"• KYC Status : {data.get('kyc_status', '')}\n"
                    f"• Status     : {data.get('status', '')}"
                )
 
            elif tool_name == "send_invoice":
                response_parts.append(
                    f"**Invoice Sent ✅**\n"
                    f"• Invoice ID : {data.get('invoice_id', '')}\n"
                    f"• Amount     : ₹{data.get('amount', 0):,.2f}\n"
                    f"• Total      : ₹{data.get('total_amount', 0):,.2f}\n"
                    f"• Status     : {data.get('status', '')}"
                )
 
            elif tool_name == "get_received_invoices":
                lines = [f"**Received Invoices ({data.get('total', 0)} total):**"]
                for inv in data.get("invoices", [])[:5]:
                    lines.append(
                        f"• [{inv.get('invoice_id')}] {inv.get('partner')} — "
                        f"₹{inv.get('amount', 0):,.2f} | Due: {inv.get('due_date')} | {inv.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "acknowledge_payment":
                response_parts.append(
                    f"**Payment Acknowledged ✅**\n"
                    f"• ACK ID     : {data.get('acknowledgment_id', '')}\n"
                    f"• Invoice ID : {data.get('invoice_id', '')}\n"
                    f"• Status     : {data.get('status', '')}"
                )
 
            elif tool_name == "create_proforma_invoice":
                response_parts.append(
                    f"**Proforma Invoice Created ✅**\n"
                    f"• Proforma ID : {data.get('proforma_id', '')}\n"
                    f"• Amount      : ₹{data.get('amount', 0):,.2f}\n"
                    f"• Valid Until : {data.get('validity_date', '')}"
                )
 
            elif tool_name == "create_cd_note":
                response_parts.append(
                    f"**{data.get('note_type', '')} Note Created ✅**\n"
                    f"• Note ID : {data.get('note_id', '')}\n"
                    f"• Amount  : ₹{data.get('amount', 0):,.2f}\n"
                    f"• Reason  : {data.get('reason', '')}"
                )
 
            elif tool_name == "create_purchase_order":
                response_parts.append(
                    f"**Purchase Order Raised ✅**\n"
                    f"• PO ID    : {data.get('po_id', '')}\n"
                    f"• Amount   : ₹{data.get('amount', 0):,.2f}\n"
                    f"• Delivery : {data.get('delivery_date', '')}"
                )
 
            # ── INSURANCE ─────────────────────────────────────────────
            elif tool_name == "fetch_insurance_dues":
                lines = [f"**Upcoming Insurance Dues ({len(data.get('dues', []))} policies):**"]
                for d in data.get("dues", []):
                    lines.append(
                        f"• [{d.get('policy_number')}] {d.get('insurer')} — "
                        f"₹{d.get('premium', 0):,.2f} | Due: {d.get('due_date')} | {d.get('type')}"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "pay_insurance_premium":
                response_parts.append(
                    f"**Insurance Premium Paid ✅**\n"
                    f"• Transaction ID : {data.get('transaction_id', '')}\n"
                    f"• Policy         : {data.get('policy_number', '')}\n"
                    f"• Amount         : ₹{data.get('amount', 0):,.2f}\n"
                    f"• Status         : {data.get('status', '')}"
                )
 
            elif tool_name == "get_insurance_payment_history":
                lines = [f"**Insurance Payment History ({data.get('total', 0)} records)**"]
                for p in data.get("payments", [])[:5]:
                    lines.append(
                        f"• {p.get('policy_number')} — ₹{p.get('amount', 0):,.2f} | {p.get('paid_on')} | {p.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            # ── BANK STATEMENT ────────────────────────────────────────
            elif tool_name == "fetch_bank_statement":
                response_parts.append(
                    f"**Bank Statement**\n"
                    f"• Account       : {data.get('account_number', '')}\n"
                    f"• Period        : {data.get('from_date', '')} → {data.get('to_date', '')}\n"
                    f"• Opening Bal   : ₹{data.get('opening_balance', 0):,.2f}\n"
                    f"• Closing Bal   : ₹{data.get('closing_balance', 0):,.2f}\n"
                    f"• Total Credits : ₹{data.get('total_credits', 0):,.2f}\n"
                    f"• Total Debits  : ₹{data.get('total_debits', 0):,.2f}"
                )
 
            elif tool_name == "download_bank_statement":
                response_parts.append(
                    f"**Statement Download Ready**\n"
                    f"• Format : {data.get('format', '')}\n"
                    f"• [Download Statement]({data.get('download_url', '')})"
                )
 
            elif tool_name == "get_account_balance":
                response_parts.append(
                    f"**Account Balance**\n"
                    f"• Account           : {data.get('account_number', '')}\n"
                    f"• Available Balance : ₹{data.get('available_balance', 0):,.2f}\n"
                    f"• Current Balance   : ₹{data.get('current_balance', 0):,.2f}"
                )
 
            elif tool_name == "get_transaction_history":
                txns  = data.get("transactions", [])
                lines = [
                    f"**Transaction History — {data.get('from_date', '')} → {data.get('to_date', '')}**\n"
                    f"• Account  : {data.get('account_number', '') or 'Default'}\n"
                    f"• Showing  : {data.get('returned', len(txns))} of {data.get('total', 0)} transactions"
                ]
                for t in txns:
                    symbol = "⬆️" if t.get("type") == "CREDIT" else "⬇️"
                    lines.append(
                        f"{symbol} {t.get('date')} | {t.get('description', '')[:45]}\n"
                        f"   ₹{t.get('amount', 0):,.2f} {t.get('type')} | {t.get('mode')} | Bal: ₹{t.get('balance', 0):,.2f}"
                    )
                response_parts.append("\n".join(lines))
 
            # ── CUSTOM / SEZ ──────────────────────────────────────────
            elif tool_name == "pay_custom_duty":
                response_parts.append(
                    f"**Custom Duty Paid ✅**\n"
                    f"• Transaction ID : {data.get('transaction_id', '')}\n"
                    f"• BOE Number     : {data.get('bill_of_entry_number', '')}\n"
                    f"• Amount         : ₹{data.get('amount', 0):,.2f}\n"
                    f"• Challan        : {data.get('challan_number', '')}\n"
                    f"• Status         : {data.get('status', '')}"
                )
 
            elif tool_name == "track_custom_duty_payment":
                response_parts.append(
                    f"**Custom Duty Payment Status**\n"
                    f"• Transaction ID : {data.get('transaction_id', '')}\n"
                    f"• Status         : {data.get('status', '')}\n"
                    f"• Challan        : {data.get('challan_number', '')}"
                )
 
            elif tool_name == "get_custom_duty_history":
                lines = [f"**Custom Duty History ({data.get('total', 0)} records)**"]
                for p in data.get("payments", [])[:5]:
                    lines.append(
                        f"• TXN: {p.get('transaction_id')} — ₹{p.get('amount', 0):,.2f} | {p.get('paid_on')} | {p.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            # ── GST (bank server — pay/fetch/history/challan) ─────────
            elif tool_name == "fetch_gst_dues":
                lines = [f"**GST Dues for {data.get('gstin', '')}:**"]
                for d in data.get("dues", []):
                    lines.append(
                        f"• [{d.get('return_type')}] {d.get('period')} — "
                        f"₹{d.get('amount', 0):,.2f} | Due: {d.get('due_date')} | {d.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "pay_gst":
                response_parts.append(
                    f"**GST Payment Successful ✅**\n"
                    f"• Transaction ID    : {data.get('transaction_id', '')}\n"
                    f"• GSTIN             : {data.get('gstin', '')}\n"
                    f"• Amount            : ₹{data.get('amount', 0):,.2f}\n"
                    f"• Tax Type          : {data.get('tax_type', '')}\n"
                    f"• Payment Reference : {data.get('payment_reference', '')}"
                )
 
            elif tool_name == "create_gst_challan":
                response_parts.append(
                    f"**GST Challan (PMT-06) Created ✅**\n"
                    f"• CPIN         : {data.get('cpin', '')}\n"
                    f"• GSTIN        : {data.get('gstin', '')}\n"
                    f"• Total Amount : ₹{data.get('total_amount', 0):,.2f}\n"
                    f"• IGST         : ₹{data.get('igst', 0):,.2f}\n"
                    f"• CGST         : ₹{data.get('cgst', 0):,.2f}\n"
                    f"• SGST         : ₹{data.get('sgst', 0):,.2f}\n"
                    f"• CESS         : ₹{data.get('cess', 0):,.2f}\n"
                    f"• Valid Until  : {data.get('valid_until', '')}"
                )
 
            elif tool_name == "get_gst_payment_history":
                lines = [f"**GST Payment History — {data.get('gstin', '')} ({data.get('total', 0)} records)**"]
                for p in data.get("payments", [])[:5]:
                    lines.append(
                        f"• CPIN: {p.get('cpin')} — ₹{p.get('amount', 0):,.2f} | {p.get('paid_on')} | {p.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            # ── ESIC ──────────────────────────────────────────────────
            elif tool_name == "fetch_esic_dues":
                response_parts.append(
                    f"**ESIC Dues — {data.get('establishment_code', '')} ({data.get('month', '')})**\n"
                    f"• Employees            : {data.get('employee_count', 0)}\n"
                    f"• Employer Contribution: ₹{data.get('employer_contribution', 0):,.2f}\n"
                    f"• Employee Contribution: ₹{data.get('employee_contribution', 0):,.2f}\n"
                    f"• Total Due            : ₹{data.get('total_due', 0):,.2f}\n"
                    f"• Due Date             : {data.get('due_date', '')}"
                )
 
            elif tool_name == "pay_esic":
                response_parts.append(
                    f"**ESIC Payment Successful ✅**\n"
                    f"• Transaction ID : {data.get('transaction_id', '')}\n"
                    f"• Establishment  : {data.get('establishment_code', '')}\n"
                    f"• Month          : {data.get('month', '')}\n"
                    f"• Amount         : ₹{data.get('amount', 0):,.2f}\n"
                    f"• Challan        : {data.get('challan_number', '')}"
                )
 
            elif tool_name == "get_esic_payment_history":
                lines = [f"**ESIC Payment History ({data.get('total', 0)} records)**"]
                for p in data.get("payments", [])[:5]:
                    lines.append(
                        f"• {p.get('month')} — ₹{p.get('amount', 0):,.2f} | {p.get('paid_on')} | {p.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            # ── EPF ───────────────────────────────────────────────────
            elif tool_name == "fetch_epf_dues":
                response_parts.append(
                    f"**EPF Dues — {data.get('establishment_id', '')} ({data.get('month', '')})**\n"
                    f"• Employees            : {data.get('employee_count', 0)}\n"
                    f"• Employer Contribution: ₹{data.get('employer_contribution', 0):,.2f}\n"
                    f"• Employee Contribution: ₹{data.get('employee_contribution', 0):,.2f}\n"
                    f"• Admin Charges        : ₹{data.get('admin_charges', 0):,.2f}\n"
                    f"• Total Due            : ₹{data.get('total_due', 0):,.2f}\n"
                    f"• Due Date             : {data.get('due_date', '')}"
                )
 
            elif tool_name == "pay_epf":
                response_parts.append(
                    f"**EPF Payment Successful ✅**\n"
                    f"• Transaction ID : {data.get('transaction_id', '')}\n"
                    f"• Establishment  : {data.get('establishment_id', '')}\n"
                    f"• Month          : {data.get('month', '')}\n"
                    f"• Amount         : ₹{data.get('amount', 0):,.2f}\n"
                    f"• TRRN           : {data.get('trrn', '')}"
                )
 
            elif tool_name == "get_epf_payment_history":
                lines = [f"**EPF Payment History ({data.get('total', 0)} records)**"]
                for p in data.get("payments", [])[:5]:
                    lines.append(
                        f"• {p.get('month')} — ₹{p.get('amount', 0):,.2f} | TRRN: {p.get('trrn')} | {p.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            # ── PAYROLL ───────────────────────────────────────────────
            elif tool_name == "fetch_payroll_summary":
                response_parts.append(
                    f"**Payroll Summary — {data.get('month', '')}**\n"
                    f"• Total Employees : {data.get('total_employees', 0)}\n"
                    f"• Gross           : ₹{data.get('total_gross', 0):,.2f}\n"
                    f"• Deductions      : ₹{data.get('total_deductions', 0):,.2f}\n"
                    f"• Net Payable     : ₹{data.get('total_net', 0):,.2f}\n"
                    f"• Status          : {data.get('status', '')}"
                )
 
            elif tool_name == "process_payroll":
                response_parts.append(
                    f"**Payroll Processing Started ✅**\n"
                    f"• Batch ID        : {data.get('batch_id', '')}\n"
                    f"• Month           : {data.get('month', '')}\n"
                    f"• Total Employees : {data.get('total_employees', 0)}\n"
                    f"• Total Amount    : ₹{data.get('total_amount', 0):,.2f}\n"
                    f"• Status          : {data.get('status', '')}"
                )
 
            elif tool_name == "get_payroll_history":
                lines = [f"**Payroll History ({data.get('total', 0)} records)**"]
                for p in data.get("payrolls", [])[:5]:
                    lines.append(
                        f"• {p.get('month')} — ₹{p.get('total_amount', 0):,.2f} | {p.get('employees')} employees | {p.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            # ── TAXES ─────────────────────────────────────────────────
            elif tool_name == "fetch_tax_dues":
                lines = [f"**Pending Tax Dues — PAN: {data.get('pan', '')}**"]
                for d in data.get("dues", []):
                    lines.append(
                        f"• [{d.get('type')}] {d.get('period', d.get('state', ''))} — "
                        f"₹{d.get('amount', 0):,.2f} | Due: {d.get('due_date')}"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "pay_direct_tax":
                response_parts.append(
                    f"**Direct Tax Payment Successful ✅**\n"
                    f"• Transaction ID  : {data.get('transaction_id', '')}\n"
                    f"• Tax Type        : {data.get('tax_type', '')}\n"
                    f"• Assessment Year : {data.get('assessment_year', '')}\n"
                    f"• Amount          : ₹{data.get('amount', 0):,.2f}\n"
                    f"• CIN             : {data.get('cin', '')}"
                )
 
            elif tool_name == "pay_state_tax":
                response_parts.append(
                    f"**State Tax Payment Successful ✅**\n"
                    f"• Transaction ID : {data.get('transaction_id', '')}\n"
                    f"• State          : {data.get('state', '')}\n"
                    f"• Category       : {data.get('tax_category', '')}\n"
                    f"• Amount         : ₹{data.get('amount', 0):,.2f}"
                )
 
            elif tool_name == "pay_bulk_tax":
                response_parts.append(
                    f"**Bulk Tax Payment Queued ✅**\n"
                    f"• Batch ID      : {data.get('batch_id', '')}\n"
                    f"• Tax Type      : {data.get('tax_type', '')}\n"
                    f"• Total Records : {data.get('total_records', 0)}\n"
                    f"• Total Amount  : ₹{data.get('total_amount', 0):,.2f}\n"
                    f"• Status        : {data.get('status', '')}"
                )
 
            elif tool_name == "get_tax_payment_history":
                lines = [f"**Tax Payment History — PAN: {data.get('pan', '')} ({data.get('total', 0)} records)**"]
                for p in data.get("payments", [])[:5]:
                    lines.append(
                        f"• [{p.get('type')}] ₹{p.get('amount', 0):,.2f} | CIN: {p.get('cin')} | {p.get('paid_on')} | {p.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            # ── ACCOUNT MANAGEMENT ────────────────────────────────────
            elif tool_name == "get_account_summary":
                lines = [f"**Linked Accounts ({len(data.get('accounts', []))}):**"]
                for acc in data.get("accounts", []):
                    lines.append(
                        f"• {acc.get('account_number')} | {acc.get('type')} | "
                        f"₹{acc.get('balance', 0):,.2f} | {acc.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "get_account_details":
                response_parts.append(
                    f"**Account Details**\n"
                    f"• Account : {data.get('account_number', '')}\n"
                    f"• Type    : {data.get('type', '')}\n"
                    f"• Bank    : {data.get('bank', '')}\n"
                    f"• Branch  : {data.get('branch', '')}\n"
                    f"• IFSC    : {data.get('ifsc', '')}\n"
                    f"• Holder  : {data.get('holder_name', '')}\n"
                    f"• Status  : {data.get('status', '')}"
                )
 
            elif tool_name == "get_linked_accounts":
                lines = [f"**Linked Accounts ({data.get('total', 0)}):**"]
                for acc in data.get("accounts", []):
                    lines.append(f"• {acc.get('account_number')} | {acc.get('bank')} | {acc.get('type')}")
                response_parts.append("\n".join(lines))
 
            elif tool_name == "set_default_account":
                response_parts.append(
                    f"**Default Account Updated ✅**\n"
                    f"• Account : {data.get('account_number', '')}\n"
                    f"• Default : {data.get('is_default', False)}"
                )
 
            # ── TRANSACTION & HISTORY ─────────────────────────────────
            elif tool_name == "search_transactions":
                response_parts.append(f"**Transactions Found: {data.get('total', 0)}**")
 
            elif tool_name == "get_transaction_details":
                response_parts.append(
                    f"**Transaction Details**\n"
                    f"• TXN ID      : {data.get('transaction_id', '')}\n"
                    f"• Amount      : ₹{data.get('amount', 0):,.2f}\n"
                    f"• Type        : {data.get('txn_type', '')}\n"
                    f"• Mode        : {data.get('mode', '')}\n"
                    f"• Beneficiary : {data.get('beneficiary', '')}\n"
                    f"• UTR         : {data.get('utr', '')}\n"
                    f"• Status      : {data.get('status', '')}"
                )
 
            elif tool_name == "download_transaction_report":
                response_parts.append(
                    f"**Transaction Report Ready**\n"
                    f"• Format : {data.get('format', '')}\n"
                    f"• [Download Report]({data.get('download_url', '')})"
                )
 
            elif tool_name == "get_pending_transactions":
                lines = [f"**Pending Transactions: {data.get('total', 0)}**"]
                for txn in data.get("transactions", [])[:5]:
                    lines.append(
                        f"• {txn.get('transaction_id')} — ₹{txn.get('amount', 0):,.2f} | {txn.get('mode')} | {txn.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            # ── DUES & REMINDERS ──────────────────────────────────────
            elif tool_name == "get_upcoming_dues":
                lines = [f"**Upcoming Dues (Next {data.get('days_ahead', 30)} days):**"]
                for d in data.get("dues", []):
                    lines.append(
                        f"• [{d.get('type')}] ₹{d.get('amount', 0):,.2f} | Due: {d.get('due_date')} | {d.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "get_overdue_payments":
                lines = [f"**Overdue Payments ({data.get('total', 0)}):**"]
                for o in data.get("overdue", []):
                    lines.append(
                        f"• [{o.get('type')}] ₹{o.get('amount', 0):,.2f} | "
                        f"Due: {o.get('due_date')} | {o.get('days_overdue')} days overdue ⚠️"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "set_payment_reminder":
                response_parts.append(
                    f"**Reminder Set ✅**\n"
                    f"• Reminder ID   : {data.get('reminder_id', '')}\n"
                    f"• Title         : {data.get('title', '')}\n"
                    f"• Due Date      : {data.get('due_date', '')}\n"
                    f"• Notify Before : {data.get('notify_days_before', 3)} days"
                )
 
            elif tool_name == "get_reminder_list":
                lines = [f"**Active Reminders ({data.get('total', 0)}):**"]
                for r in data.get("reminders", []):
                    lines.append(f"• [{r.get('reminder_id')}] {r.get('title')} | Due: {r.get('due_date')}")
                response_parts.append("\n".join(lines))
 
            elif tool_name == "delete_reminder":
                response_parts.append(
                    f"**Reminder Deleted ✅**\n"
                    f"• Reminder ID : {data.get('reminder_id', '')}"
                )
 
            # ── DASHBOARD & ANALYTICS ─────────────────────────────────
            elif tool_name == "get_dashboard_summary":
                response_parts.append(
                    f"**Dashboard Summary**\n"
                    f"• Total Balance    : ₹{data.get('total_balance', 0):,.2f}\n"
                    f"• Pending Dues     : ₹{data.get('pending_dues', 0):,.2f}\n"
                    f"• Overdue Amount   : ₹{data.get('overdue_amount', 0):,.2f}\n"
                    f"• Payments (Month) : ₹{data.get('payments_this_month', 0):,.2f}\n"
                    f"• Upcoming Dues    : {data.get('upcoming_dues_count', 0)}\n"
                    f"• Account Health   : {data.get('account_health', '')}"
                )
 
            elif tool_name == "get_spending_analytics":
                lines = ["**Spending Analytics:**"]
                for cat in data.get("categories", []):
                    lines.append(
                        f"• {cat.get('category')}: ₹{cat.get('amount', 0):,.2f} ({cat.get('percentage')}%)"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "get_cashflow_summary":
                response_parts.append(
                    f"**Cash Flow Summary — {data.get('month', '')}**\n"
                    f"• Total Inflow  : ₹{data.get('total_inflow', 0):,.2f}\n"
                    f"• Total Outflow : ₹{data.get('total_outflow', 0):,.2f}\n"
                    f"• Net Cash Flow : ₹{data.get('net_cashflow', 0):,.2f}"
                )
 
            elif tool_name == "get_monthly_report":
                response_parts.append(
                    f"**Monthly Report — {data.get('month', '')}**\n"
                    f"• Total Payments  : {data.get('total_payments', 0)}\n"
                    f"• Total Amount    : ₹{data.get('total_amount', 0):,.2f}\n"
                    f"• Compliance Paid : ₹{data.get('compliance_paid', 0):,.2f}\n"
                    f"• [Download Report]({data.get('download_url', '')})"
                )
 
            elif tool_name == "get_vendor_payment_summary":
                lines = ["**Vendor Payment Summary:**"]
                for v in data.get("vendors", []):
                    lines.append(
                        f"• {v.get('name')}: ₹{v.get('total_paid', 0):,.2f} ({v.get('payment_count')} payments)"
                    )
                response_parts.append("\n".join(lines))
 
            # ── COMPANY MANAGEMENT ────────────────────────────────────
            elif tool_name == "get_company_profile":
                response_parts.append(
                    f"**Company Profile**\n"
                    f"• Name       : {data.get('company_name', '')}\n"
                    f"• PAN        : {data.get('pan', '')}\n"
                    f"• GSTIN      : {data.get('gstin', '')}\n"
                    f"• CIN        : {data.get('cin', '')}\n"
                    f"• KYC Status : {data.get('kyc_status', '')}"
                )
 
            elif tool_name == "update_company_details":
                response_parts.append(
                    f"**Company Details Updated ✅**\n"
                    f"• Field   : {data.get('field', '')}\n"
                    f"• Value   : {data.get('value', '')}\n"
                    f"• Updated : {data.get('updated', False)}"
                )
 
            elif tool_name == "get_gst_profile":
                lines = [f"**Linked GST Numbers ({len(data.get('gst_numbers', []))}):**"]
                for g in data.get("gst_numbers", []):
                    lines.append(f"• {g.get('gstin')} | {g.get('state')} | {g.get('status')}")
                response_parts.append("\n".join(lines))
 
            elif tool_name == "get_authorized_signatories":
                lines = [f"**Authorized Signatories ({len(data.get('signatories', []))}):**"]
                for s in data.get("signatories", []):
                    lines.append(
                        f"• {s.get('name')} | {s.get('role')} | PAN: {s.get('pan')} | {s.get('status')}"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "manage_user_roles":
                response_parts.append(
                    f"**User Role Updated ✅**\n"
                    f"• User ID : {data.get('user_id', '')}\n"
                    f"• Role    : {data.get('role', '')}\n"
                    f"• Action  : {data.get('action', '')}"
                )
 
            # ── SUPPORT ───────────────────────────────────────────────
            elif tool_name == "raise_support_ticket":
                response_parts.append(
                    f"**Support Ticket Raised ✅**\n"
                    f"• Ticket ID : {data.get('ticket_id', '')}\n"
                    f"• Category  : {data.get('category', '')}\n"
                    f"• Subject   : {data.get('subject', '')}\n"
                    f"• Priority  : {data.get('priority', '')}\n"
                    f"• Status    : {data.get('status', '')}"
                )
 
            elif tool_name == "get_ticket_history":
                lines = [f"**Support Tickets ({data.get('total', 0)}):**"]
                for t in data.get("tickets", [])[:5]:
                    lines.append(
                        f"• [{t.get('ticket_id')}] {t.get('subject')} | {t.get('status')} | {t.get('created_at')}"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "chat_with_support":
                response_parts.append(
                    f"**Live Support Connected ✅**\n"
                    f"• Session ID : {data.get('session_id', '')}\n"
                    f"• Agent      : {data.get('agent', '')}\n"
                    f"• Wait Time  : {data.get('wait_time_minutes', 0)} mins\n"
                    f"• Status     : {data.get('status', '')}"
                )
 
            elif tool_name == "get_contact_details":
                response_parts.append(
                    f"**Support Contacts — {data.get('category', '')}**\n"
                    f"• Phone          : {data.get('phone', '')}\n"
                    f"• Email          : {data.get('email', '')}\n"
                    f"• Hours          : {data.get('hours', '')}\n"
                    f"• Chat Available : {data.get('chat_available', False)}"
                )
 
            # ── GST CALCULATOR (gst_client_manager / server.py) ───────
            # FIX 1+2: These were (a) placed after `else` as a new if/elif
            # block — making them unreachable — and (b) get_bank_onboarding_guide
            # had `response_parts.ap\npend(...)` split across two lines.
            # Both fixed by merging into the main elif chain here.
            elif tool_name == "calculate_gst":
                response_parts.append(
                    f"**GST Calculation @ {data.get('gst_rate', 0)}%**\n"
                    f"• Base Amount  : ₹{data.get('base_amount', 0):,.2f}\n"
                    f"• GST Amount   : ₹{data.get('gst_amount', 0):,.2f}\n"
                    f"• Total Amount : ₹{data.get('total_amount', 0):,.2f}"
                )
 
            elif tool_name == "reverse_calculate_gst":
                response_parts.append(
                    f"**Reverse GST Calculation @ {data.get('gst_rate', 0)}%**\n"
                    f"• Total Amount            : ₹{data.get('total_amount', 0):,.2f}\n"
                    f"• Base Amount (excl. GST) : ₹{data.get('base_amount', 0):,.2f}\n"
                    f"• GST Amount              : ₹{data.get('gst_amount', 0):,.2f}"
                )
 
            elif tool_name == "gst_breakdown":
                breakdown = data.get("breakdown", {})
                response_parts.append(
                    f"**GST Breakdown ({breakdown.get('type', '')})**\n"
                    f"• Base Amount : ₹{data.get('base_amount', 0):,.2f}\n"
                    f"• CGST        : ₹{breakdown.get('cgst', 0):,.2f}\n"
                    f"• SGST        : ₹{breakdown.get('sgst', 0):,.2f}\n"
                    f"• IGST        : ₹{breakdown.get('igst', 0):,.2f}"
                )
 
            elif tool_name == "compare_gst_rates":
                lines = [f"**GST Rate Comparison for ₹{data.get('base_amount', 0):,.2f}:**"]
                for comp in data.get("comparisons", []):
                    lines.append(
                        f"• {comp.get('rate')}% → Total: ₹{comp.get('total_amount', 0):,.2f}"
                        f"  (+₹{comp.get('difference_from_lowest', 0):,.2f} vs lowest)"
                    )
                response_parts.append("\n".join(lines))
 
            elif tool_name == "validate_gstin":
                valid      = data.get("valid", False)
                symbol     = "✅ Valid" if valid else "❌ Invalid"
                components = data.get("components", {})
                if valid:
                    response_parts.append(
                        f"**GSTIN Validation: {symbol}**\n"
                        f"• GSTIN      : {data.get('gstin', '')}\n"
                        f"• State Code : {components.get('state_code', '')}\n"
                        f"• PAN        : {components.get('pan_number', '')}"
                    )
                else:
                    response_parts.append(
                        f"**GSTIN Validation: {symbol}**\n"
                        f"• GSTIN  : {data.get('gstin', '')}\n"
                        f"• Reason : {data.get('error', 'Invalid format')}"
                    )
 
            # ── ONBOARDING INFO (info_client_manager / info_server.py) ─
            elif tool_name == "get_company_onboarding_guide":
                lines = [f"**{data.get('title', 'Company Onboarding Guide')}**"]
                for step in data.get("steps", []):
                    lines.append(f"\n**Step {step.get('step_number')}: {step.get('title', '')}**")
                    for action in step.get("actions", []):
                        lines.append(f"  • {action}")
                    for field in step.get("required_fields", [])[:5]:
                        lines.append(f"    - {field.get('field', '')}")
                if data.get("completion_message"):
                    lines.append(f"\n{data['completion_message']}")
                response_parts.append("\n".join(lines))
 
            elif tool_name == "get_company_required_documents":
                lines = [f"**{data.get('title', 'Required Documents')}**"]
                for doc in data.get("documents", []):
                    lines.append(f"• {doc.get('name', '')} — {doc.get('description', '')}")
                response_parts.append("\n".join(lines))
 
            elif tool_name == "get_validation_formats":
                lines = ["**Validation Formats:**"]
                for doc_type, doc_data in list(data.get("formats", {}).items())[:8]:
                    lines.append(f"• {doc_type}: `{doc_data.get('pattern', '')}`")
                    if doc_data.get("example"):
                        lines.append(f"  Example: {doc_data['example']}")
                response_parts.append("\n".join(lines))
 
            elif tool_name == "get_onboarding_faq":
                lines = [f"**{data.get('title', 'Onboarding FAQ')}**"]
                for faq in data.get("faqs", [])[:5]:
                    lines.append(f"**Q: {faq.get('question', '')}**")
                    lines.append(f"A: {faq.get('answer', '')}")
                response_parts.append("\n".join(lines))
 
            elif tool_name == "get_bank_onboarding_guide":
                # FIX 1: was `response_parts.ap\n                pend(...)` — broken across two lines
                lines = [f"**{data.get('title', 'Bank Onboarding Guide')}**"]
                for step in data.get("steps", []):
                    lines.append(f"\n**Step {step.get('step_number')}: {step.get('title', '')}**")
                    for action in step.get("actions", []):
                        lines.append(f"  • {action}")
                if data.get("completion_message"):
                    lines.append(f"\n{data['completion_message']}")
                response_parts.append("\n".join(lines))
 
            elif tool_name == "get_vendor_onboarding_guide":
                lines = [f"**{data.get('title', 'Vendor Onboarding Guide')}**"]
                for step in data.get("steps", []):
                    lines.append(f"\n**Step {step.get('step_number')}: {step.get('title', '')}**")
                    for action in step.get("actions", []):
                        lines.append(f"  • {action}")
                if data.get("completion_message"):
                    lines.append(f"\n{data['completion_message']}")
                response_parts.append("\n".join(lines))
 
            else:
                response_parts.append(f"✓ {tool_name} executed successfully.")
 
        return "\n\n".join(response_parts)
 
 
# Global service instance
claude_service = LocalMLService()