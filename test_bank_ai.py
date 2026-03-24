#!/usr/bin/env python3
"""
Bank AI Assistant — Comprehensive Automated Test Suite
Tests all intents, conflict resolution, multi-intent, entity extraction,
API-level integration, memory/context, and edge cases.

Usage:
    python3 test_bank_ai.py                    # full suite against localhost:8000
    python3 test_bank_ai.py --unit-only        # unit tests (no server needed)
    python3 test_bank_ai.py --url http://...   # custom server URL
    python3 test_bank_ai.py --category gst     # run only one category
"""

import sys
import json
import time
import argparse
import asyncio
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ─────────────────────────────────────────────────────────────────────────────
# TEST RESULT STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    message: str = ""
    actual: Any = None
    expected: Any = None
    duration_ms: float = 0.0
    skipped: bool = False

@dataclass
class TestSuite:
    name: str
    results: List[TestResult] = field(default_factory=list)

    @property
    def passed(self):  return sum(1 for r in self.results if r.passed and not r.skipped)
    @property
    def failed(self):  return sum(1 for r in self.results if not r.passed and not r.skipped)
    @property
    def skipped(self): return sum(1 for r in self.results if r.skipped)
    @property
    def total(self):   return len(self.results)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000"
SESSION_COUNTER = 1000

def next_session():
    global SESSION_COUNTER
    SESSION_COUNTER += 1
    return f"auto-test-{SESSION_COUNTER}"

def chat(message: str, session_id: str = None, user_id: str = "test-user-001",
         url: str = None) -> Optional[Dict]:
    """Send a chat message to the API and return parsed response."""
    target = (url or BASE_URL) + "/api/chat"
    payload = {
        "message":    message,
        "session_id": session_id or next_session(),
        "user_id":    user_id,
    }
    try:
        resp = requests.post(target, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        return {"_error": str(e)}

def assert_intents(resp: Dict, expected: List[str], exact: bool = False) -> tuple:
    """Check intents. Returns (passed, message)."""
    if resp is None:
        return False, "Server unreachable"
    if "_error" in resp:
        return False, f"Request error: {resp['_error']}"
    actual = resp.get("intents_detected", [])
    if exact:
        if sorted(actual) == sorted(expected):
            return True, f"✓ intents={actual}"
        return False, f"Expected exactly {expected}, got {actual}"
    # subset check — all expected must be present
    missing = [e for e in expected if e not in actual]
    if missing:
        return False, f"Missing intents {missing} in {actual}"
    return True, f"✓ intents={actual}"

def assert_no_intents(resp: Dict, unwanted: List[str]) -> tuple:
    if resp is None:
        return False, "Server unreachable"
    actual = resp.get("intents_detected", [])
    found = [u for u in unwanted if u in actual]
    if found:
        return False, f"Unwanted intents {found} appeared in {actual}"
    return True, f"✓ no unwanted intents (got {actual})"

def assert_tool_called(resp: Dict, tool_name: str) -> tuple:
    if resp is None:
        return False, "Server unreachable"
    tools = [t["tool"] for t in resp.get("tool_calls", [])]
    if tool_name in tools:
        return True, f"✓ tool '{tool_name}' called"
    return False, f"Tool '{tool_name}' not called. Called: {tools}"

def assert_tool_success(resp: Dict, tool_name: str) -> tuple:
    if resp is None:
        return False, "Server unreachable"
    for t in resp.get("tool_calls", []):
        if t["tool"] == tool_name:
            if t.get("success"):
                return True, f"✓ '{tool_name}' succeeded"
            return False, f"'{tool_name}' failed: {t.get('error', t.get('result', ''))}"
    return False, f"Tool '{tool_name}' not found in response"

def assert_no_error(resp: Dict, tool_name: str) -> tuple:
    """Assert a tool did not return a pydantic/validation error."""
    if resp is None:
        return False, "Server unreachable"
    for t in resp.get("tool_calls", []):
        if t["tool"] == tool_name:
            result = str(t.get("result", ""))
            if "validation error" in result.lower() or "unexpected keyword" in result.lower():
                return False, f"Validation error in '{tool_name}': {result[:120]}"
            return True, f"✓ no validation error"
    return True, "✓ (tool not called)"

def assert_multi_intent(resp: Dict, expected_flag: bool) -> tuple:
    if resp is None:
        return False, "Server unreachable"
    actual = resp.get("is_multi_intent", False)
    if actual == expected_flag:
        return True, f"✓ is_multi_intent={actual}"
    return False, f"Expected is_multi_intent={expected_flag}, got {actual}"

def assert_context_used(resp: Dict) -> tuple:
    if resp is None:
        return False, "Server unreachable"
    if resp.get("context_used"):
        return True, "✓ context_used=true"
    return False, "context_used=false (expected true)"

def assert_memory_field(resp: Dict, field: str, value: str) -> tuple:
    if resp is None:
        return False, "Server unreachable"
    snap = resp.get("memory_snapshot", {})
    actual = snap.get(field)
    if actual == value:
        return True, f"✓ memory.{field}={actual!r}"
    return False, f"memory.{field}: expected {value!r}, got {actual!r}"

def mk(name, category, passed, message, actual=None, expected=None, duration_ms=0.0):
    return TestResult(name=name, category=category, passed=passed,
                      message=message, actual=actual, expected=expected,
                      duration_ms=duration_ms)

def skip(name, category, reason):
    return TestResult(name=name, category=category, passed=False,
                      message=reason, skipped=True)

def run_chat_test(suite: TestSuite, name: str, category: str,
                  message: str, checks, session_id=None, url=None):
    """Run a single chat test with one or more check functions."""
    if not HAS_REQUESTS:
        suite.results.append(skip(name, category, "requests not installed"))
        return None

    t0 = time.time()
    resp = chat(message, session_id=session_id, url=url)
    dur = (time.time() - t0) * 1000

    overall_pass = True
    messages = []

    for check_fn in (checks if isinstance(checks, list) else [checks]):
        passed, msg = check_fn(resp)
        if not passed:
            overall_pass = False
        messages.append(msg)

    suite.results.append(mk(
        name=name, category=category,
        passed=overall_pass,
        message=" | ".join(messages),
        actual=resp.get("intents_detected") if resp else None,
        duration_ms=dur
    ))
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS — classifier only (no HTTP)
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests() -> TestSuite:
    suite = TestSuite("Unit: Intent Classifier")

    try:
        sys.path.insert(0, ".")
        import ml_intent_classifier as _clf_mod
        clf = _clf_mod.intent_classifier

        # Auto-train if model not yet loaded (datasets available on-server)
        if clf.classifier is None:
            try:
                import logging
                logging.disable(logging.CRITICAL)
                clf.load_datasets()
                clf.train()
                logging.disable(logging.NOTSET)
                suite.results.append(mk("classifier_train", "unit", True, "✓ Model auto-trained"))
            except Exception as train_err:
                suite.results.append(mk("classifier_train", "unit", False,
                                        f"Auto-train failed: {train_err}"))
                return suite
    except ImportError as e:
        suite.results.append(mk("import_classifier", "unit", False, f"Import failed: {e}"))
        return suite

    def check(name, query, expected_intents=None, forbidden_intents=None,
              exact=False, expect_multi=None):
        t0 = time.time()
        result = clf.process_query(query)
        dur = (time.time() - t0) * 1000
        actual = result.get("intents_detected", [])
        msgs = []
        passed = True

        if expected_intents:
            if exact:
                if sorted(actual) != sorted(expected_intents):
                    passed = False
                    msgs.append(f"Expected exactly {expected_intents}, got {actual}")
                else:
                    msgs.append(f"✓ exact={actual}")
            else:
                missing = [e for e in expected_intents if e not in actual]
                if missing:
                    passed = False
                    msgs.append(f"Missing {missing} in {actual}")
                else:
                    msgs.append(f"✓ contains {expected_intents}")

        if forbidden_intents:
            found = [f for f in forbidden_intents if f in actual]
            if found:
                passed = False
                msgs.append(f"Forbidden {found} found in {actual}")
            else:
                msgs.append(f"✓ no forbidden {forbidden_intents}")

        if expect_multi is not None:
            is_multi = result.get("is_multi_intent", len(actual) > 1)
            if is_multi != expect_multi:
                passed = False
                msgs.append(f"is_multi_intent: expected {expect_multi}, got {is_multi}")
            else:
                msgs.append(f"✓ multi={is_multi}")

        suite.results.append(mk(name, "unit", passed, " | ".join(msgs),
                                actual=actual, duration_ms=dur))

    # ── CORE PAYMENT ───────────────────────────────────────────────
    check("payment_initiate",        "send ₹50000 to vendor via NEFT",               ["initiate_payment"])
    check("payment_status",          "check payment status TXN001",                   ["get_payment_status"])
    check("payment_cancel",          "cancel payment TXN002",                         ["cancel_payment"])
    check("payment_retry",           "retry failed payment TXN003",                   ["retry_payment"])
    check("payment_receipt",         "get payment receipt for TXN004",                ["get_payment_receipt"])
    check("validate_beneficiary",    "validate beneficiary account 1234567890",       ["validate_beneficiary"])
    check("upload_bulk",             "upload bulk payment file",                       ["upload_bulk_payment"])
    check("validate_file",           "validate my payment file before upload",        ["validate_payment_file"])

    # ── B2B ────────────────────────────────────────────────────────
    check("onboard_partner",         "onboard new business partner ABC Corp",         ["onboard_business_partner"])
    check("send_invoice",            "send invoice to client for ₹100000",            ["send_invoice"])
    check("received_invoices",       "show all received invoices",                    ["get_received_invoices"])
    check("ack_payment",             "acknowledge payment for invoice INV001",        ["acknowledge_payment"])
    check("proforma_invoice",        "create proforma invoice for ₹200000",          ["create_proforma_invoice"])
    check("cd_note",                 "create credit note for client",                 ["create_cd_note"])
    check("purchase_order",          "raise purchase order for ₹500000",             ["create_purchase_order"])

    # ── GST CLASSIFIER ─────────────────────────────────────────────
    check("fetch_gst_dues",          "show my GST dues",                              ["fetch_gst_dues"],
          forbidden_intents=["pay_gst"])
    check("pay_gst",                 "pay GST 27AAAPD1234F1ZK amount 50000",         ["pay_gst"],
          forbidden_intents=["fetch_gst_dues"])
    check("pay_gst_dues_ambiguous",  "pay GST dues for this month",                  ["pay_gst"],
          forbidden_intents=["fetch_gst_dues"])
    check("gst_challan",             "create GST challan for GSTIN 27AAAPD1234F1ZK", ["create_gst_challan"])
    check("gst_history",             "show GST payment history",                      ["get_gst_payment_history"])
    check("calculate_gst",           "calculate GST on 50000 at 18%",                ["calculate_gst"])
    check("reverse_gst",             "reverse calculate GST on total 59000 at 18%",  ["reverse_gst"])
    check("gst_breakdown",           "GST breakdown for ₹100000",                    ["gst_breakdown"])
    check("compare_gst",             "compare GST rates 5% 12% 18% on ₹50000",      ["compare_rates"])
    check("validate_gstin",          "validate GSTIN 27AAAPD1234F1ZK",               ["validate_gstin"])

    # ── ESIC ───────────────────────────────────────────────────────
    check("fetch_esic_dues",         "show my ESIC dues",                             ["fetch_esic_dues"],
          forbidden_intents=["pay_esic"])
    check("pay_esic",                "pay ESIC for 02-2026",                          ["pay_esic"],
          forbidden_intents=["fetch_esic_dues"])
    check("pay_esic_dues_word",      "pay ESIC dues for 02-2026",                    ["pay_esic"],
          forbidden_intents=["fetch_esic_dues"])
    check("esic_history",            "show ESIC payment history",                     ["get_esic_payment_history"])

    # ── EPF ────────────────────────────────────────────────────────
    check("fetch_epf_dues",          "show my EPF dues",                              ["fetch_epf_dues"],
          forbidden_intents=["pay_epf"])
    check("pay_epf",                 "pay EPF for 02-2026",                           ["pay_epf"],
          forbidden_intents=["fetch_epf_dues"])
    check("pay_epf_dues_word",       "pay EPF dues for this month",                  ["pay_epf"],
          forbidden_intents=["fetch_epf_dues"])
    check("epf_history",             "show EPF payment history",                      ["get_epf_payment_history"])

    # ── PAYROLL ────────────────────────────────────────────────────
    check("payroll_summary",         "show payroll summary for March 2026",           ["fetch_payroll_summary"])
    check("process_payroll",         "process payroll for March 2026",                ["process_payroll"])
    check("payroll_history",         "show payroll history",                          ["get_payroll_history"])

    # ── TAXES ──────────────────────────────────────────────────────
    check("fetch_tax_dues",          "show my pending tax dues",                      ["fetch_tax_dues"],
          forbidden_intents=["pay_direct_tax"])
    check("pay_direct_tax",          "pay advance tax ₹100000",                      ["pay_direct_tax"])
    check("pay_direct_tax_dues",     "pay direct tax dues",                           ["pay_direct_tax"],
          forbidden_intents=["fetch_tax_dues"])
    check("pay_state_tax",           "pay state tax Karnataka",                       ["pay_state_tax"])
    check("pay_bulk_tax",            "pay bulk tax for multiple challans",            ["pay_bulk_tax"])
    check("tax_history",             "show tax payment history",                      ["get_tax_payment_history"])

    # ── INSURANCE ──────────────────────────────────────────────────
    check("fetch_insurance_dues",    "show insurance premium dues",                   ["fetch_insurance_dues"],
          forbidden_intents=["pay_insurance_premium"])
    check("pay_insurance",           "pay insurance premium for policy LIC001",       ["pay_insurance_premium"])
    check("pay_insurance_dues",      "pay insurance dues this month",                 ["pay_insurance_premium"],
          forbidden_intents=["fetch_insurance_dues"])
    check("insurance_history",       "show insurance payment history",                ["get_insurance_payment_history"])

    # ── CUSTOM / SEZ ───────────────────────────────────────────────
    check("pay_custom_duty",         "pay custom duty for bill of entry BOE2026",     ["pay_custom_duty"])
    check("track_custom_duty",       "track custom duty payment status",              ["track_custom_duty_payment"])
    check("custom_duty_history",     "show custom duty payment history",              ["get_custom_duty_history"])

    # ── BANK / ACCOUNT ─────────────────────────────────────────────
    check("account_balance",         "what is my account balance",                    ["get_account_balance"],
          forbidden_intents=["get_account_details"])
    check("account_details",         "show account details IFSC branch",              ["get_account_details"],
          forbidden_intents=["get_account_balance"])
    check("account_summary",         "show all my linked accounts summary",           ["get_account_summary"])
    check("linked_accounts",         "how many linked accounts do I have",            ["get_linked_accounts"])
    check("set_default_account",     "set account 12345 as default primary account",  ["set_default_account"])

    # ── BANK STATEMENT ─────────────────────────────────────────────
    check("fetch_bank_statement",    "show bank statement for last month",             ["fetch_bank_statement"])
    check("download_bank_statement", "download bank statement PDF",                   ["download_bank_statement"],
          forbidden_intents=["fetch_bank_statement"])

    # ── TRANSACTIONS ───────────────────────────────────────────────
    check("transaction_history",     "show transaction history last 30 days",         ["get_transaction_history"])
    check("search_transactions",     "search transactions for vendor ABC",             ["search_transactions"])
    check("transaction_details",     "show transaction details for TXN001",           ["get_transaction_details"])
    check("download_txn_report",     "download transaction report as Excel",          ["download_transaction_report"])
    check("pending_transactions",    "show pending transactions in queue",            ["get_pending_transactions"])

    # ── DUES & REMINDERS ───────────────────────────────────────────
    check("upcoming_dues",           "show all upcoming dues next 30 days",           ["get_upcoming_dues"])
    check("overdue_payments",        "show all overdue missed payments",              ["get_overdue_payments"])
    check("set_reminder",            "set reminder for GST due date",                 ["set_payment_reminder"])
    check("reminder_list",           "show all my reminders",                         ["get_reminder_list"])
    check("delete_reminder",         "delete reminder REM001",                        ["delete_reminder"])

    # ── DASHBOARD & ANALYTICS ──────────────────────────────────────
    check("dashboard_only",          "show my dashboard",                             ["get_dashboard_summary"],
          forbidden_intents=["get_upcoming_dues", "get_overdue_payments",
                              "get_cashflow_summary", "get_spending_analytics"])
    check("spending_analytics",      "show spending analytics breakdown",             ["get_spending_analytics"])
    check("cashflow_summary",        "show cashflow summary for this month",          ["get_cashflow_summary"])
    check("monthly_report",          "show monthly report for February 2026",         ["get_monthly_report"])
    check("vendor_payment_summary",  "show vendor payment summary",                   ["get_vendor_payment_summary"])

    # ── COMPANY MANAGEMENT ─────────────────────────────────────────
    check("company_profile",         "show my company profile",                       ["get_company_profile"])
    check("update_company",          "update company email address",                  ["update_company_details"])
    check("gst_profile",             "show all linked GST numbers",                   ["get_gst_profile"])
    check("signatories",             "show authorized signatories",                   ["get_authorized_signatories"])
    check("manage_roles",            "update user role for employee EMP001",          ["manage_user_roles"])

    # ── SUPPORT ────────────────────────────────────────────────────
    check("raise_ticket",            "raise support ticket for payment issue",        ["raise_support_ticket"])
    check("ticket_history",          "show all my support tickets",                   ["get_ticket_history"])
    check("chat_support",            "chat with support agent",                       ["chat_with_support"])
    check("contact_details",         "show support contact details",                  ["get_contact_details"])

    # ── ONBOARDING INFO ────────────────────────────────────────────
    check("company_onboarding",      "how to onboard my company",                     ["company_guide"])
    check("required_documents",      "what documents do I need for onboarding",       ["company_documents"])
    check("validation_formats",      "show validation formats for PAN GSTIN",        ["company_field"])
    check("onboarding_faq",          "onboarding FAQ questions",                      ["company_process"])
    check("bank_onboarding_guide",   "how to do bank account onboarding",             ["bank_guide"])
    check("vendor_onboarding_guide", "how to onboard a vendor",                       ["vendor_guide"])

    # ── MULTI-INTENT ───────────────────────────────────────────────
    check("multi_epf_esic_pay",      "pay EPF and ESIC dues for 02-2026",
          expected_intents=["pay_epf", "pay_esic"],
          forbidden_intents=["fetch_epf_dues", "fetch_esic_dues"],
          expect_multi=True)
    check("multi_gst_tax",           "pay GST and direct tax for Q4",
          expected_intents=["pay_gst", "pay_direct_tax"],
          expect_multi=True)
    check("multi_epf_history",       "show EPF and ESIC payment history",
          expected_intents=["get_epf_payment_history", "get_esic_payment_history"],
          expect_multi=True)
    check("multi_balance_dashboard", "show my balance and dashboard",
          expected_intents=["get_account_balance", "get_dashboard_summary"],
          expect_multi=True)

    # ── CONFLICT RESOLUTION ────────────────────────────────────────
    check("conflict_show_gst_dues",  "show my GST dues",
          forbidden_intents=["pay_gst"])
    check("conflict_pay_gst_no_dues","pay 50000 GST for GSTIN 27AAAPD1234F1ZK",
          forbidden_intents=["fetch_gst_dues"])
    check("conflict_balance_only",   "what is my current balance",
          forbidden_intents=["get_account_details", "get_account_summary"])
    check("conflict_download_stmt",  "download bank statement",
          forbidden_intents=["fetch_bank_statement"])
    check("conflict_view_stmt",      "show bank statement last 3 months",
          forbidden_intents=["download_bank_statement"])
    check("conflict_dashboard_nodues", "open my dashboard",
          forbidden_intents=["get_upcoming_dues", "get_cashflow_summary"])

    # ── EDGE CASES ─────────────────────────────────────────────────
    check("edge_empty_like",         "hello",                                         [])
    check("edge_ambiguous_pay",      "make a payment",                                ["initiate_payment"])
    check("edge_show_all_dues",      "show all my upcoming dues",
          expected_intents=["get_upcoming_dues"],
          forbidden_intents=["fetch_gst_dues", "fetch_epf_dues", "fetch_esic_dues"])
    check("edge_overdue_not_upcoming", "show overdue payments only",
          expected_intents=["get_overdue_payments"],
          forbidden_intents=["get_upcoming_dues"])

    return suite


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS — require live server
# ─────────────────────────────────────────────────────────────────────────────

def run_integration_tests(url: str) -> TestSuite:
    suite = TestSuite("Integration: API End-to-End")
    base = url or BASE_URL

    # ── HEALTH CHECK ───────────────────────────────────────────────
    def check_health():
        try:
            r = requests.get(base + "/health", timeout=10)
            data = r.json()
            passed = data.get("status") == "healthy" and data.get("agent_ready") is True
            return TestResult("health_check", "integration",
                              passed, f"status={data.get('status')} agent_ready={data.get('agent_ready')}")
        except Exception as e:
            return TestResult("health_check", "integration", False, str(e))

    suite.results.append(check_health())

    # ── GST CALCULATOR — was broken (api_key injection bug) ────────
    run_chat_test(suite, "api_calculate_gst", "integration",
                  "calculate GST on 50000 at 18%",
                  [lambda r: assert_intents(r, ["calculate_gst"], exact=True),
                   lambda r: assert_tool_success(r, "calculate_gst"),
                   lambda r: assert_no_error(r, "calculate_gst")])

    run_chat_test(suite, "api_reverse_gst", "integration",
                  "reverse calculate GST on 59000 at 18%",
                  [lambda r: assert_intents(r, ["reverse_gst"], exact=True),
                   lambda r: assert_no_error(r, "reverse_gst")])

    run_chat_test(suite, "api_validate_gstin", "integration",
                  "validate GSTIN 27AAAPD1234F1ZK",
                  [lambda r: assert_intents(r, ["validate_gstin"], exact=True),
                   lambda r: assert_no_error(r, "validate_gstin")])

    run_chat_test(suite, "api_gst_breakdown", "integration",
                  "GST breakdown for 100000 at 18%",
                  [lambda r: assert_intents(r, ["gst_breakdown"], exact=True),
                   lambda r: assert_no_error(r, "gst_breakdown")])

    run_chat_test(suite, "api_compare_gst", "integration",
                  "compare GST at 5% 12% and 18% on 50000",
                  [lambda r: assert_intents(r, ["compare_rates"], exact=True),
                   lambda r: assert_no_error(r, "compare_rates")])

    # ── ACCOUNT BALANCE ────────────────────────────────────────────
    run_chat_test(suite, "api_account_balance", "integration",
                  "what is my account balance",
                  [lambda r: assert_intents(r, ["get_account_balance"]),
                   lambda r: assert_tool_called(r, "get_account_balance"),
                   lambda r: assert_tool_success(r, "get_account_balance")])

    # ── DASHBOARD — clean (no leakage) ─────────────────────────────
    run_chat_test(suite, "api_dashboard_clean", "integration",
                  "show my dashboard",
                  [lambda r: assert_intents(r, ["get_dashboard_summary"], exact=True),
                   lambda r: assert_no_intents(r, ["get_upcoming_dues", "get_overdue_payments",
                                                    "get_cashflow_summary", "get_spending_analytics"]),
                   lambda r: assert_tool_success(r, "get_dashboard_summary")])

    # ── GST DUES (single intent, fetch only) ───────────────────────
    run_chat_test(suite, "api_fetch_gst_dues_clean", "integration",
                  "show my GST dues",
                  [lambda r: assert_no_intents(r, ["pay_gst"]),
                   lambda r: assert_intents(r, ["fetch_gst_dues"])])

    # ── EPF DUES ───────────────────────────────────────────────────
    run_chat_test(suite, "api_fetch_epf_dues", "integration",
                  "show my EPF dues",
                  [lambda r: assert_intents(r, ["fetch_epf_dues"], exact=True),
                   lambda r: assert_tool_success(r, "fetch_epf_dues")])

    # ── ESIC DUES ──────────────────────────────────────────────────
    run_chat_test(suite, "api_fetch_esic_dues", "integration",
                  "show my ESIC dues",
                  [lambda r: assert_intents(r, ["fetch_esic_dues"], exact=True),
                   lambda r: assert_tool_success(r, "fetch_esic_dues")])

    # ── MULTI-INTENT: pay EPF + ESIC (was broken) ──────────────────
    run_chat_test(suite, "api_multi_pay_epf_esic", "integration",
                  "pay EPF and ESIC dues for 02-2026",
                  [lambda r: assert_intents(r, ["pay_epf", "pay_esic"]),
                   lambda r: assert_no_intents(r, ["fetch_epf_dues", "fetch_esic_dues"]),
                   lambda r: assert_multi_intent(r, True),
                   lambda r: assert_tool_success(r, "pay_epf"),
                   lambda r: assert_tool_success(r, "pay_esic")])

    # ── MULTI-INTENT: pay EPF only ─────────────────────────────────
    run_chat_test(suite, "api_pay_epf_only", "integration",
                  "pay EPF for 02-2026",
                  [lambda r: assert_intents(r, ["pay_epf"]),
                   lambda r: assert_no_intents(r, ["fetch_epf_dues"]),
                   lambda r: assert_tool_success(r, "pay_epf")])

    # ── MULTI-INTENT: GST + Tax ────────────────────────────────────
    run_chat_test(suite, "api_multi_gst_tax", "integration",
                  "pay GST and advance tax for this quarter",
                  [lambda r: assert_intents(r, ["pay_gst", "pay_direct_tax"]),
                   lambda r: assert_multi_intent(r, True)])

    # ── PAYROLL ────────────────────────────────────────────────────
    run_chat_test(suite, "api_payroll_summary", "integration",
                  "show payroll summary for March 2026",
                  [lambda r: assert_intents(r, ["fetch_payroll_summary"]),
                   lambda r: assert_tool_success(r, "fetch_payroll_summary")])

    # ── ONBOARDING INFO ────────────────────────────────────────────
    run_chat_test(suite, "api_onboarding_guide", "integration",
                  "how do I onboard my company to the platform",
                  [lambda r: assert_intents(r, ["company_guide"]),
                   lambda r: assert_tool_called(r, "get_company_onboarding_guide")])

    # ── SUPPORT TICKET ─────────────────────────────────────────────
    run_chat_test(suite, "api_support_ticket", "integration",
                  "raise a support ticket for payment failure issue",
                  [lambda r: assert_intents(r, ["raise_support_ticket"]),
                   lambda r: assert_tool_success(r, "raise_support_ticket")])

    # ── TRANSACTION HISTORY ────────────────────────────────────────
    run_chat_test(suite, "api_transaction_history", "integration",
                  "show my transaction history for last 30 days",
                  [lambda r: assert_intents(r, ["get_transaction_history"]),
                   lambda r: assert_tool_success(r, "get_transaction_history")])

    # ── SPENDING ANALYTICS ─────────────────────────────────────────
    run_chat_test(suite, "api_spending_analytics", "integration",
                  "show spending analytics",
                  [lambda r: assert_intents(r, ["get_spending_analytics"]),
                   lambda r: assert_tool_success(r, "get_spending_analytics")])

    # ── UPCOMING DUES ──────────────────────────────────────────────
    run_chat_test(suite, "api_upcoming_dues", "integration",
                  "show all upcoming dues next 30 days",
                  [lambda r: assert_intents(r, ["get_upcoming_dues"]),
                   lambda r: assert_tool_success(r, "get_upcoming_dues")])

    return suite


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY / CONTEXT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_memory_tests(url: str) -> TestSuite:
    suite = TestSuite("Memory: Context & Session")

    sid = next_session()

    # Turn 1: Provide company info — check it's stored in memory
    resp1 = chat("My company is Demo Pvt Ltd with GSTIN 27AAAPD1234F1ZK", session_id=sid, url=url)
    if resp1 is None:
        suite.results.append(skip("memory_turn1_store", "memory", "Server unreachable"))
        suite.results.append(skip("memory_turn2_recall", "memory", "Server unreachable"))
        suite.results.append(skip("memory_turn3_context", "memory", "Server unreachable"))
        return suite

    p1, m1 = assert_memory_field(resp1, "gstin", "27AAAPD1234F1ZK")
    suite.results.append(mk("memory_turn1_store_gstin", "memory", p1, m1))

    p2, m2 = assert_memory_field(resp1, "company_name", "Demo Pvt Ltd")
    suite.results.append(mk("memory_turn1_store_company", "memory", p2, m2))

    # Turn 2: Use context — GST dues should use stored GSTIN
    resp2 = chat("show my GST dues", session_id=sid, url=url)
    p3, m3 = assert_intents(resp2, ["fetch_gst_dues"])
    suite.results.append(mk("memory_turn2_gst_intent", "memory", p3, m3))

    p4, m4 = assert_context_used(resp2)
    suite.results.append(mk("memory_turn2_context_used", "memory", p4, m4))

    # Turn 3: Tool should now be called (GSTIN available in memory)
    p5, m5 = assert_tool_called(resp2, "fetch_gst_dues")
    suite.results.append(mk("memory_turn2_tool_called", "memory", p5, m5))

    # Turn 4: New session should NOT have context
    sid2 = next_session()
    resp3 = chat("show my GST dues", session_id=sid2, url=url)
    p6, m6 = assert_memory_field(resp3, "gstin", None)
    suite.results.append(mk("memory_isolation_new_session", "memory", p6, m6))

    # Turn 5: Multi-turn continuity — set account, then use it
    sid3 = next_session()
    chat("My account number is 9876543210", session_id=sid3, url=url)
    resp4 = chat("show my account balance", session_id=sid3, url=url)
    p7, m7 = assert_context_used(resp4)
    suite.results.append(mk("memory_account_context", "memory", p7, m7))

    return suite


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION TESTS — previously known bugs
# ─────────────────────────────────────────────────────────────────────────────

def run_regression_tests(url: str) -> TestSuite:
    suite = TestSuite("Regression: Known Bug Fixes")

    # BUG 1: calculate_gst had api_key injected → pydantic validation error
    run_chat_test(suite, "reg_calc_gst_no_apikey_error", "regression",
                  "calculate GST on 75000 at 12%",
                  lambda r: assert_no_error(r, "calculate_gst"), url=url)

    # BUG 2: "show my GST dues" was triggering pay_gst
    run_chat_test(suite, "reg_show_gst_dues_no_pay", "regression",
                  "show my GST dues",
                  lambda r: assert_no_intents(r, ["pay_gst"]), url=url)

    # BUG 3: "pay EPF and ESIC dues" was triggering fetch_esic_dues instead of pay_esic
    run_chat_test(suite, "reg_pay_epf_esic_no_fetch", "regression",
                  "pay EPF and ESIC dues for 02-2026",
                  [lambda r: assert_no_intents(r, ["fetch_esic_dues", "fetch_epf_dues"]),
                   lambda r: assert_intents(r, ["pay_epf", "pay_esic"])], url=url)

    # BUG 4: "show dashboard" was leaking get_upcoming_dues
    run_chat_test(suite, "reg_dashboard_no_dues_leak", "regression",
                  "show my dashboard",
                  lambda r: assert_no_intents(r, ["get_upcoming_dues", "get_overdue_payments"]),
                  url=url)

    # BUG 5: "show cashflow" was returning cashflow + dashboard together
    run_chat_test(suite, "reg_cashflow_no_dashboard_leak", "regression",
                  "show cashflow summary for this month",
                  lambda r: assert_no_intents(r, ["get_dashboard_summary"]), url=url)

    # BUG 6: "show balance" was leaking get_account_details
    run_chat_test(suite, "reg_balance_no_details_leak", "regression",
                  "what is my current account balance",
                  lambda r: assert_no_intents(r, ["get_account_details"]), url=url)

    # BUG 7: pay_esic + fetch_esic_dues conflict with "dues" in query
    run_chat_test(suite, "reg_pay_esic_dues_word", "regression",
                  "pay ESIC dues for 03-2026",
                  [lambda r: assert_intents(r, ["pay_esic"]),
                   lambda r: assert_no_intents(r, ["fetch_esic_dues"])], url=url)

    # BUG 8: validate_gstin / gst_breakdown / compare_gst_rates had api_key injection
    run_chat_test(suite, "reg_validate_gstin_no_apikey", "regression",
                  "validate GSTIN 27AAAPD1234F1ZK",
                  lambda r: assert_no_error(r, "validate_gstin"), url=url)

    run_chat_test(suite, "reg_gst_breakdown_no_apikey", "regression",
                  "GST breakdown for 100000 at 18%",
                  lambda r: assert_no_error(r, "gst_breakdown"), url=url)

    # BUG 9: reverse_calculate_gst also had api_key injected
    run_chat_test(suite, "reg_reverse_gst_no_apikey", "regression",
                  "reverse GST calculation on 59000 at 18%",
                  lambda r: assert_no_error(r, "reverse_gst"), url=url)

    return suite


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

PASS  = "\033[92m✓ PASS\033[0m"
FAIL  = "\033[91m✗ FAIL\033[0m"
SKIP  = "\033[93m⊘ SKIP\033[0m"
BOLD  = "\033[1m"
RESET = "\033[0m"

def print_suite(suite: TestSuite, verbose: bool = False):
    print(f"\n{BOLD}{'─'*70}{RESET}")
    print(f"{BOLD}{suite.name}{RESET}  —  "
          f"{PASS}: {suite.passed}  "
          f"{FAIL}: {suite.failed}  "
          f"{SKIP}: {suite.skipped}  "
          f"/ {suite.total} total")
    print(f"{'─'*70}")

    for r in suite.results:
        if r.skipped:
            print(f"  {SKIP}  {r.name:<50}  {r.message}")
        elif r.passed:
            if verbose:
                print(f"  {PASS}  {r.name:<50}  {r.message}  ({r.duration_ms:.0f}ms)")
        else:
            print(f"  {FAIL}  {r.name:<50}  {r.message}")

def print_summary(all_suites: List[TestSuite]):
    total_pass  = sum(s.passed  for s in all_suites)
    total_fail  = sum(s.failed  for s in all_suites)
    total_skip  = sum(s.skipped for s in all_suites)
    total_total = sum(s.total   for s in all_suites)

    print(f"\n{'═'*70}")
    print(f"{BOLD}OVERALL RESULTS{RESET}")
    print(f"{'═'*70}")
    print(f"  Passed  : {total_pass:>4} / {total_total}")
    print(f"  Failed  : {total_fail:>4}")
    print(f"  Skipped : {total_skip:>4}")
    pct = (total_pass / max(total_total - total_skip, 1)) * 100
    color = "\033[92m" if pct == 100 else "\033[93m" if pct >= 80 else "\033[91m"
    print(f"  Score   : {color}{pct:.1f}%{RESET}")

    if total_fail:
        print(f"\n{BOLD}Failed tests:{RESET}")
        for s in all_suites:
            for r in s.results:
                if not r.passed and not r.skipped:
                    print(f"  [{s.name}] {r.name}: {r.message}")
    print(f"{'═'*70}\n")

    return total_fail == 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bank AI Automated Test Suite")
    parser.add_argument("--url",        default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--unit-only",  action="store_true",             help="Run unit tests only (no server)")
    parser.add_argument("--category",   default="all",
                        choices=["all", "unit", "integration", "memory", "regression"],
                        help="Which category to run")
    parser.add_argument("--verbose",    action="store_true",             help="Show passed tests too")
    args = parser.parse_args()

    print(f"\n{BOLD}Bank AI Assistant — Automated Test Suite{RESET}")
    print(f"Server : {args.url}")
    print(f"Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not HAS_REQUESTS and not args.unit_only:
        print("\n⚠️  'requests' not installed — running unit tests only")
        print("   pip install requests  to enable integration tests\n")

    all_suites = []

    run_unit = args.category in ("all", "unit") or args.unit_only
    run_intg = args.category in ("all", "integration") and not args.unit_only and HAS_REQUESTS
    run_mem  = args.category in ("all", "memory")      and not args.unit_only and HAS_REQUESTS
    run_reg  = args.category in ("all", "regression")  and not args.unit_only and HAS_REQUESTS

    if run_unit:
        s = run_unit_tests()
        all_suites.append(s)
        print_suite(s, verbose=args.verbose)

    if run_intg:
        s = run_integration_tests(args.url)
        all_suites.append(s)
        print_suite(s, verbose=args.verbose)

    if run_mem:
        s = run_memory_tests(args.url)
        all_suites.append(s)
        print_suite(s, verbose=args.verbose)

    if run_reg:
        s = run_regression_tests(args.url)
        all_suites.append(s)
        print_suite(s, verbose=args.verbose)

    success = print_summary(all_suites)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()