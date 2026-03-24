"""
Fintech ML Model — Training + Comprehensive Test Suite
Trains ProductionIntentClassifier and validates all intents, conflict
resolution, multi-intent detection, entity extraction, and edge cases.

Usage:
    python train_model.py              # train + full test run
    python train_model.py --test-only  # skip retraining, load saved model
    python train_model.py --verbose    # print all results including passes
"""

import sys
import logging
import argparse
from ml_intent_classifier import ProductionIntentClassifier

logging.basicConfig(
    level=logging.WARNING,           # suppress classifier noise during tests
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class TestRunner:
    def __init__(self, classifier: ProductionIntentClassifier, verbose: bool = False):
        self.clf     = classifier
        self.verbose = verbose
        self.passed  = 0
        self.failed  = 0
        self.results = []

    def run(self, name: str, query: str,
            expected_intents: list  = None,
            forbidden_intents: list = None,
            exact_intents: bool     = False,
            expected_tools: int     = None,
            min_tools: int          = None,
            expect_multi: bool      = None):
        """
        Run a single test case.
        - expected_intents  : all must appear (or exact match if exact_intents=True)
        - forbidden_intents : none of these may appear
        - exact_intents     : intents_detected must equal exactly expected_intents (sorted)
        - expected_tools    : exact tool count
        - min_tools         : minimum tool count
        - expect_multi      : is_multi_intent flag expected value
        """
        result   = self.clf.process_query(query)
        detected = result["intents_detected"]
        tools    = result["tool_calls"]
        is_multi = result["is_multi_intent"]

        errors = []

        # Intent checks
        if expected_intents is not None:
            if exact_intents:
                if sorted(detected) != sorted(expected_intents):
                    errors.append(
                        f"Intents exact mismatch — expected {sorted(expected_intents)}, "
                        f"got {sorted(detected)}"
                    )
            else:
                missing = [i for i in expected_intents if i not in detected]
                if missing:
                    errors.append(f"Missing intents: {missing}  (detected={detected})")

        if forbidden_intents:
            found = [i for i in forbidden_intents if i in detected]
            if found:
                errors.append(f"Forbidden intents found: {found}")

        # Tool count checks
        if expected_tools is not None and len(tools) != expected_tools:
            errors.append(
                f"Tool count: expected {expected_tools}, got {len(tools)} "
                f"({[t['tool_name'] for t in tools]})"
            )
        if min_tools is not None and len(tools) < min_tools:
            errors.append(
                f"Tool count too low: min {min_tools}, got {len(tools)} "
                f"({[t['tool_name'] for t in tools]})"
            )

        # Multi-intent flag
        if expect_multi is not None and is_multi != expect_multi:
            errors.append(f"is_multi_intent: expected {expect_multi}, got {is_multi}")

        passed = len(errors) == 0
        if passed:
            self.passed += 1
        else:
            self.failed += 1

        self.results.append({
            "name":     name,
            "query":    query,
            "passed":   passed,
            "errors":   errors,
            "detected": detected,
            "tools":    [t["tool_name"] for t in tools],
            "entities": result.get("entities", {}),
        })

        return passed

    def section(self, title: str):
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")

    def print_results(self):
        """Print failed tests + summary."""
        failures = [r for r in self.results if not r["passed"]]

        if failures:
            print(f"\n{'═' * 70}")
            print("  FAILED TESTS")
            print(f"{'═' * 70}")
            for r in failures:
                print(f"\n  ✗ [{r['name']}]  {r['query']}")
                for e in r["errors"]:
                    print(f"      → {e}")
                print(f"      detected={r['detected']}  tools={r['tools']}")
                if r["entities"]:
                    print(f"      entities={r['entities']}")

        if self.verbose:
            passes = [r for r in self.results if r["passed"]]
            if passes:
                print(f"\n{'─' * 70}")
                print("  PASSED TESTS")
                print(f"{'─' * 70}")
                for r in passes:
                    print(f"  ✓ [{r['name']}]  {r['query']}")
                    print(f"      intents={r['detected']}  tools={r['tools']}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests(clf: ProductionIntentClassifier, verbose: bool) -> TestRunner:
    t = TestRunner(clf, verbose=verbose)

    # ══════════════════════════════════════════════════════════════════════
    # 1. CORE PAYMENT
    # ══════════════════════════════════════════════════════════════════════
    t.section("1. Core Payment")
    t.run("payment_initiate",    "send ₹50000 to vendor via NEFT",
          expected_intents=["initiate_payment"], expected_tools=1)
    t.run("payment_status",      "check payment status for TXN123456",
          expected_intents=["get_payment_status"])
    t.run("payment_cancel",      "cancel payment TXN999",
          expected_intents=["cancel_payment"])
    t.run("payment_retry",       "retry failed payment TXN888",
          expected_intents=["retry_payment"])
    t.run("payment_receipt",     "get receipt for payment TXN777",
          expected_intents=["get_payment_receipt"])
    t.run("validate_beneficiary","validate beneficiary account 9876543210",
          expected_intents=["validate_beneficiary"])
    t.run("upload_bulk",         "upload bulk payment file",
          expected_intents=["upload_bulk_payment"])
    t.run("validate_file",       "validate payment file before upload",
          expected_intents=["validate_payment_file"])

    # ══════════════════════════════════════════════════════════════════════
    # 2. B2B
    # ══════════════════════════════════════════════════════════════════════
    t.section("2. B2B")
    t.run("onboard_partner",   "onboard new business partner Infosys",
          expected_intents=["onboard_business_partner"])
    t.run("send_invoice",      "send invoice to client for ₹200000",
          expected_intents=["send_invoice"])
    t.run("recv_invoices",     "show all received invoices from vendors",
          expected_intents=["get_received_invoices"])
    t.run("ack_payment",       "acknowledge payment for invoice INV001",
          expected_intents=["acknowledge_payment"])
    t.run("proforma_invoice",  "create proforma invoice for ₹500000",
          expected_intents=["create_proforma_invoice"])
    t.run("cd_note",           "create credit note for client refund",
          expected_intents=["create_cd_note"])
    t.run("purchase_order",    "raise purchase order for office supplies ₹250000",
          expected_intents=["create_purchase_order"])

    # ══════════════════════════════════════════════════════════════════════
    # 3. GST — BANK TOOLS (fetch/pay/challan/history)
    # ══════════════════════════════════════════════════════════════════════
    t.section("3. GST — Bank Tools")
    t.run("fetch_gst_dues",    "show my GST dues",
          expected_intents=["fetch_gst_dues"],
          forbidden_intents=["pay_gst"], expected_tools=1)
    t.run("pay_gst_clean",     "pay GST for GSTIN 27AAAPD1234F1ZK amount 50000",
          expected_intents=["pay_gst"],
          forbidden_intents=["fetch_gst_dues"])
    t.run("pay_gst_dues_word", "pay GST dues for this month",
          expected_intents=["pay_gst"],
          forbidden_intents=["fetch_gst_dues"])
    t.run("gst_challan",       "create GST challan PMT-06 for GSTIN 27AAAPD1234F1ZK",
          expected_intents=["create_gst_challan"])
    t.run("gst_history",       "show GST payment history for last quarter",
          expected_intents=["get_gst_payment_history"])

    # ══════════════════════════════════════════════════════════════════════
    # 4. GST CALCULATOR — GST Server Tools
    # ══════════════════════════════════════════════════════════════════════
    t.section("4. GST Calculator")
    t.run("calc_gst_18",       "calculate GST on 10000 at 18%",
          expected_intents=["calculate_gst"], expected_tools=1)
    t.run("calc_gst_12",       "calculate GST on 5000 at 12%",
          expected_intents=["calculate_gst"], expected_tools=1)
    t.run("reverse_gst",       "reverse calculate GST total amount 59000 at 18%",
          expected_intents=["reverse_gst"], expected_tools=1)
    t.run("gst_breakdown",     "show GST breakdown for 100000 at 18%",
          expected_intents=["gst_breakdown"], expected_tools=1)
    t.run("compare_rates",     "compare GST rates 5% 12% 18% on 50000",
          expected_intents=["compare_rates"], expected_tools=1)
    t.run("validate_gstin",    "validate GSTIN 27AAAPD1234F1ZK",
          expected_intents=["validate_gstin"], expected_tools=1)

    # GST multi-intent combinations
    t.run("calc_and_breakdown","calculate GST on 5000 at 12% and show breakdown",
          expected_intents=["calculate_gst", "gst_breakdown"],
          expect_multi=True, min_tools=2)
    t.run("calc_and_compare",  "calculate GST on 10000 at 18% and compare with 12%",
          expected_intents=["calculate_gst", "compare_rates"],
          expect_multi=True, min_tools=2)
    t.run("validate_and_calc", "validate GSTIN 29ABCDE1234F1Z5 and calculate GST on 5000 at 12%",
          expected_intents=["validate_gstin", "calculate_gst"],
          expect_multi=True, min_tools=2)
    t.run("breakdown_and_guide","show GST breakdown for 10000 at 18% along with company registration",
          expected_intents=["gst_breakdown", "company_guide"],
          expect_multi=True, min_tools=2)

    # multi-rate calculate_gst: 3 rates → 3 tool calls (+ 1 compare = 4 total)
    t.run("calc_multi_rate",   "calculate GST on 5000 at 5% and 12% and 18%",
          expected_intents=["calculate_gst"],
          min_tools=1)   # at minimum 1 gst call; compare_rates may also fire

    # ══════════════════════════════════════════════════════════════════════
    # 5. ESIC
    # ══════════════════════════════════════════════════════════════════════
    t.section("5. ESIC")
    t.run("fetch_esic_dues",   "show my ESIC dues",
          expected_intents=["fetch_esic_dues"],
          forbidden_intents=["pay_esic"], expected_tools=1)
    t.run("pay_esic_clean",    "pay ESIC for 02-2026",
          expected_intents=["pay_esic"],
          forbidden_intents=["fetch_esic_dues"])
    t.run("pay_esic_dues_word","pay ESIC dues for 03-2026",
          expected_intents=["pay_esic"],
          forbidden_intents=["fetch_esic_dues"])
    t.run("esic_history",      "show ESIC payment history",
          expected_intents=["get_esic_payment_history"])

    # ══════════════════════════════════════════════════════════════════════
    # 6. EPF
    # ══════════════════════════════════════════════════════════════════════
    t.section("6. EPF")
    t.run("fetch_epf_dues",    "show my EPF dues",
          expected_intents=["fetch_epf_dues"],
          forbidden_intents=["pay_epf"], expected_tools=1)
    t.run("pay_epf_clean",     "pay EPF for 02-2026",
          expected_intents=["pay_epf"],
          forbidden_intents=["fetch_epf_dues"])
    t.run("pay_epf_dues_word", "pay EPF dues for this month",
          expected_intents=["pay_epf"],
          forbidden_intents=["fetch_epf_dues"])
    t.run("epf_history",       "show EPF payment history",
          expected_intents=["get_epf_payment_history"])

    # ══════════════════════════════════════════════════════════════════════
    # 7. PAYROLL
    # ══════════════════════════════════════════════════════════════════════
    t.section("7. Payroll")
    t.run("payroll_summary",   "show payroll summary for March 2026",
          expected_intents=["fetch_payroll_summary"],
          forbidden_intents=["process_payroll"])
    t.run("process_payroll",   "process payroll for March 2026",
          expected_intents=["process_payroll"],
          forbidden_intents=["fetch_payroll_summary"])
    t.run("payroll_history",   "show payroll payment history",
          expected_intents=["get_payroll_history"])

    # ══════════════════════════════════════════════════════════════════════
    # 8. TAXES
    # ══════════════════════════════════════════════════════════════════════
    t.section("8. Taxes")
    t.run("fetch_tax_dues",    "show my pending tax dues",
          expected_intents=["fetch_tax_dues"],
          forbidden_intents=["pay_direct_tax"])
    t.run("pay_direct_tax",    "pay advance tax ₹100000",
          expected_intents=["pay_direct_tax"])
    t.run("pay_tax_dues_word", "pay direct tax dues",
          expected_intents=["pay_direct_tax"],
          forbidden_intents=["fetch_tax_dues"])
    t.run("pay_state_tax",     "pay professional tax Karnataka",
          expected_intents=["pay_state_tax"])
    t.run("pay_bulk_tax",      "pay bulk TDS for multiple challans",
          expected_intents=["pay_bulk_tax"])
    t.run("tax_history",       "show direct tax payment history",
          expected_intents=["get_tax_payment_history"])

    # ══════════════════════════════════════════════════════════════════════
    # 9. INSURANCE
    # ══════════════════════════════════════════════════════════════════════
    t.section("9. Insurance")
    t.run("fetch_insurance",   "show insurance premium dues",
          expected_intents=["fetch_insurance_dues"],
          forbidden_intents=["pay_insurance_premium"])
    t.run("pay_insurance",     "pay insurance premium for policy LIC001",
          expected_intents=["pay_insurance_premium"])
    t.run("pay_ins_dues_word", "pay insurance dues this month",
          expected_intents=["pay_insurance_premium"],
          forbidden_intents=["fetch_insurance_dues"])
    t.run("insurance_history", "show insurance payment history",
          expected_intents=["get_insurance_payment_history"])

    # ══════════════════════════════════════════════════════════════════════
    # 10. CUSTOM / SEZ
    # ══════════════════════════════════════════════════════════════════════
    t.section("10. Custom Duty / SEZ")
    t.run("pay_custom_duty",   "pay custom duty for bill of entry BOE2026",
          expected_intents=["pay_custom_duty"])
    t.run("track_custom_duty", "track status of custom duty payment",
          expected_intents=["track_custom_duty_payment"],
          forbidden_intents=["pay_custom_duty"])
    t.run("custom_history",    "show custom duty payment history",
          expected_intents=["get_custom_duty_history"])

    # ══════════════════════════════════════════════════════════════════════
    # 11. BANK / ACCOUNT
    # ══════════════════════════════════════════════════════════════════════
    t.section("11. Bank / Account")
    t.run("account_balance",   "what is my current account balance",
          expected_intents=["get_account_balance"],
          forbidden_intents=["get_account_details", "get_account_summary"])
    t.run("account_details",   "show account details IFSC and branch",
          expected_intents=["get_account_details"],
          forbidden_intents=["get_account_balance"])
    t.run("account_summary",   "show summary of all my linked accounts",
          expected_intents=["get_account_summary"])
    t.run("linked_accounts",   "how many linked accounts do I have",
          expected_intents=["get_linked_accounts"])
    t.run("set_default_acct",  "set account 9876543210 as default primary account",
          expected_intents=["set_default_account"])
    t.run("bank_statement",    "show bank statement for last 3 months",
          expected_intents=["fetch_bank_statement"],
          forbidden_intents=["download_bank_statement"])
    t.run("download_statement","download bank statement as PDF",
          expected_intents=["download_bank_statement"],
          forbidden_intents=["fetch_bank_statement"])

    # ══════════════════════════════════════════════════════════════════════
    # 12. TRANSACTIONS
    # ══════════════════════════════════════════════════════════════════════
    t.section("12. Transactions")
    t.run("txn_history",       "show transaction history for last 30 days",
          expected_intents=["get_transaction_history"])
    t.run("search_txns",       "search transactions for vendor ABC",
          expected_intents=["search_transactions"])
    t.run("txn_details",       "show transaction details for TXN123456",
          expected_intents=["get_transaction_details"])
    t.run("download_txn",      "download transaction report as Excel",
          expected_intents=["download_transaction_report"])
    t.run("pending_txns",      "show all pending transactions in queue",
          expected_intents=["get_pending_transactions"])

    # ══════════════════════════════════════════════════════════════════════
    # 13. DUES & REMINDERS
    # ══════════════════════════════════════════════════════════════════════
    t.section("13. Dues & Reminders")
    t.run("upcoming_dues",     "show all upcoming dues next 30 days",
          expected_intents=["get_upcoming_dues"],
          forbidden_intents=["fetch_gst_dues", "fetch_epf_dues", "fetch_esic_dues"])
    t.run("overdue_payments",  "show all overdue missed payments",
          expected_intents=["get_overdue_payments"],
          forbidden_intents=["get_upcoming_dues"])
    t.run("set_reminder",      "set reminder for GST payment due date",
          expected_intents=["set_payment_reminder"])
    t.run("reminder_list",     "show all my active reminders",
          expected_intents=["get_reminder_list"])
    t.run("delete_reminder",   "delete reminder REM001",
          expected_intents=["delete_reminder"])

    # ══════════════════════════════════════════════════════════════════════
    # 14. DASHBOARD & ANALYTICS
    # ══════════════════════════════════════════════════════════════════════
    t.section("14. Dashboard & Analytics")
    t.run("dashboard_clean",   "show my dashboard",
          expected_intents=["get_dashboard_summary"],
          forbidden_intents=["get_upcoming_dues", "get_overdue_payments",
                              "get_cashflow_summary", "get_spending_analytics"],
          exact_intents=True)
    t.run("spending_analytics","show spending analytics breakdown by category",
          expected_intents=["get_spending_analytics"])
    t.run("cashflow_summary",  "show cashflow summary for this month",
          expected_intents=["get_cashflow_summary"],
          forbidden_intents=["get_dashboard_summary"])
    t.run("monthly_report",    "show monthly report for February 2026",
          expected_intents=["get_monthly_report"])
    t.run("vendor_summary",    "show vendor payment summary",
          expected_intents=["get_vendor_payment_summary"])

    # ══════════════════════════════════════════════════════════════════════
    # 15. COMPANY MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════
    t.section("15. Company Management")
    t.run("company_profile",   "show my company profile",
          expected_intents=["get_company_profile"])
    t.run("update_company",    "update company email address",
          expected_intents=["update_company_details"])
    t.run("gst_profile",       "show all linked GST numbers",
          expected_intents=["get_gst_profile"])
    t.run("signatories",       "show authorized signatories for my company",
          expected_intents=["get_authorized_signatories"])
    t.run("manage_roles",      "update user role for employee EMP001",
          expected_intents=["manage_user_roles"])

    # ══════════════════════════════════════════════════════════════════════
    # 16. SUPPORT
    # ══════════════════════════════════════════════════════════════════════
    t.section("16. Support")
    t.run("raise_ticket",      "raise support ticket for payment failure",
          expected_intents=["raise_support_ticket"])
    t.run("ticket_history",    "show all my support tickets",
          expected_intents=["get_ticket_history"])
    t.run("chat_support",      "chat with support agent now",
          expected_intents=["chat_with_support"])
    t.run("contact_details",   "show support contact phone and email",
          expected_intents=["get_contact_details"])

    # ══════════════════════════════════════════════════════════════════════
    # 17. ONBOARDING INFO
    # ══════════════════════════════════════════════════════════════════════
    t.section("17. Onboarding Info")
    t.run("company_guide",       "how do I register my company",
          expected_intents=["company_guide"], expected_tools=1)
    t.run("company_documents",   "what documents do I need for company onboarding",
          expected_intents=["company_documents"])
    t.run("validation_formats",  "show validation formats for PAN GSTIN IFSC",
          expected_intents=["company_field"])
    t.run("onboarding_faq",      "show onboarding FAQ frequently asked questions",
          expected_intents=["company_process"])
    t.run("bank_guide",          "how to do bank account onboarding",
          expected_intents=["bank_guide"])
    t.run("vendor_guide",        "how do I onboard a new vendor to the platform",
          expected_intents=["vendor_guide"])

    # ══════════════════════════════════════════════════════════════════════
    # 18. MULTI-INTENT — KEY COMBINATIONS
    # ══════════════════════════════════════════════════════════════════════
    t.section("18. Multi-Intent Detection")
    t.run("multi_epf_esic_pay",  "pay EPF and ESIC dues for 02-2026",
          expected_intents=["pay_epf", "pay_esic"],
          forbidden_intents=["fetch_epf_dues", "fetch_esic_dues"],
          expect_multi=True, min_tools=2)
    t.run("multi_gst_tax",       "pay GST and advance tax for Q4",
          expected_intents=["pay_gst", "pay_direct_tax"],
          expect_multi=True)
    t.run("multi_epf_esic_hist", "show EPF and ESIC payment history",
          expected_intents=["get_epf_payment_history", "get_esic_payment_history"],
          expect_multi=True)
    t.run("multi_balance_dash",  "show my account balance and dashboard",
          expected_intents=["get_account_balance", "get_dashboard_summary"],
          expect_multi=True)
    t.run("multi_three",         "pay EPF, ESIC and GST for February",
          expected_intents=["pay_epf", "pay_esic", "pay_gst"],
          expect_multi=True, min_tools=3)

    # ══════════════════════════════════════════════════════════════════════
    # 19. CONFLICT RESOLUTION
    # ══════════════════════════════════════════════════════════════════════
    t.section("19. Conflict Resolution")
    # GROUP 4: pay vs fetch-dues with "dues" in query
    t.run("conf_pay_gst_dues",    "pay GST dues 27AAAPD1234F1ZK",
          expected_intents=["pay_gst"],
          forbidden_intents=["fetch_gst_dues"])
    t.run("conf_show_gst_dues",   "show my GST dues",
          expected_intents=["fetch_gst_dues"],
          forbidden_intents=["pay_gst"])
    t.run("conf_pay_epf_dues",    "pay EPF dues for 02-2026",
          expected_intents=["pay_epf"],
          forbidden_intents=["fetch_epf_dues"])
    t.run("conf_pay_esic_dues",   "pay ESIC dues for 03-2026",
          expected_intents=["pay_esic"],
          forbidden_intents=["fetch_esic_dues"])
    t.run("conf_pay_tax_dues",    "pay direct tax dues",
          expected_intents=["pay_direct_tax"],
          forbidden_intents=["fetch_tax_dues"])
    t.run("conf_pay_ins_dues",    "pay insurance dues this month",
          expected_intents=["pay_insurance_premium"],
          forbidden_intents=["fetch_insurance_dues"])
    # GROUP 3: account balance vs details
    t.run("conf_balance_only",    "what is my current balance",
          expected_intents=["get_account_balance"],
          forbidden_intents=["get_account_details", "get_account_summary"])
    t.run("conf_details_only",    "show account details with IFSC code",
          expected_intents=["get_account_details"],
          forbidden_intents=["get_account_balance"])
    # Dashboard no leakage
    t.run("conf_dash_nodues",     "open dashboard",
          expected_intents=["get_dashboard_summary"],
          forbidden_intents=["get_upcoming_dues", "get_cashflow_summary",
                              "get_spending_analytics"])
    # Statement: view vs download
    t.run("conf_view_statement",  "show bank statement last month",
          expected_intents=["fetch_bank_statement"],
          forbidden_intents=["download_bank_statement"])
    t.run("conf_dl_statement",    "download bank statement PDF",
          expected_intents=["download_bank_statement"],
          forbidden_intents=["fetch_bank_statement"])
    # Payroll: process vs summary
    t.run("conf_payroll_view",    "show payroll summary",
          expected_intents=["fetch_payroll_summary"],
          forbidden_intents=["process_payroll"])
    t.run("conf_payroll_run",     "run payroll disbursement for March",
          expected_intents=["process_payroll"],
          forbidden_intents=["fetch_payroll_summary"])
    # Custom duty: pay vs track
    t.run("conf_custom_track",    "track status of my custom duty payment",
          expected_intents=["track_custom_duty_payment"],
          forbidden_intents=["pay_custom_duty"])

    # ══════════════════════════════════════════════════════════════════════
    # 20. ENTITY EXTRACTION
    # ══════════════════════════════════════════════════════════════════════
    t.section("20. Entity Extraction")

    def check_entity(name, query, entity_key, expected_value=None):
        result   = clf.process_query(query)
        entities = result.get("entities", {})
        val      = entities.get(entity_key)
        passed   = (val is not None) if expected_value is None else (val == expected_value)
        errors   = [] if passed else [
            f"entity '{entity_key}': expected {expected_value!r}, got {val!r}"
        ]
        if passed:
            t.passed += 1
        else:
            t.failed += 1
        t.results.append({
            "name": name, "query": query, "passed": passed,
            "errors": errors,
            "detected": result["intents_detected"],
            "tools": [tc["tool_name"] for tc in result["tool_calls"]],
            "entities": entities,
        })

    check_entity("entity_gstin",   "show GST dues for 27AAAPD1234F1ZK",
                 "gstin", "27AAAPD1234F1ZK")
    check_entity("entity_amount",  "calculate GST on 50000 at 18%",
                 "amount", 50000.0)
    check_entity("entity_gst_rate","calculate GST on 10000 at 12%",
                 "gst_rate", 12.0)
    check_entity("entity_month",   "pay ESIC dues for 02-2026",
                 "month", "02-2026")
    check_entity("entity_txn_id",  "check payment status TXN123456",
                 "transaction_id")
    check_entity("entity_account", "check balance for account 9876543210",
                 "account_number", "9876543210")

    # ══════════════════════════════════════════════════════════════════════
    # 21. EDGE CASES
    # ══════════════════════════════════════════════════════════════════════
    t.section("21. Edge Cases")
    t.run("edge_empty",         "hello",
          expected_intents=[], exact_intents=True, expected_tools=0)
    t.run("edge_gibberish",     "asdfgh xyz 123",
          expected_intents=[], exact_intents=True, expected_tools=0)
    t.run("edge_all_dues",      "show all my upcoming dues",
          expected_intents=["get_upcoming_dues"],
          forbidden_intents=["fetch_gst_dues", "fetch_epf_dues",
                              "fetch_esic_dues", "fetch_tax_dues"])
    t.run("edge_overdue_only",  "which payments are overdue",
          expected_intents=["get_overdue_payments"],
          forbidden_intents=["get_upcoming_dues"])
    t.run("edge_ambiguous_pay", "make a payment",
          expected_intents=["initiate_payment"])
    t.run("edge_gst_only",      "GST",  # single word
          expected_intents=[])          # should not trigger without context

    return t


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train + test Bank AI intent classifier")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if saved model exists")
    parser.add_argument("--verbose", action="store_true",
                        help="Print passing tests too")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  FINTECH ML — Intent Classifier Training & Test")
    print("=" * 70)

    # ── LOAD OR TRAIN ─────────────────────────────────────────────
    clf = ProductionIntentClassifier()   # auto-loads pkl if it exists

    if clf.classifier is not None and not args.retrain:
        # Model already loaded from pkl — skip training
        print("\n✓ Loaded saved model from models/production_classifier.pkl")
        print("  (Use --retrain to force retraining from datasets)")
    elif args.retrain or clf.classifier is None:
        if clf.classifier is None:
            print("\n  No saved model found — training from scratch...")
        else:
            print("\n  --retrain flag set — retraining from datasets...")
        print("\n[ Training ]")
        logging.getLogger().setLevel(logging.INFO)
        clf.train()
        logging.getLogger().setLevel(logging.WARNING)
        print("✓ Training complete — model saved to models/production_classifier.pkl\n")

    # ── TESTS ─────────────────────────────────────────────────────
    print("\n[ Running Tests ]")
    runner = run_all_tests(clf, verbose=args.verbose)
    runner.print_results()

    total = runner.passed + runner.failed
    pct   = (runner.passed / total * 100) if total else 0

    print("\n" + "=" * 70)
    print("  TEST RESULTS")
    print("=" * 70)
    print(f"  Passed : {runner.passed:>4} / {total}")
    print(f"  Failed : {runner.failed:>4}")
    symbol = "✓" if pct >= 95 else "⚠"
    print(f"  Score  : {symbol} {pct:.1f}%")
    print("=" * 70)

    if runner.failed == 0:
        print("\n  ✓ ALL TESTS PASSED — model is ready for production.\n")
    else:
        print(f"\n  ⚠ {runner.failed} test(s) failed — review failures above.\n")
        print("  To fix: update conflict rules in ml_intent_classifier.py,")
        print("  then run:  python train_model.py --retrain\n")

    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())