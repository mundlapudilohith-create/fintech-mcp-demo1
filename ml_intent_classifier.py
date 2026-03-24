"""
Production-Ready Intent Classifier — Bank AI Assistant
Covers: Payments, B2B, GST, EPF, ESIC, Payroll, Taxes,
        Insurance, Custom/SEZ, Bank Statement,
        Account Management, Transactions, Dues, Dashboard & Support
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import pickle
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProductionIntentClassifier:

    def __init__(self, model_path: str = "models/", datasets_path: str = "datasets/"):
        self.model_path = model_path
        self.datasets_path = datasets_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[OneVsRestClassifier] = None
        self.mlb: Optional[MultiLabelBinarizer] = None
        self.intent_mappings = self._load_intent_mappings()
        self.entity_patterns = self._load_entity_patterns()

        model_file = os.path.join(model_path, "production_classifier.pkl")
        if os.path.exists(model_file):
            logger.info("Loading pre-trained model...")
            self.load_model()
        else:
            logger.info("No pre-trained model found. Please run train() first.")

    # ========================
    # INTENT CONFIG
    # ========================

    def _load_intent_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {

            # ── CORE PAYMENT ──────────────────────────────────────────
            "initiate_payment": {
                "tool": "initiate_payment",
                "required_params": ["beneficiary_id", "amount", "payment_mode"],
                "keywords": [
                    "send money", "transfer money", "initiate payment", "make payment",
                    "pay to", "transfer to", "send funds", "fund transfer",
                    "neft payment", "rtgs payment", "imps payment", "upi payment"
                ],
                "multi_triggers": ["initiate payment", "send money", "fund transfer", "transfer money", "pay to"]
            },
            "get_payment_status": {
                "tool": "get_payment_status",
                "required_params": ["transaction_id"],
                "keywords": [
                    "payment status", "track payment", "transaction status",
                    "check payment", "payment update", "utr status"
                ],
                "multi_triggers": ["payment status", "transaction status", "track payment", "check payment"]
            },
            "cancel_payment": {
                "tool": "cancel_payment",
                "required_params": ["transaction_id"],
                "keywords": [
                    "cancel payment", "stop payment", "abort payment", "revoke payment"
                ],
                "multi_triggers": ["cancel payment", "stop payment", "abort payment"]
            },
            "retry_payment": {
                "tool": "retry_payment",
                "required_params": ["transaction_id"],
                "keywords": [
                    "retry payment", "resend payment", "redo payment", "payment failed retry"
                ],
                "multi_triggers": ["retry payment", "resend payment", "redo payment"]
            },
            "get_payment_receipt": {
                "tool": "get_payment_receipt",
                "required_params": ["transaction_id"],
                "keywords": [
                    "payment receipt", "download receipt", "payment acknowledgment",
                    "receipt download", "transaction receipt"
                ],
                "multi_triggers": ["payment receipt", "download receipt", "transaction receipt"]
            },
            "validate_beneficiary": {
                "tool": "validate_beneficiary",
                "required_params": [],
                "keywords": [
                    "validate account", "verify account", "check account",
                    "validate upi", "verify beneficiary", "validate beneficiary"
                ],
                "multi_triggers": ["validate beneficiary", "verify account", "validate account"]
            },

            # ── UPLOAD PAYMENT ────────────────────────────────────────
            "upload_bulk_payment": {
                "tool": "upload_bulk_payment",
                "required_params": ["file_name", "file_base64"],
                "keywords": [
                    "bulk payment", "upload payment", "batch payment",
                    "bulk transfer", "upload file payment", "multiple payments"
                ],
                "multi_triggers": ["bulk payment", "upload payment", "batch payment", "bulk transfer"]
            },
            "validate_payment_file": {
                "tool": "validate_payment_file",
                "required_params": ["upload_id"],
                "keywords": [
                    "validate payment file", "check payment file", "verify upload"
                ],
                "multi_triggers": ["validate payment file", "verify upload", "check payment file"]
            },

            # ── B2B ───────────────────────────────────────────────────
            "onboard_business_partner": {
                "tool": "onboard_business_partner",
                "required_params": ["company_name", "gstin", "pan", "contact_email", "contact_phone"],
                "keywords": [
                    "onboard partner", "add partner", "register partner",
                    "new business partner", "b2b onboarding", "partner registration"
                ],
                "multi_triggers": ["onboard partner", "add partner", "b2b onboarding", "partner registration"]
            },
            "send_invoice": {
                "tool": "send_invoice",
                "required_params": ["partner_id", "invoice_number", "invoice_date", "due_date", "amount"],
                "keywords": [
                    "send invoice", "create invoice", "raise invoice",
                    "generate invoice", "invoice to partner"
                ],
                "multi_triggers": ["send invoice", "raise invoice", "generate invoice", "create invoice"]
            },
            "get_received_invoices": {
                "tool": "get_received_invoices",
                "required_params": [],
                "keywords": [
                    "received invoices", "incoming invoices", "bills received",
                    "pending invoices", "view invoices", "all invoices"
                ],
                "multi_triggers": ["received invoices", "incoming invoices", "pending invoices"]
            },
            "acknowledge_payment": {
                "tool": "acknowledge_payment",
                "required_params": ["invoice_id", "transaction_id"],
                "keywords": [
                    "acknowledge payment", "payment acknowledgment",
                    "confirm payment", "payment confirmation"
                ],
                "multi_triggers": ["acknowledge payment", "payment acknowledgment", "confirm payment"]
            },
            "create_proforma_invoice": {
                "tool": "create_proforma_invoice",
                "required_params": ["partner_id", "validity_date", "amount", "description"],
                "keywords": [
                    "proforma invoice", "pre-sale invoice", "create proforma",
                    "proforma document", "quotation invoice"
                ],
                "multi_triggers": ["proforma invoice", "create proforma", "pre-sale invoice"]
            },
            "create_cd_note": {
                "tool": "create_cd_note",
                "required_params": ["partner_id", "note_type", "original_invoice_id", "amount", "reason"],
                "keywords": [
                    "credit note", "debit note", "cd note", "adjustment note",
                    "create credit note", "create debit note"
                ],
                "multi_triggers": ["credit note", "debit note", "cd note", "adjustment note"]
            },
            "create_purchase_order": {
                "tool": "create_purchase_order",
                "required_params": ["partner_id", "po_date", "delivery_date", "amount", "description"],
                "keywords": [
                    "purchase order", "create po", "raise po",
                    "new purchase order", "vendor order"
                ],
                "multi_triggers": ["purchase order", "create po", "raise po", "vendor order"]
            },

            # ── INSURANCE ─────────────────────────────────────────────
            "fetch_insurance_dues": {
                "tool": "fetch_insurance_dues",
                "required_params": [],
                "keywords": [
                    "insurance dues", "premium due", "insurance premium",
                    "policy due", "insurance payment due"
                ],
                "multi_triggers": ["insurance dues", "premium due", "policy due", "insurance premium due"]
            },
            "pay_insurance_premium": {
                "tool": "pay_insurance_premium",
                "required_params": ["policy_number", "amount"],
                "keywords": [
                    "pay insurance", "pay premium", "insurance payment",
                    "premium payment", "policy payment", "renew insurance",
                    "insurance renewal", "pay policy", "settle premium",
                    "clear insurance", "insurance due payment", "pay insurance premium",
                    "premium due pay", "policy renewal payment"
                ],
                "multi_triggers": [
                    "pay insurance", "pay premium", "insurance payment",
                    "premium payment", "renew insurance", "pay policy", "insurance renewal"
                ]
            },
            "get_insurance_payment_history": {
                "tool": "get_insurance_payment_history",
                "required_params": [],
                "keywords": [
                    "insurance history", "premium history", "insurance payments",
                    "past insurance", "policy payment history"
                ],
                "multi_triggers": ["insurance history", "premium history", "insurance payment history"]
            },

            # ── BANK STATEMENT ────────────────────────────────────────
            "fetch_bank_statement": {
                "tool": "fetch_bank_statement",
                "required_params": ["account_number", "from_date", "to_date"],
                "keywords": [
                    "bank statement", "account statement", "fetch statement",
                    "view statement", "statement for"
                ],
                "multi_triggers": ["bank statement", "account statement", "fetch statement"]
            },
            "download_bank_statement": {
                "tool": "download_bank_statement",
                "required_params": ["account_number", "from_date", "to_date"],
                "keywords": [
                    "download statement", "export statement", "statement pdf",
                    "statement excel", "statement download"
                ],
                "multi_triggers": ["download statement", "export statement", "statement pdf", "statement download"]
            },
            "get_account_balance": {
                "tool": "get_account_balance",
                "required_params": ["account_number"],
                "keywords": [
                    "account balance", "check balance", "available balance",
                    "current balance", "balance inquiry"
                ],
                "multi_triggers": ["account balance", "check balance", "available balance"]
            },
            "get_transaction_history": {
                "tool": "get_transaction_history",
                "required_params": ["account_number"],
                "keywords": [
                    "transaction history", "recent transactions", "past transactions",
                    "view transactions", "transaction list"
                ],
                "multi_triggers": ["transaction history", "recent transactions", "past transactions"]
            },

            # ── CUSTOM / SEZ ──────────────────────────────────────────
            "pay_custom_duty": {
                "tool": "pay_custom_duty",
                "required_params": ["bill_of_entry_number", "amount", "port_code", "importer_code"],
                "keywords": [
                    "custom duty", "pay custom", "customs payment",
                    "import duty", "sez payment", "customs duty"
                ],
                "multi_triggers": ["custom duty", "pay custom", "customs payment", "import duty"]
            },
            "track_custom_duty_payment": {
                "tool": "track_custom_duty_payment",
                "required_params": ["transaction_id"],
                "keywords": [
                    "track customs", "custom payment status", "duty payment status",
                    "customs tracking"
                ],
                "multi_triggers": ["track customs", "custom payment status", "customs tracking"]
            },
            "get_custom_duty_history": {
                "tool": "get_custom_duty_history",
                "required_params": [],
                "keywords": [
                    "customs history", "custom duty history", "past customs payments",
                    "import duty history"
                ],
                "multi_triggers": ["customs history", "custom duty history", "import duty history"]
            },

            # ── GST ───────────────────────────────────────────────────
            "fetch_gst_dues": {
                "tool": "fetch_gst_dues",
                "required_params": ["gstin"],
                "keywords": [
                    "gst dues", "gst pending", "gst liability",
                    "gst return due", "pending gst"
                ],
                "multi_triggers": ["gst dues", "pending gst", "gst return due", "gst liability"]
            },
            "pay_gst": {
                "tool": "pay_gst",
                "required_params": ["gstin", "challan_number", "amount", "tax_type"],
                "keywords": [
                    "pay gst", "gst payment", "pay igst",
                    "pay cgst", "pay sgst", "pay cess"
                ],
                "multi_triggers": ["pay gst", "gst payment", "pay igst", "pay cgst", "epf esic and gst", "esic and gst", "and gst for"]
            },
            "create_gst_challan": {
                "tool": "create_gst_challan",
                "required_params": ["gstin", "return_period"],
                "keywords": [
                    "gst challan", "create challan", "pmt-06", "generate challan",
                    "gst challan creation"
                ],
                "multi_triggers": ["gst challan", "create challan", "pmt-06", "generate challan"]
            },
            "get_gst_payment_history": {
                "tool": "get_gst_payment_history",
                "required_params": ["gstin"],
                "keywords": [
                    "gst payment history", "past gst payments",
                    "gst history", "previous gst payments"
                ],
                "multi_triggers": ["gst payment history", "gst history", "past gst payments"]
            },

            # ── ESIC ──────────────────────────────────────────────────
            "fetch_esic_dues": {
                "tool": "fetch_esic_dues",
                "required_params": ["establishment_code", "month"],
                "keywords": [
                    "esic dues", "esic contribution", "esic pending",
                    "esic payment due", "employee state insurance",
                    "esic amount due", "how much esic", "esic liability",
                    "esic this month", "esic challan amount", "check esic dues",
                    "esic outstanding", "esic payable", "esi dues", "esi contribution"
                ],
                "multi_triggers": [
                    "esic dues", "esic contribution", "esic pending",
                    "esic payment due", "esic outstanding", "esic payable",
                    "esi dues", "esi contribution", "check esic"
                ]
            },
            "pay_esic": {
                "tool": "pay_esic",
                "required_params": ["establishment_code", "month", "amount"],
                "keywords": [
                    "pay esic", "esic payment", "esic challan",
                    "employee insurance payment"
                ],
                "multi_triggers": ["pay esic", "esic payment", "esic challan", "pay epf and esic",
                                   "esic dues for", "esic for 0", "epf esic and", "esic and gst",
                                   "epf, esic", "esic,", "pay esic and"]
            },
            "get_esic_payment_history": {
                "tool": "get_esic_payment_history",
                "required_params": ["establishment_code"],
                "keywords": [
                    "esic history", "esic payment history",
                    "past esic", "esic records"
                ],
                "multi_triggers": ["esic history", "esic payment history", "past esic",
                                   "epf and esic history", "epf and esic payment",
                                   "esic history and", "and esic history"]
            },

            # ── EPF ───────────────────────────────────────────────────
            "fetch_epf_dues": {
                "tool": "fetch_epf_dues",
                "required_params": ["establishment_id", "month"],
                "keywords": [
                    "epf dues", "pf dues", "epf contribution",
                    "provident fund due", "pf pending"
                ],
                "multi_triggers": ["epf dues", "pf dues", "epf contribution", "provident fund due"]
            },
            "pay_epf": {
                "tool": "pay_epf",
                "required_params": ["establishment_id", "month", "amount"],
                "keywords": [
                    "pay epf", "pf payment", "epf challan",
                    "provident fund payment", "pay pf"
                ],
                "multi_triggers": ["pay epf", "pf payment", "epf challan", "pay pf"]
            },
            "get_epf_payment_history": {
                "tool": "get_epf_payment_history",
                "required_params": ["establishment_id"],
                "keywords": [
                    "epf history", "pf history", "epf payment history",
                    "past pf payments", "epf records"
                ],
                "multi_triggers": ["epf history", "pf history", "epf payment history",
                                   "epf and esic history", "epf and esic payment",
                                   "show epf and", "epf history and"]
            },

            # ── PAYROLL ───────────────────────────────────────────────
            "fetch_payroll_summary": {
                "tool": "fetch_payroll_summary",
                "required_params": ["month"],
                "keywords": [
                    "payroll summary", "salary summary", "payroll report",
                    "employee salary", "payroll details"
                ],
                "multi_triggers": ["payroll summary", "salary summary", "payroll report"]
            },
            "process_payroll": {
                "tool": "process_payroll",
                "required_params": ["month", "account_number", "approved_by"],
                "keywords": [
                    "process payroll", "run payroll", "salary disbursement",
                    "disburse salary", "pay salaries", "payroll processing"
                ],
                "multi_triggers": ["process payroll", "run payroll", "salary disbursement", "disburse salary"]
            },
            "get_payroll_history": {
                "tool": "get_payroll_history",
                "required_params": [],
                "keywords": [
                    "payroll history", "salary history", "past payroll",
                    "payroll records", "previous salaries"
                ],
                "multi_triggers": ["payroll history", "salary history", "past payroll"]
            },

            # ── TAXES ─────────────────────────────────────────────────
            "fetch_tax_dues": {
                "tool": "fetch_tax_dues",
                "required_params": ["pan"],
                "keywords": [
                    "tax dues", "pending tax", "tax liability",
                    "tds dues", "advance tax due", "tax outstanding",
                    "how much tax", "tax payable", "income tax dues",
                    "check tax dues", "tax pending", "any tax due",
                    "corporate tax dues", "what tax is due", "tax owed",
                    "pending tds", "tds outstanding", "tds liability",
                    "advance tax pending", "tax dues check", "tax balance due",
                    "how much tds", "remaining tax", "tax to be paid"
                ],
                "multi_triggers": [
                    "tax dues", "pending tax", "tds dues", "advance tax due",
                    "tax outstanding", "tax payable", "income tax dues",
                    "tds outstanding", "tax pending", "how much tax",
                    "tax owed", "pending tds", "tax to be paid"
                ]
            },
            "pay_direct_tax": {
                "tool": "pay_direct_tax",
                "required_params": ["pan", "tax_type", "assessment_year", "amount", "challan_type"],
                "keywords": [
                    "pay tds", "direct tax", "pay advance tax",
                    "income tax payment", "self assessment tax"
                ],
                "multi_triggers": ["pay tds", "direct tax", "pay advance tax", "income tax payment"]
            },
            "pay_state_tax": {
                "tool": "pay_state_tax",
                "required_params": ["state", "tax_category", "amount", "assessment_period"],
                "keywords": [
                    "state tax", "professional tax", "pay state tax",
                    "vat payment", "state tax payment"
                ],
                "multi_triggers": ["state tax", "professional tax", "pay state tax", "vat payment"]
            },
            "pay_bulk_tax": {
                "tool": "pay_bulk_tax",
                "required_params": ["file_name", "file_base64", "tax_type"],
                "keywords": [
                    "bulk tax", "bulk tds", "tax bulk payment",
                    "multiple tax payments", "bulk tax payment"
                ],
                "multi_triggers": ["bulk tax", "bulk tds", "tax bulk payment", "multiple tax payments"]
            },
            "get_tax_payment_history": {
                "tool": "get_tax_payment_history",
                "required_params": ["pan"],
                "keywords": [
                    "tax history", "tax payment history", "past tax payments",
                    "tds history", "tax records"
                ],
                "multi_triggers": ["tax history", "tax payment history", "tds history", "past tax payments"]
            },

            # ── ACCOUNT MANAGEMENT ────────────────────────────────────
            "get_account_summary": {
                "tool": "get_account_summary",
                "required_params": [],
                "keywords": [
                    "account summary", "all accounts", "my accounts",
                    "linked accounts summary", "accounts overview"
                ],
                "multi_triggers": ["account summary", "all accounts", "accounts overview"]
            },
            "get_account_details": {
                "tool": "get_account_details",
                "required_params": ["account_number"],
                "keywords": [
                    "account details", "account info", "bank account details",
                    "ifsc details", "account information",
                    "show account details", "account holder name",
                    "branch details", "ifsc code", "bank branch info",
                    "account type details", "who is account holder",
                    "account number details", "bank details for account"
                ],
                "multi_triggers": [
                    "account details", "account info", "bank account details",
                    "ifsc details", "branch details", "ifsc code",
                    "account holder name", "account information"
                ]
            },
            "get_linked_accounts": {
                "tool": "get_linked_accounts",
                "required_params": [],
                "keywords": [
                    "linked accounts", "all linked", "connected accounts",
                    "my bank accounts", "list accounts"
                ],
                "multi_triggers": ["linked accounts", "connected accounts", "list accounts"]
            },
            "set_default_account": {
                "tool": "set_default_account",
                "required_params": ["account_number"],
                "keywords": [
                    "set default account", "primary account", "default bank account",
                    "make default", "set primary"
                ],
                "multi_triggers": ["set default account", "primary account", "make default"]
            },

            # ── TRANSACTION & HISTORY ─────────────────────────────────
            "search_transactions": {
                "tool": "search_transactions",
                "required_params": [],
                "keywords": [
                    "search transactions", "find transaction", "filter transactions",
                    "transaction search", "look up transaction"
                ],
                "multi_triggers": ["search transactions", "find transaction", "filter transactions"]
            },
            "get_transaction_details": {
                "tool": "get_transaction_details",
                "required_params": ["transaction_id"],
                "keywords": [
                    "transaction details", "transaction info",
                    "detail of transaction", "transaction breakdown"
                ],
                "multi_triggers": ["transaction details", "transaction info", "transaction breakdown"]
            },
            "download_transaction_report": {
                "tool": "download_transaction_report",
                "required_params": ["from_date", "to_date"],
                "keywords": [
                    "transaction report", "download transactions", "export transactions",
                    "transaction export", "transactions excel"
                ],
                "multi_triggers": ["transaction report", "download transactions", "export transactions"]
            },
            "get_pending_transactions": {
                "tool": "get_pending_transactions",
                "required_params": [],
                "keywords": [
                    "pending transactions", "in-process payments",
                    "outstanding transactions", "pending payments"
                ],
                "multi_triggers": ["pending transactions", "in-process payments", "outstanding transactions"]
            },

            # ── DUES & REMINDERS ──────────────────────────────────────
            "get_upcoming_dues": {
                "tool": "get_upcoming_dues",
                "required_params": [],
                "keywords": [
                    "upcoming dues", "all dues", "what is due",
                    "payment dues", "scheduled dues", "due payments",
                    "dues this month", "what payments are due", "show all dues",
                    "pending dues", "dues next month", "what is coming due",
                    "upcoming payments", "scheduled payments", "dues overview",
                    "due soon", "payments coming up", "next dues",
                    "show upcoming dues", "list all dues", "dues summary",
                    "what dues are pending", "upcoming compliance dues",
                    "all upcoming payments", "payments due this month",
                    "dues next 30 days", "what needs to be paid", "due list",
                    "payment schedule", "upcoming liabilities", "dues calendar",
                    "what do i need to pay", "payments coming", "show due payments"
                ],
                "multi_triggers": [
                    "upcoming dues", "all dues", "payment dues", "due payments",
                    "dues this month", "upcoming payments", "show all dues",
                    "pending dues", "due soon", "dues summary", "due list",
                    "what is due", "payments due", "dues next"
                ]
            },
            "get_overdue_payments": {
                "tool": "get_overdue_payments",
                "required_params": [],
                "keywords": [
                    "overdue payments", "missed payments", "overdue dues",
                    "late payments", "payment overdue"
                ],
                "multi_triggers": ["overdue payments", "missed payments", "late payments"]
            },
            "set_payment_reminder": {
                "tool": "set_payment_reminder",
                "required_params": ["title", "due_date"],
                "keywords": [
                    "set reminder", "payment reminder", "remind me",
                    "add reminder", "due date reminder"
                ],
                "multi_triggers": ["set reminder", "payment reminder", "due date reminder"]
            },
            "get_reminder_list": {
                "tool": "get_reminder_list",
                "required_params": [],
                "keywords": [
                    "reminder list", "my reminders", "all reminders",
                    "view reminders", "active reminders"
                ],
                "multi_triggers": ["reminder list", "my reminders", "view reminders"]
            },
            "delete_reminder": {
                "tool": "delete_reminder",
                "required_params": ["reminder_id"],
                "keywords": [
                    "delete reminder", "remove reminder",
                    "cancel reminder", "clear reminder"
                ],
                "multi_triggers": ["delete reminder", "remove reminder", "cancel reminder"]
            },

            # ── DASHBOARD & ANALYTICS ─────────────────────────────────
            "get_dashboard_summary": {
                "tool": "get_dashboard_summary",
                "required_params": [],
                "keywords": [
                    "dashboard", "overview", "account health",
                    "financial summary", "dashboard summary"
                ],
                "multi_triggers": ["dashboard summary", "account health", "financial summary", "overview"]
            },
            "get_spending_analytics": {
                "tool": "get_spending_analytics",
                "required_params": [],
                "keywords": [
                    "spending analytics", "expense breakdown", "category wise spending",
                    "spending report", "where am i spending"
                ],
                "multi_triggers": ["spending analytics", "expense breakdown", "spending report"]
            },
            "get_cashflow_summary": {
                "tool": "get_cashflow_summary",
                "required_params": [],
                "keywords": [
                    "cashflow", "cash flow", "inflow outflow",
                    "net cashflow", "cash summary"
                ],
                "multi_triggers": ["cashflow", "cash flow", "inflow outflow", "net cashflow"]
            },
            "get_monthly_report": {
                "tool": "get_monthly_report",
                "required_params": ["month"],
                "keywords": [
                    "monthly report", "month report", "financial report",
                    "monthly summary", "report for month"
                ],
                "multi_triggers": ["monthly report", "monthly summary", "financial report"]
            },
            "get_vendor_payment_summary": {
                "tool": "get_vendor_payment_summary",
                "required_params": [],
                "keywords": [
                    "vendor payment summary", "vendor wise payment",
                    "top vendors", "vendor payments"
                ],
                "multi_triggers": ["vendor payment summary", "vendor wise payment", "top vendors"]
            },

            # ── COMPANY MANAGEMENT ────────────────────────────────────
            "get_company_profile": {
                "tool": "get_company_profile",
                "required_params": [],
                "keywords": [
                    "company profile", "company details", "business profile",
                    "company info", "organization details"
                ],
                "multi_triggers": ["company profile", "company details", "business profile"]
            },
            "update_company_details": {
                "tool": "update_company_details",
                "required_params": ["field", "value"],
                "keywords": [
                    "update company", "change company details", "edit company",
                    "modify company info", "update business details",
                    "change company address", "update company name",
                    "edit company info", "change business details",
                    "update company profile", "modify company details",
                    "change company email", "update company phone",
                    "edit business info", "update contact details",
                    "change registered address", "update company data",
                    "modify business info", "edit company profile",
                    "change company information", "update company contact",
                    "modify company profile", "change company data",
                    "update my company", "edit my company details",
                    "change my business details", "update business profile",
                    "modify business profile", "change company field"
                ],
                "multi_triggers": [
                    "update company", "change company details", "edit company",
                    "modify company", "change company address", "update company name",
                    "change business details", "update company profile",
                    "edit company info", "update contact details",
                    "change registered address", "update my company"
                ]
            },
            "get_gst_profile": {
                "tool": "get_gst_profile",
                "required_params": [],
                "keywords": [
                    "gst profile", "linked gst", "gst numbers",
                    "company gstin", "registered gst",
                    "my gst numbers", "all gstin", "show gst profile",
                    "gst registrations", "linked gstin numbers",
                    "company gst details", "how many gst", "gst linked accounts",
                    "view gst profile", "gst number list", "registered gstin"
                ],
                "multi_triggers": [
                    "gst profile", "linked gst", "company gstin",
                    "my gst numbers", "all gstin", "gst registrations",
                    "linked gstin", "registered gstin", "gst number list"
                ]
            },
            "get_authorized_signatories": {
                "tool": "get_authorized_signatories",
                "required_params": [],
                "keywords": [
                    "authorized signatories", "authorized persons",
                    "company signatories", "who can sign"
                ],
                "multi_triggers": ["authorized signatories", "authorized persons", "company signatories"]
            },
            "manage_user_roles": {
                "tool": "manage_user_roles",
                "required_params": ["user_id", "role", "action"],
                "keywords": [
                    "user role", "assign role", "change role",
                    "user permission", "maker checker"
                ],
                "multi_triggers": ["user role", "assign role", "change role", "user permission"]
            },

            # ── GST CALCULATOR (→ gst_client_manager / server.py) ─────
            "calculate_gst": {
                "tool": "calculate_gst",
                "required_params": ["base_amount", "gst_rate"],
                "keywords": [
                    "calculate gst", "add gst", "gst on", "apply gst",
                    "gst for", "add tax", "gst amount", "how much gst",
                    "compute gst", "find gst", "what is gst on",
                    "total with gst", "price with gst", "gst calculation",
                    "calculate tax on", "what will be gst", "gst total",
                    "including gst", "with 18% gst", "with 12% gst",
                    "with 5% gst", "with 28% gst", "gst inclusive price",
                    "final price with gst", "amount after gst"
                ],
                "multi_triggers": [
                    "calculate gst", "compute gst", "find gst", "add gst",
                    "gst on", "gst amount", "gst calculation", "total with gst",
                    "price with gst", "what is gst on"
                ]
            },
            "reverse_gst": {
                "tool": "reverse_calculate_gst",
                "required_params": ["total_amount", "gst_rate"],
                "keywords": [
                    "reverse gst", "remove gst", "exclude gst", "before gst",
                    "without gst", "inclusive gst", "gst included",
                    "base price from total", "base amount from total", "excluding gst"
                ],
                "multi_triggers": ["reverse gst", "remove gst", "exclude gst", "without gst", "base price"]
            },
            "gst_breakdown": {
                "tool": "gst_breakdown",
                "required_params": ["base_amount", "gst_rate"],
                "keywords": [
                    "gst breakdown", "split gst", "cgst sgst", "igst breakdown",
                    "tax split", "show breakdown", "cgst and sgst", "show cgst",
                    "show sgst", "intra state gst", "inter state gst"
                ],
                "multi_triggers": ["gst breakdown", "show breakdown", "split gst", "cgst sgst", "igst breakdown"]
            },
            "compare_rates": {
                "tool": "compare_gst_rates",
                "required_params": ["base_amount", "rates"],
                "keywords": [
                    "compare gst", "compare rates", "compare gst rates",
                    "which gst rate", "rate comparison", "different gst rates",
                    "better rate", "gst rate difference"
                ],
                "multi_triggers": ["compare gst", "compare rates", "rate comparison", "gst rate difference"]
            },
            "validate_gstin": {
                "tool": "validate_gstin",
                "required_params": ["gstin"],
                "keywords": [
                    "validate gstin", "check gstin", "gstin valid",
                    "verify gstin", "is gstin valid", "gstin check",
                    "gstin verification", "validate gst number"
                ],
                "multi_triggers": ["validate gstin", "verify gstin", "check gstin", "gstin valid"]
            },

            # ── ONBOARDING INFO (→ info_client_manager / info_server.py) ──
            "company_guide": {
                "tool": "get_company_onboarding_guide",
                "required_params": [],
                "keywords": [
                    "company onboarding", "register company", "company registration",
                    "how to onboard company", "start company", "onboard organization",
                    "company setup", "register my company", "register a company",
                    "setting up a company", "set up company", "company register",
                    "explain the company onboarding", "company onboarding process",
                    "onboard my company", "how do i onboard my company", "onboard my company to", "how do i onboard my",
                    "onboard my organization", "company registration process"
                ],
                "multi_triggers": [
                    "company onboarding", "register company", "company registration",
                    "register my company", "register a company", "company setup",
                    "set up company", "registration process", "onboard my company",
                    "company onboarding process", "explain the company onboarding"
                ]
            },
            "company_documents": {
                "tool": "get_company_required_documents",
                "required_params": [],
                "keywords": [
                    "documents needed", "required documents", "company documents",
                    "documents for company", "what documents", "document checklist"
                ],
                "multi_triggers": ["required documents", "document checklist", "what documents", "documents needed"]
            },
            "company_field": {
                "tool": "get_validation_formats",
                "required_params": [],
                "keywords": [
                    "pan number format", "gst number format", "mandatory fields",
                    "field format", "validation format", "field validation"
                ],
                "multi_triggers": ["pan number format", "mandatory fields", "field format", "validation format"]
            },
            "company_process": {
                "tool": "get_onboarding_faq",
                "required_params": [],
                "keywords": [
                    "how long onboarding", "onboarding timeline", "processing time",
                    "approval time", "how many days to register", "onboarding duration",
                    "onboarding faq", "faq onboarding", "frequently asked",
                    "common questions onboarding", "onboarding questions"
                ],
                "multi_triggers": ["processing time", "approval time", "how many days",
                                   "onboarding timeline", "onboarding faq", "frequently asked"]
            },
            "bank_guide": {
                "tool": "get_bank_onboarding_guide",
                "required_params": [],
                "keywords": [
                    "bank onboarding", "register bank", "bank registration",
                    "add bank account", "supported banks", "connect bank account",
                    "how to add bank", "bank account onboarding"
                ],
                "multi_triggers": ["bank onboarding", "register bank", "add bank account", "supported banks"]
            },
            "vendor_guide": {
                "tool": "get_vendor_onboarding_guide",
                "required_params": [],
                "keywords": [
                    "vendor onboarding", "add vendor", "register vendor",
                    "supplier onboarding", "how to add vendor", "vendor registration",
                    "onboard supplier", "create vendor", "how do i onboard a vendor",
                    "how to onboard vendor", "onboard a new vendor", "vendor setup guide"
                ],
                "multi_triggers": ["vendor onboarding", "add vendor", "register vendor",
                                   "supplier onboarding", "how do i onboard a vendor", "onboard a new vendor"]
            },

            # ── SUPPORT ───────────────────────────────────────────────
            "raise_support_ticket": {
                "tool": "raise_support_ticket",
                "required_params": ["category", "subject", "description"],
                "keywords": [
                    "support ticket", "raise ticket", "create ticket",
                    "report issue", "raise complaint", "log issue"
                ],
                "multi_triggers": ["support ticket", "raise ticket", "create ticket", "report issue"]
            },
            "get_ticket_history": {
                "tool": "get_ticket_history",
                "required_params": [],
                "keywords": [
                    "ticket history", "my tickets", "all tickets",
                    "past tickets", "support history"
                ],
                "multi_triggers": ["ticket history", "my tickets", "past tickets"]
            },
            "chat_with_support": {
                "tool": "chat_with_support",
                "required_params": ["issue_summary"],
                "keywords": [
                    "chat support", "live support", "talk to agent",
                    "chat with agent", "live chat"
                ],
                "multi_triggers": ["chat support", "live support", "talk to agent", "live chat"]
            },
            "get_contact_details": {
                "tool": "get_contact_details",
                "required_params": [],
                "keywords": [
                    "contact details", "support contact", "helpline",
                    "customer care", "contact number"
                ],
                "multi_triggers": ["contact details", "support contact", "helpline", "customer care"]
            },
        }

    def _load_entity_patterns(self) -> Dict[str, str]:
        return {
            "amount":     r"(?:₹|rs\.?|inr|rupees?)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
            "percentage": r"(\d+(?:\.\d+)?)\s*(?:%|percent)",
            "gstin":      r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b",
            "pan":        r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
            "account":    r"\b\d{9,18}\b",
            "ifsc":       r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
            "date":       r"\b(\d{4}-\d{2}-\d{2})\b",
            "month":      r"\b(0[1-9]|1[0-2])-(\d{4})\b",
        }

    # ========================
    # DATA LOADING
    # ========================

    def load_datasets(self) -> Tuple[List[str], List[List[str]]]:
        logger.info(f"Loading datasets from {self.datasets_path}")

        queries = []
        labels  = []

        dataset_mapping = {
            # ── Payment ───────────────────────────────────────────────────────
            "payment_initiate_500.csv":                  ["initiate_payment"],
            "payment_status_300.csv":                    ["get_payment_status"],
            "payment_bulk_upload_300.csv":               ["upload_bulk_payment"],
            "acknowledge_payment_200.csv":               ["acknowledge_payment"],
            "cancel_payment_200.csv":                    ["cancel_payment"],
            "retry_payment_200.csv":                     ["retry_payment"],
            "validate_beneficiary_200.csv":              ["validate_beneficiary"],
            "validate_payment_file_200.csv":             ["validate_payment_file"],
            "get_payment_receipt_200.csv":               ["get_payment_receipt"],
            "search_transactions_200.csv":               ["search_transactions"],
            "get_transaction_details_200.csv":           ["get_transaction_details"],
            "get_pending_transactions_200.csv":          ["get_pending_transactions"],
            "get_account_summary_200.csv":               ["get_account_summary"],
            "get_linked_accounts_200.csv":               ["get_linked_accounts"],
            "set_default_account_200.csv":               ["set_default_account"],
            "get_authorized_signatories_200.csv":        ["get_authorized_signatories"],
            "get_cashflow_summary_200.csv":              ["get_cashflow_summary"],
            "get_spending_analytics_200.csv":            ["get_spending_analytics"],
            "get_vendor_payment_summary_200.csv":        ["get_vendor_payment_summary"],
            "get_monthly_report_200.csv":                ["get_monthly_report"],
            "get_overdue_payments_200.csv":              ["get_overdue_payments"],
            "get_reminder_list_200.csv":                 ["get_reminder_list"],
            "delete_reminder_200.csv":                   ["delete_reminder"],
            "get_gst_payment_history_200.csv":           ["get_gst_payment_history"],
            "get_epf_payment_history_200.csv":           ["get_epf_payment_history"],
            "get_esic_payment_history_200.csv":          ["get_esic_payment_history"],
            "get_insurance_payment_history_200.csv":     ["get_insurance_payment_history"],
            "get_tax_payment_history_200.csv":           ["get_tax_payment_history"],
            "get_payroll_history_200.csv":               ["get_payroll_history"],
            "get_custom_duty_history_200.csv":           ["get_custom_duty_history"],
            "get_ticket_history_200.csv":                ["get_ticket_history"],
            "fetch_payroll_summary_200.csv":             ["fetch_payroll_summary"],
            "fetch_insurance_dues_200.csv":              ["fetch_insurance_dues"],
            "pay_custom_duty_200.csv":                   ["pay_custom_duty"],
            "track_custom_duty_payment_200.csv":         ["track_custom_duty_payment"],
            "create_cd_note_200.csv":                    ["create_cd_note"],
            "create_proforma_invoice_200.csv":           ["create_proforma_invoice"],
            "manage_user_roles_200.csv":                 ["manage_user_roles"],
            "pay_bulk_tax_200.csv":                      ["pay_bulk_tax"],
            "download_bank_statement_200.csv":           ["download_bank_statement"],
            "download_transaction_report_200.csv":       ["download_transaction_report"],
            "chat_with_support_200.csv":                 ["chat_with_support"],
            "get_contact_details_200.csv":               ["get_contact_details"],

            # ── B2B ───────────────────────────────────────────────────────────
            "b2b_partner_onboard_400.csv":               ["onboard_business_partner"],
            "b2b_invoice_send_300.csv":                  ["send_invoice"],
            "b2b_invoice_receive_300.csv":               ["get_received_invoices"],
            "b2b_purchase_order_300.csv":                ["create_purchase_order"],

            # ── Compliance ────────────────────────────────────────────────────
            "gst_pay_400.csv":                           ["pay_gst"],
            "gst_challan_300.csv":                       ["create_gst_challan"],
            "epf_pay_400.csv":                           ["pay_epf"],
            "esic_pay_400.csv":                          ["pay_esic"],
            "payroll_process_400.csv":                   ["process_payroll"],
            "tax_direct_400.csv":                        ["pay_direct_tax"],
            "tax_state_300.csv":                         ["pay_state_tax"],
            "pay_insurance_premium_200.csv":             ["pay_insurance_premium"],

            # ── Account & Transactions ────────────────────────────────────────
            "account_balance_300.csv":                   ["get_account_balance"],
            "account_statement_300.csv":                 ["fetch_bank_statement"],
            "transaction_history_300.csv":               ["get_transaction_history"],
            "get_account_details_200.csv":               ["get_account_details"],

            # ── Dashboard & Dues ──────────────────────────────────────────────
            "dashboard_400.csv":                         ["get_dashboard_summary"],
            "dues_upcoming_300.csv":                     ["get_upcoming_dues"],
            "dues_upcoming_boost_400.csv":               ["get_upcoming_dues"],

            # ── Fetch Dues ────────────────────────────────────────────────────
            "fetch_epf_dues_200.csv":                    ["fetch_epf_dues"],
            "fetch_gst_dues_200.csv":                    ["fetch_gst_dues"],
            "fetch_esic_dues_200.csv":                   ["fetch_esic_dues"],
            "fetch_tax_dues_200.csv":                    ["fetch_tax_dues"],

            # ── Reminders ─────────────────────────────────────────────────────
            "set_payment_reminder_200.csv":              ["set_payment_reminder"],

            # ── Support ───────────────────────────────────────────────────────
            "support_ticket_300.csv":                    ["raise_support_ticket"],

            # ── GST Calculator ────────────────────────────────────────────────
            "gst_variations.csv":                        ["calculate_gst"],
            "reverse_gst_variations.csv":                ["reverse_gst"],
            "gst_breakdown_variations.csv":              ["gst_breakdown"],
            "D_rate_comparison_400.csv":                 ["compare_rates"],
            "E_gstin_validation_300.csv":                ["validate_gstin"],
            "get_gst_profile_200.csv":                   ["get_gst_profile"],
            "calc_compare_boost_600.csv":                ["calculate_gst", "compare_rates"],

            # ── Onboarding — Company ──────────────────────────────────────────
            "Company_A_General_Onboarding_500.csv":      ["company_guide"],
            "Company_B_Required_Documents_300.csv":      ["company_documents"],
            "Company_C_Field_Questions_300.csv":         ["company_field"],
            "Company_D_Process_Questions_300.csv":       ["company_process"],
            "company_process_boost_300.csv":             ["company_process"],
            "company_profile_300.csv":                   ["get_company_profile"],
            "company_update_300.csv":                    ["update_company_details"],

            # ── Onboarding — Bank ─────────────────────────────────────────────
            "csv-export-2026-02-19__3_.csv":             ["bank_guide"],
            "csv-export-2026-02-19__4_.csv":             ["bank_guide"],
            "csv-export-2026-02-19__5_.csv":             ["bank_guide"],
            "csv-export-2026-02-19__6_.csv":             ["bank_guide"],
            "csv-export-2026-02-19__7_.csv":             ["bank_guide"],

            # ── Onboarding — Vendor ───────────────────────────────────────────
            "csv-export-2026-02-19.csv":                 ["vendor_guide"],
            "csv-export-2026-02-19__1_.csv":             ["vendor_guide"],
            "csv-export-2026-02-19__2_.csv":             ["vendor_guide"],

            # ── Multi-Intent ──────────────────────────────────────────────────
            "multi_intent_bank_600.csv":                 "MULTI",
            "F_multi_intent_400.csv":                    "MULTI",
            "_MConverter_eu_Multi_Intent_1500.csv":      "MULTI",
        }

        for filename, intent_label in dataset_mapping.items():
            filepath = os.path.join(self.datasets_path, filename)

            if not os.path.exists(filepath):
                logger.warning(f"⚠ Missing file: {filename}")
                continue

            try:
                df = pd.read_csv(filepath, header=None, names=["query"], on_bad_lines="skip")
                df["query"] = df["query"].astype(str)
                df["query"] = df["query"].str.replace(r"^\d+\.\s*", "", regex=True).str.strip()
                df = df[df["query"].str.len() > 5]

                for query in df["query"].tolist():
                    if intent_label == "MULTI":
                        detected = self._detect_multi_intents_from_query(query)
                        if detected:
                            queries.append(query)
                            labels.append(detected)
                    else:
                        queries.append(query)
                        labels.append(intent_label)

                logger.info(f"✓ Loaded {len(df)} examples from {filename}")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

        logger.info(f"Total training examples: {len(queries)}")
        return queries, labels

    def _detect_multi_intents_from_query(self, query: str) -> List[str]:
        """Multi-intent detection for training data using multi_triggers."""
        query_lower = query.lower()
        detected = []

        for intent_name, intent_data in self.intent_mappings.items():
            triggers = intent_data.get("multi_triggers", [])
            for trigger in triggers:
                if trigger in query_lower:
                    detected.append(intent_name)
                    break

        return list(set(detected))[:3]

    # ========================
    # TRAINING
    # ========================

    def train(self):
        texts, labels = self.load_datasets()

        if len(texts) == 0:
            raise ValueError("No training data loaded!")

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=None
        )

        logger.info(f"Training: {len(X_train)} | Test: {len(X_test)}")

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words="english",
            min_df=2,
            sublinear_tf=True
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec  = self.vectorizer.transform(X_test)

        self.mlb = MultiLabelBinarizer()
        y_train_bin = self.mlb.fit_transform(y_train)
        y_test_bin  = self.mlb.transform(y_test)

        logger.info(f"Intent classes: {list(self.mlb.classes_)}")

        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                C=1.5,
                n_jobs=-1,
                class_weight="balanced"
            )
        )

        self.classifier.fit(X_train_vec, y_train_bin)

        y_pred = self.classifier.predict(X_test_vec)

        from sklearn.metrics import hamming_loss, f1_score

        exact_match = accuracy_score(y_test_bin, y_pred)
        hamming     = 1 - hamming_loss(y_test_bin, y_pred)
        macro_f1    = f1_score(y_test_bin, y_pred, average="macro",  zero_division=0)
        micro_f1    = f1_score(y_test_bin, y_pred, average="micro",  zero_division=0)

        logger.info("=" * 70)
        logger.info(f"Exact Match Accuracy : {exact_match * 100:.2f}%")
        logger.info(f"Hamming Score        : {hamming     * 100:.2f}%  ← PRIMARY metric")
        logger.info(f"Macro F1             : {macro_f1    * 100:.2f}%")
        logger.info(f"Micro F1             : {micro_f1    * 100:.2f}%")
        logger.info("-" * 70)
        logger.info("Per-Intent Scores (F1 | Accuracy):")

        for idx, intent in enumerate(self.mlb.classes_):
            intent_acc = accuracy_score(y_test_bin[:, idx], y_pred[:, idx])
            intent_f1  = f1_score(y_test_bin[:, idx], y_pred[:, idx], zero_division=0)
            logger.info(f"  {intent:<35} F1: {intent_f1 * 100:.2f}%  |  Acc: {intent_acc * 100:.2f}%")

        logger.info("=" * 70)

        self.save_model()

    # ========================
    # PREDICTION
    # ========================

    def predict_intents(self, query: str) -> List[str]:
        """Hybrid ML + keyword multi-intent prediction."""
        if not self.vectorizer or not self.classifier:
            raise ValueError("Model not loaded. Run train() first.")

        query_lower = query.lower()

        # Step 1: ML prediction
        X = self.vectorizer.transform([query])
        probabilities = self.classifier.predict_proba(X)[0]

        query_words = len(query.split())
        if query_words > 15:
            threshold = 0.25
        elif query_words > 10:
            threshold = 0.30
        else:
            threshold = 0.35

        predicted = [
            self.mlb.classes_[idx]
            for idx, prob in enumerate(probabilities)
            if prob > threshold
        ]

        if not predicted:
            predicted.append(self.mlb.classes_[np.argmax(probabilities)])

        # Step 2: Keyword enhancement
        keyword_matched = []

        multi_indicators = [" and ", " also ", " then ", " additionally ", " as well as ", " along with "]
        has_multi_indicator = any(ind in query_lower for ind in multi_indicators)

        if has_multi_indicator:
            for intent, config in self.intent_mappings.items():
                for trigger in config.get("multi_triggers", []):
                    if trigger in query_lower:
                        keyword_matched.append(intent)
                        break

        for intent, config in self.intent_mappings.items():
            for kw in config["keywords"]:
                if kw in query_lower and intent not in keyword_matched:
                    keyword_matched.append(intent)
                    break

        combined = list(set(predicted + keyword_matched))

        if len(keyword_matched) >= 2:
            combined = keyword_matched + [p for p in predicted if p not in keyword_matched]

        combined = self._resolve_intent_conflicts(query_lower, combined)
        return combined[:3]

    def _resolve_intent_conflicts(self, query_lower: str, intents: List[str]) -> List[str]:
        """Comprehensive conflict resolution — suppresses all spurious co-intents."""
        resolved = list(intents)

        # ═══════════════════════════════════════════════════════════════
        # GROUP 1: PAYMENT CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # initiate_payment needs an explicit send/transfer verb — not just a number
        if "initiate_payment" in resolved:
            pay_signals = ["send", "transfer", "pay to", "initiate", "make payment",
                           "make a payment", "fund transfer", "neft", "rtgs", "imps", "upi",
                           "remit", "wire", "dispatch payment", "i want to pay",
                           "need to pay", "want to transfer"]
            if not any(sig in query_lower for sig in pay_signals):
                resolved = [i for i in resolved if i != "initiate_payment"]

        # initiate_payment should NOT fire alongside get_payment_status
        if "initiate_payment" in resolved and "get_payment_status" in resolved:
            pay_signals = ["send", "transfer", "pay to", "initiate", "make payment", "fund transfer"]
            if not any(sig in query_lower for sig in pay_signals):
                resolved = [i for i in resolved if i != "initiate_payment"]

        # cancel_payment needs explicit cancel signal — NOT on general payment queries
        if "cancel_payment" in resolved:
            cancel_signals = ["cancel", "stop", "abort", "revoke", "halt"]
            if not any(sig in query_lower for sig in cancel_signals):
                resolved = [i for i in resolved if i != "cancel_payment"]

        # retry_payment needs explicit retry signal
        if "retry_payment" in resolved:
            retry_signals = ["retry", "resend", "redo", "re-initiate", "again", "re-attempt", "re-send"]
            if not any(sig in query_lower for sig in retry_signals):
                resolved = [i for i in resolved if i != "retry_payment"]

        # acknowledge_payment: must have explicit acknowledge/confirm signal
        if "acknowledge_payment" in resolved:
            ack_signals = ["acknowledge", "ack", "confirm receipt", "mark received", "confirm payment received"]
            if not any(sig in query_lower for sig in ack_signals):
                resolved = [i for i in resolved if i != "acknowledge_payment"]

        # get_payment_receipt: must have receipt/proof signal
        if "get_payment_receipt" in resolved:
            receipt_signals = ["receipt", "proof", "acknowledgement receipt", "download receipt"]
            if not any(sig in query_lower for sig in receipt_signals):
                resolved = [i for i in resolved if i != "get_payment_receipt"]

        # validate_beneficiary should NOT fire alongside initiate_payment unless explicit
        if "validate_beneficiary" in resolved and "initiate_payment" in resolved:
            validate_signals = ["validate", "verify", "check account", "validate beneficiary"]
            if not any(sig in query_lower for sig in validate_signals):
                resolved = [i for i in resolved if i != "validate_beneficiary"]

        # validate_payment_file needs explicit file/validate signal
        if "validate_payment_file" in resolved:
            file_signals = ["validate file", "check file", "verify file", "payment file", "bulk file"]
            if not any(sig in query_lower for sig in file_signals):
                resolved = [i for i in resolved if i != "validate_payment_file"]

        # upload_bulk_payment vs initiate_payment: bulk needs explicit bulk signal
        if "upload_bulk_payment" in resolved and "initiate_payment" in resolved:
            bulk_signals = ["bulk", "batch", "multiple", "upload", "csv", "many vendors"]
            if not any(sig in query_lower for sig in bulk_signals):
                resolved = [i for i in resolved if i != "upload_bulk_payment"]
            else:
                resolved = [i for i in resolved if i != "initiate_payment"]

        # ═══════════════════════════════════════════════════════════════
        # GROUP 2: TRANSACTION CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # get_transaction_history is the primary — suppress noise when it fires
        if "get_transaction_history" in resolved:
            # suppress search_transactions unless explicitly searching
            search_signals = ["search", "find", "look up", "filter", "lookup"]
            if not any(sig in query_lower for sig in search_signals):
                resolved = [i for i in resolved if i != "search_transactions"]
            # suppress get_transaction_details unless txn id or explicit detail request
            detail_signals = ["txn id", "transaction id", "ref no", "utr", "detail of", "info of"]
            if not any(sig in query_lower for sig in detail_signals):
                resolved = [i for i in resolved if i != "get_transaction_details"]
            # suppress download_transaction_report unless explicitly downloading
            dl_signals = ["download", "export", "report", "excel", "pdf"]
            if not any(sig in query_lower for sig in dl_signals):
                resolved = [i for i in resolved if i != "download_transaction_report"]

        # search_transactions standalone: needs explicit search signal
        if "search_transactions" in resolved and "get_transaction_history" not in resolved:
            search_signals = ["search", "find", "look up", "filter", "lookup"]
            if not any(sig in query_lower for sig in search_signals):
                resolved = [i for i in resolved if i != "search_transactions"]

        # get_transaction_details standalone: needs txn id or explicit detail signal
        if "get_transaction_details" in resolved and "get_transaction_history" not in resolved:
            detail_signals = ["txn id", "transaction id", "ref no", "utr",
                              "detail of", "info of", "breakdown of", "specific transaction",
                              "txnref", "txn ref", "details for txn", "details of txn"]
            # also fire if query contains a TXN-like reference token
            import re as _re2
            has_txn_ref = bool(_re2.search(r'(txn|ref|tran)\w*\d+', query_lower))
            if not any(sig in query_lower for sig in detail_signals) and not has_txn_ref:
                resolved = [i for i in resolved if i != "get_transaction_details"]

        # get_pending_transactions: needs explicit pending signal
        if "get_pending_transactions" in resolved:
            pending_signals = ["pending", "in progress", "processing", "queue", "outstanding", "in-process"]
            if not any(sig in query_lower for sig in pending_signals):
                resolved = [i for i in resolved if i != "get_pending_transactions"]

        # download_transaction_report vs fetch_bank_statement: need explicit signals
        if "download_transaction_report" in resolved and "fetch_bank_statement" in resolved:
            txn_report_signals = ["transaction report", "txn report", "payment report"]
            statement_signals  = ["bank statement", "account statement", "statement"]
            has_txn    = any(sig in query_lower for sig in txn_report_signals)
            has_stmt   = any(sig in query_lower for sig in statement_signals)
            if has_stmt and not has_txn:
                resolved = [i for i in resolved if i != "download_transaction_report"]
            elif has_txn and not has_stmt:
                resolved = [i for i in resolved if i != "fetch_bank_statement"]

        # fetch_bank_statement vs download_bank_statement: view vs download
        if "fetch_bank_statement" in resolved and "download_bank_statement" in resolved:
            dl_signals = ["download", "export", "pdf", "excel", "save"]
            if any(sig in query_lower for sig in dl_signals):
                resolved = [i for i in resolved if i != "fetch_bank_statement"]
            else:
                resolved = [i for i in resolved if i != "download_bank_statement"]

        # ═══════════════════════════════════════════════════════════════
        # GROUP 3: ACCOUNT CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # get_account_summary fires → suppress get_account_balance unless balance is explicit
        if "get_account_summary" in resolved and "get_account_balance" in resolved:
            if "balance" not in query_lower and "available" not in query_lower:
                resolved = [i for i in resolved if i != "get_account_balance"]

        # get_account_balance fires → suppress get_account_details unless details explicit
        if "get_account_balance" in resolved and "get_account_details" in resolved:
            # "balance" keyword → user wants balance only, suppress details
            if "balance" in query_lower or "available" in query_lower or "current balance" in query_lower:
                resolved = [i for i in resolved if i != "get_account_details"]
            # "details/ifsc/branch" keyword → user wants details, suppress balance
            elif any(sig in query_lower for sig in ("details", "ifsc", "branch", "account info", "holder name")):
                resolved = [i for i in resolved if i != "get_account_balance"]
            # default → suppress details (balance is more common intent)
            else:
                resolved = [i for i in resolved if i != "get_account_details"]

        # get_linked_accounts: needs explicit "linked" or "connected" signal
        if "get_linked_accounts" in resolved:
            linked_signals = ["linked", "connected", "all accounts", "multiple accounts", "how many accounts"]
            if not any(sig in query_lower for sig in linked_signals):
                resolved = [i for i in resolved if i != "get_linked_accounts"]

        # set_default_account: needs explicit set/change/default signal
        if "set_default_account" in resolved:
            set_signals = ["set default", "change default", "make default", "primary account", "set primary"]
            if not any(sig in query_lower for sig in set_signals):
                resolved = [i for i in resolved if i != "set_default_account"]

        # get_authorized_signatories: needs explicit signatory signal
        if "get_authorized_signatories" in resolved:
            sig_signals = ["signator", "who can sign", "signing authority", "authorized person"]
            if not any(sig in query_lower for sig in sig_signals):
                resolved = [i for i in resolved if i != "get_authorized_signatories"]

        # ═══════════════════════════════════════════════════════════════
        # GROUP 4: COMPLIANCE PAYMENT CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # Pay vs Fetch dues — if only paying, suppress fetch unless dues explicitly mentioned
        if "pay_gst" in resolved and "fetch_gst_dues" in resolved:
            # "show/view/check/fetch dues" → only fetch, suppress pay
            if any(w in query_lower for w in ("show", "view", "check", "fetch", "get", "display", "what", "how much", "pending")):
                resolved = [i for i in resolved if i != "pay_gst"]
            # explicit pay verb → keep pay, suppress fetch
            elif any(w in query_lower for w in ("pay ", "paying", "payment", "submit", "remit")):
                resolved = [i for i in resolved if i != "fetch_gst_dues"]
            # no fetch signal and no dues → suppress fetch
            elif "dues" not in query_lower:
                resolved = [i for i in resolved if i != "fetch_gst_dues"]

        if "pay_epf" in resolved and "fetch_epf_dues" in resolved:
            if any(w in query_lower for w in ("show", "view", "check", "fetch", "get", "display", "what", "how much", "pending")):
                resolved = [i for i in resolved if i != "pay_epf"]
            elif any(w in query_lower for w in ("pay ", "paying", "payment", "submit", "remit")):
                resolved = [i for i in resolved if i != "fetch_epf_dues"]
            elif "dues" not in query_lower:
                resolved = [i for i in resolved if i != "fetch_epf_dues"]

        if "pay_esic" in resolved and "fetch_esic_dues" in resolved:
            if any(w in query_lower for w in ("show", "view", "check", "fetch", "get", "display", "what", "how much", "pending")):
                resolved = [i for i in resolved if i != "pay_esic"]
            elif any(w in query_lower for w in ("pay ", "paying", "payment", "submit", "remit")):
                resolved = [i for i in resolved if i != "fetch_esic_dues"]
            elif "dues" not in query_lower:
                resolved = [i for i in resolved if i != "fetch_esic_dues"]

        if "pay_direct_tax" in resolved and "fetch_tax_dues" in resolved:
            if any(w in query_lower for w in ("show", "view", "check", "fetch", "get", "display", "what", "how much", "pending", "liability", "outstanding")):
                resolved = [i for i in resolved if i != "pay_direct_tax"]
            elif any(w in query_lower for w in ("pay ", "paying", "payment", "submit", "remit")):
                resolved = [i for i in resolved if i != "fetch_tax_dues"]
            elif "dues" not in query_lower:
                resolved = [i for i in resolved if i != "fetch_tax_dues"]

        if "pay_insurance_premium" in resolved and "fetch_insurance_dues" in resolved:
            if any(w in query_lower for w in ("show", "view", "check", "fetch", "get", "display", "what", "how much", "pending", "liability", "outstanding")):
                resolved = [i for i in resolved if i != "pay_insurance_premium"]
            elif any(w in query_lower for w in ("pay ", "paying", "payment", "submit", "remit")):
                resolved = [i for i in resolved if i != "fetch_insurance_dues"]
            elif "dues" not in query_lower:
                resolved = [i for i in resolved if i != "fetch_insurance_dues"]

        # Pay vs History — paying should not trigger history unless explicit
        if "pay_gst" in resolved and "get_gst_payment_history" in resolved:
            hist_signals = ["history", "past", "previous", "records", "last month"]
            if not any(sig in query_lower for sig in hist_signals):
                resolved = [i for i in resolved if i != "get_gst_payment_history"]

        if "pay_epf" in resolved and "get_epf_payment_history" in resolved:
            hist_signals = ["history", "past", "previous", "records", "last month"]
            pay_signals  = ["pay ", "paying", "payment of"]
            if any(sig in query_lower for sig in hist_signals):
                resolved = [i for i in resolved if i != "pay_epf"]
            elif not any(sig in query_lower for sig in pay_signals):
                resolved = [i for i in resolved if i != "get_epf_payment_history"]

        if "pay_esic" in resolved and "get_esic_payment_history" in resolved:
            hist_signals = ["history", "past", "previous", "records", "last month"]
            pay_signals  = ["pay ", "paying", "payment of"]
            if any(sig in query_lower for sig in hist_signals):
                # history keyword → user wants history, suppress pay
                resolved = [i for i in resolved if i != "pay_esic"]
            elif not any(sig in query_lower for sig in pay_signals):
                resolved = [i for i in resolved if i != "get_esic_payment_history"]

        if "pay_direct_tax" in resolved and "get_tax_payment_history" in resolved:
            hist_signals = ["history", "past", "previous", "records", "last month"]
            if not any(sig in query_lower for sig in hist_signals):
                resolved = [i for i in resolved if i != "get_tax_payment_history"]

        if "pay_insurance_premium" in resolved and "get_insurance_payment_history" in resolved:
            hist_signals = ["history", "past", "previous", "records", "last month"]
            if not any(sig in query_lower for sig in hist_signals):
                resolved = [i for i in resolved if i != "get_insurance_payment_history"]

        if "process_payroll" in resolved and "get_payroll_history" in resolved:
            hist_signals = ["history", "past", "previous", "records", "last month"]
            if not any(sig in query_lower for sig in hist_signals):
                resolved = [i for i in resolved if i != "get_payroll_history"]

        # pay_custom_duty vs track_custom_duty_payment: pay vs track
        if "pay_custom_duty" in resolved and "track_custom_duty_payment" in resolved:
            track_signals = ["track", "status", "check", "where is", "tracing"]
            pay_signals   = ["pay ", "paying", "payment of", "remit", "submit duty"]
            has_track = any(sig in query_lower for sig in track_signals)
            has_pay   = any(sig in query_lower for sig in pay_signals)
            if has_track and not has_pay:
                resolved = [i for i in resolved if i != "pay_custom_duty"]
            elif has_pay and not has_track:
                resolved = [i for i in resolved if i != "track_custom_duty_payment"]

        # pay_custom_duty standalone: needs explicit pay signal (not just "custom duty")
        if "pay_custom_duty" in resolved and "track_custom_duty_payment" not in resolved:
            pay_signals = ["pay ", "paying", "payment of", "remit", "submit", "clear duty"]
            if not any(sig in query_lower for sig in pay_signals):
                resolved = [i for i in resolved if i != "pay_custom_duty"]

        # get_payment_status should NOT fire on custom duty tracking queries
        if "get_payment_status" in resolved and "track_custom_duty_payment" in resolved:
            resolved = [i for i in resolved if i != "get_payment_status"]

        # pay_gst vs create_gst_challan: challan is creation, pay is action
        if "pay_gst" in resolved and "create_gst_challan" in resolved:
            challan_signals = ["challan", "create challan", "generate challan"]
            if not any(sig in query_lower for sig in challan_signals):
                resolved = [i for i in resolved if i != "create_gst_challan"]

        # pay_bulk_tax vs pay_direct_tax: bulk needs explicit bulk signal
        if "pay_bulk_tax" in resolved and "pay_direct_tax" in resolved:
            bulk_signals = ["bulk", "multiple", "batch", "all tax", "all tds"]
            if any(sig in query_lower for sig in bulk_signals):
                resolved = [i for i in resolved if i != "pay_direct_tax"]
            else:
                resolved = [i for i in resolved if i != "pay_bulk_tax"]

        # process_payroll vs fetch_payroll_summary: processing vs viewing
        if "process_payroll" in resolved and "fetch_payroll_summary" in resolved:
            process_signals = ["process", "run", "disburse", "execute", "pay salary", "pay employee"]
            summary_signals = ["summary", "view", "show", "check", "how much", "total"]
            has_process = any(sig in query_lower for sig in process_signals)
            has_summary = any(sig in query_lower for sig in summary_signals)
            if has_process and not has_summary:
                resolved = [i for i in resolved if i != "fetch_payroll_summary"]
            elif has_summary and not has_process:
                resolved = [i for i in resolved if i != "process_payroll"]

        # ═══════════════════════════════════════════════════════════════
        # GROUP 5: DUES CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # get_upcoming_dues fires → suppress specific fetch_*_dues unless those are explicit
        if "get_upcoming_dues" in resolved:
            # Only suppress specific dues if user is asking for all dues, not a specific one
            specific_gst  = ["gst dues", "gst payable", "gst pending"]
            specific_epf  = ["epf dues", "pf dues", "epf payable"]
            specific_esic = ["esic dues", "esi dues", "esic payable"]
            specific_tax  = ["tax dues", "tds dues", "tax payable"]
            specific_ins  = ["insurance dues", "premium due", "policy due"]
            if not any(sig in query_lower for sig in specific_gst):
                resolved = [i for i in resolved if i != "fetch_gst_dues"]
            if not any(sig in query_lower for sig in specific_epf):
                resolved = [i for i in resolved if i != "fetch_epf_dues"]
            if not any(sig in query_lower for sig in specific_esic):
                resolved = [i for i in resolved if i != "fetch_esic_dues"]
            if not any(sig in query_lower for sig in specific_tax):
                resolved = [i for i in resolved if i != "fetch_tax_dues"]
            if not any(sig in query_lower for sig in specific_ins):
                resolved = [i for i in resolved if i != "fetch_insurance_dues"]

        # get_overdue_payments vs get_upcoming_dues: overdue vs upcoming
        if "get_overdue_payments" in resolved and "get_upcoming_dues" in resolved:
            overdue_signals = ["overdue", "missed", "late", "past due", "already due"]
            upcoming_signals = ["upcoming", "next", "coming", "future", "this month"]
            has_overdue  = any(sig in query_lower for sig in overdue_signals)
            has_upcoming = any(sig in query_lower for sig in upcoming_signals)
            if has_overdue and not has_upcoming:
                resolved = [i for i in resolved if i != "get_upcoming_dues"]
            elif has_upcoming and not has_overdue:
                resolved = [i for i in resolved if i != "get_overdue_payments"]

        # set_payment_reminder vs get_reminder_list: setting vs viewing
        if "set_payment_reminder" in resolved and "get_reminder_list" in resolved:
            set_signals  = ["set", "add", "create", "remind me", "schedule"]
            list_signals = ["show", "list", "view", "all reminders", "my reminders"]
            has_set  = any(sig in query_lower for sig in set_signals)
            has_list = any(sig in query_lower for sig in list_signals)
            if has_set and not has_list:
                resolved = [i for i in resolved if i != "get_reminder_list"]
            elif has_list and not has_set:
                resolved = [i for i in resolved if i != "set_payment_reminder"]

        # delete_reminder vs get_reminder_list: deleting vs viewing
        if "delete_reminder" in resolved and "get_reminder_list" in resolved:
            del_signals = ["delete", "remove", "cancel reminder", "clear reminder"]
            if any(sig in query_lower for sig in del_signals):
                resolved = [i for i in resolved if i != "get_reminder_list"]

        # ═══════════════════════════════════════════════════════════════
        # GROUP 6: DASHBOARD / ANALYTICS CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # get_dashboard_summary fires → suppress overlapping analytics unless explicit
        if "get_dashboard_summary" in resolved:
            cashflow_signals = ["cashflow", "cash flow", "inflow", "outflow", "net cash"]
            spending_signals = ["spending", "expense", "where am i spending", "category wise"]
            vendor_signals   = ["vendor payment", "vendor wise", "top vendor"]
            monthly_signals  = ["monthly report", "month report", "report for", "month end"]
            if not any(sig in query_lower for sig in cashflow_signals):
                resolved = [i for i in resolved if i != "get_cashflow_summary"]
            if not any(sig in query_lower for sig in spending_signals):
                resolved = [i for i in resolved if i != "get_spending_analytics"]
            if not any(sig in query_lower for sig in vendor_signals):
                resolved = [i for i in resolved if i != "get_vendor_payment_summary"]
            if not any(sig in query_lower for sig in monthly_signals):
                resolved = [i for i in resolved if i != "get_monthly_report"]
            # Also suppress get_upcoming_dues / get_overdue_payments unless explicit
            dues_signals = ["dues", "overdue", "upcoming payment", "what is due"]
            if not any(sig in query_lower for sig in dues_signals):
                resolved = [i for i in resolved if i not in ("get_upcoming_dues", "get_overdue_payments")]

        # standalone cashflow / spending — suppress if keyword not explicitly present
        if "get_cashflow_summary" in resolved and "get_dashboard_summary" not in resolved:
            if not any(w in query_lower for w in ("cashflow", "cash flow", "inflow", "outflow", "net cash")):
                resolved = [i for i in resolved if i != "get_cashflow_summary"]

        if "get_spending_analytics" in resolved and "get_dashboard_summary" not in resolved:
            if not any(w in query_lower for w in ("spending", "expense", "category wise", "where am i spending")):
                resolved = [i for i in resolved if i != "get_spending_analytics"]

        # get_monthly_report should only fire with explicit monthly/report signal
        if "get_monthly_report" in resolved:
            monthly_signals = ["monthly", "month report", "month end", "report for", "monthly summary"]
            if not any(sig in query_lower for sig in monthly_signals):
                resolved = [i for i in resolved if i != "get_monthly_report"]

        # ═══════════════════════════════════════════════════════════════
        # GROUP 7: COMPANY CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # get_company_profile vs update_company_details: viewing vs editing
        if "get_company_profile" in resolved and "update_company_details" in resolved:
            update_signals = ["update", "change", "edit", "modify", "new address", "new email"]
            view_signals   = ["show", "view", "profile", "details", "info"]
            has_update = any(sig in query_lower for sig in update_signals)
            has_view   = any(sig in query_lower for sig in view_signals)
            if has_update and not has_view:
                resolved = [i for i in resolved if i != "get_company_profile"]
            elif has_view and not has_update:
                resolved = [i for i in resolved if i != "update_company_details"]

        # update_company_details: must have explicit update/change/edit signal
        if "update_company_details" in resolved:
            update_signals = ["update", "change", "edit", "modify", "correct", "new ", "revise"]
            if not any(sig in query_lower for sig in update_signals):
                resolved = [i for i in resolved if i != "update_company_details"]

        # get_gst_profile vs validate_gstin: profile vs validation
        if "get_gst_profile" in resolved and "validate_gstin" in resolved:
            validate_signals = ["validate", "verify", "check gstin", "is valid", "gstin valid"]
            if not any(sig in query_lower for sig in validate_signals):
                resolved = [i for i in resolved if i != "validate_gstin"]

        # get_gst_profile vs calculate_gst: profile vs calculation
        if "get_gst_profile" in resolved and "calculate_gst" in resolved:
            calc_signals = ["calculate", "compute", "how much gst", "gst on", "add gst"]
            if not any(sig in query_lower for sig in calc_signals):
                resolved = [i for i in resolved if i != "calculate_gst"]

        # manage_user_roles: must have explicit role/permission signal
        if "manage_user_roles" in resolved:
            role_signals = ["role", "permission", "access", "maker", "checker", "assign", "user management"]
            if not any(sig in query_lower for sig in role_signals):
                resolved = [i for i in resolved if i != "manage_user_roles"]

        # ═══════════════════════════════════════════════════════════════
        # GROUP 8: B2B CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # send_invoice vs get_received_invoices: sending vs viewing received
        if "send_invoice" in resolved and "get_received_invoices" in resolved:
            send_signals     = ["send", "create", "raise", "generate invoice", "issue"]
            received_signals = ["received", "incoming", "bills from", "vendor invoice"]
            has_send     = any(sig in query_lower for sig in send_signals)
            has_received = any(sig in query_lower for sig in received_signals)
            if has_send and not has_received:
                resolved = [i for i in resolved if i != "get_received_invoices"]
            elif has_received and not has_send:
                resolved = [i for i in resolved if i != "send_invoice"]

        # create_purchase_order vs send_invoice: PO vs invoice
        if "create_purchase_order" in resolved and "send_invoice" in resolved:
            po_signals      = ["purchase order", "po", "order for vendor", "buy"]
            invoice_signals = ["invoice", "bill to client", "raise bill"]
            has_po      = any(sig in query_lower for sig in po_signals)
            has_invoice = any(sig in query_lower for sig in invoice_signals)
            if has_po and not has_invoice:
                resolved = [i for i in resolved if i != "send_invoice"]
            elif has_invoice and not has_po:
                resolved = [i for i in resolved if i != "create_purchase_order"]

        # create_cd_note: must have explicit credit/debit note signal
        if "create_cd_note" in resolved:
            cdn_signals = ["credit note", "debit note", "cd note", "adjustment note", "cn", "dn"]
            if not any(sig in query_lower for sig in cdn_signals):
                resolved = [i for i in resolved if i != "create_cd_note"]

        # create_proforma_invoice vs send_invoice: proforma vs regular
        if "create_proforma_invoice" in resolved and "send_invoice" in resolved:
            proforma_signals = ["proforma", "pro forma", "advance invoice", "pre-sale", "estimate"]
            if not any(sig in query_lower for sig in proforma_signals):
                resolved = [i for i in resolved if i != "create_proforma_invoice"]

        # onboard_business_partner vs company_guide: action vs guide/info
        if "onboard_business_partner" in resolved and "company_guide" in resolved:
            guide_signals  = ["how to", "how do i", "explain", "guide", "process",
                              "steps", "what is", "tell me", "describe", "procedure"]
            if any(sig in query_lower for sig in guide_signals):
                resolved = [i for i in resolved if i != "onboard_business_partner"]

        # onboard_business_partner vs vendor_guide: action vs guide
        if "onboard_business_partner" in resolved and "vendor_guide" in resolved:
            # "how do I / how to / steps / guide" → user wants guidance, not to actually onboard
            guide_signals   = ["how do i", "how to", "guide", "process", "steps", "what is",
                               "explain", "tell me about", "procedure"]
            onboard_signals = ["onboard abc", "add vendor", "register vendor", "new partner ABC",
                               "onboard company", "add partner"]  # specific action signals
            has_guide   = any(sig in query_lower for sig in guide_signals)
            if has_guide:
                resolved = [i for i in resolved if i != "onboard_business_partner"]

        # ═══════════════════════════════════════════════════════════════
        # GROUP 9: GST CALCULATOR CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # compare_rates: must have explicit compare signal
        if "compare_rates" in resolved:
            compare_signals = ["compare", "comparison", "vs", "versus",
                               "rate table", "which rate", "different rates",
                               "all rates", "rate comparison", "compare rates", "compare gst"]
            if not any(sig in query_lower for sig in compare_signals):
                resolved = [i for i in resolved if i != "compare_rates"]

        # When compare_rates + calculate_gst both fire: keep both only if calc is explicit
        if "compare_rates" in resolved and "calculate_gst" in resolved:
            calc_signals = ["calculate gst", "compute gst", "find gst", "add gst",
                            "how much gst", "and calculate", "also calculate", "gst on"]
            if not any(sig in query_lower for sig in calc_signals):
                resolved = [i for i in resolved if i != "calculate_gst"]

        # gst_breakdown + calculate_gst: suppress calculate if only breakdown requested
        if "gst_breakdown" in resolved and "calculate_gst" in resolved:
            calc_signals = ["calculate gst", "compute gst", "find gst", "add gst",
                            "how much gst", "and calculate", "also calculate"]
            if not any(sig in query_lower for sig in calc_signals):
                resolved = [i for i in resolved if i != "calculate_gst"]

        # reverse_gst vs calculate_gst: reverse verb always wins — suppress calculate_gst
        if "reverse_gst" in resolved and "calculate_gst" in resolved:
            # "reverse" keyword unambiguously indicates reverse calculation — always suppress forward calc
            reverse_signals = ["reverse", "remove gst", "exclude gst", "without gst",
                               "before gst", "base price from", "inclusive", "reverse calculate"]
            if any(sig in query_lower for sig in reverse_signals):
                resolved = [i for i in resolved if i != "calculate_gst"]
            else:
                resolved = [i for i in resolved if i != "reverse_gst"]

        # gst_breakdown: needs explicit breakdown/component signal (not just a rate+amount)
        if "gst_breakdown" in resolved:
            breakdown_signals = ["breakdown", "break down", "split", "components",
                                 "cgst sgst igst", "how is gst split", "intra state",
                                 "inter state", "component wise", "gst split", "along with breakdown"]
            if not any(sig in query_lower for sig in breakdown_signals):
                resolved = [i for i in resolved if i != "gst_breakdown"]

        # validate_gstin: must have explicit GSTIN or validate signal
        if "validate_gstin" in resolved:
            gstin_signals = ["validate gstin", "verify gstin", "check gstin",
                             "gstin valid", "gstin check", "is gstin"]
            # also allow if a GSTIN pattern is present in original query
            gstin_pattern = r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}"
            import re as _re
            has_gstin_pattern = bool(_re.search(gstin_pattern, query_lower.upper()))
            if not any(sig in query_lower for sig in gstin_signals) and not has_gstin_pattern:
                resolved = [i for i in resolved if i != "validate_gstin"]

        # pay_gst vs calculate_gst: paying GST ≠ calculating it
        if "pay_gst" in resolved and "calculate_gst" in resolved:
            calc_signals = ["calculate", "compute", "how much gst", "gst on", "add gst", "find gst"]
            if not any(sig in query_lower for sig in calc_signals):
                resolved = [i for i in resolved if i != "calculate_gst"]

        # ═══════════════════════════════════════════════════════════════
        # GROUP 10: ONBOARDING / GUIDE CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # company_guide sub-intents: only fire when explicitly asked
        if "company_guide" in resolved:
            doc_signals     = ["document", "checklist", "what documents", "required documents", "papers"]
            process_signals = ["how long", "timeline", "processing time", "approval",
                                "how many days", "days to", "faq", "frequently asked",
                                "common questions", "onboarding faq", "questions about onboarding"]
            field_signals   = ["pan format", "din", "cin", "field", "what is cin", "what is din"]
            if not any(sig in query_lower for sig in doc_signals):
                resolved = [i for i in resolved if i != "company_documents"]
            if not any(sig in query_lower for sig in process_signals):
                resolved = [i for i in resolved if i != "company_process"]
            if not any(sig in query_lower for sig in field_signals):
                resolved = [i for i in resolved if i != "company_field"]

        # company_process standalone: needs explicit FAQ/process signal
        if "company_process" in resolved and "company_guide" not in resolved:
            process_standalone = ["faq", "frequently asked", "onboarding faq", "common question",
                                   "how long", "timeline", "approval time", "processing time",
                                   "how many days", "steps for", "onboarding questions",
                                   "common questions"]
            if not any(sig in query_lower for sig in process_standalone):
                resolved = [i for i in resolved if i != "company_process"]

        # bank_guide vs vendor_guide: bank vs vendor context
        if "bank_guide" in resolved and "vendor_guide" in resolved:
            bank_signals   = ["bank account", "bank onboarding", "add bank", "open account"]
            vendor_signals = ["vendor", "supplier", "partner onboard"]
            has_bank   = any(sig in query_lower for sig in bank_signals)
            has_vendor = any(sig in query_lower for sig in vendor_signals)
            if has_bank and not has_vendor:
                resolved = [i for i in resolved if i != "vendor_guide"]
            elif has_vendor and not has_bank:
                resolved = [i for i in resolved if i != "bank_guide"]

        # onboard_business_partner vs bank_guide: action vs guide
        if "onboard_business_partner" in resolved and "bank_guide" in resolved:
            guide_signals = ["guide", "how to", "process", "steps", "what is", "explain"]
            if not any(sig in query_lower for sig in guide_signals):
                resolved = [i for i in resolved if i != "bank_guide"]

        # bank_guide must have explicit onboarding/registration signal
        # "show bank statement" / "bank account balance" should NOT trigger bank_guide
        if "bank_guide" in resolved:
            bank_guide_signals = [
                "bank onboarding", "register bank", "bank registration", "add bank account",
                "supported banks", "connect bank", "how to add bank", "onboard bank",
                "link bank", "new bank account", "bank account onboarding"
            ]
            if not any(sig in query_lower for sig in bank_guide_signals):
                resolved = [i for i in resolved if i != "bank_guide"]

        # vendor_guide must have explicit vendor/supplier onboarding signal
        if "vendor_guide" in resolved:
            vendor_guide_signals = [
                "vendor onboarding", "add vendor", "register vendor", "supplier onboarding",
                "how to add vendor", "vendor registration", "onboard supplier", "create vendor",
                "new vendor", "vendor guide", "onboard a vendor", "onboard new vendor",
                "how do i onboard a vendor", "vendor setup", "add a vendor", "vendor process",
                "how to onboard vendor", "vendor setup guide", "onboard a new vendor"
            ]
            if not any(sig in query_lower for sig in vendor_guide_signals):
                resolved = [i for i in resolved if i != "vendor_guide"]

        # company_guide must have explicit company onboarding signal
        # "company profile" / "company details" should NOT trigger company_guide
        if "company_guide" in resolved:
            company_guide_signals = [
                "company onboarding", "register company", "company registration",
                "how to onboard", "start company", "onboard organization",
                "company setup", "register my company", "set up company",
                "company register", "registration process",
                "onboard my company", "how do i onboard my", "explain the company onboarding",
                "company onboarding process", "onboard my organization"
            ]
            if not any(sig in query_lower for sig in company_guide_signals):
                resolved = [i for i in resolved if i != "company_guide"]

        # ═══════════════════════════════════════════════════════════════
        # GROUP 11: SUPPORT CONFLICTS
        # ═══════════════════════════════════════════════════════════════

        # raise_support_ticket vs get_ticket_history: creating vs viewing
        if "raise_support_ticket" in resolved and "get_ticket_history" in resolved:
            raise_signals = ["raise", "create", "open", "new ticket", "report issue", "file complaint"]
            hist_signals  = ["history", "past", "my tickets", "all tickets", "previous"]
            has_raise = any(sig in query_lower for sig in raise_signals)
            has_hist  = any(sig in query_lower for sig in hist_signals)
            if has_raise and not has_hist:
                resolved = [i for i in resolved if i != "get_ticket_history"]
            elif has_hist and not has_raise:
                resolved = [i for i in resolved if i != "raise_support_ticket"]

        # chat_with_support vs raise_support_ticket: chat vs ticket
        if "chat_with_support" in resolved and "raise_support_ticket" in resolved:
            chat_signals   = ["chat", "talk to", "speak with", "live support", "agent"]
            ticket_signals = ["ticket", "complaint", "report issue", "log issue"]
            has_chat   = any(sig in query_lower for sig in chat_signals)
            has_ticket = any(sig in query_lower for sig in ticket_signals)
            if has_chat and not has_ticket:
                resolved = [i for i in resolved if i != "raise_support_ticket"]
            elif has_ticket and not has_chat:
                resolved = [i for i in resolved if i != "chat_with_support"]

        # get_contact_details: must have explicit contact/helpline signal
        if "get_contact_details" in resolved:
            contact_signals = ["contact", "helpline", "phone number", "customer care", "reach", "call"]
            if not any(sig in query_lower for sig in contact_signals):
                resolved = [i for i in resolved if i != "get_contact_details"]

        return resolved

    # ========================
    # ENTITY EXTRACTION
    # ========================

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract banking entities from query."""
        entities = {}
        cleaned_query = query

        # GSTIN
        gstin_match = re.search(self.entity_patterns["gstin"], query)
        if gstin_match:
            entities["gstin"] = gstin_match.group(0)
            cleaned_query = re.sub(self.entity_patterns["gstin"], "", cleaned_query, flags=re.IGNORECASE)

        # PAN
        pan_match = re.search(self.entity_patterns["pan"], query)
        if pan_match:
            entities["pan"] = pan_match.group(0)
            cleaned_query = re.sub(self.entity_patterns["pan"], "", cleaned_query, flags=re.IGNORECASE)

        # IFSC
        ifsc_match = re.search(self.entity_patterns["ifsc"], query)
        if ifsc_match:
            entities["ifsc_code"] = ifsc_match.group(0)

        # Dates
        date_matches = re.findall(self.entity_patterns["date"], query)
        if date_matches:
            entities["from_date"] = date_matches[0]
            if len(date_matches) > 1:
                entities["to_date"] = date_matches[1]

        # Month
        month_match = re.search(self.entity_patterns["month"], query)
        if month_match:
            entities["month"] = month_match.group(0)

        # Percentages
        percent_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", cleaned_query)
        if percent_matches:
            entities["gst_rates"] = [float(p) for p in percent_matches]
            entities["gst_rate"]  = float(percent_matches[0])
            for m in percent_matches:
                cleaned_query = cleaned_query.replace(f"{m}%", "").replace(f"{m} percent", "")

        # Amounts
        amount_matches = re.findall(r"(?:₹|rs\.?|inr|rupees?)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)", cleaned_query)
        if amount_matches:
            amounts = [float(a.replace(",", "")) for a in amount_matches if float(a.replace(",", "")) > 0]
            if amounts:
                entities["amounts"]      = amounts
                entities["amount"]       = amounts[0]
                entities["base_amount"]  = amounts[0]
            if len(amounts) > 1:
                entities["total_amount"] = amounts[1]

        # Transaction / record limit — e.g. "show 10 transactions", "last 25 records"
        limit_match = re.search(
            r"\b(?:show|last|recent|top|first|get|fetch)?\s*(\d{1,3})\s*(?:transactions?|txns?|records?|entries|payments?)\b",
            query, re.IGNORECASE
        )
        if not limit_match:
            # plain "show 10" — number directly before/after keyword
            limit_match = re.search(r"\b(\d{1,3})\s+transactions?\b", query, re.IGNORECASE)
        if limit_match:
            entities["limit"] = int(limit_match.group(1))

        # Account number (long digit string not already matched)
        acct_match = re.search(r"\b(\d{9,18})\b", cleaned_query)
        if acct_match and "account_number" not in entities:
            entities["account_number"] = acct_match.group(1)

        # Payment mode
        for mode in ["NEFT", "RTGS", "IMPS", "UPI"]:
            if mode.lower() in query.lower():
                entities["payment_mode"] = mode
                break

        # Transaction ID
        txn_match = re.search(r"\b(TXN\w+)\b", query, re.IGNORECASE)
        if txn_match:
            entities["transaction_id"] = txn_match.group(1)

        # Intra / Inter state
        if "inter" in query.lower() or "interstate" in query.lower():
            entities["is_intra_state"] = False
        elif "intra" in query.lower() or "intrastate" in query.lower():
            entities["is_intra_state"] = True

        return entities

    # ========================
    # FULL PIPELINE
    # ========================

    def process_query(self, user_message: str) -> Dict[str, Any]:
        """Detect intents, extract entities, and build tool calls."""
        detected_intents = self.predict_intents(user_message)
        entities         = self.extract_entities(user_message)

        logger.info(f"Detected intents : {detected_intents}")
        logger.info(f"Extracted entities: {entities}")

        tool_calls = []

        amount          = entities.get("amount") or entities.get("base_amount")
        gst_rates       = entities.get("gst_rates", [18.0])
        gstin           = entities.get("gstin")
        pan             = entities.get("pan")
        account_number  = entities.get("account_number", "")
        transaction_id  = entities.get("transaction_id", "")
        month           = entities.get("month", "")
        from_date       = entities.get("from_date", "")
        to_date         = entities.get("to_date", "")
        payment_mode    = entities.get("payment_mode", "NEFT")

        # ── CORE PAYMENT ──────────────────────────────────────────────
        if "initiate_payment" in detected_intents:
            tool_calls.append({"tool_name": "initiate_payment", "parameters": {
                "beneficiary_id": entities.get("beneficiary_id", ""),
                "amount":         amount or 0,
                "payment_mode":   payment_mode,
            }})

        if "get_payment_status" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "get_payment_status", "parameters": {"transaction_id": transaction_id}})

        if "cancel_payment" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "cancel_payment", "parameters": {"transaction_id": transaction_id}})

        if "retry_payment" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "retry_payment", "parameters": {"transaction_id": transaction_id}})

        if "get_payment_receipt" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "get_payment_receipt", "parameters": {"transaction_id": transaction_id}})

        if "validate_beneficiary" in detected_intents:
            tool_calls.append({"tool_name": "validate_beneficiary", "parameters": {
                "account_number": account_number,
                "ifsc_code":      entities.get("ifsc_code", ""),
            }})

        # ── UPLOAD PAYMENT ────────────────────────────────────────────
        if "upload_bulk_payment" in detected_intents:
            tool_calls.append({"tool_name": "upload_bulk_payment", "parameters": {
                "file_name":   entities.get("file_name", ""),
                "file_base64": "",
                "file_format": "CSV",
            }})

        if "validate_payment_file" in detected_intents:
            tool_calls.append({"tool_name": "validate_payment_file", "parameters": {
                "upload_id": entities.get("upload_id", ""),
            }})

        # ── B2B ───────────────────────────────────────────────────────
        if "onboard_business_partner" in detected_intents:
            tool_calls.append({"tool_name": "onboard_business_partner", "parameters": {
                "company_name":  entities.get("company_name", ""),
                "gstin":         gstin or "",
                "pan":           pan or "",
                "contact_email": entities.get("contact_email", ""),
                "contact_phone": entities.get("contact_phone", ""),
            }})

        if "send_invoice" in detected_intents:
            tool_calls.append({"tool_name": "send_invoice", "parameters": {
                "partner_id":     entities.get("partner_id", ""),
                "invoice_number": entities.get("invoice_number", ""),
                "invoice_date":   from_date,
                "due_date":       to_date,
                "amount":         amount or 0,
            }})

        if "get_received_invoices" in detected_intents:
            tool_calls.append({"tool_name": "get_received_invoices", "parameters": {"status": "ALL"}})

        if "acknowledge_payment" in detected_intents:
            tool_calls.append({"tool_name": "acknowledge_payment", "parameters": {
                "invoice_id":     entities.get("invoice_id", ""),
                "transaction_id": transaction_id,
            }})

        if "create_proforma_invoice" in detected_intents:
            tool_calls.append({"tool_name": "create_proforma_invoice", "parameters": {
                "partner_id":    entities.get("partner_id", ""),
                "validity_date": to_date,
                "amount":        amount or 0,
                "description":   "",
            }})

        if "create_cd_note" in detected_intents:
            tool_calls.append({"tool_name": "create_cd_note", "parameters": {
                "partner_id":          entities.get("partner_id", ""),
                "note_type":           "CREDIT",
                "original_invoice_id": entities.get("invoice_id", ""),
                "amount":              amount or 0,
                "reason":              "",
            }})

        if "create_purchase_order" in detected_intents:
            tool_calls.append({"tool_name": "create_purchase_order", "parameters": {
                "partner_id":    entities.get("partner_id", ""),
                "po_date":       from_date,
                "delivery_date": to_date,
                "amount":        amount or 0,
                "description":   "",
            }})

        # ── INSURANCE ─────────────────────────────────────────────────
        if "fetch_insurance_dues" in detected_intents:
            tool_calls.append({"tool_name": "fetch_insurance_dues", "parameters": {}})

        if "pay_insurance_premium" in detected_intents:
            tool_calls.append({"tool_name": "pay_insurance_premium", "parameters": {
                "policy_number": entities.get("policy_number", ""),
                "amount":        amount or 0,
            }})

        if "get_insurance_payment_history" in detected_intents:
            tool_calls.append({"tool_name": "get_insurance_payment_history", "parameters": {}})

        # ── BANK STATEMENT ────────────────────────────────────────────
        if "fetch_bank_statement" in detected_intents:
            tool_calls.append({"tool_name": "fetch_bank_statement", "parameters": {
                "account_number": account_number,
                "from_date":      from_date,
                "to_date":        to_date,
            }})

        if "download_bank_statement" in detected_intents:
            tool_calls.append({"tool_name": "download_bank_statement", "parameters": {
                "account_number": account_number,
                "from_date":      from_date,
                "to_date":        to_date,
                "format":         "PDF",
            }})

        if "get_account_balance" in detected_intents:
            tool_calls.append({"tool_name": "get_account_balance", "parameters": {
                "account_number": account_number,
            }})

        if "get_transaction_history" in detected_intents:
            tool_calls.append({"tool_name": "get_transaction_history", "parameters": {
                "account_number": account_number,
                "from_date":      from_date,
                "to_date":        to_date,
                "limit":          entities.get("limit", 10),
            }})

        # ── CUSTOM / SEZ ──────────────────────────────────────────────
        if "pay_custom_duty" in detected_intents:
            tool_calls.append({"tool_name": "pay_custom_duty", "parameters": {
                "bill_of_entry_number": entities.get("bill_of_entry_number", ""),
                "amount":               amount or 0,
                "port_code":            entities.get("port_code", ""),
                "importer_code":        entities.get("importer_code", ""),
            }})

        if "track_custom_duty_payment" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "track_custom_duty_payment", "parameters": {"transaction_id": transaction_id}})

        if "get_custom_duty_history" in detected_intents:
            tool_calls.append({"tool_name": "get_custom_duty_history", "parameters": {}})

        # ── GST ───────────────────────────────────────────────────────
        if "fetch_gst_dues" in detected_intents:
            tool_calls.append({"tool_name": "fetch_gst_dues", "parameters": {"gstin": gstin or ""}})

        if "pay_gst" in detected_intents:
            tool_calls.append({"tool_name": "pay_gst", "parameters": {
                "gstin":          gstin,
                "challan_number": entities.get("challan_number", ""),
                "amount":         amount or 0,
                "tax_type":       entities.get("tax_type", "CGST"),
            }})

        if "create_gst_challan" in detected_intents:
            tool_calls.append({"tool_name": "create_gst_challan", "parameters": {
                "gstin":         gstin,
                "return_period": entities.get("return_period", month.replace("-", "") if month else ""),
            }})

        if "get_gst_payment_history" in detected_intents:
            tool_calls.append({"tool_name": "get_gst_payment_history", "parameters": {"gstin": gstin}})

        # ── ESIC ──────────────────────────────────────────────────────
        if "fetch_esic_dues" in detected_intents:
            tool_calls.append({"tool_name": "fetch_esic_dues", "parameters": {
                "establishment_code": entities.get("establishment_code", ""),
                "month":              month,
            }})

        if "pay_esic" in detected_intents:
            tool_calls.append({"tool_name": "pay_esic", "parameters": {
                "establishment_code": entities.get("establishment_code", ""),
                "month":              month,
                "amount":             amount or 0,
            }})

        if "get_esic_payment_history" in detected_intents:
            tool_calls.append({"tool_name": "get_esic_payment_history", "parameters": {
                "establishment_code": entities.get("establishment_code", ""),
            }})

        # ── EPF ───────────────────────────────────────────────────────
        if "fetch_epf_dues" in detected_intents:
            tool_calls.append({"tool_name": "fetch_epf_dues", "parameters": {
                "establishment_id": entities.get("establishment_id", ""),
                "month":            month,
            }})

        if "pay_epf" in detected_intents:
            tool_calls.append({"tool_name": "pay_epf", "parameters": {
                "establishment_id": entities.get("establishment_id", ""),
                "month":            month,
                "amount":           amount or 0,
            }})

        if "get_epf_payment_history" in detected_intents:
            tool_calls.append({"tool_name": "get_epf_payment_history", "parameters": {
                "establishment_id": entities.get("establishment_id", ""),
            }})

        # ── PAYROLL ───────────────────────────────────────────────────
        if "fetch_payroll_summary" in detected_intents:
            tool_calls.append({"tool_name": "fetch_payroll_summary", "parameters": {"month": month}})

        if "process_payroll" in detected_intents:
            tool_calls.append({"tool_name": "process_payroll", "parameters": {
                "month":          month,
                "account_number": account_number,
                "approved_by":    entities.get("approved_by", ""),
            }})

        if "get_payroll_history" in detected_intents:
            tool_calls.append({"tool_name": "get_payroll_history", "parameters": {}})

        # ── TAXES ─────────────────────────────────────────────────────
        if "fetch_tax_dues" in detected_intents and pan:
            tool_calls.append({"tool_name": "fetch_tax_dues", "parameters": {"pan": pan}})

        if "pay_direct_tax" in detected_intents and pan:
            tool_calls.append({"tool_name": "pay_direct_tax", "parameters": {
                "pan":             pan,
                "tax_type":        entities.get("tax_type", "TDS"),
                "assessment_year": entities.get("assessment_year", "2026-27"),
                "amount":          amount or 0,
                "challan_type":    entities.get("challan_type", "281"),
            }})

        if "pay_state_tax" in detected_intents:
            tool_calls.append({"tool_name": "pay_state_tax", "parameters": {
                "state":             entities.get("state", ""),
                "tax_category":      entities.get("tax_category", "Professional Tax"),
                "amount":            amount or 0,
                "assessment_period": entities.get("assessment_period", ""),
            }})

        if "pay_bulk_tax" in detected_intents:
            tool_calls.append({"tool_name": "pay_bulk_tax", "parameters": {
                "file_name":   entities.get("file_name", ""),
                "file_base64": "",
                "tax_type":    entities.get("tax_type", "TDS"),
            }})

        if "get_tax_payment_history" in detected_intents and pan:
            tool_calls.append({"tool_name": "get_tax_payment_history", "parameters": {"pan": pan}})

        # ── ACCOUNT MANAGEMENT ────────────────────────────────────────
        if "get_account_summary" in detected_intents:
            tool_calls.append({"tool_name": "get_account_summary", "parameters": {}})

        if "get_account_details" in detected_intents:
            tool_calls.append({"tool_name": "get_account_details", "parameters": {"account_number": account_number}})

        if "get_linked_accounts" in detected_intents:
            tool_calls.append({"tool_name": "get_linked_accounts", "parameters": {}})

        if "set_default_account" in detected_intents:
            tool_calls.append({"tool_name": "set_default_account", "parameters": {"account_number": account_number}})

        # ── TRANSACTION & HISTORY ─────────────────────────────────────
        # search_transactions: only fire when user explicitly searches (not just "show transactions")
        if "search_transactions" in detected_intents:
            search_keywords = ["search", "find", "look up", "filter", "lookup"]
            query_l = user_message.lower()
            if any(kw in query_l for kw in search_keywords):
                params = {"from_date": from_date, "to_date": to_date}
                if amount:
                    params["amount"] = amount
                if account_number:
                    params["beneficiary_id"] = account_number
                tool_calls.append({"tool_name": "search_transactions", "parameters": params})

        # get_transaction_details: only fire when a specific transaction_id is present
        if "get_transaction_details" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "get_transaction_details", "parameters": {"transaction_id": transaction_id}})

        if "download_transaction_report" in detected_intents:
            tool_calls.append({"tool_name": "download_transaction_report", "parameters": {
                "from_date": from_date, "to_date": to_date, "format": "XLSX",
            }})

        if "get_pending_transactions" in detected_intents:
            tool_calls.append({"tool_name": "get_pending_transactions", "parameters": {}})

        # ── DUES & REMINDERS ──────────────────────────────────────────
        if "get_upcoming_dues" in detected_intents:
            tool_calls.append({"tool_name": "get_upcoming_dues", "parameters": {"days_ahead": 30}})

        if "get_overdue_payments" in detected_intents:
            tool_calls.append({"tool_name": "get_overdue_payments", "parameters": {}})

        if "set_payment_reminder" in detected_intents:
            tool_calls.append({"tool_name": "set_payment_reminder", "parameters": {
                "title":    entities.get("reminder_title", ""),
                "due_date": to_date or from_date,
            }})

        if "get_reminder_list" in detected_intents:
            tool_calls.append({"tool_name": "get_reminder_list", "parameters": {}})

        if "delete_reminder" in detected_intents:
            tool_calls.append({"tool_name": "delete_reminder", "parameters": {
                "reminder_id": entities.get("reminder_id", ""),
            }})

        # ── DASHBOARD & ANALYTICS ─────────────────────────────────────
        if "get_dashboard_summary" in detected_intents:
            tool_calls.append({"tool_name": "get_dashboard_summary", "parameters": {}})

        if "get_spending_analytics" in detected_intents:
            tool_calls.append({"tool_name": "get_spending_analytics", "parameters": {
                "from_date": from_date, "to_date": to_date,
            }})

        if "get_cashflow_summary" in detected_intents:
            tool_calls.append({"tool_name": "get_cashflow_summary", "parameters": {"month": month}})

        if "get_monthly_report" in detected_intents:
            tool_calls.append({"tool_name": "get_monthly_report", "parameters": {"month": month}})

        if "get_vendor_payment_summary" in detected_intents:
            tool_calls.append({"tool_name": "get_vendor_payment_summary", "parameters": {}})

        # ── COMPANY MANAGEMENT ────────────────────────────────────────
        if "get_company_profile" in detected_intents:
            tool_calls.append({"tool_name": "get_company_profile", "parameters": {}})

        if "update_company_details" in detected_intents:
            tool_calls.append({"tool_name": "update_company_details", "parameters": {
                "field": entities.get("field", ""),
                "value": entities.get("value", ""),
            }})

        if "get_gst_profile" in detected_intents:
            tool_calls.append({"tool_name": "get_gst_profile", "parameters": {}})

        if "get_authorized_signatories" in detected_intents:
            tool_calls.append({"tool_name": "get_authorized_signatories", "parameters": {}})

        if "manage_user_roles" in detected_intents:
            tool_calls.append({"tool_name": "manage_user_roles", "parameters": {
                "user_id": entities.get("user_id", ""),
                "role":    entities.get("role", "VIEWER"),
                "action":  entities.get("action", "ASSIGN"),
            }})

        # ── SUPPORT ───────────────────────────────────────────────────
        if "raise_support_ticket" in detected_intents:
            tool_calls.append({"tool_name": "raise_support_ticket", "parameters": {
                "category":    entities.get("ticket_category", "OTHER"),
                "subject":     entities.get("subject", ""),
                "description": user_message,
            }})

        if "get_ticket_history" in detected_intents:
            tool_calls.append({"tool_name": "get_ticket_history", "parameters": {"status": "ALL"}})

        if "chat_with_support" in detected_intents:
            tool_calls.append({"tool_name": "chat_with_support", "parameters": {
                "issue_summary": user_message[:200],
            }})

        if "get_contact_details" in detected_intents:
            tool_calls.append({"tool_name": "get_contact_details", "parameters": {"category": "GENERAL"}})

        # ── GST CALCULATOR (→ gst_client_manager) ─────────────────────
        if "calculate_gst" in detected_intents and amount:
            # When compare_rates also detected, emit exactly 1 calculate call (primary rate only)
            if "compare_rates" in detected_intents:
                tool_calls.append({"tool_name": "calculate_gst", "parameters": {
                    "base_amount": amount,
                    "gst_rate":    gst_rates[0],
                }})
            else:
                for rate in gst_rates:
                    tool_calls.append({"tool_name": "calculate_gst", "parameters": {
                        "base_amount": amount,
                        "gst_rate":    rate,
                    }})

        if "reverse_gst" in detected_intents:
            amt = entities.get("total_amount") or amount
            if amt:
                tool_calls.append({"tool_name": "reverse_calculate_gst", "parameters": {
                    "total_amount": amt,
                    "gst_rate":     gst_rates[0],
                }})

        if "gst_breakdown" in detected_intents and amount:
            params = {"base_amount": amount, "gst_rate": gst_rates[0]}
            if "is_intra_state" in entities:
                params["is_intra_state"] = entities["is_intra_state"]
            tool_calls.append({"tool_name": "gst_breakdown", "parameters": params})

        if "compare_rates" in detected_intents and amount:
            rates_to_compare = gst_rates if len(gst_rates) > 1 else [5, 12, 18, 28]
            tool_calls.append({"tool_name": "compare_gst_rates", "parameters": {
                "base_amount": amount,
                "rates":       rates_to_compare,
            }})

        if "validate_gstin" in detected_intents and gstin:
            tool_calls.append({"tool_name": "validate_gstin", "parameters": {"gstin": gstin}})

        # ── ONBOARDING INFO (→ info_client_manager) ───────────────────
        if "company_guide" in detected_intents:
            tool_calls.append({"tool_name": "get_company_onboarding_guide", "parameters": {}})

        if "company_documents" in detected_intents:
            tool_calls.append({"tool_name": "get_company_required_documents", "parameters": {}})

        if "company_field" in detected_intents:
            tool_calls.append({"tool_name": "get_validation_formats", "parameters": {}})

        if "company_process" in detected_intents:
            tool_calls.append({"tool_name": "get_onboarding_faq", "parameters": {}})

        if "bank_guide" in detected_intents:
            tool_calls.append({"tool_name": "get_bank_onboarding_guide", "parameters": {}})

        if "vendor_guide" in detected_intents:
            tool_calls.append({"tool_name": "get_vendor_onboarding_guide", "parameters": {}})

        return {
            "intents_detected": detected_intents,
            "tool_calls":       tool_calls,
            "entities":         entities,
            "is_multi_intent":  len(detected_intents) > 1,
            "total_tools":      len(tool_calls),
        }

    # ========================
    # SAVE / LOAD
    # ========================

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "mlb":        self.mlb,
            "version":    "3.0.0"
        }
        filepath = os.path.join(self.model_path, "production_classifier.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"✓ Model saved to {filepath}")

    def load_model(self):
        filepath = os.path.join(self.model_path, "production_classifier.pkl")
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        self.vectorizer = model_data["vectorizer"]
        self.classifier = model_data["classifier"]
        self.mlb        = model_data["mlb"]
        logger.info(f"✓ Model loaded (v{model_data.get('version', '1.0.0')})")


# Global instance
intent_classifier = ProductionIntentClassifier()