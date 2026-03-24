"""
Onboarding Information MCP Server
Provides step-by-step guides for Company, Bank, and Vendor onboarding
NO actual registration - information only
"""
from fastmcp import FastMCP
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Onboarding Info Server")



@mcp.tool()
def get_company_onboarding_guide() -> dict:
    """
    Get complete step-by-step guide for company onboarding on Vanghee B2B platform.
    
    Returns:
        Detailed onboarding steps with all required information
    """
    logger.info("Providing company onboarding guide")
    
    return {
        "title": "Company Onboarding Guide - Vanghee B2B Platform",
        "platform": "Vanghee B2B",
        "total_steps": 3,
        "steps": [
            {
                "step_number": 1,
                "title": "Create Business Account",
                "description": "Initial platform registration",
                "actions": [
                    "Visit Vanghee B2B platform",
                    "Click 'Register Company'",
                    "Enter official email address",
                    "Create secure password",
                    "Verify email via OTP"
                ]
            },
            {
                "step_number": 2,
                "title": "Company Basic Information",
                "description": "Enter company details",
                "required_fields": [
                    {"field": "PAN Number", "format": "10 characters (ABCDE1234F)", "required": True},
                    {"field": "Company Name", "format": "Full registered name", "required": True},
                    {"field": "Type of Business", "options": ["Goods", "Service"], "required": True},
                    {"field": "GST Number", "format": "15 characters (29ABCDE1234F1Z5)", "required": True},
                    {"field": "TAN Number", "format": "10 characters (ABCD12345E)", "required": True},
                    {"field": "Mobile Number", "format": "10 digits", "required": True},
                    {"field": "Website URL", "format": "https://example.com", "required": False},
                    {"field": "Year of Establishment", "format": "YYYY", "required": True}
                ]
            },
            {
                "step_number": 3,
                "title": "Company Address Information",
                "description": "Enter registered office address",
                "required_fields": [
                    {"field": "Type of Organization", "options": ["Private Limited", "LLP", "Partnership", "Proprietorship", "Public Limited"], "required": True},
                    {"field": "Line of Business", "format": "e.g., Manufacturing, Trading, Services", "required": True},
                    {"field": "Flat / Door Number", "required": True},
                    {"field": "Premise / Building Name", "required": True},
                    {"field": "Road / Street", "required": True},
                    {"field": "Area / Locality", "required": True},
                    {"field": "State", "format": "Select from dropdown", "required": True},
                    {"field": "District", "format": "Select from dropdown", "required": True},
                    {"field": "Pincode", "format": "6 digits", "required": True}
                ]
            }
        ],
        "completion_message": "✅ Your company onboarding is completed!",
        "next_steps": "After completion, you can proceed with Bank and Vendor onboarding",
        "support_contact": "Contact Vanghee B2B support for assistance"
    }


@mcp.tool()
def get_company_required_documents() -> dict:
    """
    Get list of documents required for company onboarding.
    
    Returns:
        List of required documents and their specifications
    """
    logger.info("Providing company required documents")
    
    return {
        "title": "Required Documents for Company Onboarding",
        "documents": [
            {
                "document": "PAN Card",
                "description": "Company PAN card issued by Income Tax Department",
                "format": "10 characters (e.g., ABCDE1234F)",
                "mandatory": True
            },
            {
                "document": "GST Certificate",
                "description": "GST registration certificate",
                "format": "15 characters (e.g., 29ABCDE1234F1Z5)",
                "mandatory": True
            },
            {
                "document": "TAN Number",
                "description": "Tax Deduction Account Number",
                "format": "10 characters (e.g., ABCD12345E)",
                "mandatory": True
            },
            {
                "document": "Certificate of Incorporation",
                "description": "Company registration certificate",
                "mandatory": True
            },
            {
                "document": "Registered Office Address Proof",
                "description": "Utility bill / Rent agreement",
                "mandatory": True
            }
        ]
    }



# BANK ONBOARDING GUIDES


@mcp.tool()
def get_bank_onboarding_guide() -> dict:
    """
    Get complete step-by-step guide for bank account onboarding on Vanghee B2B platform.
    
    Returns:
        Detailed bank onboarding steps
    """
    logger.info("Providing bank onboarding guide")
    
    return {
        "title": "Bank Account Onboarding Guide - Vanghee B2B Platform",
        "platform": "Vanghee B2B",
        "total_steps": 2,
        "prerequisite": "Complete company onboarding first",
        "steps": [
            {
                "step_number": 1,
                "title": "Add Bank",
                "description": "Select your bank from available options",
                "actions": [
                    "Log in to Vanghee B2B platform",
                    "Navigate to Bank Onboarding section",
                    "Click 'Add Bank Account'",
                    "Select your respective bank from the available bank list"
                ],
                "supported_banks": [
                    "State Bank of India (SBI)",
                    "HDFC Bank",
                    "ICICI Bank",
                    "Axis Bank",
                    "Kotak Mahindra Bank",
                    "Punjab National Bank",
                    "Bank of Baroda",
                    "Canara Bank",
                    "Union Bank of India",
                    "IDBI Bank",
                    "Yes Bank",
                    "IndusInd Bank",
                    "Bank of India",
                    "Federal Bank",
                    "Indian Bank"
                ]
            },
            {
                "step_number": 2,
                "title": "Bank Account Information",
                "description": "Enter your bank account details",
                "required_fields": [
                    {"field": "Customer ID", "format": "Bank-specific format", "required": False, "note": "If applicable"},
                    {"field": "Account Holder Name", "format": "Full name as per bank records", "required": True},
                    {"field": "Bank Account Number", "format": "9-18 digits", "required": True},
                    {"field": "Confirm Bank Account Number", "format": "Must match above", "required": True},
                    {"field": "IFSC Code", "format": "11 characters (e.g., SBIN0001234)", "required": True},
                    {"field": "Registered Email Address", "format": "Valid email", "required": True}
                ]
            }
        ],
        "completion_message": "✅ Your Bank onboarding is completed!",
        "verification_note": "Bank account may undergo penny drop verification",
        "next_steps": "You can now proceed with vendor onboarding or start transactions"
    }


@mcp.tool()
def get_supported_banks() -> dict:
    """
    Get list of all banks supported on Vanghee B2B platform.
    
    Returns:
        List of supported banks with their IFSC prefixes
    """
    logger.info("Providing supported banks list")
    
    return {
        "title": "Supported Banks on Vanghee B2B Platform",
        "total_banks": 15,
        "banks": [
            {"name": "State Bank of India", "short_name": "SBI", "ifsc_prefix": "SBIN"},
            {"name": "HDFC Bank", "short_name": "HDFC", "ifsc_prefix": "HDFC"},
            {"name": "ICICI Bank", "short_name": "ICICI", "ifsc_prefix": "ICIC"},
            {"name": "Axis Bank", "short_name": "Axis", "ifsc_prefix": "UTIB"},
            {"name": "Kotak Mahindra Bank", "short_name": "Kotak", "ifsc_prefix": "KKBK"},
            {"name": "Punjab National Bank", "short_name": "PNB", "ifsc_prefix": "PUNB"},
            {"name": "Bank of Baroda", "short_name": "BOB", "ifsc_prefix": "BARB"},
            {"name": "Canara Bank", "short_name": "Canara", "ifsc_prefix": "CNRB"},
            {"name": "Union Bank of India", "short_name": "Union", "ifsc_prefix": "UBIN"},
            {"name": "IDBI Bank", "short_name": "IDBI", "ifsc_prefix": "IBKL"},
            {"name": "Yes Bank", "short_name": "Yes", "ifsc_prefix": "YESB"},
            {"name": "IndusInd Bank", "short_name": "IndusInd", "ifsc_prefix": "INDB"},
            {"name": "Bank of India", "short_name": "BOI", "ifsc_prefix": "BKID"},
            {"name": "Federal Bank", "short_name": "Federal", "ifsc_prefix": "FDRL"},
            {"name": "Indian Bank", "short_name": "Indian", "ifsc_prefix": "IDIB"}
        ],
        "note": "More banks may be added in the future"
    }



# VENDOR ONBOARDING GUIDES


@mcp.tool()
def get_vendor_onboarding_guide() -> dict:
    """
    Get complete step-by-step guide for vendor registration on Vanghee B2B platform.
    
    Returns:
        Detailed vendor onboarding steps
    """
    logger.info("Providing vendor onboarding guide")
    
    return {
        "title": "Vendor Onboarding Guide - Vanghee B2B Platform",
        "platform": "Vanghee B2B",
        "total_steps": 3,
        "prerequisite": "Complete company onboarding first",
        "steps": [
            {
                "step_number": 1,
                "title": "Start Vendor Registration",
                "description": "Navigate to vendor registration",
                "actions": [
                    "Log in to Vanghee B2B platform",
                    "Navigate to Vendor Onboarding",
                    "Click 'Register Vendor'"
                ]
            },
            {
                "step_number": 2,
                "title": "Vendor Business Details",
                "description": "Enter vendor contact information",
                "required_fields": [
                    {"field": "Vendor Name", "format": "Full legal name or business name", "required": True},
                    {"field": "Mobile Number", "format": "10 digits", "required": True},
                    {"field": "Email", "format": "Valid email address", "required": True}
                ]
            },
            {
                "step_number": 3,
                "title": "Bank Details",
                "description": "Enter vendor bank account information",
                "required_fields": [
                    {"field": "Bank Account Number", "format": "9-18 digits", "required": True},
                    {"field": "Confirm Bank Account Number", "format": "Must match above", "required": True},
                    {"field": "IFSC Code", "format": "11 characters (e.g., SBIN0001234)", "required": True}
                ]
            }
        ],
        "completion_message": "✅ Once approved, the vendor will be activated for transactions",
        "approval_process": "Vendor registration goes through approval process",
        "timeline": "Approval typically takes 1-2 business days",
        "next_steps": "After approval, vendor can start receiving purchase orders"
    }



# VALIDATION FORMAT HELPERS


@mcp.tool()
def get_validation_formats() -> dict:
    """
    Get format specifications for all validation fields (PAN, GST, TAN, IFSC, etc.).
    
    Returns:
        Validation formats and examples for all document types
    """
    logger.info("Providing validation formats")
    
    return {
        "title": "Validation Format Guide",
        "formats": {
            "PAN": {
                "description": "Permanent Account Number",
                "format": "5 letters + 4 digits + 1 letter",
                "pattern": "ABCDE1234F",
                "example": "ABCDE1234F",
                "length": 10,
                "notes": "4th character indicates entity type: C=Company, P=Individual, etc."
            },
            "GST": {
                "description": "Goods and Services Tax Identification Number",
                "format": "2 digits (state code) + 10 char PAN + 1 letter + Z + 1 alphanumeric",
                "pattern": "29ABCDE1234F1Z5",
                "example": "29ABCDE1234F1Z5",
                "length": 15,
                "notes": "First 2 digits are state code (e.g., 29=Karnataka, 27=Maharashtra)"
            },
            "TAN": {
                "description": "Tax Deduction Account Number",
                "format": "4 letters + 5 digits + 1 letter",
                "pattern": "ABCD12345E",
                "example": "MUMB12345E",
                "length": 10,
                "notes": "First 4 letters represent city code"
            },
            "IFSC": {
                "description": "Indian Financial System Code",
                "format": "4 letters (bank code) + 0 + 6 alphanumeric (branch code)",
                "pattern": "SBIN0001234",
                "example": "SBIN0001234",
                "length": 11,
                "notes": "5th character is always 0"
            },
            "Mobile": {
                "description": "Indian Mobile Number",
                "format": "10 digits starting with 6-9",
                "pattern": "9876543210",
                "example": "9876543210",
                "length": 10,
                "notes": "First digit must be 6, 7, 8, or 9"
            },
            "Pincode": {
                "description": "Indian Postal Code",
                "format": "6 digits",
                "pattern": "560001",
                "example": "560001",
                "length": 6,
                "notes": "First digit cannot be 0"
            },
            "BankAccount": {
                "description": "Bank Account Number",
                "format": "9-18 digits",
                "pattern": "123456789012",
                "example": "1234567890123456",
                "min_length": 9,
                "max_length": 18,
                "notes": "Exact length varies by bank"
            }
        }
    }



# FAQ AND TROUBLESHOOTING


@mcp.tool()
def get_onboarding_faq() -> dict:
    """
    Get frequently asked questions about onboarding process.
    
    Returns:
        Common questions and answers about company, bank, and vendor onboarding
    """
    logger.info("Providing onboarding FAQ")
    
    return {
        "title": "Onboarding FAQ - Vanghee B2B Platform",
        "categories": {
            "Company Onboarding": [
                {
                    "q": "Do I need GST registration to onboard?",
                    "a": "Yes, a valid GST number is mandatory for company onboarding."
                },
                {
                    "q": "Can I change company details after onboarding?",
                    "a": "Yes, but changes to PAN and GST require admin approval."
                },
                {
                    "q": "How long does company onboarding take?",
                    "a": "Typically 15-30 minutes if all documents are ready."
                }
            ],
            "Bank Onboarding": [
                {
                    "q": "Can I add multiple bank accounts?",
                    "a": "Yes, you can register multiple bank accounts for different purposes."
                },
                {
                    "q": "What is penny drop verification?",
                    "a": "A small amount (₹1) is deposited to verify account authenticity."
                },
                {
                    "q": "My bank is not in the list. What should I do?",
                    "a": "Contact Vanghee support to request addition of your bank."
                }
            ],
            "Vendor Onboarding": [
                {
                    "q": "How long does vendor approval take?",
                    "a": "Typically 1-2 business days after submission."
                },
                {
                    "q": "Can vendors self-register?",
                    "a": "No, only companies can register vendors on their behalf."
                },
                {
                    "q": "What documents does a vendor need?",
                    "a": "Basic business details, contact info, and bank account information."
                }
            ]
        }
    }


@mcp.tool()
def get_common_errors() -> dict:
    """
    Get common errors during onboarding and their solutions.
    
    Returns:
        List of common errors and troubleshooting steps
    """
    logger.info("Providing common errors guide")
    
    return {
        "title": "Common Onboarding Errors and Solutions",
        "errors": [
            {
                "error": "Invalid PAN format",
                "cause": "PAN doesn't match required format",
                "solution": "Ensure 10 characters: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)"
            },
            {
                "error": "GST number already registered",
                "cause": "This GST is already in use",
                "solution": "Contact support if you believe this is an error"
            },
            {
                "error": "IFSC code not found",
                "cause": "Invalid or incorrect IFSC code",
                "solution": "Verify IFSC from bank statement or passbook. Must be 11 characters."
            },
            {
                "error": "Bank account number mismatch",
                "cause": "Account numbers in two fields don't match",
                "solution": "Carefully re-enter and confirm account number"
            },
            {
                "error": "Mobile number already in use",
                "cause": "This number is registered to another account",
                "solution": "Use a different mobile number or contact support"
            },
            {
                "error": "Email verification failed",
                "cause": "OTP not received or expired",
                "solution": "Check spam folder, request new OTP, ensure correct email"
            }
        ]
    }


if __name__ == "__main__":
    logger.info("Starting Onboarding Info MCP Server...")
    mcp.run()