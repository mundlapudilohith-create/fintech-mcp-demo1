-- ================================================================
-- Bank AI Backend — Complete Schema (v2)
-- Mobile OTP authentication + user data
-- ================================================================
-- Setup:
--   psql -U postgres -c "CREATE DATABASE bank_ai_test;"
--   psql -U postgres -d bank_ai_test -f migrations/001_schema.sql
-- ================================================================
 
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
 
-- ─────────────────────────────────────────────
-- USERS
-- Identity = mobile number only
-- All sensitive fields stored AES-256-GCM encrypted
-- _hash fields = SHA-256 for indexed search
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mobile          TEXT NOT NULL,          -- encrypted
    mobile_hash     TEXT NOT NULL UNIQUE,   -- SHA-256 for lookup
    name            TEXT,
    email           TEXT,                   -- encrypted
    pan             TEXT,                   -- encrypted
    pan_hash        TEXT,
    aadhaar         TEXT,                   -- encrypted
    kyc_status      TEXT DEFAULT 'PENDING', -- PENDING | VERIFIED | REJECTED
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_mobile_hash ON users(mobile_hash);
 
-- ─────────────────────────────────────────────
-- OTP TABLE
-- One active OTP per mobile at a time
-- Expires in 10 minutes, max 5 attempts
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS otps (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mobile_hash     TEXT NOT NULL,          -- SHA-256 of mobile
    otp_hash        TEXT NOT NULL,          -- SHA-256 of OTP (never store plain OTP)
    purpose         TEXT DEFAULT 'LOGIN',   -- LOGIN | VERIFY_PAN | VERIFY_BANK
    attempts        INTEGER DEFAULT 0,
    max_attempts    INTEGER DEFAULT 5,
    expires_at      TIMESTAMPTZ NOT NULL,
    used            BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_otps_mobile_hash ON otps(mobile_hash);
CREATE INDEX IF NOT EXISTS idx_otps_expires ON otps(expires_at);
 
-- ─────────────────────────────────────────────
-- SESSIONS
-- Created after successful OTP verification
-- TTL = 60 minutes, refreshed on activity
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_token   TEXT NOT NULL UNIQUE,   -- secure random token
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    mobile_hash     TEXT NOT NULL,
    company_id      UUID,                   -- linked company (if any)
    memory_data     JSONB DEFAULT '{}',     -- pre-seeded context for AI agent
    is_active       BOOLEAN DEFAULT TRUE,
    expires_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
 
-- ─────────────────────────────────────────────
-- COMPANIES
-- Linked to user via user_id
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS companies (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    company_name    TEXT NOT NULL,
    gstin           TEXT,                   -- encrypted
    gstin_hash      TEXT,                   -- SHA-256 for lookup
    pan             TEXT,                   -- encrypted
    pan_hash        TEXT,
    cin             TEXT,                   -- encrypted
    company_type    TEXT DEFAULT 'PRIVATE',
    industry        TEXT,
    address         TEXT,
    state           TEXT,
    pincode         TEXT,
    kyc_status      TEXT DEFAULT 'PENDING',
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_companies_user_id    ON companies(user_id);
CREATE INDEX IF NOT EXISTS idx_companies_gstin_hash ON companies(gstin_hash);
 
-- ─────────────────────────────────────────────
-- BANK ACCOUNTS
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bank_accounts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID REFERENCES companies(id) ON DELETE CASCADE,
    user_id         UUID REFERENCES users(id),
    account_number  TEXT NOT NULL,          -- encrypted
    account_hash    TEXT NOT NULL,          -- SHA-256 for lookup
    ifsc_code       TEXT,                   -- encrypted
    bank_name       TEXT,
    branch          TEXT,
    account_type    TEXT DEFAULT 'CURRENT',
    available_balance NUMERIC(18,2) DEFAULT 0,
    total_balance   NUMERIC(18,2) DEFAULT 0,
    currency        TEXT DEFAULT 'INR',
    is_default      BOOLEAN DEFAULT FALSE,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_bank_accounts_company_id ON bank_accounts(company_id);
CREATE INDEX IF NOT EXISTS idx_bank_accounts_hash ON bank_accounts(account_hash);
 
-- ─────────────────────────────────────────────
-- TRANSACTIONS
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transactions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id      TEXT NOT NULL UNIQUE,
    company_id          UUID REFERENCES companies(id),
    user_id             UUID REFERENCES users(id),
    account_number      TEXT,               -- encrypted
    beneficiary_account TEXT,               -- encrypted
    beneficiary_name    TEXT,
    beneficiary_ifsc    TEXT,
    amount              NUMERIC(18,2) NOT NULL,
    currency            TEXT DEFAULT 'INR',
    payment_mode        TEXT,               -- NEFT | RTGS | IMPS | UPI
    txn_type            TEXT DEFAULT 'DEBIT',
    utr_number          TEXT,               -- encrypted
    status              TEXT DEFAULT 'PENDING',
    remarks             TEXT,
    scheduled_date      DATE,
    initiated_at        TIMESTAMPTZ DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_transactions_company_id ON transactions(company_id);
CREATE INDEX IF NOT EXISTS idx_transactions_txn_id     ON transactions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_transactions_status     ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_created    ON transactions(created_at DESC);
 
-- ─────────────────────────────────────────────
-- GST RECORDS
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gst_records (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID REFERENCES companies(id),
    gstin           TEXT,                   -- encrypted
    return_type     TEXT,
    period          TEXT,
    igst            NUMERIC(18,2) DEFAULT 0,
    cgst            NUMERIC(18,2) DEFAULT 0,
    sgst            NUMERIC(18,2) DEFAULT 0,
    cess            NUMERIC(18,2) DEFAULT 0,
    total_amount    NUMERIC(18,2) DEFAULT 0,
    cpin            TEXT,
    status          TEXT DEFAULT 'PENDING',
    due_date        DATE,
    paid_at         TIMESTAMPTZ,
    transaction_id  UUID REFERENCES transactions(id),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gst_company_id ON gst_records(company_id);
 
-- ─────────────────────────────────────────────
-- EPF RECORDS
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS epf_records (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id            UUID REFERENCES companies(id),
    establishment_id      TEXT,
    wage_month            TEXT,
    employee_count        INTEGER DEFAULT 0,
    employer_contribution NUMERIC(18,2) DEFAULT 0,
    employee_contribution NUMERIC(18,2) DEFAULT 0,
    admin_charges         NUMERIC(18,2) DEFAULT 0,
    total_amount          NUMERIC(18,2) DEFAULT 0,
    trrn                  TEXT,
    challan_number        TEXT,
    status                TEXT DEFAULT 'PENDING',
    due_date              DATE,
    paid_at               TIMESTAMPTZ,
    transaction_id        UUID REFERENCES transactions(id),
    created_at            TIMESTAMPTZ DEFAULT NOW()
);
 
-- ─────────────────────────────────────────────
-- ESIC RECORDS
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS esic_records (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id            UUID REFERENCES companies(id),
    establishment_code    TEXT,
    contribution_month    TEXT,
    employee_count        INTEGER DEFAULT 0,
    employer_contribution NUMERIC(18,2) DEFAULT 0,
    employee_contribution NUMERIC(18,2) DEFAULT 0,
    total_amount          NUMERIC(18,2) DEFAULT 0,
    challan_number        TEXT,
    status                TEXT DEFAULT 'PENDING',
    due_date              DATE,
    paid_at               TIMESTAMPTZ,
    transaction_id        UUID REFERENCES transactions(id),
    created_at            TIMESTAMPTZ DEFAULT NOW()
);
 
-- ─────────────────────────────────────────────
-- INVOICES
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS invoices (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID REFERENCES companies(id),
    partner_id      TEXT,
    invoice_number  TEXT NOT NULL,
    invoice_type    TEXT DEFAULT 'OUTBOUND',
    invoice_date    DATE,
    due_date        DATE,
    amount          NUMERIC(18,2),
    gst_amount      NUMERIC(18,2) DEFAULT 0,
    total_amount    NUMERIC(18,2),
    status          TEXT DEFAULT 'PENDING',
    transaction_id  UUID REFERENCES transactions(id),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_invoices_company_id ON invoices(company_id);
 
-- ─────────────────────────────────────────────
-- REMINDERS
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reminders (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reminder_id         TEXT NOT NULL UNIQUE,
    company_id          UUID REFERENCES companies(id),
    user_id             UUID REFERENCES users(id),
    title               TEXT NOT NULL,
    payment_type        TEXT,
    amount              NUMERIC(18,2) DEFAULT 0,
    due_date            DATE NOT NULL,
    notify_days_before  INTEGER DEFAULT 3,
    is_active           BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
 
-- ─────────────────────────────────────────────
-- CONVERSATION HISTORY (AI Agent storage)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_token   TEXT,
    user_id         UUID REFERENCES users(id),
    company_id      UUID,
    role            TEXT NOT NULL,          -- user | assistant
    message         TEXT NOT NULL,
    intents         JSONB DEFAULT '[]',
    tool_calls      JSONB DEFAULT '[]',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_token);
CREATE INDEX IF NOT EXISTS idx_conv_user    ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at DESC);
 
-- ─────────────────────────────────────────────
-- AUDIT LOG
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_logs (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID REFERENCES users(id),
    action      TEXT NOT NULL,
    entity      TEXT,
    entity_id   TEXT,
    ip_address  TEXT,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_user    ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC);
 
-- ─────────────────────────────────────────────
-- AUTO updated_at trigger
-- ─────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;
 
DO $$ BEGIN
  CREATE TRIGGER users_updated_at     BEFORE UPDATE ON users     FOR EACH ROW EXECUTE FUNCTION update_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  CREATE TRIGGER companies_updated_at BEFORE UPDATE ON companies FOR EACH ROW EXECUTE FUNCTION update_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  CREATE TRIGGER accounts_updated_at  BEFORE UPDATE ON bank_accounts FOR EACH ROW EXECUTE FUNCTION update_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
 
SELECT 'Schema v2 created successfully ✓' AS status;