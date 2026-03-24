"""
config/config.py — Bank AI Assistant
Configuration Management.
Local ML model — NO external LLM API keys required.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings for ML-based Bank AI Assistant."""

    # ── ML Model ──────────────────────────────────────────────
    ml_model_path: str = "models/"
    llm_provider:  str = "local_ml"

    # ── Bank MCP Server ───────────────────────────────────────
    bank_api_key: str = Field(default="", alias="BANK_API_KEY")

    # ── GST API (optional) ────────────────────────────────────
    gst_api_url: Optional[str] = None
    gst_api_key: Optional[str] = None

    # ── Server ────────────────────────────────────────────────
    host:  str  = "0.0.0.0"
    port:  int  = 8000
    debug: bool = True

    # ── Logging ───────────────────────────────────────────────
    log_level: str = "INFO"

    # ── PostgreSQL (agent persistence) ────────────────────────
    postgres_enabled: bool         = True
    db_host:          str          = "localhost"
    db_port:          int          = 5432
    db_name:          str          = "bankdb"
    db_user:          str          = "postgres"
    db_password:      str          = ""

    # ── Redis (optional distributed session memory) ────────────
    use_redis:      bool = False
    redis_host:     str  = "localhost"
    redis_port:     int  = 6379
    redis_db:       int  = 0
    redis_password: str  = ""

    # ── Agent ─────────────────────────────────────────────────
    agent_memory_ttl_minutes: int = 60

    # ── API / CORS ────────────────────────────────────────────
    # Comma-separated origins or * for all.
    # Production: set to your frontend domain e.g. https://app.yourbank.com
    allowed_origins:       str = "*"
    request_timeout_secs:  int = 30

    # ── Security ──────────────────────────────────────────────
    enable_audit_log:    bool = True
    data_retention_days: int  = 90

    # ── Legacy (kept for backward compatibility) ───────────────
    database_url: Optional[str] = None

    model_config = {
        "env_file":         ".env",
        "case_sensitive":   False,
        "extra":            "ignore",
        "populate_by_name": True,
    }


settings = Settings()