import os
from typing import Optional


class Config:
    """Configuration loaded from environment variables."""

    # Backend OpenAI-compatible API
    BACKEND_URL: str = os.getenv("OPENAI2CLAUDE_BACKEND", "http://localhost:8080/v1")
    API_KEY: str = os.getenv("OPENAI2CLAUDE_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxx")

    # Local proxy server
    HOST: str = os.getenv("OPENAI2CLAUDE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("OPENAI2CLAUDE_PORT", "8000"))

    # Logging
    LOG_LEVEL: str = os.getenv("OPENAI2CLAUDE_LOG_LEVEL", "INFO")

    # Request settings
    TIMEOUT: float = float(os.getenv("OPENAI2CLAUDE_TIMEOUT", "300.0"))
    VERIFY_SSL: bool = os.getenv("OPENAI2CLAUDE_VERIFY_SSL", "true").lower() != "false"


def load_config() -> Config:
    """Load and return configuration instance."""
    return Config()
