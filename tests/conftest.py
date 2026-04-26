"""Pytest fixtures for openai2claude tests."""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

from openai2claude.config import Config
from openai2claude.proxy import create_app


@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = Config()
    config.BACKEND_URL = "http://test-backend/v1"
    config.API_KEY = "test-key"
    config.HOST = "127.0.0.1"
    config.PORT = 8000
    config.LOG_LEVEL = "DEBUG"
    config.TIMEOUT = 30.0
    config.VERIFY_SSL = True
    return config


@pytest.fixture
def test_app(test_config):
    """Create a test FastAPI app."""
    return create_app(test_config)


@pytest.fixture
async def async_client(test_app):
    """Create an async test client."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client."""
    mock = AsyncMock(spec=AsyncClient)
    return mock
