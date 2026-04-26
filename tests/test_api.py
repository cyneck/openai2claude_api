"""Tests for API endpoints."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from openai2claude.config import Config
from openai2claude.proxy import create_app


@pytest.fixture
def app():
    """Create app with test config."""
    config = Config()
    config.BACKEND_URL = "http://test-backend/v1"
    config.API_KEY = "test-key"
    return create_app(config)


@pytest.fixture
async def client(app):
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    async def test_health_get(self, client):
        """GET / returns health status."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "backend" in data

    async def test_health_head(self, client):
        """HEAD / returns 200."""
        response = await client.head("/")
        assert response.status_code == 200


class TestProxyEndpoint:
    """Tests for /v1/messages proxy endpoint."""

    async def test_proxy_non_streaming(self, client, app):
        """Test non-streaming proxy."""
        # Mock the upstream response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": None
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }

        # Mock the HTTP client
        with patch.object(app.state, 'http_client') as mock_client:
            mock_client.post.return_value = mock_response

            response = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert len(data["content"]) == 1
            assert data["content"][0]["text"] == "Hello!"

    async def test_proxy_with_tools(self, client, app):
        """Test proxy with tool calls."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-456",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "Bash",
                                "arguments": '{"command": "ls"}'
                            }
                        }
                    ]
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }

        with patch.object(app.state, 'http_client') as mock_client:
            mock_client.post.return_value = mock_response

            response = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "List files"}],
                    "tools": [
                        {
                            "name": "Bash",
                            "description": "Run bash",
                            "input_schema": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}}
                            }
                        }
                    ]
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["content"]) == 1
            assert data["content"][0]["type"] == "tool_use"
            assert data["content"][0]["name"] == "Bash"

    async def test_proxy_invalid_json(self, client):
        """Test proxy with invalid JSON."""
        response = await client.post(
            "/v1/messages",
            content="not json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 400

    async def test_proxy_upstream_error(self, client, app):
        """Test handling of upstream errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal error"}

        with patch.object(app.state, 'http_client') as mock_client:
            mock_client.post.return_value = mock_response

            response = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            )

            # Should return the error from upstream
            assert response.status_code == 500

    async def test_proxy_system_prompt_injection(self, client, app):
        """Test that system prompt gets tool instruction injected."""
        captured_payload = None

        async def capture_post(*args, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "chatcmpl-789",
                "model": "gpt-4",
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}
            }
            return mock_resp

        with patch.object(app.state, 'http_client') as mock_client:
            mock_client.post.side_effect = capture_post

            response = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            )

            assert response.status_code == 200
            # System prompt should be injected
            assert captured_payload is not None
            messages = captured_payload["messages"]
            system_msg = messages[0]
            assert system_msg["role"] == "system"
            assert "Bash" in system_msg["content"]
            assert "command" in system_msg["content"]


class TestConfigEnvVars:
    """Tests for environment variable configuration."""

    def test_config_defaults(self):
        """Test config has correct defaults."""
        import os
        # Clear any env vars
        env_vars = [
            "OPENAI2CLAUDE_BACKEND",
            "OPENAI2CLAUDE_API_KEY",
            "OPENAI2CLAUDE_HOST",
            "OPENAI2CLAUDE_PORT",
            "OPENAI2CLAUDE_LOG_LEVEL",
            "OPENAI2CLAUDE_TIMEOUT",
            "OPENAI2CLAUDE_VERIFY_SSL",
        ]
        saved = {k: os.environ.get(k) for k in env_vars}
        try:
            for k in env_vars:
                if k in os.environ:
                    del os.environ[k]

            config = Config()
            assert config.BACKEND_URL == "http://localhost:8080/v1"
            assert config.HOST == "0.0.0.0"
            assert config.PORT == 8000
            assert config.LOG_LEVEL == "INFO"
            assert config.TIMEOUT == 300.0
            assert config.VERIFY_SSL is True
        finally:
            # Restore env vars
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
