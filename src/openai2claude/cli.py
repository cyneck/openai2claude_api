#!/usr/bin/env python3
"""CLI entry point for openai2claude."""

import click
import uvicorn

from .config import Config, load_config
from .proxy import create_app, setup_logging


@click.command()
@click.option(
    "--backend",
    "-b",
    envvar="OPENAI2CLAUDE_BACKEND",
    default=Config.BACKEND_URL,
    help="OpenAI-compatible backend URL"
)
@click.option(
    "--api-key",
    "-k",
    envvar="OPENAI2CLAUDE_API_KEY",
    default=None,
    help="API key for the backend"
)
@click.option(
    "--host",
    "-h",
    envvar="OPENAI2CLAUDE_HOST",
    default=Config.HOST,
    help="Host to bind the proxy server"
)
@click.option(
    "--port",
    "-p",
    envvar="OPENAI2CLAUDE_PORT",
    type=int,
    default=Config.PORT,
    help="Port to bind the proxy server"
)
@click.option(
    "--log-level",
    "-l",
    envvar="OPENAI2CLAUDE_LOG_LEVEL",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default=Config.LOG_LEVEL.lower(),
    help="Logging level"
)
@click.option(
    "--timeout",
    "-t",
    envvar="OPENAI2CLAUDE_TIMEOUT",
    type=float,
    default=Config.TIMEOUT,
    help="Request timeout in seconds"
)
@click.option(
    "--no-verify-ssl",
    is_flag=True,
    default=False,
    help="Disable SSL certificate verification"
)
@click.version_option(version="0.1.0")
def main(
    backend: str,
    api_key: str | None,
    host: str,
    port: int,
    log_level: str,
    timeout: float,
    no_verify_ssl: bool
):
    """Proxy server that converts Claude API requests to OpenAI-compatible format.

    Configure via environment variables or command-line arguments:

        OPENAI2CLAUDE_BACKEND     Backend URL (default: http://localhost:8080/v1)
        OPENAI2CLAUDE_API_KEY     API key for backend
        OPENAI2CLAUDE_HOST        Proxy host (default: 0.0.0.0)
        OPENAI2CLAUDE_PORT        Proxy port (default: 8000)
        OPENAI2CLAUDE_LOG_LEVEL   Log level (default: INFO)
        OPENAI2CLAUDE_TIMEOUT     Request timeout (default: 300)
        OPENAI2CLAUDE_VERIFY_SSL  Verify SSL (default: true)

    Example usage:

        openai2claude --backend https://api.openai.com/v1 --api-key sk-xxx

    Then use Claude SDK with:

        ANTHROPIC_API_BASE=http://localhost:8000
    """
    # Setup logging first
    setup_logging(log_level)

    # Build config
    config = Config()
    config.BACKEND_URL = backend
    if api_key:
        config.API_KEY = api_key
    config.HOST = host
    config.PORT = port
    config.LOG_LEVEL = log_level.upper()
    config.TIMEOUT = timeout
    config.VERIFY_SSL = not no_verify_ssl

    click.echo(f"Starting openai2claude proxy")
    click.echo(f"  Backend:   {config.BACKEND_URL}")
    click.echo(f"  Listen:    {config.HOST}:{config.PORT}")
    click.echo(f"  Log level: {config.LOG_LEVEL}")

    # Create app and run
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
