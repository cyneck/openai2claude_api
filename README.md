# openai2claude

Proxy server that converts Claude API requests to OpenAI-compatible backends. This allows you to use the Anthropic Claude SDK with any OpenAI-compatible API provider.

## Installation

```bash
pip install openai2claude
```

## Quick Start

1. **Start the proxy server:**

```bash
export OPENAI2CLAUDE_BACKEND=https://api.openai.com/v1
export OPENAI2CLAUDE_API_KEY=sk-your-api-key
openai2claude
```

Or with command-line arguments:

```bash
openai2claude --backend https://api.openai.com/v1 --api-key sk-your-api-key
```

2. **Configure your Claude SDK code to use the proxy:**

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8000"
)

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

Or via environment variable:

```bash
export ANTHROPIC_API_BASE="http://localhost:8000"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI2CLAUDE_BACKEND` | `http://localhost:8080/v1` | OpenAI-compatible backend URL |
| `OPENAI2CLAUDE_API_KEY` | `sk-xxx...` | API key for the backend |
| `OPENAI2CLAUDE_HOST` | `0.0.0.0` | Proxy server host |
| `OPENAI2CLAUDE_PORT` | `8000` | Proxy server port |
| `OPENAI2CLAUDE_LOG_LEVEL` | `INFO` | Logging level |
| `OPENAI2CLAUDE_TIMEOUT` | `300` | Request timeout in seconds |
| `OPENAI2CLAUDE_VERIFY_SSL` | `true` | Verify SSL certificates |

### Command-Line Options

```
openai2claude [OPTIONS]

Options:
  -b, --backend TEXT         OpenAI-compatible backend URL
  -k, --api-key TEXT         API key for the backend
  -h, --host TEXT            Host to bind the proxy server
  -p, --port INTEGER         Port to bind the proxy server
  -l, --log-level TEXT       Logging level (debug, info, warning, error)
  -t, --timeout FLOAT        Request timeout in seconds
  --no-verify-ssl            Disable SSL certificate verification
  --version                  Show version
```

## Architecture

```
┌─────────────┐    ┌─────────────────┐    ┌────────────────────┐
│  Your Code  │───>│  openai2claude  │───>│  OpenAI Backend    │
│ Claude SDK  │    │    (localhost   │    │  (api.openai.com,  │
│             │<───│     :8000)      │<───│  ollama, etc.)     │
└─────────────┘    └─────────────────┘    └────────────────────┘
```

The proxy:
1. Accepts requests in Claude API format (`/v1/messages`)
2. Converts them to OpenAI API format (`/v1/chat/completions`)
3. Forwards to the configured backend
4. Converts responses back to Claude format

## Features

- **Format Conversion**: Automatically converts between Claude and OpenAI API formats
- **Tool Calling Fix**: Fixes malformed tool calls (e.g., `cmd`/`input` -> `command`) for Bash tools
- **Streaming Support**: Full SSE streaming support for real-time responses
- **System Prompt Injection**: Adds tool usage instructions to prevent common errors
- **Environment Variables**: Full configuration via environment variables
- **CLI Interface**: Easy-to-use command-line interface with `--help`

## Use Cases

1. **Use Claude SDK with local models**: Point to Ollama, LM Studio, or other OpenAI-compatible local servers
2. **Multi-provider support**: Switch between OpenAI, Azure, or other OpenAI-compatible providers
3. **Development and testing**: Test Claude code against mock or local backends

## License

MIT
