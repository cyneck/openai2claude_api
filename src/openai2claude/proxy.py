import json
import logging
import sys
from typing import AsyncGenerator
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from httpx import AsyncClient

from .config import Config


logger = logging.getLogger("openai2claude")


def setup_logging(level: str):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def convert_tools(claude_tools: list) -> list:
    """Convert Claude tools format to OpenAI tools format."""
    if not isinstance(claude_tools, list):
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {})
            }
        }
        for tool in claude_tools
    ]


def convert_messages(claude_messages: list) -> list:
    """Convert Claude messages format to OpenAI messages format."""
    openai_messages = []

    for msg in claude_messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, list):
            current_text, tool_calls, tool_results = "", [], []

            for block in content:
                if block.get("type") == "text":
                    current_text += block["text"]
                elif block.get("type") == "tool_use":
                    tool_name = block["name"]
                    tool_input = block.get("input", {})

                    # Fix malformed tool calls that use 'cmd' or 'input' instead of 'command'
                    if "bash" in tool_name.lower() and "command" not in tool_input:
                        tool_input["command"] = tool_input.get("cmd", tool_input.get("input", "echo 'History patched'"))

                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input, ensure_ascii=False)
                        }
                    })
                elif block.get("type") == "tool_result":
                    tool_results.append(block)

            if current_text or tool_calls:
                msg_obj = {"role": role, "content": current_text if current_text else None}
                if tool_calls:
                    msg_obj["tool_calls"] = tool_calls
                openai_messages.append(msg_obj)

            for result in tool_results:
                res_content = result.get("content")
                res_text = "".join(
                    b.get("text", "") for b in res_content if b.get("type") == "text"
                ) if isinstance(res_content, list) else str(res_content)
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_use_id"],
                    "content": res_text
                })
        else:
            openai_messages.append({"role": role, "content": content})

    return openai_messages


def fix_tool_arguments(tool_name: str, args_str: str) -> str:
    """Fix common tool argument issues for Bash-style tools."""
    if not args_str:
        return json.dumps({"command": "echo 'Empty command fixed by proxy'"}, ensure_ascii=False)

    try:
        args_json = json.loads(args_str)
    except json.JSONDecodeError:
        args_json = {}

    tool_name_lower = tool_name.lower()
    if "bash" in tool_name_lower and "command" not in args_json:
        logger.warning(f"Intercepted broken {tool_name} call. Fixing: {args_str}")
        fallback_cmd = args_json.get("cmd", args_json.get("input", ""))
        if not fallback_cmd and args_str and not args_str.strip().startswith("{"):
            fallback_cmd = args_str.strip()
        args_json["command"] = fallback_cmd if fallback_cmd else "echo 'Empty command fixed by proxy'"

    return json.dumps(args_json, ensure_ascii=False)


def create_app(config: Config) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = AsyncClient(
            verify=config.VERIFY_SSL,
            timeout=config.TIMEOUT,
            follow_redirects=True
        )
        yield
        await app.state.http_client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.head("/")
    @app.get("/")
    async def health():
        return {"status": "ok", "backend": config.BACKEND_URL}

    @app.post("/v1/messages")
    async def proxy_handler(request: Request):
        """Handle Claude API /v1/messages endpoint."""
        try:
            claude_body = await request.json()
        except Exception:
            return Response(content="Invalid JSON", status_code=400)

        # Build system prompt with tool usage instructions
        system_prompt = claude_body.get("system", "")
        system_addon = "\n\n[CRITICAL INSTRUCTION: When calling the Bash tool, your arguments MUST be a valid JSON object strictly containing the 'command' string key.]"

        if isinstance(system_prompt, list):
            system_prompt.append({"type": "text", "text": system_addon})
        else:
            system_prompt = str(system_prompt) + system_addon

        # Convert to OpenAI format
        openai_payload = {
            "model": claude_body.get("model"),
            "messages": convert_messages(claude_body.get("messages", [])),
            "temperature": claude_body.get("temperature", 1.0),
            "stream": claude_body.get("stream", False),
        }

        if system_prompt:
            openai_payload["messages"].insert(0, {"role": "system", "content": system_prompt})

        tools = claude_body.get("tools")
        if tools and isinstance(tools, list) and len(tools) > 0:
            openai_payload["tools"] = convert_tools(tools)

        headers = {
            "Authorization": f"Bearer {config.API_KEY}",
            "Content-Type": "application/json"
        }

        client: AsyncClient = request.app.state.http_client

        if openai_payload.get("stream"):
            return StreamingResponse(
                _stream_generator(client, config.BACKEND_URL, openai_payload, headers),
                media_type="text/event-stream"
            )
        else:
            return await _handle_non_stream(client, config.BACKEND_URL, openai_payload, headers)

    return app


async def _handle_non_stream(client: AsyncClient, backend_url: str, payload: dict, headers: dict) -> Response:
    """Handle non-streaming responses."""
    resp = await client.post(
        f"{backend_url}/chat/completions",
        json=payload,
        headers=headers
    )
    res = resp.json()

    if "choices" not in res:
        return Response(content=json.dumps(res), status_code=resp.status_code)

    choice = res["choices"][0]["message"]
    content_blocks = []

    if choice.get("content"):
        content_blocks.append({"type": "text", "text": choice["content"]})

    if choice.get("tool_calls"):
        for tc in choice["tool_calls"]:
            content_blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"])
            })

    return {
        "id": res.get("id", "chatcmpl-local"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": res.get("model", "unknown"),
        "usage": {
            "input_tokens": res.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": res.get("usage", {}).get("completion_tokens", 0)
        }
    }


async def _stream_generator(client: AsyncClient, backend_url: str, payload: dict, headers: dict) -> AsyncGenerator[str, None]:
    """Generate SSE stream for streaming responses."""
    async with client.stream(
        "POST",
        f"{backend_url}/chat/completions",
        json=payload,
        headers=headers
    ) as r:
        if r.status_code != 200:
            err_val = await r.aread()
            logger.error(f"Upstream Error: {err_val.decode()}")
            yield f"event: error\ndata: {err_val.decode()}\n\n"
            return

        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'role': 'assistant', 'content': []}})}\n\n"

        c_op = False
        active_tools = {}

        async for line in r.aiter_lines():
            if not line.startswith("data: ") or "[DONE]" in line:
                continue

            chunk = json.loads(line[6:])
            if not chunk.get("choices"):
                continue

            delta = chunk["choices"][0].get("delta", {})

            if "content" in delta and delta["content"]:
                if not c_op:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    c_op = True
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"

            # Handle tool calls
            tcs = delta.get("tool_calls")
            if tcs and isinstance(tcs, list):
                for tc in tcs:
                    idx = tc.get("index", 0)
                    if idx not in active_tools:
                        active_tools[idx] = {
                            "id": tc.get("id", f"call_{idx}"),
                            "name": tc.get("function", {}).get("name", "Bash"),
                            "args": ""
                        }
                    if tc.get("function", {}).get("arguments"):
                        active_tools[idx]["args"] += tc["function"]["arguments"]

            if chunk["choices"][0].get("finish_reason"):
                # Emit fixed tool calls
                for idx, tool in active_tools.items():
                    fixed_args_str = fix_tool_arguments(tool["name"], tool["args"])
                    block_idx = idx + 1

                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_idx, 'content_block': {'type': 'tool_use', 'id': tool['id'], 'name': tool['name'], 'input': {}}})}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'input_json_delta', 'partial_json': fixed_args_str}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"

                if c_op:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
