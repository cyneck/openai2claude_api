"""Tests for message and tool conversion functions."""

import json
import pytest
from openai2claude.proxy import convert_tools, convert_messages, fix_tool_arguments


class TestConvertTools:
    """Tests for convert_tools function."""

    def test_convert_empty_list(self):
        """Empty input returns empty list."""
        assert convert_tools([]) == []
        assert convert_tools(None) == []
        assert convert_tools("not a list") == []

    def test_convert_basic_tool(self):
        """Test converting a basic tool."""
        claude_tools = [
            {
                "name": "bash",
                "description": "Run a bash command",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        ]
        result = convert_tools(claude_tools)
        assert len(result) == 1
        assert result[0] == {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a bash command",
                "parameters": claude_tools[0]["input_schema"]
            }
        }

    def test_convert_multiple_tools(self):
        """Test converting multiple tools."""
        claude_tools = [
            {"name": "tool1", "description": "Desc 1", "input_schema": {}},
            {"name": "tool2", "description": "Desc 2", "input_schema": {}},
        ]
        result = convert_tools(claude_tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["function"]["name"] == "tool2"


class TestConvertMessages:
    """Tests for convert_messages function."""

    def test_convert_simple_text_message(self):
        """Test converting a simple text message."""
        claude_messages = [
            {"role": "user", "content": "Hello"}
        ]
        result = convert_messages(claude_messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_convert_user_message_with_text_blocks(self):
        """Test converting user message with text blocks."""
        claude_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"}
                ]
            }
        ]
        result = convert_messages(claude_messages)
        assert result == [{"role": "user", "content": "Hello world"}]

    def test_convert_message_with_tool_use(self):
        """Test converting message with tool_use block."""
        claude_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "Bash",
                        "input": {"command": "ls -la"}
                    }
                ]
            }
        ]
        result = convert_messages(claude_messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "toolu_123"
        assert result[0]["tool_calls"][0]["function"]["name"] == "Bash"
        assert '"command": "ls -la"' in result[0]["tool_calls"][0]["function"]["arguments"]

    def test_convert_message_with_tool_result(self):
        """Test converting message with tool_result block."""
        claude_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": [{"type": "text", "text": "file1.txt\nfile2.txt"}]
                    }
                ]
            }
        ]
        result = convert_messages(claude_messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "toolu_123"
        assert result[0]["content"] == "file1.txt\nfile2.txt"

    def test_convert_message_with_mixed_content(self):
        """Test converting message with text and tool_use."""
        claude_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running command:"},
                    {"type": "tool_use", "id": "toolu_456", "name": "Bash", "input": {"command": "pwd"}}
                ]
            }
        ]
        result = convert_messages(claude_messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Running command:"
        assert len(result[0]["tool_calls"]) == 1

    def test_convert_message_with_text_and_tool_result(self):
        """Test converting message with text and tool_result (tool results come separately)."""
        claude_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the result:"},
                    {"type": "tool_result", "tool_use_id": "toolu_789", "content": [{"type": "text", "text": "success"}]}
                ]
            }
        ]
        result = convert_messages(claude_messages)
        # Should produce: one message with text, one message with tool result
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Here is the result:"
        assert result[1]["role"] == "tool"
        assert result[1]["content"] == "success"

    def test_fix_bash_tool_cmd_to_command(self):
        """Test that Bash tool with 'cmd' is fixed to 'command'."""
        claude_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_001", "name": "Bash", "input": {"cmd": "echo hello"}}
                ]
            }
        ]
        result = convert_messages(claude_messages)
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        # The 'cmd' should be converted to 'command'
        assert '"command": "echo hello"' in args


class TestFixToolArguments:
    """Tests for fix_tool_arguments function."""

    def test_fix_valid_arguments(self):
        """Valid arguments pass through unchanged."""
        result = fix_tool_arguments("Bash", '{"command": "ls"}')
        assert result == '{"command": "ls"}'

    def test_fix_empty_arguments(self):
        """Empty arguments get default command."""
        result = fix_tool_arguments("Bash", "")
        parsed = json.loads(result)
        assert "command" in parsed

    def test_fix_bash_with_cmd(self):
        """Bash tool with 'cmd' gets fixed to 'command'."""
        result = fix_tool_arguments("Bash", '{"cmd": "echo hi"}')
        parsed = json.loads(result)
        assert parsed.get("command") == "echo hi"

    def test_fix_bash_with_input(self):
        """Bash tool with 'input' gets fixed to 'command'."""
        result = fix_tool_arguments("bash", '{"input": "echo hi"}')
        parsed = json.loads(result)
        assert parsed.get("command") == "echo hi"

    def test_fix_non_bash_unchanged(self):
        """Non-Bash tools pass through unchanged."""
        result = fix_tool_arguments("SomeTool", '{"foo": "bar"}')
        assert result == '{"foo": "bar"}'

    def test_fix_invalid_json(self):
        """Invalid JSON returns default command."""
        result = fix_tool_arguments("Bash", "not json")
        parsed = json.loads(result)
        assert "command" in parsed
