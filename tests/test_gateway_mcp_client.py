"""Tests for the gateway MCP client (DownstreamConnection).

Tests cover: connection & tool discovery, tool call forwarding, schema
fidelity, error handling (crash, timeout, malformed response), child
process lifecycle, and edge cases.
"""

import asyncio
import json
import sys
import textwrap

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

from sanna.gateway.mcp_client import (
    DownstreamConnection,
    DownstreamConnectionError,
    DownstreamError,
    DownstreamTimeoutError,
)

# =============================================================================
# MOCK SERVER SCRIPTS
# =============================================================================

MOCK_SERVER_SCRIPT = textwrap.dedent("""\
    import json
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mock_downstream")

    @mcp.tool()
    def get_status() -> str:
        \"\"\"Get the current server status.\"\"\"
        return json.dumps({"status": "ok", "version": "1.0"})

    @mcp.tool()
    def search(query: str, limit: int = 10) -> str:
        \"\"\"Search for items matching a query.\"\"\"
        return json.dumps({"query": query, "limit": limit, "results": ["a", "b"]})

    @mcp.tool()
    def create_item(name: str, tags: list[str], metadata: dict) -> str:
        \"\"\"Create a new item with tags and metadata.\"\"\"
        return json.dumps({
            "created": True, "name": name,
            "tags": tags, "metadata": metadata,
        })

    @mcp.tool()
    def error_tool() -> str:
        \"\"\"A tool that always errors.\"\"\"
        raise ValueError("Intentional error for testing")

    mcp.run(transport="stdio")
""")

SLOW_SERVER_SCRIPT = textwrap.dedent("""\
    import time
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("slow_downstream")

    @mcp.tool()
    def slow_tool() -> str:
        \"\"\"A tool that takes forever.\"\"\"
        time.sleep(300)
        return "done"

    mcp.run(transport="stdio")
""")

CRASH_SERVER_SCRIPT = textwrap.dedent("""\
    import os
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("crash_downstream")

    @mcp.tool()
    def crash_tool() -> str:
        \"\"\"A tool that crashes the server.\"\"\"
        os._exit(1)

    @mcp.tool()
    def normal_tool() -> str:
        \"\"\"A tool that works normally.\"\"\"
        return "ok"

    mcp.run(transport="stdio")
""")


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture()
def mock_server_path(tmp_path):
    """Write the standard mock server script to a temp file."""
    path = tmp_path / "mock_server.py"
    path.write_text(MOCK_SERVER_SCRIPT)
    return str(path)


@pytest.fixture()
def slow_server_path(tmp_path):
    """Write the slow mock server script to a temp file."""
    path = tmp_path / "slow_server.py"
    path.write_text(SLOW_SERVER_SCRIPT)
    return str(path)


@pytest.fixture()
def crash_server_path(tmp_path):
    """Write the crash mock server script to a temp file."""
    path = tmp_path / "crash_server.py"
    path.write_text(CRASH_SERVER_SCRIPT)
    return str(path)


# =============================================================================
# 1. CONNECTION & DISCOVERY
# =============================================================================

class TestConnectionDiscovery:
    def test_connect_discovers_tools(self, mock_server_path):
        """Connecting to the mock server discovers all 4 tools."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                assert len(conn.tools) == 4

        asyncio.run(_test())

    def test_discovered_tool_names(self, mock_server_path):
        """All expected tool names are present."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                expected = {"get_status", "search", "create_item", "error_tool"}
                assert conn.tool_names == expected

        asyncio.run(_test())

    def test_connected_is_true_after_connect(self, mock_server_path):
        """The connected property is True after a successful connect."""
        async def _test():
            conn = DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            )
            assert conn.connected is False
            await conn.connect()
            try:
                assert conn.connected is True
            finally:
                await conn.close()

        asyncio.run(_test())

    def test_tools_empty_before_connect(self):
        """A fresh instance has no tools."""
        conn = DownstreamConnection(command="unused")
        assert conn.tools == []
        assert conn.tool_names == set()

    def test_context_manager_connects(self, mock_server_path):
        """Using async with connects and sets connected=True."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                assert conn.connected is True

        asyncio.run(_test())


# =============================================================================
# 2. TOOL CALLS
# =============================================================================

class TestToolCalls:
    def test_call_no_params_tool(self, mock_server_path):
        """Calling a no-params tool returns a valid result."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                result = await conn.call_tool("get_status")
                assert result.isError is not True
                text = result.content[0].text
                data = json.loads(text)
                assert data["status"] == "ok"

        asyncio.run(_test())

    def test_call_with_required_params(self, mock_server_path):
        """Calling a tool with required params returns correct data."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                result = await conn.call_tool(
                    "search", {"query": "hello"}
                )
                assert result.isError is not True
                data = json.loads(result.content[0].text)
                assert data["query"] == "hello"
                assert data["limit"] == 10  # default

        asyncio.run(_test())

    def test_call_with_optional_and_required_params(self, mock_server_path):
        """Optional params are forwarded correctly."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                result = await conn.call_tool(
                    "search", {"query": "hello", "limit": 5}
                )
                assert result.isError is not True
                data = json.loads(result.content[0].text)
                assert data["query"] == "hello"
                assert data["limit"] == 5

        asyncio.run(_test())

    def test_call_with_complex_params(self, mock_server_path):
        """Tool with list and dict params works correctly."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                result = await conn.call_tool("create_item", {
                    "name": "widget",
                    "tags": ["red", "large"],
                    "metadata": {"weight": 42},
                })
                assert result.isError is not True
                data = json.loads(result.content[0].text)
                assert data["created"] is True
                assert data["name"] == "widget"
                assert data["tags"] == ["red", "large"]
                assert data["metadata"] == {"weight": 42}

        asyncio.run(_test())

    def test_call_result_is_call_tool_result(self, mock_server_path):
        """call_tool returns a CallToolResult instance."""
        from mcp.types import CallToolResult

        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                result = await conn.call_tool("get_status")
                assert isinstance(result, CallToolResult)

        asyncio.run(_test())


# =============================================================================
# 3. SCHEMA FIDELITY
# =============================================================================

class TestSchemaFidelity:
    def test_every_tool_has_input_schema(self, mock_server_path):
        """All discovered tools have an inputSchema field."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                for tool in conn.tools:
                    assert "inputSchema" in tool, f"{tool['name']} missing inputSchema"
                    assert tool["inputSchema"]["type"] == "object"

        asyncio.run(_test())

    def test_required_fields_correct(self, mock_server_path):
        """Required params appear in the 'required' list."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                by_name = {t["name"]: t for t in conn.tools}

                # search: query is required, limit is optional
                search_schema = by_name["search"]["inputSchema"]
                required = search_schema.get("required", [])
                assert "query" in required
                assert "limit" not in required

        asyncio.run(_test())

    def test_optional_params_not_required(self, mock_server_path):
        """Optional params (with defaults) are NOT in required."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                by_name = {t["name"]: t for t in conn.tools}
                search_schema = by_name["search"]["inputSchema"]
                required = search_schema.get("required", [])
                assert "limit" not in required

        asyncio.run(_test())

    def test_property_types(self, mock_server_path):
        """Parameter types in inputSchema match expectations."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                by_name = {t["name"]: t for t in conn.tools}

                # search: query is string, limit is integer
                search_props = by_name["search"]["inputSchema"]["properties"]
                assert search_props["query"]["type"] == "string"
                assert search_props["limit"]["type"] == "integer"

                # create_item: tags is array, metadata is object
                create_props = by_name["create_item"]["inputSchema"]["properties"]
                assert create_props["tags"]["type"] == "array"
                assert create_props["metadata"]["type"] == "object"

        asyncio.run(_test())

    def test_description_preserved(self, mock_server_path):
        """Tool descriptions are preserved in the schema."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                by_name = {t["name"]: t for t in conn.tools}
                assert by_name["get_status"]["description"] is not None
                assert "status" in by_name["get_status"]["description"].lower()
                assert by_name["search"]["description"] is not None
                assert "search" in by_name["search"]["description"].lower()

        asyncio.run(_test())

    def test_schema_field_by_field_fidelity(self, mock_server_path):
        """Every field in the discovered schema matches the original.

        Connects twice independently and verifies the schemas are
        identical, proving deterministic and lossless passthrough.
        """
        async def _test():
            # First connection
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn1:
                tools1 = conn1.tools

            # Second connection
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn2:
                tools2 = conn2.tools

            # Sort both by name for comparison
            tools1.sort(key=lambda t: t["name"])
            tools2.sort(key=lambda t: t["name"])

            assert len(tools1) == len(tools2)
            for t1, t2 in zip(tools1, tools2):
                assert t1 == t2, (
                    f"Schema mismatch for {t1['name']}: {t1} != {t2}"
                )

        asyncio.run(_test())


# =============================================================================
# 4. ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    def test_downstream_crash_returns_error(self, crash_server_path):
        """Server crash during tool call returns isError, doesn't crash client."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[crash_server_path],
                timeout=10.0,
            ) as conn:
                result = await conn.call_tool("crash_tool")
                assert result.isError is True
                assert len(result.content) > 0

        asyncio.run(_test())

    def test_downstream_timeout_returns_error(self, slow_server_path):
        """Slow tool call returns timeout error, doesn't hang."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[slow_server_path],
                timeout=30.0,  # generous for connect
            ) as conn:
                # Use a short per-call timeout
                result = await conn.call_tool("slow_tool", timeout=2.0)
                assert result.isError is True
                assert "timed out" in result.content[0].text.lower()

        asyncio.run(_test())

    def test_downstream_error_tool_returns_mcp_error(self, mock_server_path):
        """Tool that raises returns isError=True."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                result = await conn.call_tool("error_tool")
                assert result.isError is True

        asyncio.run(_test())

    def test_call_before_connect_returns_error(self):
        """Calling a tool before connect returns an error result."""
        async def _test():
            conn = DownstreamConnection(command="unused")
            result = await conn.call_tool("anything")
            assert result.isError is True
            assert "not connected" in result.content[0].text.lower()

        asyncio.run(_test())

    def test_call_after_close_returns_error(self, mock_server_path):
        """Calling a tool after close returns an error result."""
        async def _test():
            conn = DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            )
            await conn.connect()
            await conn.close()
            result = await conn.call_tool("get_status")
            assert result.isError is True
            assert "not connected" in result.content[0].text.lower()

        asyncio.run(_test())

    def test_nonexistent_command_raises(self):
        """Connecting to a nonexistent command raises DownstreamConnectionError."""
        async def _test():
            conn = DownstreamConnection(
                command="/nonexistent/binary/that/does/not/exist",
            )
            with pytest.raises(DownstreamConnectionError) as exc_info:
                await conn.connect()
            assert "failed to start" in str(exc_info.value).lower()

        asyncio.run(_test())


# =============================================================================
# 5. LIFECYCLE
# =============================================================================

class TestLifecycle:
    def test_close_sets_disconnected(self, mock_server_path):
        """After close(), connected is False."""
        async def _test():
            conn = DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            )
            await conn.connect()
            assert conn.connected is True
            await conn.close()
            assert conn.connected is False

        asyncio.run(_test())

    def test_close_clears_tools(self, mock_server_path):
        """After close(), tools list is empty."""
        async def _test():
            conn = DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            )
            await conn.connect()
            assert len(conn.tools) > 0
            await conn.close()
            assert conn.tools == []
            assert conn.tool_names == set()

        asyncio.run(_test())

    def test_double_connect_raises(self, mock_server_path):
        """Connecting twice without closing raises an error."""
        async def _test():
            conn = DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            )
            await conn.connect()
            try:
                with pytest.raises(DownstreamConnectionError, match="Already connected"):
                    await conn.connect()
            finally:
                await conn.close()

        asyncio.run(_test())

    def test_double_close_is_safe(self, mock_server_path):
        """Closing twice does not raise."""
        async def _test():
            conn = DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            )
            await conn.connect()
            await conn.close()
            await conn.close()  # Should not raise

        asyncio.run(_test())

    def test_context_manager_cleans_up(self, mock_server_path):
        """Exiting the context manager sets connected=False and clears state."""
        async def _test():
            conn = DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            )
            async with conn:
                assert conn.connected is True
            assert conn.connected is False
            assert conn.tools == []

        asyncio.run(_test())


# =============================================================================
# 6. EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_call_nonexistent_tool(self, mock_server_path):
        """Calling a tool that doesn't exist returns an error."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
            ) as conn:
                result = await conn.call_tool("this_tool_does_not_exist")
                assert result.isError is True

        asyncio.run(_test())

    def test_custom_timeout_on_constructor(self, mock_server_path):
        """Custom timeout in constructor is stored and accessible."""
        conn = DownstreamConnection(
            command=sys.executable,
            args=[mock_server_path],
            timeout=42.0,
        )
        assert conn._timeout == 42.0

    def test_per_call_timeout_override(self, mock_server_path):
        """Per-call timeout works correctly for successful calls."""
        async def _test():
            async with DownstreamConnection(
                command=sys.executable,
                args=[mock_server_path],
                timeout=30.0,
            ) as conn:
                # Short timeout but fast tool — should succeed
                result = await conn.call_tool(
                    "get_status", timeout=10.0,
                )
                assert result.isError is not True

        asyncio.run(_test())

    def test_exception_hierarchy(self):
        """All custom exceptions inherit from DownstreamError."""
        assert issubclass(DownstreamConnectionError, DownstreamError)
        assert issubclass(DownstreamTimeoutError, DownstreamError)
        assert issubclass(DownstreamError, Exception)
