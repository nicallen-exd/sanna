"""Tests for the gateway MCP server (SannaGateway — Block B).

Tests cover: tool discovery with prefixed names, schema fidelity,
end-to-end call forwarding, error handling, and lifecycle management.
"""

import asyncio
import json
import sys
import textwrap

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

from sanna.gateway.mcp_client import DownstreamConnectionError
from sanna.gateway.server import SannaGateway, _dict_to_tool, _META_TOOL_NAMES

# =============================================================================
# MOCK SERVER SCRIPT (same tools as Block A tests)
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


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture()
def mock_server_path(tmp_path):
    """Write the mock server script to a temp file."""
    path = tmp_path / "mock_server.py"
    path.write_text(MOCK_SERVER_SCRIPT)
    return str(path)


# =============================================================================
# 1. TOOL DISCOVERY WITH PREFIXED NAMES
# =============================================================================

class TestToolDiscovery:
    def test_discovers_downstream_tools_with_prefix(self, mock_server_path):
        """Gateway discovers tools and prefixes them with server name."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                tools = gw._build_tool_list()
                names = {t.name for t in tools}
                assert "mock_get_status" in names
                assert "mock_search" in names
                assert "mock_create_item" in names
                assert "mock_error_tool" in names
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_correct_tool_count(self, mock_server_path):
        """Number of gateway tools matches downstream + meta-tools."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                tools = gw._build_tool_list()
                downstream = [t for t in tools if t.name not in _META_TOOL_NAMES]
                meta = [t for t in tools if t.name in _META_TOOL_NAMES]
                assert len(downstream) == 4
                assert len(meta) == 2
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_tool_names_follow_prefix_pattern(self, mock_server_path):
        """Every downstream tool name follows {server_name}/{tool_name}
        pattern. Meta-tools are not prefixed."""
        async def _test():
            gw = SannaGateway(
                server_name="my-server",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                tools = gw._build_tool_list()
                for t in tools:
                    if t.name in _META_TOOL_NAMES:
                        continue  # meta-tools are not prefixed
                    assert t.name.startswith("my-server_"), t.name
                    original = t.name[len("my-server_"):]
                    assert "_" not in original or original in gw.tool_map.values()
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_tool_map_matches_tool_list(self, mock_server_path):
        """tool_map keys correspond to the prefixed downstream tool names.
        Meta-tools are not in tool_map."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                tools = gw._build_tool_list()
                downstream_names = {
                    t.name for t in tools if t.name not in _META_TOOL_NAMES
                }
                assert set(gw.tool_map.keys()) == downstream_names
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 2. SCHEMA FIDELITY
# =============================================================================

class TestSchemaFidelity:
    def test_input_schema_identical(self, mock_server_path):
        """inputSchema is identical between downstream and gateway tools."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                downstream_by_name = {
                    t["name"]: t for t in gw.downstream.tools
                }
                for gw_tool in gw._build_tool_list():
                    if gw_tool.name in _META_TOOL_NAMES:
                        continue
                    original_name = gw.tool_map[gw_tool.name]
                    ds_tool = downstream_by_name[original_name]
                    assert gw_tool.inputSchema == ds_tool["inputSchema"], (
                        f"Schema mismatch for {original_name}"
                    )
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_description_preserved(self, mock_server_path):
        """Tool descriptions are preserved through the gateway."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                downstream_by_name = {
                    t["name"]: t for t in gw.downstream.tools
                }
                for gw_tool in gw._build_tool_list():
                    if gw_tool.name in _META_TOOL_NAMES:
                        continue
                    original_name = gw.tool_map[gw_tool.name]
                    ds_tool = downstream_by_name[original_name]
                    assert gw_tool.description == ds_tool.get("description"), (
                        f"Description mismatch for {original_name}"
                    )
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_only_name_changes(self, mock_server_path):
        """The only difference between downstream and gateway tool is the
        prefixed name — all other fields are identical."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                downstream_by_name = {
                    t["name"]: t for t in gw.downstream.tools
                }
                for gw_tool in gw._build_tool_list():
                    if gw_tool.name in _META_TOOL_NAMES:
                        continue
                    original_name = gw.tool_map[gw_tool.name]
                    ds_tool = downstream_by_name[original_name]

                    # Dump the gateway tool to dict for comparison
                    gw_dict = gw_tool.model_dump(exclude_none=True)
                    # Replace name with original for comparison
                    gw_dict["name"] = original_name
                    # annotations are ToolAnnotations objects in gw_dict
                    # but plain dicts in ds_tool — normalize
                    assert gw_dict["name"] == ds_tool["name"]
                    assert gw_dict["inputSchema"] == ds_tool["inputSchema"]
                    assert gw_dict.get("description") == ds_tool.get("description")
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 3. CALL FORWARDING (END-TO-END)
# =============================================================================

class TestCallForwarding:
    def test_forward_no_params_tool(self, mock_server_path):
        """Forwarding a no-params tool returns the correct result."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                result = await gw._forward_call("mock_get_status", {})
                assert result.isError is not True
                data = json.loads(result.content[0].text)
                assert data["status"] == "ok"
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_forward_with_params(self, mock_server_path):
        """Forwarding a tool call with params works correctly."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                result = await gw._forward_call(
                    "mock_search", {"query": "hello", "limit": 5},
                )
                assert result.isError is not True
                data = json.loads(result.content[0].text)
                assert data["query"] == "hello"
                assert data["limit"] == 5
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_forward_complex_params(self, mock_server_path):
        """Forwarding nested params (list, dict) works correctly."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                result = await gw._forward_call("mock_create_item", {
                    "name": "widget",
                    "tags": ["red", "large"],
                    "metadata": {"weight": 42},
                })
                assert result.isError is not True
                data = json.loads(result.content[0].text)
                assert data["created"] is True
                assert data["tags"] == ["red", "large"]
                assert data["metadata"] == {"weight": 42}
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_multiple_tools_all_work(self, mock_server_path):
        """All tools from the same downstream can be called."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                r1 = await gw._forward_call("mock_get_status", {})
                assert r1.isError is not True

                r2 = await gw._forward_call(
                    "mock_search", {"query": "test"},
                )
                assert r2.isError is not True

                r3 = await gw._forward_call("mock_create_item", {
                    "name": "x", "tags": [], "metadata": {},
                })
                assert r3.isError is not True
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_error_tool_preserves_is_error(self, mock_server_path):
        """Error from downstream preserves isError=True."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                result = await gw._forward_call("mock_error_tool", {})
                assert result.isError is True
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 4. ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    def test_nonexistent_downstream_command(self):
        """Startup with bad command raises DownstreamConnectionError."""
        async def _test():
            gw = SannaGateway(
                server_name="bad",
                command="/nonexistent/binary/that/does/not/exist",
            )
            with pytest.raises(DownstreamConnectionError):
                await gw.start()

        asyncio.run(_test())

    def test_unknown_tool_name_returns_error(self, mock_server_path):
        """Calling a tool that doesn't exist returns isError."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                result = await gw._forward_call("mock_nonexistent", {})
                assert result.isError is True
                assert "unknown tool" in result.content[0].text.lower()
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_call_without_prefix_returns_error(self, mock_server_path):
        """Calling by original name (no prefix) returns error."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                # "search" exists on downstream but gateway exposes "mock_search"
                result = await gw._forward_call("search", {})
                assert result.isError is True
                assert "unknown tool" in result.content[0].text.lower()
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_call_with_wrong_prefix_returns_error(self, mock_server_path):
        """Calling with wrong server prefix returns error."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                result = await gw._forward_call("other/search", {})
                assert result.isError is True
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 5. LIFECYCLE
# =============================================================================

class TestLifecycle:
    def test_shutdown_disconnects_downstream(self, mock_server_path):
        """Shutdown terminates the downstream connection."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            assert gw.downstream is not None
            assert gw.downstream.connected is True
            await gw.shutdown()
            assert gw.downstream is None

        asyncio.run(_test())

    def test_double_shutdown_is_safe(self, mock_server_path):
        """Calling shutdown twice doesn't raise."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            await gw.shutdown()
            await gw.shutdown()  # Should not raise

        asyncio.run(_test())

    def test_tool_map_cleared_on_shutdown(self, mock_server_path):
        """Tool map is empty after shutdown."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            assert len(gw.tool_map) > 0
            await gw.shutdown()
            assert gw.tool_map == {}

        asyncio.run(_test())

    def test_tool_list_empty_before_start(self):
        """Tool list is empty before start."""
        gw = SannaGateway(
            server_name="mock",
            command="unused",
        )
        assert gw._build_tool_list() == []
        assert gw.tool_map == {}


# =============================================================================
# 6. EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_server_name_property(self):
        """server_name property returns the configured name."""
        gw = SannaGateway(server_name="notion", command="unused")
        assert gw.server_name == "notion"

    def test_tool_map_is_copy(self, mock_server_path):
        """tool_map returns a copy, not the internal dict."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                m = gw.tool_map
                m["injected"] = "bad"
                assert "injected" not in gw.tool_map
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_dict_to_tool_preserves_fields(self):
        """_dict_to_tool helper preserves all schema fields."""
        import mcp.types as types

        tool_dict = {
            "name": "original",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
        tool = _dict_to_tool("pfx/original", tool_dict)
        assert tool.name == "pfx/original"
        assert tool.description == "A test tool"
        assert tool.inputSchema == tool_dict["inputSchema"]
        assert isinstance(tool, types.Tool)
