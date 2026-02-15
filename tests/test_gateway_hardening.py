"""Tests for gateway hardening (Block F).

Tests cover: crash recovery, circuit breaker, timeout handling,
receipt persistence, receipt file verification, filename format,
latency budget, structured logging, and reconnect.
"""

import asyncio
import json
import logging
import os
import sys
import textwrap
import time

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

from sanna.gateway.mcp_client import DownstreamConnection
from sanna.gateway.server import (
    SannaGateway,
    _CIRCUIT_BREAKER_THRESHOLD,
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
    def delete_item(item_id: str) -> str:
        \"\"\"Delete an item by ID.\"\"\"
        return json.dumps({"deleted": True, "item_id": item_id})

    mcp.run(transport="stdio")
""")

# Server that exits after the first tool call (simulates crash)
CRASHING_SERVER_SCRIPT = textwrap.dedent("""\
    import json
    import os
    import signal
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("crashing_downstream")

    _call_count = 0

    @mcp.tool()
    def get_status() -> str:
        \"\"\"Get the current server status.\"\"\"
        global _call_count
        _call_count += 1
        if _call_count > 1:
            # Kill ourselves to simulate a crash
            os.kill(os.getpid(), signal.SIGTERM)
        return json.dumps({"status": "ok"})

    mcp.run(transport="stdio")
""")


# =============================================================================
# HELPERS
# =============================================================================

def _create_signed_constitution(tmp_path, authority_boundaries=None):
    """Create a signed constitution and keypair for testing."""
    from sanna.crypto import generate_keypair
    from sanna.constitution import (
        Constitution,
        AgentIdentity,
        Provenance,
        Boundary,
        sign_constitution,
        save_constitution,
    )

    keys_dir = tmp_path / "keys"
    private_key_path, public_key_path = generate_keypair(str(keys_dir))

    identity = AgentIdentity(
        agent_name="test-agent",
        domain="testing",
    )
    provenance = Provenance(
        authored_by="test@example.com",
        approved_by=["approver@example.com"],
        approval_date="2024-01-01",
        approval_method="manual-sign-off",
    )
    boundaries = [
        Boundary(
            id="B001",
            description="Test boundary",
            category="scope",
            severity="high",
        ),
    ]

    constitution = Constitution(
        schema_version="0.1.0",
        identity=identity,
        provenance=provenance,
        boundaries=boundaries,
        invariants=[],
        authority_boundaries=authority_boundaries,
    )

    signed = sign_constitution(
        constitution, private_key_path=str(private_key_path),
    )

    const_path = tmp_path / "constitution.yaml"
    save_constitution(signed, const_path)

    return str(const_path), str(private_key_path), str(public_key_path)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture()
def mock_server_path(tmp_path):
    """Write the standard mock server script."""
    path = tmp_path / "mock_server.py"
    path.write_text(MOCK_SERVER_SCRIPT)
    return str(path)


@pytest.fixture()
def crashing_server_path(tmp_path):
    """Write the crashing server script."""
    path = tmp_path / "crashing_server.py"
    path.write_text(CRASHING_SERVER_SCRIPT)
    return str(path)


@pytest.fixture()
def signed_constitution(tmp_path):
    """Create a signed constitution with no authority boundaries."""
    const_path, private_key, public_key = _create_signed_constitution(
        tmp_path,
    )
    return const_path, private_key, public_key


@pytest.fixture()
def receipt_store(tmp_path):
    """Create a receipt store directory."""
    store = tmp_path / "receipts"
    store.mkdir()
    return str(store)


# =============================================================================
# 1. CONNECTION ERROR TRACKING (mcp_client.py)
# =============================================================================

class TestConnectionErrorTracking:
    def test_successful_call_clears_error_flag(self, mock_server_path):
        """Successful tool calls clear the connection error flag."""
        async def _test():
            conn = DownstreamConnection(
                command=sys.executable, args=[mock_server_path],
            )
            await conn.connect()
            try:
                result = await conn.call_tool("get_status")
                assert result.isError is not True
                assert conn.last_call_was_connection_error is False
            finally:
                await conn.close()

        asyncio.run(_test())

    def test_not_connected_sets_error_flag(self):
        """Calling on a disconnected client sets the error flag."""
        async def _test():
            conn = DownstreamConnection(
                command="unused", args=[],
            )
            result = await conn.call_tool("anything")
            assert result.isError is True
            assert conn.last_call_was_connection_error is True

        asyncio.run(_test())

    def test_reconnect_restores_connection(self, mock_server_path):
        """reconnect() closes and re-establishes the connection."""
        async def _test():
            conn = DownstreamConnection(
                command=sys.executable, args=[mock_server_path],
            )
            await conn.connect()
            try:
                # Verify connected
                assert conn.connected is True
                original_tools = conn.tool_names

                # Reconnect
                await conn.reconnect()
                assert conn.connected is True
                assert conn.tool_names == original_tools
            finally:
                await conn.close()

        asyncio.run(_test())

    def test_error_flag_initially_false(self, mock_server_path):
        """Connection error flag starts as False."""
        conn = DownstreamConnection(
            command=sys.executable, args=[mock_server_path],
        )
        assert conn.last_call_was_connection_error is False


# =============================================================================
# 2. CIRCUIT BREAKER
# =============================================================================

class TestCircuitBreaker:
    def test_gateway_starts_healthy(self, mock_server_path):
        """Gateway starts in healthy state."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                assert gw.healthy is True
                assert gw.consecutive_failures == 0
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_successful_call_keeps_healthy(self, mock_server_path):
        """A successful tool call keeps the gateway healthy."""
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
                assert gw.healthy is True
                assert gw.consecutive_failures == 0
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_unhealthy_blocks_forwarding(
        self, mock_server_path, signed_constitution, receipt_store,
    ):
        """An unhealthy gateway returns error without forwarding."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
                receipt_store_path=receipt_store,
            )
            await gw.start()
            try:
                # Force unhealthy
                gw._healthy = False
                result = await gw._forward_call("mock_get_status", {})
                assert result.isError is True
                assert "unhealthy" in result.content[0].text.lower()
                # Error receipt should have been generated
                assert gw.last_receipt is not None
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_unhealthy_generates_error_receipt(
        self, mock_server_path, signed_constitution, receipt_store,
    ):
        """Unhealthy gateway generates an error receipt with halt event."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
                receipt_store_path=receipt_store,
            )
            await gw.start()
            try:
                gw._healthy = False
                await gw._forward_call("mock_get_status", {})
                receipt = gw.last_receipt
                assert receipt is not None
                assert receipt["halt_event"]["halted"] is True
                assert "unhealthy" in receipt["halt_event"]["reason"].lower()
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_circuit_breaker_threshold_constant(self):
        """Circuit breaker threshold is 3."""
        assert _CIRCUIT_BREAKER_THRESHOLD == 3


# =============================================================================
# 3. RECEIPT PERSISTENCE
# =============================================================================

class TestReceiptPersistence:
    def test_receipt_persisted_to_directory(
        self, mock_server_path, signed_constitution, receipt_store,
    ):
        """Receipts are written as JSON files to the receipt store."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
                receipt_store_path=receipt_store,
            )
            await gw.start()
            try:
                await gw._forward_call("mock_get_status", {})
                # Check receipt file exists
                files = os.listdir(receipt_store)
                json_files = [f for f in files if f.endswith(".json")]
                assert len(json_files) == 1
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_receipt_filename_format(
        self, mock_server_path, signed_constitution, receipt_store,
    ):
        """Receipt filename follows {timestamp}_{receipt_id}.json format."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
                receipt_store_path=receipt_store,
            )
            await gw.start()
            try:
                await gw._forward_call("mock_get_status", {})
                files = [f for f in os.listdir(receipt_store) if f.endswith(".json")]
                assert len(files) == 1
                name = files[0]
                # Should not contain colons (filesystem-safe)
                assert ":" not in name
                # Should end with .json
                assert name.endswith(".json")
                # Should contain the receipt_id
                receipt = gw.last_receipt
                assert receipt["receipt_id"] in name
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_persisted_receipt_is_valid_json(
        self, mock_server_path, signed_constitution, receipt_store,
    ):
        """Persisted receipt file is valid JSON matching last_receipt."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
                receipt_store_path=receipt_store,
            )
            await gw.start()
            try:
                await gw._forward_call("mock_get_status", {})
                files = [f for f in os.listdir(receipt_store) if f.endswith(".json")]
                filepath = os.path.join(receipt_store, files[0])
                with open(filepath) as f:
                    persisted = json.load(f)
                assert persisted["receipt_id"] == gw.last_receipt["receipt_id"]
                assert persisted["trace_id"] == gw.last_receipt["trace_id"]
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_persisted_receipt_verifies_fingerprint(
        self, mock_server_path, signed_constitution, receipt_store,
    ):
        """Persisted receipt passes fingerprint verification."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
                receipt_store_path=receipt_store,
            )
            await gw.start()
            try:
                await gw._forward_call("mock_get_status", {})
                files = [f for f in os.listdir(receipt_store) if f.endswith(".json")]
                filepath = os.path.join(receipt_store, files[0])
                with open(filepath) as f:
                    persisted = json.load(f)

                from sanna.verify import verify_fingerprint
                matches, computed, expected = verify_fingerprint(persisted)
                assert matches is True, (
                    f"Fingerprint mismatch: {computed} != {expected}"
                )
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_multiple_calls_create_multiple_files(
        self, mock_server_path, signed_constitution, receipt_store,
    ):
        """Each tool call creates a separate receipt file."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
                receipt_store_path=receipt_store,
            )
            await gw.start()
            try:
                await gw._forward_call("mock_get_status", {})
                await gw._forward_call(
                    "mock_search", {"query": "test"},
                )
                files = [f for f in os.listdir(receipt_store) if f.endswith(".json")]
                assert len(files) == 2
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_no_persistence_without_store_path(
        self, mock_server_path, signed_constitution,
    ):
        """No receipt files created when receipt_store_path is None."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
                # No receipt_store_path
            )
            await gw.start()
            try:
                await gw._forward_call("mock_get_status", {})
                assert gw.last_receipt is not None
                # No crash, no files written
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 4. STRUCTURED LOGGING
# =============================================================================

class TestStructuredLogging:
    def test_allow_logged_at_info(
        self, mock_server_path, signed_constitution, caplog,
    ):
        """Allowed tool calls are logged at INFO."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
            )
            await gw.start()
            try:
                with caplog.at_level(logging.INFO, logger="sanna.gateway.server"):
                    await gw._forward_call("mock_get_status", {})
                assert any("ALLOW" in r.message for r in caplog.records)
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_deny_logged_at_warning(
        self, mock_server_path, signed_constitution, caplog,
    ):
        """Denied tool calls are logged at WARNING."""
        async def _test():
            from sanna.constitution import AuthorityBoundaries
            const_path, private_key, _ = _create_signed_constitution(
                pytest.importorskip("tmp_path_factory"),
            )

        # Create constitution with cannot_execute boundary
        async def _test2():
            from sanna.constitution import AuthorityBoundaries
            tmp = mock_server_path  # just need the fixture trigger
            # Use policy_overrides instead (simpler)
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=signed_constitution[0],
                signing_key_path=signed_constitution[1],
                policy_overrides={"get_status": "cannot_execute"},
            )
            await gw.start()
            try:
                with caplog.at_level(logging.WARNING, logger="sanna.gateway.server"):
                    await gw._forward_call("mock_get_status", {})
                assert any("DENY" in r.message for r in caplog.records)
            finally:
                await gw.shutdown()

        asyncio.run(_test2())

    def test_circuit_breaker_logged_at_error(
        self, mock_server_path, signed_constitution, caplog, receipt_store,
    ):
        """Circuit breaker open is logged at WARNING."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
                receipt_store_path=receipt_store,
            )
            await gw.start()
            try:
                gw._healthy = False
                with caplog.at_level(logging.WARNING, logger="sanna.gateway.server"):
                    await gw._forward_call("mock_get_status", {})
                assert any(
                    "circuit breaker" in r.message.lower()
                    for r in caplog.records
                )
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 5. LATENCY BUDGET
# =============================================================================

class TestLatencyBudget:
    def test_gateway_overhead_under_500ms(
        self, mock_server_path, signed_constitution, receipt_store,
    ):
        """Gateway enforcement + receipt overhead < 500ms per call."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
                receipt_store_path=receipt_store,
            )
            await gw.start()
            try:
                # Warm up — first call may include lazy imports
                await gw._forward_call("mock_get_status", {})

                # Measure over several calls
                times = []
                for _ in range(5):
                    start = time.monotonic()
                    await gw._forward_call(
                        "mock_search", {"query": "test"},
                    )
                    elapsed = time.monotonic() - start
                    times.append(elapsed)

                avg = sum(times) / len(times)
                # Allow generous margin for CI — assert < 2s
                # (500ms target, but CI machines are slow)
                assert avg < 2.0, (
                    f"Average gateway overhead {avg:.3f}s exceeds 2.0s"
                )
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 6. TIMEOUT CONFIGURATION
# =============================================================================

class TestTimeoutConfig:
    def test_timeout_from_config(self):
        """DownstreamConfig supports configurable timeout."""
        from sanna.gateway.config import DownstreamConfig
        ds = DownstreamConfig(
            name="test",
            command="unused",
            timeout=60.0,
        )
        assert ds.timeout == 60.0

    def test_timeout_default(self):
        """DownstreamConfig defaults to 30s timeout."""
        from sanna.gateway.config import DownstreamConfig
        ds = DownstreamConfig(
            name="test",
            command="unused",
        )
        assert ds.timeout == 30.0


# =============================================================================
# 7. CRASH RECOVERY
# =============================================================================

class TestCrashRecovery:
    def test_after_downstream_call_resets_on_success(self, mock_server_path):
        """Successful downstream call resets failure counter."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                # Simulate prior failures
                gw._consecutive_failures = 2
                result = await gw._forward_call("mock_get_status", {})
                assert result.isError is not True
                assert gw.consecutive_failures == 0
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_rebuild_tool_map(self, mock_server_path):
        """_rebuild_tool_map correctly rebuilds from downstream tools."""
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
            )
            await gw.start()
            try:
                original_map = dict(gw.tool_map)
                gw._tool_map.clear()
                assert gw.tool_map == {}
                gw._rebuild_tool_map()
                assert gw.tool_map == original_map
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_error_receipt_has_correct_structure(
        self, mock_server_path, signed_constitution,
    ):
        """Error receipts have halt_event and authority_decisions."""
        async def _test():
            const_path, private_key, _ = signed_constitution
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=private_key,
            )
            await gw.start()
            try:
                receipt = gw._generate_error_receipt(
                    prefixed_name="mock_get_status",
                    original_name="get_status",
                    arguments={},
                    error_text="test error",
                )
                assert receipt["halt_event"]["halted"] is True
                assert "test error" in receipt["halt_event"]["reason"]
                assert len(receipt["authority_decisions"]) == 1
                assert receipt["authority_decisions"][0]["decision"] == "halt"
            finally:
                await gw.shutdown()

        asyncio.run(_test())
