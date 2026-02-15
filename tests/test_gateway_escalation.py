"""Tests for gateway must_escalate UX (Block E).

Tests cover: escalation lifecycle (create, approve, deny), receipt chain,
expiry, meta-tool registration, concurrent escalations, and edge cases.
"""

import asyncio
import json
import sys
import textwrap

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

from sanna.gateway.server import (
    SannaGateway,
    EscalationStore,
    _META_TOOL_APPROVE,
    _META_TOOL_DENY,
)


# =============================================================================
# MOCK SERVER SCRIPT
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
    def update_item(item_id: str, name: str) -> str:
        \"\"\"Update an item.\"\"\"
        return json.dumps({"updated": True, "item_id": item_id, "name": name})

    @mcp.tool()
    def delete_item(item_id: str) -> str:
        \"\"\"Delete an item by ID.\"\"\"
        return json.dumps({"deleted": True, "item_id": item_id})

    mcp.run(transport="stdio")
""")


# =============================================================================
# HELPERS
# =============================================================================

def _create_signed_constitution(
    tmp_path,
    authority_boundaries=None,
):
    """Create a signed constitution and keypair for testing.

    Returns (constitution_path, private_key_path, public_key_path).
    """
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
    """Write the mock server script to a temp file."""
    path = tmp_path / "mock_server.py"
    path.write_text(MOCK_SERVER_SCRIPT)
    return str(path)


@pytest.fixture()
def signed_constitution(tmp_path):
    """Create a signed constitution with must_escalate for update_item."""
    from sanna.constitution import (
        AuthorityBoundaries,
        EscalationRule,
    )

    ab = AuthorityBoundaries(
        cannot_execute=["delete_item"],
        must_escalate=[
            EscalationRule(condition="update"),
        ],
        can_execute=["get_status", "search"],
    )
    return _create_signed_constitution(tmp_path, authority_boundaries=ab)


# =============================================================================
# 1. ESCALATION REQUIRED — CREATING PENDING ESCALATION
# =============================================================================

class TestEscalationRequired:
    def test_escalation_returns_structured_json(
        self, mock_server_path, signed_constitution,
    ):
        """must_escalate tool call returns ESCALATION_REQUIRED with
        escalation_id (not a deny)."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                # Not an error — it's an escalation prompt
                assert result.isError is not True
                data = json.loads(result.content[0].text)
                assert data["status"] == "ESCALATION_REQUIRED"
                assert data["escalation_id"].startswith("esc_")
                assert data["tool"] == "mock_update_item"
                assert data["parameters"] == {"item_id": "1", "name": "new"}
                assert "reason" in data
                assert "instruction" in data
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_escalation_creates_pending_entry(
        self, mock_server_path, signed_constitution,
    ):
        """Escalation adds entry to the escalation store."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                data = json.loads(result.content[0].text)
                esc_id = data["escalation_id"]
                entry = gw.escalation_store.get(esc_id)
                assert entry is not None
                assert entry.original_name == "update_item"
                assert entry.arguments == {"item_id": "1", "name": "new"}
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 2. APPROVE ESCALATION
# =============================================================================

class TestApproveEscalation:
    def test_approve_forwards_to_downstream(
        self, mock_server_path, signed_constitution,
    ):
        """Approve valid escalation → original request forwarded to
        downstream, result returned."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                # Trigger escalation
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "42", "name": "approved-name"},
                )
                esc_data = json.loads(esc_result.content[0].text)
                esc_id = esc_data["escalation_id"]

                # Approve it
                approve_result = await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": esc_id},
                )
                assert approve_result.isError is not True
                data = json.loads(approve_result.content[0].text)
                assert data["updated"] is True
                assert data["item_id"] == "42"
                assert data["name"] == "approved-name"
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_approve_receipt_has_chain(
        self, mock_server_path, signed_constitution,
    ):
        """Approve valid escalation → receipt includes full approval chain."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                # Trigger escalation
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_data = json.loads(esc_result.content[0].text)
                esc_id = esc_data["escalation_id"]
                esc_receipt = gw.last_receipt
                esc_receipt_id = esc_receipt["receipt_id"]

                # Approve
                await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": esc_id},
                )
                approval_receipt = gw.last_receipt

                # Chain fields in extensions
                gw_ext = approval_receipt["extensions"]["gateway"]
                assert gw_ext["escalation_id"] == esc_id
                assert gw_ext["escalation_receipt_id"] == esc_receipt_id
                assert gw_ext["escalation_resolution"] == "approved"
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_approve_removes_from_store(
        self, mock_server_path, signed_constitution,
    ):
        """After approval, the escalation is removed from the store."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_data = json.loads(esc_result.content[0].text)
                esc_id = esc_data["escalation_id"]
                assert len(gw.escalation_store) == 1

                await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": esc_id},
                )
                assert len(gw.escalation_store) == 0
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 3. DENY ESCALATION
# =============================================================================

class TestDenyEscalation:
    def test_deny_generates_halt_receipt(
        self, mock_server_path, signed_constitution,
    ):
        """Deny valid escalation → denial receipt generated with halt_event,
        no downstream call."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                # Trigger escalation
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_data = json.loads(esc_result.content[0].text)
                esc_id = esc_data["escalation_id"]

                # Deny
                deny_result = await gw._forward_call(
                    _META_TOOL_DENY,
                    {"escalation_id": esc_id},
                )
                assert deny_result.isError is not True
                deny_data = json.loads(deny_result.content[0].text)
                assert deny_data["status"] == "denied"

                # Receipt has halt_event
                receipt = gw.last_receipt
                assert receipt["halt_event"] is not None
                assert receipt["halt_event"]["halted"] is True
                assert esc_id in receipt["halt_event"]["reason"]
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_deny_receipt_has_chain(
        self, mock_server_path, signed_constitution,
    ):
        """Denial receipt references the original escalation receipt ID."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_data = json.loads(esc_result.content[0].text)
                esc_id = esc_data["escalation_id"]
                esc_receipt_id = gw.last_receipt["receipt_id"]

                await gw._forward_call(
                    _META_TOOL_DENY,
                    {"escalation_id": esc_id},
                )
                deny_receipt = gw.last_receipt
                gw_ext = deny_receipt["extensions"]["gateway"]
                assert gw_ext["escalation_id"] == esc_id
                assert gw_ext["escalation_receipt_id"] == esc_receipt_id
                assert gw_ext["escalation_resolution"] == "denied"
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 4. EXPIRED AND NOT FOUND
# =============================================================================

class TestExpiredAndNotFound:
    def test_approve_nonexistent_returns_not_found(
        self, mock_server_path, signed_constitution,
    ):
        """Approve nonexistent escalation_id → ESCALATION_NOT_FOUND error."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                result = await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": "esc_does_not_exist"},
                )
                assert result.isError is True
                data = json.loads(result.content[0].text)
                assert data["error"] == "ESCALATION_NOT_FOUND"
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_deny_nonexistent_returns_not_found(
        self, mock_server_path, signed_constitution,
    ):
        """Deny nonexistent escalation_id → ESCALATION_NOT_FOUND error."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                result = await gw._forward_call(
                    _META_TOOL_DENY,
                    {"escalation_id": "esc_does_not_exist"},
                )
                assert result.isError is True
                data = json.loads(result.content[0].text)
                assert data["error"] == "ESCALATION_NOT_FOUND"
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_approve_expired_returns_expired(
        self, mock_server_path, signed_constitution,
    ):
        """Approve expired escalation → ESCALATION_EXPIRED error."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
                escalation_timeout=1,  # 1 second
            )
            await gw.start()
            try:
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_data = json.loads(esc_result.content[0].text)
                esc_id = esc_data["escalation_id"]

                # Wait for expiry
                await asyncio.sleep(1.2)

                result = await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": esc_id},
                )
                assert result.isError is True
                data = json.loads(result.content[0].text)
                assert data["error"] == "ESCALATION_EXPIRED"
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_deny_expired_returns_expired(
        self, mock_server_path, signed_constitution,
    ):
        """Deny expired escalation → ESCALATION_EXPIRED error."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
                escalation_timeout=1,  # 1 second
            )
            await gw.start()
            try:
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_data = json.loads(esc_result.content[0].text)
                esc_id = esc_data["escalation_id"]

                await asyncio.sleep(1.2)

                result = await gw._forward_call(
                    _META_TOOL_DENY,
                    {"escalation_id": esc_id},
                )
                assert result.isError is True
                data = json.loads(result.content[0].text)
                assert data["error"] == "ESCALATION_EXPIRED"
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 5. RECEIPT VERIFICATION
# =============================================================================

class TestReceiptVerification:
    def test_escalation_receipt_verifies_offline(
        self, mock_server_path, signed_constitution,
    ):
        """Escalation receipt fingerprint passes offline verification."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            from sanna.verify import verify_fingerprint

            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                r = gw.last_receipt
                matches, computed, expected = verify_fingerprint(r)
                assert matches, (
                    f"Escalation receipt fingerprint mismatch: "
                    f"computed={computed}, expected={expected}"
                )
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_approval_receipt_verifies_offline(
        self, mock_server_path, signed_constitution,
    ):
        """Approval receipt fingerprint passes offline verification."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            from sanna.verify import verify_fingerprint

            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_data = json.loads(esc_result.content[0].text)
                esc_id = esc_data["escalation_id"]

                await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": esc_id},
                )
                r = gw.last_receipt
                matches, computed, expected = verify_fingerprint(r)
                assert matches, (
                    f"Approval receipt fingerprint mismatch: "
                    f"computed={computed}, expected={expected}"
                )
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_denial_receipt_verifies_offline(
        self, mock_server_path, signed_constitution,
    ):
        """Denial receipt fingerprint passes offline verification."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            from sanna.verify import verify_fingerprint

            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_data = json.loads(esc_result.content[0].text)
                esc_id = esc_data["escalation_id"]

                await gw._forward_call(
                    _META_TOOL_DENY,
                    {"escalation_id": esc_id},
                )
                r = gw.last_receipt
                matches, computed, expected = verify_fingerprint(r)
                assert matches, (
                    f"Denial receipt fingerprint mismatch: "
                    f"computed={computed}, expected={expected}"
                )
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_approval_receipt_references_escalation_receipt_id(
        self, mock_server_path, signed_constitution,
    ):
        """Approval receipt references the original escalation receipt ID."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_data = json.loads(esc_result.content[0].text)
                esc_id = esc_data["escalation_id"]
                esc_receipt_id = gw.last_receipt["receipt_id"]

                await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": esc_id},
                )
                approval_receipt = gw.last_receipt
                chain_ref = approval_receipt["extensions"]["gateway"][
                    "escalation_receipt_id"
                ]
                assert chain_ref == esc_receipt_id
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 6. META-TOOL REGISTRATION
# =============================================================================

class TestMetaToolRegistration:
    def test_meta_tools_visible_in_tool_list(
        self, mock_server_path, signed_constitution,
    ):
        """Meta-tools (approve/deny) are registered and visible in
        gateway's tool list."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                tools = gw._build_tool_list()
                names = {t.name for t in tools}
                assert _META_TOOL_APPROVE in names
                assert _META_TOOL_DENY in names
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_meta_tools_not_prefixed(
        self, mock_server_path, signed_constitution,
    ):
        """Meta-tools do NOT get prefixed with a server name."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                tools = gw._build_tool_list()
                for t in tools:
                    if t.name in (_META_TOOL_APPROVE, _META_TOOL_DENY):
                        assert not t.name.startswith("mock_")
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_meta_tools_have_schemas(
        self, mock_server_path, signed_constitution,
    ):
        """Meta-tools have proper input schemas with escalation_id."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                tools = gw._build_tool_list()
                meta_tools = {t.name: t for t in tools
                              if t.name in (_META_TOOL_APPROVE, _META_TOOL_DENY)}
                for name, tool in meta_tools.items():
                    schema = tool.inputSchema
                    assert "escalation_id" in schema["properties"]
                    assert "escalation_id" in schema["required"]
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 7. CONCURRENT ESCALATIONS
# =============================================================================

class TestConcurrentEscalations:
    def test_two_escalations_approve_one_deny_other(
        self, mock_server_path, signed_constitution,
    ):
        """Two different escalation IDs: approve one, deny the other."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                # Trigger two escalations
                r1 = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "first"},
                )
                esc_id_1 = json.loads(r1.content[0].text)["escalation_id"]

                r2 = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "2", "name": "second"},
                )
                esc_id_2 = json.loads(r2.content[0].text)["escalation_id"]

                assert esc_id_1 != esc_id_2
                assert len(gw.escalation_store) == 2

                # Approve first
                approve_result = await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": esc_id_1},
                )
                data = json.loads(approve_result.content[0].text)
                assert data["updated"] is True
                assert data["item_id"] == "1"

                # Deny second
                deny_result = await gw._forward_call(
                    _META_TOOL_DENY,
                    {"escalation_id": esc_id_2},
                )
                deny_data = json.loads(deny_result.content[0].text)
                assert deny_data["status"] == "denied"

                assert len(gw.escalation_store) == 0
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 8. DOUBLE RESOLUTION
# =============================================================================

class TestDoubleResolution:
    def test_double_approve_returns_not_found(
        self, mock_server_path, signed_constitution,
    ):
        """Double-approve same escalation_id → second call returns NOT_FOUND."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_id = json.loads(
                    esc_result.content[0].text,
                )["escalation_id"]

                # First approve succeeds
                r1 = await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": esc_id},
                )
                assert r1.isError is not True

                # Second approve → NOT_FOUND
                r2 = await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": esc_id},
                )
                assert r2.isError is True
                data = json.loads(r2.content[0].text)
                assert data["error"] == "ESCALATION_NOT_FOUND"
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_approve_then_deny_returns_not_found(
        self, mock_server_path, signed_constitution,
    ):
        """Approve then deny same escalation → deny returns NOT_FOUND."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                esc_result = await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                esc_id = json.loads(
                    esc_result.content[0].text,
                )["escalation_id"]

                await gw._forward_call(
                    _META_TOOL_APPROVE,
                    {"escalation_id": esc_id},
                )

                deny_result = await gw._forward_call(
                    _META_TOOL_DENY,
                    {"escalation_id": esc_id},
                )
                assert deny_result.isError is True
                data = json.loads(deny_result.content[0].text)
                assert data["error"] == "ESCALATION_NOT_FOUND"
            finally:
                await gw.shutdown()

        asyncio.run(_test())


# =============================================================================
# 9. EDGE CASES
# =============================================================================

class TestEscalationEdgeCases:
    def test_approve_missing_escalation_id(
        self, mock_server_path, signed_constitution,
    ):
        """Approve with missing escalation_id → MISSING_PARAMETER error."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                result = await gw._forward_call(
                    _META_TOOL_APPROVE, {},
                )
                assert result.isError is True
                data = json.loads(result.content[0].text)
                assert data["error"] == "MISSING_PARAMETER"
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_deny_missing_escalation_id(
        self, mock_server_path, signed_constitution,
    ):
        """Deny with missing escalation_id → MISSING_PARAMETER error."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                result = await gw._forward_call(
                    _META_TOOL_DENY, {},
                )
                assert result.isError is True
                data = json.loads(result.content[0].text)
                assert data["error"] == "MISSING_PARAMETER"
            finally:
                await gw.shutdown()

        asyncio.run(_test())

    def test_escalation_store_cleared_on_shutdown(
        self, mock_server_path, signed_constitution,
    ):
        """Escalation store is cleared on gateway shutdown."""
        const_path, key_path, _ = signed_constitution
        async def _test():
            gw = SannaGateway(
                server_name="mock",
                command=sys.executable,
                args=[mock_server_path],
                constitution_path=const_path,
                signing_key_path=key_path,
            )
            await gw.start()
            try:
                await gw._forward_call(
                    "mock_update_item",
                    {"item_id": "1", "name": "new"},
                )
                assert len(gw.escalation_store) == 1
            finally:
                await gw.shutdown()
            assert len(gw.escalation_store) == 0

        asyncio.run(_test())
