#!/usr/bin/env python3
"""Gateway Demo — Sanna v0.10.0

Three-beat story demonstrating the MCP enforcement gateway:

  Beat 1: notion_search → can_execute → forward → receipt
  Beat 2: notion_update_page → must_escalate → user approves → forward → receipt
  Beat 3: Both receipts verified offline (fingerprint + signature)

Runs against a mock downstream — no real Notion server needed.

Run:
    python examples/gateway_demo.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanna.constitution import (
    Constitution,
    AgentIdentity,
    Provenance,
    Boundary,
    Invariant,
    HaltCondition,
    AuthorityBoundaries,
    EscalationRule,
    sign_constitution,
    save_constitution,
)
from sanna.crypto import generate_keypair
from sanna.verify import verify_receipt, load_schema

# =============================================================================
# CONSTANTS
# =============================================================================

SECTION_WIDTH = 70


def header(beat: int, title: str) -> None:
    print()
    print("=" * SECTION_WIDTH)
    print(f"  BEAT {beat}: {title}")
    print("=" * SECTION_WIDTH)
    print()


def subheader(label: str) -> None:
    print(f"  -> {label}")


def success(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# =============================================================================
# MOCK DOWNSTREAM
# =============================================================================

def _make_mock_downstream():
    """Create a mock DownstreamConnection with Notion-like tools."""
    from mcp.types import CallToolResult, TextContent

    mock = AsyncMock()
    mock.connected = True
    mock.last_call_was_connection_error = False
    mock.tools = [
        {
            "name": "search",
            "description": "Search Notion pages",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "update_page",
            "description": "Update a Notion page",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "page_id": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["page_id"],
            },
        },
        {
            "name": "read_credentials",
            "description": "Read stored credentials",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]

    # Mock call_tool responses
    async def mock_call_tool(name, args=None):
        if name == "search":
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "results": [
                            {"title": "Q1 Planning", "id": "page-001"},
                            {"title": "Q1 OKRs", "id": "page-002"},
                        ],
                    }),
                )],
            )
        elif name == "update_page":
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "updated": True,
                        "page_id": (args or {}).get("page_id", ""),
                        "title": (args or {}).get("title", ""),
                    }),
                )],
            )
        else:
            return CallToolResult(
                content=[TextContent(type="text", text="ok")],
            )

    mock.call_tool = mock_call_tool
    return mock


# =============================================================================
# DEMO
# =============================================================================

async def _run_async(tmp_path: Path) -> dict:
    """Run the three-beat demo. Returns summary dict."""
    # Import gateway after path setup
    from sanna.gateway.server import SannaGateway

    results = {
        "beats_completed": 0,
        "receipts_generated": 0,
        "receipts_verified": 0,
    }

    # -- Setup: constitution + signing keys --

    subheader("Setting up constitution and keys")

    gateway_priv, gateway_pub = generate_keypair(tmp_path / "gateway_keys")
    receipt_store = tmp_path / "receipts"
    receipt_store.mkdir()

    const = Constitution(
        schema_version="0.1.0",
        identity=AgentIdentity(
            agent_name="gateway-demo-agent",
            domain="workspace-management",
            description="Demo agent for gateway enforcement",
        ),
        provenance=Provenance(
            authored_by="demo@sanna.dev",
            approved_by=["admin@sanna.dev"],
            approval_date="2026-02-14",
            approval_method="demo",
        ),
        boundaries=[
            Boundary(
                id="B001",
                description="Operate within workspace",
                category="scope",
                severity="high",
            ),
        ],
        invariants=[
            Invariant(
                id="INV_NO_FABRICATION",
                rule="Do not claim facts absent from sources.",
                enforcement="halt",
            ),
            Invariant(
                id="INV_MARK_INFERENCE",
                rule="Mark inferences clearly.",
                enforcement="warn",
            ),
            Invariant(
                id="INV_NO_FALSE_CERTAINTY",
                rule="No false certainty.",
                enforcement="warn",
            ),
            Invariant(
                id="INV_PRESERVE_TENSION",
                rule="Preserve conflicting evidence.",
                enforcement="warn",
            ),
            Invariant(
                id="INV_NO_PREMATURE_COMPRESSION",
                rule="No oversimplification.",
                enforcement="warn",
            ),
        ],
        authority_boundaries=AuthorityBoundaries(
            can_execute=["search", "read_file"],
            must_escalate=[
                EscalationRule(condition="Update database page or CRM record"),
            ],
            cannot_execute=[
                "read_credentials",
                "delete_repo",
                "upload_external",
            ],
        ),
        halt_conditions=[
            HaltCondition(
                id="H001",
                trigger="Credential access attempt",
                escalate_to="admin@sanna.dev",
                severity="critical",
                enforcement="halt",
            ),
        ],
    )

    const_path = tmp_path / "constitution.yaml"
    signed = sign_constitution(const, private_key_path=str(gateway_priv))
    save_constitution(signed, const_path)

    print(f"  Constitution: {const.identity.agent_name}")
    print(f"  can_execute: {const.authority_boundaries.can_execute}")
    print(f"  must_escalate: [update_page, ...]")
    print(f"  cannot_execute: {const.authority_boundaries.cannot_execute}")
    print()

    # -- Create gateway with mock downstream --

    gw = SannaGateway(
        server_name="notion",
        command="echo",  # placeholder, won't be used
        args=[],
        constitution_path=str(const_path),
        signing_key_path=str(gateway_priv),
        receipt_store_path=str(receipt_store),
    )

    # Inject mock downstream — bypass real subprocess
    mock_downstream = _make_mock_downstream()
    gw._downstream = mock_downstream

    # Build tool map from mock tools
    for tool in mock_downstream.tools:
        prefixed = f"notion_{tool['name']}"
        gw._tool_map[prefixed] = tool["name"]

    # Load constitution (same as start() but without subprocess)
    from sanna.constitution import (
        load_constitution,
        constitution_to_receipt_ref,
    )
    from sanna.enforcement import configure_checks

    gw._constitution = load_constitution(str(const_path))
    gw._constitution_ref = constitution_to_receipt_ref(gw._constitution)
    gw._check_configs, gw._custom_records = configure_checks(gw._constitution)

    receipts = []

    # =================================================================
    # Beat 1: notion_search → can_execute → forward → receipt
    # =================================================================
    header(1, "READ OPERATION (can_execute)")

    search_result = await gw._forward_call(
        "notion_search", {"query": "Q1 planning"},
    )

    search_receipt = gw.last_receipt
    receipts.append(search_receipt)

    print(f"  Tool: notion_search")
    print(f"  Decision: {search_receipt['extensions']['gateway']['decision']}")
    print(f"  Result: {search_result.content[0].text[:80]}...")
    print(f"  Receipt ID: {search_receipt['receipt_id'][:16]}...")
    print(f"  Status: {search_receipt['coherence_status']}")
    halt_ev = search_receipt.get("halt_event") or {}
    print(f"  Halted: {halt_ev.get('halted', False)}")

    assert not search_result.isError, "Search should succeed"
    assert search_receipt["extensions"]["gateway"]["decision"] == "allow"
    success("Search forwarded, receipt generated")
    results["beats_completed"] += 1
    results["receipts_generated"] += 1

    # =================================================================
    # Beat 2: notion_update_page → must_escalate → approve → receipt
    # =================================================================
    header(2, "WRITE OPERATION (must_escalate → approve)")

    # 2a: Initial call triggers escalation
    subheader("Step 1: Tool call triggers escalation")

    escalation_result = await gw._forward_call(
        "notion_update_page",
        {"page_id": "page-001", "title": "Q1 Planning (updated)"},
    )

    escalation_receipt = gw.last_receipt
    results["receipts_generated"] += 1

    escalation_data = json.loads(escalation_result.content[0].text)
    escalation_id = escalation_data["escalation_id"]

    print(f"  Tool: notion_update_page")
    print(f"  Status: {escalation_data['status']}")
    print(f"  Escalation ID: {escalation_id}")
    print(f"  Reason: {escalation_data['reason']}")
    print(f"  Escalation receipt: {escalation_receipt['receipt_id'][:16]}...")

    assert escalation_data["status"] == "ESCALATION_REQUIRED"
    success("Write escalated — pending user approval")
    print()

    # 2b: User approves → forward → receipt
    subheader("Step 2: User approves, tool call forwarded")

    approve_result = await gw._forward_call(
        "sanna_approve_escalation",
        {"escalation_id": escalation_id},
    )

    approve_receipt = gw.last_receipt
    receipts.append(approve_receipt)
    results["receipts_generated"] += 1

    approve_text = approve_result.content[0].text
    approve_data = json.loads(approve_text)

    print(f"  Approval result: {approve_data}")
    print(f"  Approval receipt: {approve_receipt['receipt_id'][:16]}...")
    print(f"  Decision: {approve_receipt['extensions']['gateway']['decision']}")
    print(f"  Escalation chain: {approve_receipt['extensions']['gateway'].get('escalation_receipt_id', 'n/a')[:16]}...")
    print(f"  Resolution: {approve_receipt['extensions']['gateway'].get('escalation_resolution')}")

    assert not approve_result.isError, "Approved call should succeed"
    assert approve_receipt["extensions"]["gateway"]["escalation_resolution"] == "approved"
    success("Write forwarded after approval, receipt with chain")
    results["beats_completed"] += 1

    # =================================================================
    # Beat 3: Verify both receipts offline
    # =================================================================
    header(3, "OFFLINE VERIFICATION")

    schema = load_schema()

    for i, receipt in enumerate(receipts):
        label = ["search (allow)", "update_page (approved)"][i]
        subheader(f"Verifying: {label}")

        vr = verify_receipt(
            receipt, schema,
            public_key_path=str(gateway_pub),
        )

        fp_match = vr.computed_fingerprint == vr.expected_fingerprint

        print(f"    Valid: {vr.valid}")
        print(f"    Errors: {vr.errors}")
        print(f"    Warnings: {len(vr.warnings)}")
        print(f"    Fingerprint match: {fp_match}")
        print(f"    Status: {vr.computed_status}")

        if not vr.valid:
            for err in vr.errors:
                fail(f"    {err}")
        else:
            success(f"Receipt verified: {label}")
            results["receipts_verified"] += 1

    # Verify persisted receipts exist on disk
    persisted = list(receipt_store.glob("*.json"))
    subheader(f"Persisted receipts on disk: {len(persisted)}")
    for p in sorted(persisted):
        print(f"    {p.name}")

    assert len(persisted) >= 3, (
        f"Expected at least 3 persisted receipts, got {len(persisted)}"
    )
    success(f"{len(persisted)} receipts persisted to {receipt_store.name}/")
    results["beats_completed"] += 1

    return results


def run_demo() -> dict:
    """Run the gateway demo. Returns summary dict."""
    with tempfile.TemporaryDirectory(prefix="sanna_gateway_") as tmp:
        return asyncio.run(_run_async(Path(tmp)))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * SECTION_WIDTH)
    print("  SANNA GATEWAY DEMO — MCP Enforcement Proxy")
    print("=" * SECTION_WIDTH)

    results = run_demo()

    print()
    print("=" * SECTION_WIDTH)
    print("  GATEWAY DEMO COMPLETE")
    print("=" * SECTION_WIDTH)
    print()
    print(f"  Beats completed:      {results['beats_completed']}/3")
    print(f"  Receipts generated:   {results['receipts_generated']}")
    print(f"  Receipts verified:    {results['receipts_verified']}")
    print()

    if results["beats_completed"] < 3:
        sys.exit(1)
