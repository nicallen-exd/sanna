"""Gateway MCP server — transparent proxy to downstream MCP servers.

Re-exposes downstream tools with prefixed names ({server_name}_{tool_name}).
Full schema fidelity: inputSchema, description, annotations all preserved.
Only the tool name changes.

Uses the low-level ``mcp.server.lowlevel.Server`` so the ``call_tool``
handler can return ``CallToolResult`` directly (preserving ``isError``
from downstream).

Block C adds constitution enforcement: every tool call is evaluated
against authority boundaries before forwarding, and every action
generates a signed reasoning receipt.

Block E adds must_escalate UX: escalated tool calls are held pending
until explicitly approved or denied via gateway meta-tools
(``sanna_approve_escalation`` / ``sanna_deny_escalation``).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from sanna.gateway.mcp_client import (
    DownstreamConnection,
    DownstreamConnectionError,
)

logger = logging.getLogger("sanna.gateway.server")

# Gateway meta-tool names — never prefixed with server name
_META_TOOL_APPROVE = "sanna_approve_escalation"
_META_TOOL_DENY = "sanna_deny_escalation"
_META_TOOL_NAMES = frozenset({_META_TOOL_APPROVE, _META_TOOL_DENY})

# Default escalation timeout (seconds)
_DEFAULT_ESCALATION_TIMEOUT = 300  # 5 minutes

# Circuit breaker threshold — consecutive connection failures before
# marking the downstream as unhealthy
_CIRCUIT_BREAKER_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Pending escalation store
# ---------------------------------------------------------------------------

@dataclass
class PendingEscalation:
    """A tool call held pending user approval."""
    escalation_id: str
    prefixed_name: str
    original_name: str
    arguments: dict[str, Any]
    server_name: str
    reason: str
    created_at: str  # ISO 8601
    escalation_receipt_id: str = ""


class EscalationStore:
    """In-memory store for pending escalations with expiry.

    Not persistent — pending escalations are lost on gateway restart.
    """

    def __init__(self, timeout: float = _DEFAULT_ESCALATION_TIMEOUT) -> None:
        self._pending: dict[str, PendingEscalation] = {}
        self._timeout = timeout

    @property
    def timeout(self) -> float:
        return self._timeout

    def create(
        self,
        prefixed_name: str,
        original_name: str,
        arguments: dict[str, Any],
        server_name: str,
        reason: str,
    ) -> PendingEscalation:
        """Create and store a new pending escalation."""
        esc_id = f"esc_{uuid.uuid4().hex[:8]}"
        entry = PendingEscalation(
            escalation_id=esc_id,
            prefixed_name=prefixed_name,
            original_name=original_name,
            arguments=arguments,
            server_name=server_name,
            reason=reason,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._pending[esc_id] = entry
        return entry

    def get(self, escalation_id: str) -> PendingEscalation | None:
        """Get a pending escalation, or None if not found."""
        return self._pending.get(escalation_id)

    def is_expired(self, entry: PendingEscalation) -> bool:
        """Check if an escalation entry has expired."""
        created = datetime.fromisoformat(
            entry.created_at.replace("Z", "+00:00"),
        )
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - created).total_seconds()
        return elapsed > self._timeout

    def remove(self, escalation_id: str) -> PendingEscalation | None:
        """Remove and return a pending escalation."""
        return self._pending.pop(escalation_id, None)

    def __len__(self) -> int:
        return len(self._pending)

    def clear(self) -> None:
        self._pending.clear()


# ---------------------------------------------------------------------------
# SannaGateway
# ---------------------------------------------------------------------------

class SannaGateway:
    """MCP gateway that proxies tools from a downstream stdio MCP server.

    Discovers downstream tools on startup, re-exposes them under
    ``{server_name}_{tool_name}`` prefixed names, and forwards all
    ``tools/call`` requests transparently.

    When a constitution is configured, every tool call is evaluated
    against authority boundaries and generates a signed receipt.

    Tools matching ``must_escalate`` are held pending until the user
    approves or denies via the ``sanna_approve_escalation`` /
    ``sanna_deny_escalation`` meta-tools.

    Usage::

        gw = SannaGateway(
            server_name="notion",
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
            constitution_path="constitution.yaml",
            signing_key_path="gateway.key",
        )
        await gw.start()
        try:
            tools = gw.tool_map
            result = await gw._forward_call("notion_search", {"query": "hi"})
            receipt = gw.last_receipt  # governance receipt
        finally:
            await gw.shutdown()
    """

    def __init__(
        self,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
        # Block C: enforcement
        constitution_path: str | None = None,
        signing_key_path: str | None = None,
        policy_overrides: dict[str, str] | None = None,
        default_policy: str | None = None,
        # Block E: escalation
        escalation_timeout: float = _DEFAULT_ESCALATION_TIMEOUT,
        # Block F: hardening
        receipt_store_path: str | None = None,
    ) -> None:
        self._server_name = server_name
        self._command = command
        self._args = args or []
        self._env = env
        self._timeout = timeout
        self._downstream: DownstreamConnection | None = None
        self._tool_map: dict[str, str] = {}  # prefixed_name -> original_name
        self._server = Server("sanna_gateway")

        # Block C: enforcement state
        self._constitution_path = constitution_path
        self._signing_key_path = signing_key_path
        self._policy_overrides = policy_overrides or {}
        self._default_policy = default_policy
        self._constitution: Any = None
        self._constitution_ref: dict | None = None
        self._check_configs: list | None = None
        self._custom_records: list | None = None
        self._last_receipt: dict | None = None

        # Block E: escalation state
        self._escalation_store = EscalationStore(timeout=escalation_timeout)

        # Block F: hardening state
        self._receipt_store_path = receipt_store_path
        self._consecutive_failures = 0
        self._healthy = True

        self._setup_handlers()

    # -- properties ----------------------------------------------------------

    @property
    def server_name(self) -> str:
        """Configured name for the downstream server."""
        return self._server_name

    @property
    def downstream(self) -> DownstreamConnection | None:
        """The downstream connection, or ``None`` before start."""
        return self._downstream

    @property
    def tool_map(self) -> dict[str, str]:
        """Mapping of prefixed gateway tool names to original names."""
        return dict(self._tool_map)

    @property
    def constitution(self) -> Any:
        """The loaded constitution, or ``None``."""
        return self._constitution

    @property
    def last_receipt(self) -> dict | None:
        """The most recently generated receipt."""
        return self._last_receipt

    @property
    def escalation_store(self) -> EscalationStore:
        """The pending escalation store."""
        return self._escalation_store

    @property
    def healthy(self) -> bool:
        """Whether the downstream connection is healthy."""
        return self._healthy

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive connection-level failures."""
        return self._consecutive_failures

    # -- handler registration ------------------------------------------------

    def _setup_handlers(self) -> None:
        """Register ``list_tools`` and ``call_tool`` handlers on the
        low-level MCP server."""
        gateway = self

        @self._server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return gateway._build_tool_list()

        @self._server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any] | None,
        ) -> types.CallToolResult:
            return await gateway._forward_call(name, arguments)

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Connect to downstream, discover tools, load constitution.

        Raises:
            DownstreamConnectionError: If the downstream server cannot be
                started or the MCP handshake fails.
            SannaConstitutionError: If the constitution is not signed.
        """
        self._downstream = DownstreamConnection(
            command=self._command,
            args=self._args,
            env=self._env,
            timeout=self._timeout,
        )
        await self._downstream.connect()

        for tool in self._downstream.tools:
            prefixed = f"{self._server_name}_{tool['name']}"
            if prefixed in self._tool_map:
                logger.warning(
                    "Tool name collision: %s (already registered)", prefixed,
                )
            self._tool_map[prefixed] = tool["name"]

        # Block C: load constitution if configured
        if self._constitution_path is not None:
            from sanna.constitution import (
                load_constitution,
                constitution_to_receipt_ref,
                SannaConstitutionError,
            )
            from sanna.enforcement import configure_checks

            self._constitution = load_constitution(self._constitution_path)
            if not self._constitution.policy_hash:
                raise SannaConstitutionError(
                    f"Constitution is not signed: {self._constitution_path}. "
                    f"Run: sanna-sign-constitution {self._constitution_path}"
                )
            self._constitution_ref = constitution_to_receipt_ref(
                self._constitution,
            )
            self._check_configs, self._custom_records = configure_checks(
                self._constitution,
            )

            logger.info(
                "Constitution loaded: hash=%s, invariants=%d, checks=%d",
                self._constitution.policy_hash[:16],
                len(self._constitution.invariants),
                len(self._check_configs),
            )

        logger.info(
            "Gateway started: %d tools from '%s'",
            len(self._tool_map),
            self._server_name,
        )

    async def shutdown(self) -> None:
        """Disconnect from the downstream server and clean up."""
        if self._downstream is not None:
            await self._downstream.close()
            self._downstream = None
        self._tool_map.clear()
        self._constitution = None
        self._constitution_ref = None
        self._check_configs = None
        self._custom_records = None
        self._escalation_store.clear()

    async def run_stdio(self) -> None:
        """Start the gateway, serve on stdio, and shut down on exit."""
        await self.start()
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
        finally:
            await self.shutdown()

    # -- tool list -----------------------------------------------------------

    def _build_tool_list(self) -> list[types.Tool]:
        """Build the gateway's tool list from discovered downstream tools
        plus gateway meta-tools."""
        if self._downstream is None:
            return []
        tools: list[types.Tool] = []
        for tool_dict in self._downstream.tools:
            prefixed = f"{self._server_name}_{tool_dict['name']}"
            tools.append(_dict_to_tool(prefixed, tool_dict))

        # Block E: add gateway meta-tools (not prefixed)
        tools.extend(_build_meta_tools())
        return tools

    # -- policy resolution ---------------------------------------------------

    def _resolve_policy(self, original_name: str) -> str | None:
        """Resolve per-tool policy override.

        Priority:
            1. Per-tool override in ``_policy_overrides``
            2. ``_default_policy`` (from config ``default_policy``)
            3. ``None`` — fall through to constitution evaluation

        Returns ``"can_execute"``, ``"cannot_execute"``, or
        ``"must_escalate"``, or ``None`` to fall through to
        constitution authority boundary evaluation.
        """
        override = self._policy_overrides.get(original_name)
        if override is not None:
            return override
        return self._default_policy

    # -- crash recovery & circuit breaker (Block F) --------------------------

    async def _after_downstream_call(
        self,
        tool_result: types.CallToolResult,
    ) -> types.CallToolResult:
        """Post-call hook: track failures, attempt restart on crash.

        Counts consecutive connection-level errors. On the first connection
        error, attempts ONE restart. After ``_CIRCUIT_BREAKER_THRESHOLD``
        consecutive failures, marks the downstream as unhealthy.
        """
        if self._downstream is None:
            return tool_result

        if not self._downstream.last_call_was_connection_error:
            # Success — reset counter
            if self._consecutive_failures > 0:
                logger.info(
                    "Downstream '%s' recovered after %d failure(s)",
                    self._server_name,
                    self._consecutive_failures,
                )
            self._consecutive_failures = 0
            return tool_result

        # Connection error detected
        self._consecutive_failures += 1
        logger.warning(
            "Downstream '%s' connection error (%d/%d): %s",
            self._server_name,
            self._consecutive_failures,
            _CIRCUIT_BREAKER_THRESHOLD,
            tool_result.content[0].text if tool_result.content else "unknown",
        )

        if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            if self._healthy:
                self._healthy = False
                logger.error(
                    "Downstream '%s' marked UNHEALTHY after %d consecutive "
                    "failures — circuit breaker open",
                    self._server_name,
                    self._consecutive_failures,
                )
            return tool_result

        # Attempt ONE restart on first/second failure
        restarted = await self._attempt_restart()
        if restarted:
            self._consecutive_failures = 0
            logger.info(
                "Downstream '%s' restarted successfully, tools re-discovered",
                self._server_name,
            )
        return tool_result

    async def _attempt_restart(self) -> bool:
        """Try to reconnect to the downstream server.

        Returns ``True`` on success, ``False`` on failure.
        """
        if self._downstream is None:
            return False
        try:
            await self._downstream.reconnect()
            self._rebuild_tool_map()
            return True
        except Exception as e:
            logger.error(
                "Downstream '%s' restart failed: %s", self._server_name, e,
            )
            return False

    def _rebuild_tool_map(self) -> None:
        """Rebuild tool_map from current downstream tools after restart."""
        if self._downstream is None:
            return
        self._tool_map.clear()
        for tool in self._downstream.tools:
            prefixed = f"{self._server_name}_{tool['name']}"
            self._tool_map[prefixed] = tool["name"]

    def _make_unhealthy_result(
        self, prefixed_name: str,
    ) -> types.CallToolResult:
        """Return an error result when downstream is unhealthy."""
        msg = (
            f"Downstream '{self._server_name}' is unhealthy — "
            f"circuit breaker open. Tool '{prefixed_name}' not forwarded."
        )
        logger.warning("Circuit breaker blocked: %s", prefixed_name)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=msg)],
            isError=True,
        )

    # -- receipt persistence (Block F) ---------------------------------------

    def _persist_receipt(self, receipt: dict) -> None:
        """Write receipt JSON to the receipt store directory.

        Filename: ``{timestamp}_{receipt_id}.json`` where timestamp
        uses underscores instead of colons for filesystem safety.
        """
        if not self._receipt_store_path:
            return

        store_dir = Path(self._receipt_store_path)
        store_dir.mkdir(parents=True, exist_ok=True)

        ts = receipt.get("timestamp", "")
        # Sanitize for filesystem: replace colons and + with underscores
        safe_ts = ts.replace(":", "_").replace("+", "_")
        receipt_id = receipt.get("receipt_id", uuid.uuid4().hex[:8])
        filename = f"{safe_ts}_{receipt_id}.json"

        filepath = store_dir / filename
        try:
            tmp_path = filepath.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(receipt, indent=2))
            os.replace(str(tmp_path), str(filepath))
            logger.info("Receipt persisted: %s", filename)
        except OSError as e:
            logger.error("Failed to persist receipt %s: %s", filename, e)

    # -- call forwarding with enforcement ------------------------------------

    async def _forward_call(
        self,
        name: str,
        arguments: dict[str, Any] | None,
    ) -> types.CallToolResult:
        """Evaluate policy, enforce, forward if allowed, generate receipt."""
        # Block E: handle meta-tools
        if name == _META_TOOL_APPROVE:
            return await self._handle_approve(arguments or {})
        if name == _META_TOOL_DENY:
            return await self._handle_deny(arguments or {})

        original = self._tool_map.get(name)
        if original is None:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text", text=f"Unknown tool: {name}",
                )],
                isError=True,
            )

        if self._downstream is None:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text="Gateway not connected to downstream server",
                )],
                isError=True,
            )

        arguments = arguments or {}

        # Block F: circuit breaker — skip forwarding if unhealthy
        if not self._healthy:
            error_result = self._make_unhealthy_result(name)
            if self._constitution is not None:
                # Generate error receipt for blocked call
                receipt = self._generate_error_receipt(
                    prefixed_name=name,
                    original_name=original,
                    arguments=arguments,
                    error_text=error_result.content[0].text,
                )
                self._last_receipt = receipt
                self._persist_receipt(receipt)
            return error_result

        # No constitution -> transparent passthrough (Block B behavior)
        if self._constitution is None:
            result = await self._downstream.call_tool(original, arguments)
            await self._after_downstream_call(result)
            return result

        # -- Block C: enforcement --
        return await self._enforced_call(name, original, arguments)

    async def _enforced_call(
        self,
        prefixed_name: str,
        original_name: str,
        arguments: dict[str, Any],
    ) -> types.CallToolResult:
        """Run enforcement pipeline: evaluate, enforce, receipt."""
        from sanna.enforcement import evaluate_authority, AuthorityDecision
        from sanna.receipt import HaltEvent
        from sanna.middleware import (
            _generate_constitution_receipt,
            _build_trace_data,
        )

        # 1. Resolve policy
        policy_override = self._resolve_policy(original_name)
        if policy_override is not None:
            if policy_override == "cannot_execute":
                decision = AuthorityDecision(
                    decision="halt",
                    reason=f"Policy override: {original_name} is cannot_execute",
                    boundary_type="cannot_execute",
                )
            elif policy_override == "must_escalate":
                decision = AuthorityDecision(
                    decision="escalate",
                    reason=(
                        f"Policy override: {original_name} "
                        f"requires escalation"
                    ),
                    boundary_type="must_escalate",
                )
            else:
                decision = AuthorityDecision(
                    decision="allow",
                    reason=f"Policy override: {original_name} is can_execute",
                    boundary_type="can_execute",
                )
        else:
            decision = evaluate_authority(
                original_name, arguments, self._constitution,
            )

        # 2. Build authority_decisions record
        authority_decisions = [{
            "action": original_name,
            "decision": decision.decision,
            "reason": decision.reason,
            "boundary_type": decision.boundary_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]

        # 3. Enforce and get result
        result_text = ""
        tool_result = None
        halt_event = None

        if decision.decision == "halt":
            result_text = f"Action denied by policy: {decision.reason}"
            halt_event = HaltEvent(
                halted=True,
                reason=decision.reason,
                failed_checks=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
                enforcement_mode="halt",
            )
            logger.warning(
                "DENY %s: %s", original_name, decision.reason,
            )
        elif decision.decision == "escalate":
            logger.info(
                "ESCALATE %s: %s", original_name, decision.reason,
            )
            # Block E: create pending escalation instead of denying
            return await self._handle_escalation(
                prefixed_name, original_name, arguments, decision,
            )
        else:
            logger.info("ALLOW %s", original_name)
            # Forward to downstream
            tool_result = await self._downstream.call_tool(
                original_name, arguments,
            )
            # Block F: crash recovery
            await self._after_downstream_call(tool_result)
            if tool_result.content:
                result_text = tool_result.content[0].text

        # 4. Generate receipt
        receipt = self._generate_receipt(
            prefixed_name=prefixed_name,
            original_name=original_name,
            arguments=arguments,
            result_text=result_text,
            decision=decision,
            authority_decisions=authority_decisions,
            halt_event=halt_event,
        )

        self._last_receipt = receipt
        self._persist_receipt(receipt)

        # 5. Return result
        if decision.decision == "halt":
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text", text=result_text,
                )],
                isError=True,
            )
        else:
            return tool_result

    # -- escalation handling (Block E) ---------------------------------------

    async def _handle_escalation(
        self,
        prefixed_name: str,
        original_name: str,
        arguments: dict[str, Any],
        decision: Any,
    ) -> types.CallToolResult:
        """Create a pending escalation and return structured result."""
        # 1. Store pending escalation
        entry = self._escalation_store.create(
            prefixed_name=prefixed_name,
            original_name=original_name,
            arguments=arguments,
            server_name=self._server_name,
            reason=decision.reason,
        )

        # 2. Generate escalation receipt
        escalation_result = {
            "status": "ESCALATION_REQUIRED",
            "escalation_id": entry.escalation_id,
            "tool": prefixed_name,
            "parameters": arguments,
            "reason": decision.reason,
            "constitution_rule": (
                f"authority_boundaries.{decision.boundary_type}"
            ),
            "instruction": (
                "This action requires user approval. Please present "
                "the details of what you want to do and ask the user "
                "to confirm before proceeding."
            ),
        }
        result_text = json.dumps(escalation_result, sort_keys=True)

        authority_decisions = [{
            "action": original_name,
            "decision": decision.decision,
            "reason": decision.reason,
            "boundary_type": decision.boundary_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]

        receipt = self._generate_receipt(
            prefixed_name=prefixed_name,
            original_name=original_name,
            arguments=arguments,
            result_text=result_text,
            decision=decision,
            authority_decisions=authority_decisions,
            escalation_id=entry.escalation_id,
        )

        entry.escalation_receipt_id = receipt["receipt_id"]
        self._last_receipt = receipt
        self._persist_receipt(receipt)

        # 3. Return structured escalation result
        return types.CallToolResult(
            content=[types.TextContent(
                type="text", text=result_text,
            )],
        )

    async def _handle_approve(
        self, arguments: dict[str, Any],
    ) -> types.CallToolResult:
        """Handle sanna_approve_escalation meta-tool call."""
        escalation_id = arguments.get("escalation_id", "")
        if not escalation_id:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "MISSING_PARAMETER",
                        "detail": "escalation_id is required",
                    }),
                )],
                isError=True,
            )

        entry = self._escalation_store.get(escalation_id)
        if entry is None:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "ESCALATION_NOT_FOUND",
                        "escalation_id": escalation_id,
                    }),
                )],
                isError=True,
            )

        if self._escalation_store.is_expired(entry):
            self._escalation_store.remove(escalation_id)
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "ESCALATION_EXPIRED",
                        "escalation_id": escalation_id,
                    }),
                )],
                isError=True,
            )

        # Remove from store (resolved)
        self._escalation_store.remove(escalation_id)

        # Forward original request to downstream
        if self._downstream is None:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text="Gateway not connected to downstream server",
                )],
                isError=True,
            )

        tool_result = await self._downstream.call_tool(
            entry.original_name, entry.arguments,
        )
        # Block F: crash recovery
        await self._after_downstream_call(tool_result)
        result_text = ""
        if tool_result.content:
            result_text = tool_result.content[0].text

        # Generate approval receipt with chain to escalation receipt
        from sanna.enforcement import AuthorityDecision

        decision = AuthorityDecision(
            decision="allow",
            reason=(
                f"User approved escalation {escalation_id} for "
                f"{entry.original_name}"
            ),
            boundary_type="must_escalate",
        )
        authority_decisions = [{
            "action": entry.original_name,
            "decision": "allow",
            "reason": decision.reason,
            "boundary_type": "must_escalate",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]

        receipt = self._generate_receipt(
            prefixed_name=entry.prefixed_name,
            original_name=entry.original_name,
            arguments=entry.arguments,
            result_text=result_text,
            decision=decision,
            authority_decisions=authority_decisions,
            escalation_id=escalation_id,
            escalation_receipt_id=entry.escalation_receipt_id,
            escalation_resolution="approved",
        )

        self._last_receipt = receipt
        self._persist_receipt(receipt)

        return tool_result

    async def _handle_deny(
        self, arguments: dict[str, Any],
    ) -> types.CallToolResult:
        """Handle sanna_deny_escalation meta-tool call."""
        escalation_id = arguments.get("escalation_id", "")
        if not escalation_id:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "MISSING_PARAMETER",
                        "detail": "escalation_id is required",
                    }),
                )],
                isError=True,
            )

        entry = self._escalation_store.get(escalation_id)
        if entry is None:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "ESCALATION_NOT_FOUND",
                        "escalation_id": escalation_id,
                    }),
                )],
                isError=True,
            )

        if self._escalation_store.is_expired(entry):
            self._escalation_store.remove(escalation_id)
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "ESCALATION_EXPIRED",
                        "escalation_id": escalation_id,
                    }),
                )],
                isError=True,
            )

        # Remove from store (resolved)
        self._escalation_store.remove(escalation_id)

        result_text = (
            f"Escalation {escalation_id} denied by user. "
            f"Action {entry.original_name} was not executed."
        )

        # Generate denial receipt with chain to escalation receipt
        from sanna.enforcement import AuthorityDecision

        decision = AuthorityDecision(
            decision="halt",
            reason=f"User denied escalation {escalation_id}",
            boundary_type="must_escalate",
        )
        authority_decisions = [{
            "action": entry.original_name,
            "decision": "halt",
            "reason": decision.reason,
            "boundary_type": "must_escalate",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]

        from sanna.receipt import HaltEvent

        halt_event = HaltEvent(
            halted=True,
            reason=f"User denied escalation {escalation_id}",
            failed_checks=[],
            timestamp=datetime.now(timezone.utc).isoformat(),
            enforcement_mode="halt",
        )

        receipt = self._generate_receipt(
            prefixed_name=entry.prefixed_name,
            original_name=entry.original_name,
            arguments=entry.arguments,
            result_text=result_text,
            decision=decision,
            authority_decisions=authority_decisions,
            halt_event=halt_event,
            escalation_id=escalation_id,
            escalation_receipt_id=entry.escalation_receipt_id,
            escalation_resolution="denied",
        )

        self._last_receipt = receipt
        self._persist_receipt(receipt)

        return types.CallToolResult(
            content=[types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "denied",
                    "escalation_id": escalation_id,
                    "action": entry.original_name,
                }),
            )],
        )

    # -- error receipt generation (Block F) -----------------------------------

    def _generate_error_receipt(
        self,
        *,
        prefixed_name: str,
        original_name: str,
        arguments: dict[str, Any],
        error_text: str,
    ) -> dict:
        """Generate an error receipt for downstream failures.

        Used when the circuit breaker blocks a call or an unrecoverable
        error occurs.
        """
        from sanna.enforcement import AuthorityDecision
        from sanna.receipt import HaltEvent

        decision = AuthorityDecision(
            decision="halt",
            reason=f"Downstream error: {error_text}",
            boundary_type="cannot_execute",
        )
        authority_decisions = [{
            "action": original_name,
            "decision": "halt",
            "reason": decision.reason,
            "boundary_type": "cannot_execute",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
        halt_event = HaltEvent(
            halted=True,
            reason=error_text,
            failed_checks=[],
            timestamp=datetime.now(timezone.utc).isoformat(),
            enforcement_mode="halt",
        )

        return self._generate_receipt(
            prefixed_name=prefixed_name,
            original_name=original_name,
            arguments=arguments,
            result_text=error_text,
            decision=decision,
            authority_decisions=authority_decisions,
            halt_event=halt_event,
        )

    # -- receipt generation --------------------------------------------------

    def _generate_receipt(
        self,
        *,
        prefixed_name: str,
        original_name: str,
        arguments: dict[str, Any],
        result_text: str,
        decision: Any,
        authority_decisions: list,
        halt_event: Any = None,
        escalation_id: str | None = None,
        escalation_receipt_id: str | None = None,
        escalation_resolution: str | None = None,
    ) -> dict:
        """Generate and optionally sign a gateway receipt."""
        from sanna.middleware import (
            _generate_constitution_receipt,
            _build_trace_data,
        )

        trace_id = f"gw-{uuid.uuid4().hex[:12]}"
        context_str = json.dumps(
            arguments, sort_keys=True,
        ) if arguments else ""

        trace_data = _build_trace_data(
            trace_id=trace_id,
            query=original_name,
            context=context_str,
            output=result_text,
        )

        extensions: dict[str, Any] = {
            "gateway": {
                "server_name": self._server_name,
                "tool_name": original_name,
                "prefixed_name": prefixed_name,
                "decision": decision.decision,
                "boundary_type": decision.boundary_type,
            },
        }

        # Include escalation chain info in extensions when present
        if escalation_id is not None:
            extensions["gateway"]["escalation_id"] = escalation_id
        if escalation_receipt_id is not None:
            extensions["gateway"]["escalation_receipt_id"] = (
                escalation_receipt_id
            )
        if escalation_resolution is not None:
            extensions["gateway"]["escalation_resolution"] = (
                escalation_resolution
            )

        receipt = _generate_constitution_receipt(
            trace_data,
            check_configs=self._check_configs or [],
            custom_records=self._custom_records or [],
            constitution_ref=self._constitution_ref,
            constitution_version=(
                self._constitution.schema_version
                if self._constitution else ""
            ),
            extensions=extensions,
            halt_event=halt_event,
            authority_decisions=authority_decisions,
        )

        # Sign receipt if key provided
        if self._signing_key_path is not None:
            from sanna.crypto import sign_receipt
            receipt = sign_receipt(receipt, self._signing_key_path)

        return receipt


# ---------------------------------------------------------------------------
# Meta-tools
# ---------------------------------------------------------------------------

def _build_meta_tools() -> list[types.Tool]:
    """Build the gateway's meta-tool definitions."""
    return [
        types.Tool(
            name=_META_TOOL_APPROVE,
            description=(
                "Approve a pending escalation. Forwards the original "
                "tool call to the downstream server."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "escalation_id": {
                        "type": "string",
                        "description": (
                            "The escalation ID returned by the "
                            "ESCALATION_REQUIRED response."
                        ),
                    },
                },
                "required": ["escalation_id"],
            },
        ),
        types.Tool(
            name=_META_TOOL_DENY,
            description=(
                "Deny a pending escalation. The original tool call "
                "will not be executed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "escalation_id": {
                        "type": "string",
                        "description": (
                            "The escalation ID returned by the "
                            "ESCALATION_REQUIRED response."
                        ),
                    },
                },
                "required": ["escalation_id"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_tool(
    prefixed_name: str, tool_dict: dict[str, Any],
) -> types.Tool:
    """Convert a downstream tool dict into an MCP ``Tool`` object.

    Preserves all schema fields from the downstream tool — only the name
    is replaced with the prefixed gateway name.
    """
    kwargs: dict[str, Any] = {
        "name": prefixed_name,
        "inputSchema": tool_dict["inputSchema"],
    }
    if "description" in tool_dict:
        kwargs["description"] = tool_dict["description"]
    if "outputSchema" in tool_dict:
        kwargs["outputSchema"] = tool_dict["outputSchema"]
    if "title" in tool_dict:
        kwargs["title"] = tool_dict["title"]
    if "annotations" in tool_dict:
        kwargs["annotations"] = types.ToolAnnotations(
            **tool_dict["annotations"],
        )
    return types.Tool(**kwargs)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_gateway() -> None:
    """Parse ``--config`` and run the gateway on stdio.

    Uses :func:`sanna.gateway.config.load_gateway_config` for validated
    config parsing with env var interpolation, path expansion, and
    fail-fast validation.
    """
    import argparse
    import sys

    from sanna.gateway.config import (
        GatewayConfigError,
        build_policy_overrides,
        load_gateway_config,
    )

    parser = argparse.ArgumentParser(
        prog="sanna-gateway",
        description="Sanna MCP enforcement proxy",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to gateway YAML config file",
    )
    args = parser.parse_args()

    try:
        config = load_gateway_config(args.config)
    except GatewayConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # First downstream server (multi-server is Block D foundation)
    ds = config.downstreams[0]
    policy_overrides = build_policy_overrides(ds)

    gateway = SannaGateway(
        server_name=ds.name,
        command=ds.command,
        args=ds.args,
        env=ds.env,
        timeout=ds.timeout,
        constitution_path=config.constitution_path,
        signing_key_path=config.signing_key_path,
        policy_overrides=policy_overrides,
        default_policy=ds.default_policy,
        escalation_timeout=config.escalation_timeout,
        receipt_store_path=config.receipt_store or None,
    )

    import asyncio
    asyncio.run(gateway.run_stdio())
