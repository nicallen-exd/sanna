"""Tests for constitution diffing (Block 4, v0.9.0).

Covers diff_constitutions(), DiffResult formatting, and the sanna-diff CLI.
"""

import json
from pathlib import Path

import pytest

from sanna.constitution import (
    Constitution,
    AgentIdentity,
    Provenance,
    Boundary,
    Invariant,
    HaltCondition,
    AuthorityBoundaries,
    EscalationRule,
    TrustedSources,
    ApprovalRecord,
    ApprovalChain,
    save_constitution,
    sign_constitution,
)
from sanna.constitution_diff import diff_constitutions, DiffResult, DiffEntry
from sanna.crypto import generate_keypair


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def base_constitution():
    """A base constitution for diffing."""
    return Constitution(
        schema_version="0.1.0",
        identity=AgentIdentity(agent_name="support-bot-v2", domain="customer-support"),
        provenance=Provenance(
            authored_by="nic@company.com",
            approved_by=["jane@company.com"],
            approval_date="2026-01-01",
            approval_method="manual",
        ),
        boundaries=[
            Boundary(id="B001", description="Scope boundary", category="scope", severity="high"),
        ],
        invariants=[
            Invariant(id="INV_NO_FABRICATION", rule="No fabrication", enforcement="halt"),
            Invariant(id="INV_MARK_INFERENCE", rule="Mark inference", enforcement="warn"),
            Invariant(id="INV_LEGACY_CHECK", rule="Legacy check", enforcement="log"),
        ],
    )


@pytest.fixture
def modified_constitution():
    """A modified constitution with several governance changes."""
    return Constitution(
        schema_version="0.1.0",
        identity=AgentIdentity(agent_name="support-bot-v3", domain="customer-support-premium"),
        provenance=Provenance(
            authored_by="nic@company.com",
            approved_by=["jane@company.com"],
            approval_date="2026-02-01",
            approval_method="manual",
        ),
        boundaries=[
            Boundary(id="B001", description="Scope boundary", category="scope", severity="high"),
        ],
        invariants=[
            Invariant(id="INV_NO_FABRICATION", rule="No fabrication", enforcement="halt"),
            Invariant(id="INV_MARK_INFERENCE", rule="Mark inference", enforcement="halt"),  # warn → halt
            Invariant(id="INV_DATA_RETENTION", rule="Data retention", enforcement="halt"),  # new
        ],
        halt_conditions=[
            HaltCondition(id="H001", trigger="Production DB access", escalate_to="ops@co.com", severity="critical", enforcement="halt"),
        ],
    )


# =============================================================================
# Core diff functionality
# =============================================================================

class TestDiffConstitutions:
    def test_identical_constitutions_no_changes(self, base_constitution):
        """Diffing identical constitutions produces no entries."""
        result = diff_constitutions(base_constitution, base_constitution)
        assert result.entries == []
        assert result.has_changes is False
        assert result.summary == "No changes"

    def test_added_invariant_detected(self, base_constitution, modified_constitution):
        """Newly added invariants are detected."""
        result = diff_constitutions(base_constitution, modified_constitution)
        added = [e for e in result.entries if e.category == "invariant" and e.change_type == "added"]
        assert any(e.key == "INV_DATA_RETENTION" for e in added)

    def test_removed_invariant_detected(self, base_constitution, modified_constitution):
        """Removed invariants are detected."""
        result = diff_constitutions(base_constitution, modified_constitution)
        removed = [e for e in result.entries if e.category == "invariant" and e.change_type == "removed"]
        assert any(e.key == "INV_LEGACY_CHECK" for e in removed)

    def test_enforcement_change_detected(self, base_constitution, modified_constitution):
        """Enforcement level changes are detected."""
        result = diff_constitutions(base_constitution, modified_constitution)
        enforcement = [e for e in result.entries if e.category == "enforcement"]
        assert any(e.key == "INV_MARK_INFERENCE" and e.old_value == "warn" and e.new_value == "halt"
                    for e in enforcement)

    def test_identity_changes_detected(self, base_constitution, modified_constitution):
        """Identity field changes are detected."""
        result = diff_constitutions(base_constitution, modified_constitution)
        identity = [e for e in result.entries if e.category == "identity"]
        assert any(e.key == "agent_name" for e in identity)
        assert any(e.key == "domain" for e in identity)

    def test_halt_condition_added(self, base_constitution, modified_constitution):
        """Added halt conditions are detected."""
        result = diff_constitutions(base_constitution, modified_constitution)
        halt = [e for e in result.entries if e.category == "halt" and e.change_type == "added"]
        assert any(e.key == "H001" for e in halt)

    def test_halt_condition_removed(self):
        """Removed halt conditions are detected."""
        old = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
            halt_conditions=[
                HaltCondition(id="H001", trigger="Bad action", escalate_to="ops@co.com", severity="high", enforcement="halt"),
            ],
        )
        new = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
        )
        result = diff_constitutions(old, new)
        removed = [e for e in result.entries if e.category == "halt" and e.change_type == "removed"]
        assert any(e.key == "H001" for e in removed)

    def test_halt_condition_modified(self):
        """Modified halt condition severity is detected."""
        hc = HaltCondition(id="H001", trigger="Bad action", escalate_to="ops@co.com", severity="medium", enforcement="halt")
        old = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
            halt_conditions=[hc],
        )
        new = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
            halt_conditions=[HaltCondition(id="H001", trigger="Bad action", escalate_to="ops@co.com", severity="critical", enforcement="halt")],
        )
        result = diff_constitutions(old, new)
        modified = [e for e in result.entries if e.category == "halt" and e.change_type == "modified"]
        assert any("severity" in e.key and e.old_value == "medium" and e.new_value == "critical" for e in modified)

    def test_authority_boundary_added(self):
        """Added authority boundaries are detected."""
        old = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
        )
        new = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
            authority_boundaries=AuthorityBoundaries(
                cannot_execute=["delete_user_data"],
                must_escalate=["price_override"],
            ),
        )
        result = diff_constitutions(old, new)
        authority = [e for e in result.entries if e.category == "authority"]
        assert any("cannot_execute" in e.key and "delete_user_data" in e.key for e in authority)
        assert any("must_escalate" in e.key and "price_override" in e.key for e in authority)

    def test_authority_boundary_removed(self):
        """Removed authority boundaries are detected."""
        old = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
            authority_boundaries=AuthorityBoundaries(must_escalate=["legal_threat"]),
        )
        new = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
        )
        result = diff_constitutions(old, new)
        removed = [e for e in result.entries if e.category == "authority" and e.change_type == "removed"]
        assert any("legal_threat" in e.key for e in removed)

    def test_trusted_source_changes(self):
        """Trusted source tier changes are detected."""
        old = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
            trusted_sources=TrustedSources(
                tier_1=["old-source.com"],
                tier_2=["legacy-wiki.com"],
            ),
        )
        new = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
            trusted_sources=TrustedSources(
                tier_1=["old-source.com", "new-source.com"],
            ),
        )
        result = diff_constitutions(old, new)
        trust = [e for e in result.entries if e.category == "trust"]
        assert any("new-source.com" in e.key and e.change_type == "added" for e in trust)
        assert any("legacy-wiki.com" in e.key and e.change_type == "removed" for e in trust)

    def test_provenance_changes_detected(self):
        """Provenance authored_by changes are detected."""
        old = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="old@co.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
        )
        new = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="new@co.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
        )
        result = diff_constitutions(old, new)
        prov = [e for e in result.entries if e.category == "provenance"]
        assert any(e.key == "authored_by" for e in prov)

    def test_multiple_changes_across_categories(self, base_constitution, modified_constitution):
        """Multiple changes across categories in a single diff."""
        result = diff_constitutions(base_constitution, modified_constitution)
        categories = {e.category for e in result.entries}
        assert len(categories) >= 3  # identity, invariant/enforcement, halt

    def test_summary_line_accurate(self, base_constitution, modified_constitution):
        """Summary line contains counts matching actual changes."""
        result = diff_constitutions(base_constitution, modified_constitution)
        summary = result.summary
        assert "added" in summary or "removed" in summary or "modified" in summary
        assert result.has_changes is True


# =============================================================================
# Output formatting
# =============================================================================

class TestDiffResultFormatting:
    def test_to_dict_round_trips(self, base_constitution, modified_constitution):
        """DiffResult.to_dict() produces valid JSON that captures all changes."""
        result = diff_constitutions(base_constitution, modified_constitution)
        d = result.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert "changes" in parsed
        assert "summary" in parsed
        assert len(parsed["changes"]) == len(result.entries)

    def test_json_output_valid_structure(self, base_constitution, modified_constitution):
        """JSON output has correct structure."""
        result = diff_constitutions(base_constitution, modified_constitution)
        d = result.to_dict()
        assert "old_version" in d
        assert "new_version" in d
        assert "old_hash" in d
        assert "new_hash" in d
        assert isinstance(d["changes"], list)
        for change in d["changes"]:
            assert "category" in change
            assert "change_type" in change
            assert "key" in change

    def test_text_output_valid(self, base_constitution, modified_constitution):
        """Text output contains section headers."""
        result = diff_constitutions(base_constitution, modified_constitution)
        text = result.to_text()
        assert "Constitution Diff" in text
        assert "SUMMARY:" in text

    def test_markdown_output_valid(self, base_constitution, modified_constitution):
        """Markdown output is a valid markdown table."""
        result = diff_constitutions(base_constitution, modified_constitution)
        md = result.to_markdown()
        assert "# Constitution Diff" in md
        assert "| Category |" in md
        assert "**Summary:**" in md

    def test_empty_diff_output(self, base_constitution):
        """Empty diff produces clean output."""
        result = diff_constitutions(base_constitution, base_constitution)
        text = result.to_text()
        assert "No changes" in text
        d = result.to_dict()
        assert d["changes"] == []


# =============================================================================
# CLI entry point
# =============================================================================

class TestDiffCLI:
    @pytest.fixture
    def two_constitutions(self, tmp_path):
        """Create two constitution files on disk for CLI testing."""
        old = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="cli-agent-v1", domain="testing"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[Boundary(id="B001", description="Test", category="scope", severity="high")],
            invariants=[
                Invariant(id="INV_NO_FABRICATION", rule="No fabrication", enforcement="warn"),
            ],
        )
        new = Constitution(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="cli-agent-v2", domain="testing"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[Boundary(id="B001", description="Test", category="scope", severity="high")],
            invariants=[
                Invariant(id="INV_NO_FABRICATION", rule="No fabrication", enforcement="halt"),
                Invariant(id="INV_NEW_CHECK", rule="New check", enforcement="warn"),
            ],
        )
        old_path = tmp_path / "old.yaml"
        new_path = tmp_path / "new.yaml"
        save_constitution(old, old_path)
        save_constitution(new, new_path)
        return old_path, new_path

    def test_cli_text_output(self, two_constitutions, monkeypatch):
        """CLI with text format succeeds."""
        from sanna.cli import diff_cmd
        old_path, new_path = two_constitutions
        monkeypatch.setattr("sys.argv", ["sanna-diff", str(old_path), str(new_path)])
        result = diff_cmd()
        assert result == 0

    def test_cli_json_output(self, two_constitutions, monkeypatch, capsys):
        """CLI with --format json produces valid JSON."""
        from sanna.cli import diff_cmd
        old_path, new_path = two_constitutions
        monkeypatch.setattr("sys.argv", [
            "sanna-diff", str(old_path), str(new_path), "--format", "json"
        ])
        diff_cmd()
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert "changes" in parsed
        assert len(parsed["changes"]) > 0

    def test_cli_markdown_output(self, two_constitutions, monkeypatch, capsys):
        """CLI with --format markdown produces markdown."""
        from sanna.cli import diff_cmd
        old_path, new_path = two_constitutions
        monkeypatch.setattr("sys.argv", [
            "sanna-diff", str(old_path), str(new_path), "--format", "markdown"
        ])
        diff_cmd()
        output = capsys.readouterr().out
        assert "# Constitution Diff" in output

    def test_cli_file_not_found(self, tmp_path, monkeypatch):
        """CLI with missing file returns error."""
        from sanna.cli import diff_cmd
        monkeypatch.setattr("sys.argv", [
            "sanna-diff", str(tmp_path / "missing.yaml"), str(tmp_path / "also_missing.yaml")
        ])
        result = diff_cmd()
        assert result == 1


# =============================================================================
# HIGH-4: Authority-boundary diffing preserves EscalationRule target info
# =============================================================================

class TestAuthorityDiffTargetInfo:
    def _base(self):
        return dict(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
        )

    def test_escalation_rule_target_in_diff_key(self):
        """EscalationRule target info appears in diff key."""
        old = Constitution(**self._base())
        new = Constitution(
            **self._base(),
            authority_boundaries=AuthorityBoundaries(
                must_escalate=[EscalationRule(condition="refund > $500", target="manager")],
            ),
        )
        result = diff_constitutions(old, new)
        authority = [e for e in result.entries if e.category == "authority"]
        assert len(authority) >= 1
        assert any("refund > $500" in e.key and "manager" in e.key for e in authority)

    def test_different_targets_same_condition_are_distinct(self):
        """Changing only the target is detected as a diff."""
        old = Constitution(
            **self._base(),
            authority_boundaries=AuthorityBoundaries(
                must_escalate=[EscalationRule(condition="refund > $500", target="team_lead")],
            ),
        )
        new = Constitution(
            **self._base(),
            authority_boundaries=AuthorityBoundaries(
                must_escalate=[EscalationRule(condition="refund > $500", target="vp_finance")],
            ),
        )
        result = diff_constitutions(old, new)
        authority = [e for e in result.entries if e.category == "authority"]
        added = [e for e in authority if e.change_type == "added"]
        removed = [e for e in authority if e.change_type == "removed"]
        assert len(added) >= 1
        assert len(removed) >= 1
        assert any("vp_finance" in e.key for e in added)
        assert any("team_lead" in e.key for e in removed)

    def test_plain_string_authority_still_works(self):
        """Plain string authority boundaries still work correctly."""
        old = Constitution(**self._base())
        new = Constitution(
            **self._base(),
            authority_boundaries=AuthorityBoundaries(
                cannot_execute=["delete_data"],
                can_execute=["read_logs"],
            ),
        )
        result = diff_constitutions(old, new)
        authority = [e for e in result.entries if e.category == "authority"]
        assert any("delete_data" in e.key for e in authority)
        assert any("read_logs" in e.key for e in authority)


# =============================================================================
# MEDIUM-1: Halt-condition diffing includes escalate_to and enforcement
# =============================================================================

class TestHaltConditionDiffComplete:
    def _base(self):
        return dict(
            schema_version="0.1.0",
            identity=AgentIdentity(agent_name="test", domain="test"),
            provenance=Provenance(authored_by="a@a.com", approved_by=["b@b.com"],
                                   approval_date="2026-01-01", approval_method="manual"),
            boundaries=[],
        )

    def test_escalate_to_change_detected(self):
        """Change in halt condition escalate_to is detected."""
        old = Constitution(
            **self._base(),
            halt_conditions=[HaltCondition(id="H001", trigger="C1 fails", escalate_to="team_lead", severity="critical", enforcement="halt")],
        )
        new = Constitution(
            **self._base(),
            halt_conditions=[HaltCondition(id="H001", trigger="C1 fails", escalate_to="ciso", severity="critical", enforcement="halt")],
        )
        result = diff_constitutions(old, new)
        halt = [e for e in result.entries if e.category == "halt"]
        assert any("escalate_to" in e.key and e.old_value == "team_lead" and e.new_value == "ciso" for e in halt)

    def test_enforcement_change_detected(self):
        """Change in halt condition enforcement is detected."""
        old = Constitution(
            **self._base(),
            halt_conditions=[HaltCondition(id="H001", trigger="C1 fails", escalate_to="lead", severity="critical", enforcement="halt")],
        )
        new = Constitution(
            **self._base(),
            halt_conditions=[HaltCondition(id="H001", trigger="C1 fails", escalate_to="lead", severity="critical", enforcement="warn")],
        )
        result = diff_constitutions(old, new)
        halt = [e for e in result.entries if e.category == "halt"]
        assert any("enforcement" in e.key and e.old_value == "halt" and e.new_value == "warn" for e in halt)

    def test_no_diff_when_same(self):
        """No diff when halt conditions are identical."""
        hc = [HaltCondition(id="H001", trigger="C1 fails", escalate_to="lead", severity="critical", enforcement="halt")]
        old = Constitution(**self._base(), halt_conditions=hc)
        new = Constitution(**self._base(), halt_conditions=list(hc))
        result = diff_constitutions(old, new)
        halt = [e for e in result.entries if e.category == "halt"]
        assert len(halt) == 0
