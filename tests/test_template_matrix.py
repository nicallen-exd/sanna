"""Template matrix tests — declarative enforcement tests for all 5 gateway templates.

Covers 10+ action categories across 5 templates:
  1. Financial transactions
  2. External communications
  3. File ops (in scope)
  4. File ops (out of scope)
  5. File deletion
  6. Sensitive data access
  7. Data exfiltration
  8. System configuration
  9. Calendar & scheduling
  10. Database/CRM writes
  11. Code & deployment
  12. Destructive operations
  13. Shared resources (team only)
"""

from pathlib import Path

import pytest

from sanna.constitution import load_constitution
from sanna.enforcement.authority import evaluate_authority

# ---------------------------------------------------------------------------
# Template paths
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent.parent / "examples" / "constitutions"

_TEMPLATE_PATHS = {
    "openclaw-personal": _TEMPLATES_DIR / "openclaw-personal.yaml",
    "openclaw-developer": _TEMPLATES_DIR / "openclaw-developer.yaml",
    "cowork-personal": _TEMPLATES_DIR / "cowork-personal.yaml",
    "cowork-team": _TEMPLATES_DIR / "cowork-team.yaml",
    "claude-code-standard": _TEMPLATES_DIR / "claude-code-standard.yaml",
}


# ---------------------------------------------------------------------------
# 1. Template loading tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template_name", list(_TEMPLATE_PATHS.keys()))
def test_template_loads(template_name):
    """Every template loads without errors via load_constitution()."""
    path = _TEMPLATE_PATHS[template_name]
    c = load_constitution(str(path))
    assert c.identity.agent_name
    assert c.authority_boundaries is not None
    assert len(c.invariants) >= 5  # C1-C5 at minimum


@pytest.mark.parametrize("template_name", list(_TEMPLATE_PATHS.keys()))
def test_template_has_authority_boundaries(template_name):
    """Every template defines all three authority boundary tiers."""
    c = load_constitution(str(_TEMPLATE_PATHS[template_name]))
    ab = c.authority_boundaries
    assert len(ab.can_execute) > 0
    assert len(ab.cannot_execute) > 0
    assert len(ab.must_escalate) > 0


# ---------------------------------------------------------------------------
# 2. Enforcement matrix tests
# ---------------------------------------------------------------------------

_ENFORCEMENT_MATRIX = [
    # ===================================================================
    # openclaw-personal: individual agents on personal machines
    # ===================================================================

    # File ops in scope → allow
    ("openclaw-personal", "read_file", "allow"),
    ("openclaw-personal", "write_file", "allow"),
    ("openclaw-personal", "search_files", "allow"),
    ("openclaw-personal", "summarize_document", "allow"),

    # External communications → escalate
    ("openclaw-personal", "send_email", "escalate"),
    ("openclaw-personal", "post_message", "escalate"),

    # File deletion → escalate
    ("openclaw-personal", "delete_file", "escalate"),

    # Calendar → escalate
    ("openclaw-personal", "create_event", "escalate"),

    # Database/CRM writes → escalate
    ("openclaw-personal", "update_page", "escalate"),

    # Financial → halt
    ("openclaw-personal", "purchase_item", "halt"),

    # Sensitive data → halt
    ("openclaw-personal", "read_credentials", "halt"),
    ("openclaw-personal", "read_pii", "halt"),

    # Data exfiltration → halt
    ("openclaw-personal", "upload_external", "halt"),
    ("openclaw-personal", "send_data", "halt"),

    # System configuration → halt
    ("openclaw-personal", "modify_settings", "halt"),

    # Code & deployment → halt
    ("openclaw-personal", "deploy_to_production", "halt"),

    # Destructive → halt
    ("openclaw-personal", "delete_repo", "halt"),
    ("openclaw-personal", "force_push", "halt"),

    # ===================================================================
    # openclaw-developer: strict containment for skill builders
    # ===================================================================

    # File ops within scope → allow
    ("openclaw-developer", "read_file", "allow"),
    ("openclaw-developer", "search_files", "allow"),

    # File deletion (even within scope) → escalate
    ("openclaw-developer", "delete_file", "escalate"),

    # Database/CRM writes → escalate
    ("openclaw-developer", "update_page", "escalate"),

    # Everything else → halt
    ("openclaw-developer", "send_email", "halt"),
    ("openclaw-developer", "purchase_item", "halt"),
    ("openclaw-developer", "read_credentials", "halt"),
    ("openclaw-developer", "upload_external", "halt"),
    ("openclaw-developer", "create_event", "halt"),
    ("openclaw-developer", "modify_settings", "halt"),
    ("openclaw-developer", "deploy_to_production", "halt"),
    ("openclaw-developer", "access_external_file", "halt"),
    ("openclaw-developer", "delete_repo", "halt"),

    # ===================================================================
    # cowork-personal: knowledge workers with Claude Desktop
    # ===================================================================

    # File ops and drafting → allow
    ("cowork-personal", "read_file", "allow"),
    ("cowork-personal", "search_files", "allow"),
    ("cowork-personal", "draft_response", "allow"),

    # External communications → escalate
    ("cowork-personal", "send_email", "escalate"),

    # File deletion → escalate
    ("cowork-personal", "delete_file", "escalate"),

    # Financial → escalate (knowledge workers manage expenses)
    ("cowork-personal", "purchase_item", "escalate"),
    ("cowork-personal", "transfer_funds", "escalate"),

    # Calendar → escalate
    ("cowork-personal", "create_event", "escalate"),

    # Database/CRM writes → escalate
    ("cowork-personal", "update_page", "escalate"),

    # Sensitive data → halt
    ("cowork-personal", "read_credentials", "halt"),
    ("cowork-personal", "read_pii", "halt"),

    # Data exfiltration → halt
    ("cowork-personal", "upload_external", "halt"),

    # Destructive → halt
    ("cowork-personal", "delete_repo", "halt"),

    # ===================================================================
    # cowork-team: shared MCP infrastructure
    # ===================================================================

    # File ops → allow
    ("cowork-team", "read_file", "allow"),
    ("cowork-team", "search_files", "allow"),

    # External communications → escalate
    ("cowork-team", "send_email", "escalate"),

    # Shared resources → escalate
    ("cowork-team", "shared_drive_write", "escalate"),
    ("cowork-team", "team_channel_post", "escalate"),

    # File deletion → escalate
    ("cowork-team", "delete_file", "escalate"),

    # Financial → escalate
    ("cowork-team", "purchase_item", "escalate"),

    # Team configuration → halt
    ("cowork-team", "modify_team_config", "halt"),
    ("cowork-team", "modify_access_control", "halt"),
    ("cowork-team", "modify_permissions", "halt"),

    # Sensitive data → halt
    ("cowork-team", "read_credentials", "halt"),

    # Data exfiltration → halt
    ("cowork-team", "upload_external", "halt"),

    # Destructive → halt
    ("cowork-team", "delete_repo", "halt"),

    # ===================================================================
    # claude-code-standard: developers with MCP connectors
    # ===================================================================

    # Code ops → allow
    ("claude-code-standard", "read_file", "allow"),
    ("claude-code-standard", "git_commit", "allow"),
    ("claude-code-standard", "run_tests", "allow"),

    # Git push to main → escalate
    ("claude-code-standard", "git_push_main", "escalate"),

    # External communications → escalate
    ("claude-code-standard", "send_email", "escalate"),

    # File deletion → escalate
    ("claude-code-standard", "delete_file", "escalate"),

    # Package publishing → escalate
    ("claude-code-standard", "npm_publish", "escalate"),

    # Staging DB writes → escalate
    ("claude-code-standard", "database_write_staging", "escalate"),

    # System configuration → escalate
    ("claude-code-standard", "modify_settings", "escalate"),

    # Calendar → escalate
    ("claude-code-standard", "create_event", "escalate"),

    # Production ops → halt
    ("claude-code-standard", "deploy_production", "halt"),
    ("claude-code-standard", "database_write_production", "halt"),

    # Sensitive data → halt
    ("claude-code-standard", "read_credentials", "halt"),

    # Destructive → halt
    ("claude-code-standard", "force_push", "halt"),
    ("claude-code-standard", "terraform_apply", "halt"),
    ("claude-code-standard", "delete_repo", "halt"),
]


@pytest.mark.parametrize(
    "template,action,expected",
    _ENFORCEMENT_MATRIX,
    ids=[f"{t}/{a}" for t, a, _ in _ENFORCEMENT_MATRIX],
)
def test_template_enforcement(template, action, expected):
    """Authority boundary evaluation matches expected enforcement level."""
    c = load_constitution(str(_TEMPLATE_PATHS[template]))
    decision = evaluate_authority(action, {}, c)
    assert decision.decision == expected, (
        f"{template}/{action}: expected {expected}, "
        f"got {decision.decision} (reason: {decision.reason})"
    )


# ---------------------------------------------------------------------------
# 3. Universal invariant tests
# ---------------------------------------------------------------------------

_UNIVERSAL_HALTS = [
    "read_credentials",
    "upload_external",
    "delete_repo",
]


@pytest.mark.parametrize("template_name", list(_TEMPLATE_PATHS.keys()))
@pytest.mark.parametrize("action", _UNIVERSAL_HALTS)
def test_universal_halt(template_name, action):
    """Sensitive data, exfiltration, and destructive ops are halted in ALL templates."""
    c = load_constitution(str(_TEMPLATE_PATHS[template_name]))
    decision = evaluate_authority(action, {}, c)
    assert decision.decision == "halt", (
        f"{template_name}/{action}: expected halt, "
        f"got {decision.decision} (reason: {decision.reason})"
    )


_UNIVERSAL_ALLOWS = [
    "read_file",
    "search_files",
]


@pytest.mark.parametrize("template_name", list(_TEMPLATE_PATHS.keys()))
@pytest.mark.parametrize("action", _UNIVERSAL_ALLOWS)
def test_universal_allow(template_name, action):
    """File reads and searches are allowed in ALL templates."""
    c = load_constitution(str(_TEMPLATE_PATHS[template_name]))
    decision = evaluate_authority(action, {}, c)
    assert decision.decision == "allow", (
        f"{template_name}/{action}: expected allow, "
        f"got {decision.decision} (reason: {decision.reason})"
    )
