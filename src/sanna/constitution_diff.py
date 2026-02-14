"""
Sanna Constitution Diffing — structural/semantic comparison.

Compares two parsed Constitution objects field by field, reporting
governance-meaningful changes (invariants, enforcement levels, authority
boundaries, trusted sources, identity, halt conditions, provenance).

This is NOT a line-by-line text diff. It reports what changed in
governance terms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from .constitution import Constitution, compute_constitution_hash


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class DiffEntry:
    """A single change between two constitutions."""
    category: str       # "invariant", "enforcement", "authority", "trust",
                        # "identity", "halt", "provenance"
    change_type: str    # "added", "removed", "modified"
    key: str            # What changed (invariant ID, boundary name, etc.)
    old_value: Optional[str] = None
    new_value: Optional[str] = None


@dataclass
class DiffResult:
    """Structured result of comparing two constitutions."""
    entries: list[DiffEntry] = field(default_factory=list)
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None

    @property
    def has_changes(self) -> bool:
        """True if there are any differences."""
        return len(self.entries) > 0

    @property
    def summary(self) -> str:
        """Human-readable summary line."""
        counts: dict[str, dict[str, int]] = {}
        for entry in self.entries:
            cat = counts.setdefault(entry.category, {})
            cat[entry.change_type] = cat.get(entry.change_type, 0) + 1

        parts = []
        for category, changes in counts.items():
            for change_type, count in sorted(changes.items()):
                parts.append(f"{count} {category} {change_type}")
        return ", ".join(parts) if parts else "No changes"

    def to_dict(self) -> dict:
        """Structured output for programmatic use."""
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "changes": [asdict(e) for e in self.entries],
            "summary": self.summary,
        }

    def to_text(self) -> str:
        """Human-readable text output."""
        lines = []
        header = "Constitution Diff"
        if self.old_version or self.new_version:
            header += f": {self.old_version or '?'} → {self.new_version or '?'}"
        lines.append(header)
        lines.append("=" * len(header))

        if not self.entries:
            lines.append("No changes detected.")
            return "\n".join(lines)

        # Group by category
        by_category: dict[str, list[DiffEntry]] = {}
        for entry in self.entries:
            by_category.setdefault(entry.category, []).append(entry)

        category_labels = {
            "invariant": "INVARIANTS",
            "enforcement": "ENFORCEMENT CHANGES",
            "authority": "AUTHORITY BOUNDARIES",
            "trust": "TRUSTED SOURCES",
            "identity": "IDENTITY",
            "halt": "HALT CONDITIONS",
            "provenance": "PROVENANCE",
        }

        for cat_key, label in category_labels.items():
            if cat_key not in by_category:
                continue
            lines.append("")
            lines.append(label + ":")
            for entry in by_category[cat_key]:
                if entry.change_type == "added":
                    lines.append(f"  + Added: {entry.key}" +
                                 (f" ({entry.new_value})" if entry.new_value else ""))
                elif entry.change_type == "removed":
                    lines.append(f"  - Removed: {entry.key}" +
                                 (f" ({entry.old_value})" if entry.old_value else ""))
                elif entry.change_type == "modified":
                    lines.append(f"  ~ {entry.key}: {entry.old_value} → {entry.new_value}")

        lines.append("")
        lines.append(f"SUMMARY: {self.summary}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Markdown table output."""
        lines = []
        header = "# Constitution Diff"
        if self.old_version or self.new_version:
            header += f": {self.old_version or '?'} → {self.new_version or '?'}"
        lines.append(header)
        lines.append("")

        if not self.entries:
            lines.append("No changes detected.")
            return "\n".join(lines)

        lines.append("| Category | Change | Key | Old Value | New Value |")
        lines.append("|----------|--------|-----|-----------|-----------|")
        for entry in self.entries:
            old = entry.old_value or ""
            new = entry.new_value or ""
            lines.append(f"| {entry.category} | {entry.change_type} | {entry.key} | {old} | {new} |")

        lines.append("")
        lines.append(f"**Summary:** {self.summary}")
        return "\n".join(lines)


# =============================================================================
# DIFF LOGIC
# =============================================================================

def diff_constitutions(old: Constitution, new: Constitution) -> DiffResult:
    """Compare two constitutions and return a structured diff.

    Compares parsed Constitution objects field by field, reporting
    governance-meaningful changes.

    Args:
        old: The older (baseline) constitution.
        new: The newer (target) constitution.

    Returns:
        DiffResult with all detected changes.
    """
    entries: list[DiffEntry] = []

    # Version info
    old_version = None
    new_version = None
    if old.approval and old.approval.current:
        old_version = old.approval.current.constitution_version
    if new.approval and new.approval.current:
        new_version = new.approval.current.constitution_version

    # Hashes
    old_hash = old.policy_hash or compute_constitution_hash(old)
    new_hash = new.policy_hash or compute_constitution_hash(new)

    # --- Identity ---
    _diff_identity(old, new, entries)

    # --- Invariants ---
    _diff_invariants(old, new, entries)

    # --- Authority Boundaries ---
    _diff_authority(old, new, entries)

    # --- Trusted Sources ---
    _diff_trusted_sources(old, new, entries)

    # --- Halt Conditions ---
    _diff_halt_conditions(old, new, entries)

    # --- Provenance ---
    _diff_provenance(old, new, entries)

    return DiffResult(
        entries=entries,
        old_version=old_version,
        new_version=new_version,
        old_hash=old_hash,
        new_hash=new_hash,
    )


def _diff_identity(old: Constitution, new: Constitution, entries: list[DiffEntry]) -> None:
    """Compare identity fields."""
    if old.identity.agent_name != new.identity.agent_name:
        entries.append(DiffEntry(
            category="identity", change_type="modified", key="agent_name",
            old_value=old.identity.agent_name, new_value=new.identity.agent_name,
        ))
    if old.identity.domain != new.identity.domain:
        entries.append(DiffEntry(
            category="identity", change_type="modified", key="domain",
            old_value=old.identity.domain, new_value=new.identity.domain,
        ))
    old_desc = old.identity.description or ""
    new_desc = new.identity.description or ""
    if old_desc != new_desc:
        entries.append(DiffEntry(
            category="identity", change_type="modified", key="description",
            old_value=old_desc or "(none)", new_value=new_desc or "(none)",
        ))


def _diff_invariants(old: Constitution, new: Constitution, entries: list[DiffEntry]) -> None:
    """Compare invariants — added, removed, enforcement changes."""
    old_map = {inv.id: inv for inv in old.invariants}
    new_map = {inv.id: inv for inv in new.invariants}

    old_ids = set(old_map.keys())
    new_ids = set(new_map.keys())

    for inv_id in sorted(new_ids - old_ids):
        inv = new_map[inv_id]
        entries.append(DiffEntry(
            category="invariant", change_type="added", key=inv_id,
            new_value=f"enforcement: {inv.enforcement}",
        ))

    for inv_id in sorted(old_ids - new_ids):
        entries.append(DiffEntry(
            category="invariant", change_type="removed", key=inv_id,
        ))

    for inv_id in sorted(old_ids & new_ids):
        old_inv = old_map[inv_id]
        new_inv = new_map[inv_id]
        if old_inv.enforcement != new_inv.enforcement:
            entries.append(DiffEntry(
                category="enforcement", change_type="modified", key=inv_id,
                old_value=old_inv.enforcement, new_value=new_inv.enforcement,
            ))
        if old_inv.rule != new_inv.rule:
            entries.append(DiffEntry(
                category="invariant", change_type="modified", key=f"{inv_id} rule",
                old_value=old_inv.rule, new_value=new_inv.rule,
            ))


def _diff_authority(old: Constitution, new: Constitution, entries: list[DiffEntry]) -> None:
    """Compare authority boundaries."""
    old_auth = old.authority_boundaries
    new_auth = new.authority_boundaries
    if old_auth is None and new_auth is None:
        return
    if old_auth is None:
        old_auth_dict: dict = {}
    else:
        old_auth_dict = {
            "cannot_execute": old_auth.cannot_execute or [],
            "must_escalate": old_auth.must_escalate or [],
            "can_execute": old_auth.can_execute or [],
        }
    if new_auth is None:
        new_auth_dict: dict = {}
    else:
        new_auth_dict = {
            "cannot_execute": new_auth.cannot_execute or [],
            "must_escalate": new_auth.must_escalate or [],
            "can_execute": new_auth.can_execute or [],
        }

    def _rule_key(item) -> str:
        """Stable string representation of an authority rule, preserving target info."""
        if hasattr(item, "condition") and hasattr(item, "target"):
            target = item.target or ""
            if target:
                return f"{item.condition} → {target}"
            return item.condition
        if hasattr(item, "condition"):
            return item.condition
        return str(item)

    for boundary_type in ("cannot_execute", "must_escalate", "can_execute"):
        old_items = old_auth_dict.get(boundary_type, [])
        new_items = new_auth_dict.get(boundary_type, [])
        old_set = set(_rule_key(item) for item in old_items)
        new_set = set(_rule_key(item) for item in new_items)
        for action in sorted(new_set - old_set):
            entries.append(DiffEntry(
                category="authority", change_type="added",
                key=f"{boundary_type}: {action}",
            ))
        for action in sorted(old_set - new_set):
            entries.append(DiffEntry(
                category="authority", change_type="removed",
                key=f"{boundary_type}: {action}",
            ))


def _diff_trusted_sources(old: Constitution, new: Constitution, entries: list[DiffEntry]) -> None:
    """Compare trusted source tiers."""
    old_ts = old.trusted_sources
    new_ts = new.trusted_sources
    if old_ts is None and new_ts is None:
        return

    for tier in ("tier_1", "tier_2", "tier_3", "untrusted"):
        old_list = set(getattr(old_ts, tier, []) or []) if old_ts else set()
        new_list = set(getattr(new_ts, tier, []) or []) if new_ts else set()
        for source in sorted(new_list - old_list):
            entries.append(DiffEntry(
                category="trust", change_type="added",
                key=f"{tier}: {source}",
            ))
        for source in sorted(old_list - new_list):
            entries.append(DiffEntry(
                category="trust", change_type="removed",
                key=f"{tier}: {source}",
            ))


def _diff_halt_conditions(old: Constitution, new: Constitution, entries: list[DiffEntry]) -> None:
    """Compare halt conditions."""
    old_map = {hc.id: hc for hc in (old.halt_conditions or [])}
    new_map = {hc.id: hc for hc in (new.halt_conditions or [])}

    old_ids = set(old_map.keys())
    new_ids = set(new_map.keys())

    for hc_id in sorted(new_ids - old_ids):
        hc = new_map[hc_id]
        entries.append(DiffEntry(
            category="halt", change_type="added", key=hc_id,
            new_value=hc.trigger,
        ))

    for hc_id in sorted(old_ids - new_ids):
        entries.append(DiffEntry(
            category="halt", change_type="removed", key=hc_id,
        ))

    for hc_id in sorted(old_ids & new_ids):
        old_hc = old_map[hc_id]
        new_hc = new_map[hc_id]
        if old_hc.severity != new_hc.severity:
            entries.append(DiffEntry(
                category="halt", change_type="modified",
                key=f"{hc_id} severity",
                old_value=old_hc.severity, new_value=new_hc.severity,
            ))
        if old_hc.trigger != new_hc.trigger:
            entries.append(DiffEntry(
                category="halt", change_type="modified",
                key=f"{hc_id} trigger",
                old_value=old_hc.trigger, new_value=new_hc.trigger,
            ))
        if getattr(old_hc, "escalate_to", None) != getattr(new_hc, "escalate_to", None):
            entries.append(DiffEntry(
                category="halt", change_type="modified",
                key=f"{hc_id} escalate_to",
                old_value=getattr(old_hc, "escalate_to", None),
                new_value=getattr(new_hc, "escalate_to", None),
            ))
        if getattr(old_hc, "enforcement", None) != getattr(new_hc, "enforcement", None):
            entries.append(DiffEntry(
                category="halt", change_type="modified",
                key=f"{hc_id} enforcement",
                old_value=getattr(old_hc, "enforcement", None),
                new_value=getattr(new_hc, "enforcement", None),
            ))


def _diff_provenance(old: Constitution, new: Constitution, entries: list[DiffEntry]) -> None:
    """Compare provenance fields."""
    if old.provenance.authored_by != new.provenance.authored_by:
        entries.append(DiffEntry(
            category="provenance", change_type="modified", key="authored_by",
            old_value=old.provenance.authored_by, new_value=new.provenance.authored_by,
        ))
