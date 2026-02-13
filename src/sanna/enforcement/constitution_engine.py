"""
Constitution enforcement engine — maps invariants to check functions.

Reads a constitution's invariants and produces a check configuration
that tells the existing check engine which checks to run and at what
enforcement level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..receipt import (
    check_c1_context_contradiction,
    check_c2_unmarked_inference,
    check_c3_false_certainty,
    check_c4_conflict_collapse,
    check_c5_premature_compression,
)


# =============================================================================
# INVARIANT → CHECK MAPPING
# =============================================================================

INVARIANT_CHECK_MAP: dict[str, tuple[str, Callable]] = {
    "INV_NO_FABRICATION": ("C1", check_c1_context_contradiction),
    "INV_MARK_INFERENCE": ("C2", check_c2_unmarked_inference),
    "INV_NO_FALSE_CERTAINTY": ("C3", check_c3_false_certainty),
    "INV_PRESERVE_TENSION": ("C4", check_c4_conflict_collapse),
    "INV_NO_PREMATURE_COMPRESSION": ("C5", check_c5_premature_compression),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CheckConfig:
    """Configuration for a single check derived from a constitution invariant."""
    check_id: str              # "C1"
    check_fn: Callable         # the actual check function
    enforcement_level: str     # "halt" | "warn" | "log"
    triggered_by: str          # "INV_NO_FABRICATION"


@dataclass
class CustomInvariantRecord:
    """Record for a custom invariant that has no evaluator."""
    invariant_id: str
    rule: str
    enforcement: str
    status: str = "NOT_CHECKED"
    reason: str = "Custom invariant — no evaluator registered."


# =============================================================================
# CONFIGURATION
# =============================================================================

def configure_checks(constitution) -> tuple[list[CheckConfig], list[CustomInvariantRecord]]:
    """Read constitution invariants, return configured checks and custom invariant records.

    Rules:
    - If invariant ID matches a standard INV_* → map to check function,
      use invariant's enforcement level
    - If invariant ID starts with INV_CUSTOM_* or doesn't match standard →
      record as NOT_CHECKED
    - If constitution has no invariants → return empty (no checks run)

    Args:
        constitution: A Constitution object with an invariants field.

    Returns:
        Tuple of (check_configs, custom_records).
    """
    check_configs: list[CheckConfig] = []
    custom_records: list[CustomInvariantRecord] = []

    for invariant in constitution.invariants:
        mapping = INVARIANT_CHECK_MAP.get(invariant.id)
        if mapping is not None:
            check_id, check_fn = mapping
            check_configs.append(CheckConfig(
                check_id=check_id,
                check_fn=check_fn,
                enforcement_level=invariant.enforcement,
                triggered_by=invariant.id,
            ))
        else:
            custom_records.append(CustomInvariantRecord(
                invariant_id=invariant.id,
                rule=invariant.rule,
                enforcement=invariant.enforcement,
            ))

    return check_configs, custom_records
