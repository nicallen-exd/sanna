#!/usr/bin/env python3
"""Three Constitutions, Three Behaviors — Sanna v0.6.0 Demo

Same input, three different constitutions, three different enforcement outcomes.
This proves the constitution is the control plane for the check engine.

Run: python examples/three_constitutions_demo.py
"""

import json
import sys
import warnings
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanna.middleware import sanna_observe, SannaHaltError, SannaResult

# =============================================================================
# SHARED INPUT — identical for all three constitutions
# =============================================================================

# This context has conflicting policies (triggers C4) and conditions (triggers C3)
SHARED_CONTEXT = (
    "Our refund policy: Physical products can be returned within 30 days. "
    "Digital products are non-refundable once downloaded. "
    "Subscriptions can be cancelled anytime. "
    "If the product is defective, exceptions may apply. "
    "However, all exceptions require manager approval."
)

SHARED_QUERY = "Can I get a refund on my software purchase?"

# This output contradicts the non-refundable policy (triggers C1),
# uses definitive language without hedging (triggers C2),
# is confident without acknowledging conditions (triggers C3),
# collapses the policy tension (triggers C4),
# and compresses a complex policy into a brief answer (triggers C5).
SHARED_OUTPUT = (
    "Based on your purchase history, you are eligible to request a refund. "
    "However, since the software was downloaded, processing may take 5-7 "
    "business days."
)

# =============================================================================
# CONSTITUTION PATHS
# =============================================================================

CONSTITUTIONS_DIR = Path(__file__).parent / "constitutions"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_demo():
    """Run the three constitutions demo."""
    print("=" * 72)
    print("  Three Constitutions, Three Behaviors — Sanna v0.6.0")
    print("=" * 72)
    print()
    print(f"Query:   {SHARED_QUERY}")
    print(f"Context: {SHARED_CONTEXT[:80]}...")
    print(f"Output:  {SHARED_OUTPUT[:80]}...")
    print()

    constitutions = [
        ("strict_financial_analyst", "Strict Financial Analyst — all invariants at HALT"),
        ("permissive_support_agent", "Permissive Support Agent — 2 invariants at WARN"),
        ("research_assistant", "Research Assistant — C1 HALT, rest LOG"),
    ]

    for name, description in constitutions:
        print("-" * 72)
        print(f"  {description}")
        print("-" * 72)

        const_path = str(CONSTITUTIONS_DIR / f"{name}.yaml")

        @sanna_observe(constitution_path=const_path)
        def agent(query: str, context: str) -> str:
            return SHARED_OUTPUT

        receipt = None
        halted = False

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = agent(query=SHARED_QUERY, context=SHARED_CONTEXT)
                receipt = result.receipt

                if w:
                    for warning in w:
                        if "Sanna" in str(warning.message):
                            print(f"  WARNING: {warning.message}")

        except SannaHaltError as e:
            receipt = e.receipt
            halted = True
            print(f"  HALTED: {e}")

        if receipt:
            # Print summary
            checks = receipt.get("checks", [])
            standard_checks = [c for c in checks if c.get("status") != "NOT_CHECKED"]
            custom_checks = [c for c in checks if c.get("status") == "NOT_CHECKED"]

            print(f"\n  Status: {receipt['coherence_status']}")
            print(f"  Enforcement: {'HALTED' if halted else 'continued'}")
            print(f"  Checks run: {len(standard_checks)}")
            print(f"  Custom invariants (NOT_CHECKED): {len(custom_checks)}")

            print("\n  Check Results:")
            for c in checks:
                status_icon = "PASS" if c["passed"] else "FAIL"
                if c.get("status") == "NOT_CHECKED":
                    status_icon = "N/A"
                enforcement = c.get("enforcement_level", "?")
                triggered = c.get("triggered_by", "?")
                print(f"    [{status_icon}] {c['check_id']:6s} enforcement={enforcement:4s}  triggered_by={triggered}")

            # Write receipt to output dir
            output_path = OUTPUT_DIR / f"{name}_receipt.json"
            with open(output_path, "w") as f:
                json.dump(receipt, f, indent=2)
            print(f"\n  Receipt saved: {output_path}")

        print()

    # Summary comparison
    print("=" * 72)
    print("  SUMMARY: Same input, different constitutions, different behavior")
    print("=" * 72)
    print()
    print("  strict_financial_analyst:  5 checks, all at HALT  -> HALTED on C1")
    print("  permissive_support_agent:  2 checks at WARN + 1 custom -> WARNED")
    print("  research_assistant:        5 checks, C1=HALT rest=LOG -> HALTED on C1")
    print()
    print("  The constitution IS the control plane.")
    print()


if __name__ == "__main__":
    run_demo()
