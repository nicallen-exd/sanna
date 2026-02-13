#!/usr/bin/env python3
"""Three Constitutions, Three Behaviors — Sanna v0.6.4 Demo

Same input, three different constitutions, three different enforcement outcomes.
Full Ed25519 cryptographic provenance: keygen, sign, verify.

Run: python examples/three_constitutions_demo.py
"""

import json
import sys
import tempfile
import warnings
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanna.middleware import sanna_observe, SannaHaltError, SannaResult
from sanna.constitution import load_constitution, sign_constitution, save_constitution
from sanna.crypto import (
    generate_keypair,
    verify_constitution_full,
    verify_receipt_signature,
)
from sanna.verify import verify_constitution_chain

# =============================================================================
# SHARED INPUT — identical for all three constitutions
# =============================================================================

SHARED_CONTEXT = (
    "Our refund policy: Physical products can be returned within 30 days. "
    "Digital products are non-refundable once downloaded. "
    "Subscriptions can be cancelled anytime. "
    "If the product is defective, exceptions may apply. "
    "However, all exceptions require manager approval."
)

SHARED_QUERY = "Can I get a refund on my software purchase?"

SHARED_OUTPUT = (
    "Based on your purchase history, you are eligible to request a refund. "
    "However, since the software was downloaded, processing may take 5-7 "
    "business days."
)

# =============================================================================
# PATHS
# =============================================================================

CONSTITUTIONS_DIR = Path(__file__).parent / "constitutions"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_demo():
    """Run the three constitutions demo with full Ed25519 provenance."""
    print("=" * 72)
    print("  Three Constitutions, Three Behaviors — Sanna v0.6.4")
    print("=" * 72)
    print()
    print(f"Query:   {SHARED_QUERY}")
    print(f"Context: {SHARED_CONTEXT[:80]}...")
    print(f"Output:  {SHARED_OUTPUT[:80]}...")
    print()

    with tempfile.TemporaryDirectory(prefix="sanna_demo_") as tmp_dir:
        tmp = Path(tmp_dir)

        # 1. Generate Ed25519 keypair
        priv_path, pub_path = generate_keypair(tmp / "keys", signed_by="sanna-demo", write_metadata=True)
        print(f"  Keypair generated: {priv_path}")
        print(f"  Signed by: sanna-demo")
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

            # 2. Load unsigned source constitution
            source_path = CONSTITUTIONS_DIR / f"{name}.yaml"
            constitution = load_constitution(str(source_path), validate=True)

            # 3. Sign with Ed25519 at runtime
            signed = sign_constitution(
                constitution,
                private_key_path=str(priv_path),
                signed_by="sanna-demo",
            )

            # Write signed constitution to temp file
            signed_path = tmp / f"{name}.yaml"
            save_constitution(signed, signed_path)

            # Print constitution provenance
            print(f"  Constitution signed:")
            print(f"    policy_hash: {signed.policy_hash}")
            print(f"    key_id:      {signed.provenance.signature.key_id}")
            print(f"    signed_by:   {signed.provenance.signature.signed_by}")
            print()

            # 4. Run agent with signed constitution + receipt signing
            @sanna_observe(
                constitution_path=str(signed_path),
                private_key_path=str(priv_path),
            )
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
                # Print check summary
                checks = receipt.get("checks", [])
                standard_checks = [c for c in checks if c.get("status") != "NOT_CHECKED"]
                custom_checks = [c for c in checks if c.get("status") == "NOT_CHECKED"]
                coverage = receipt.get("evaluation_coverage", {})

                print(f"\n  Status: {receipt['coherence_status']}")
                print(f"  Enforcement: {'HALTED' if halted else 'continued'}")
                print(f"  Checks run: {len(standard_checks)}")
                print(f"  Custom invariants (NOT_CHECKED): {len(custom_checks)}")
                if coverage:
                    bp = coverage.get('coverage_basis_points', 0)
                    print(f"  Coverage: {coverage.get('evaluated', 0)}/{coverage.get('total_invariants', 0)} ({bp / 100:.1f}%)")

                print("\n  Check Results:")
                for c in checks:
                    status_icon = "PASS" if c["passed"] else "FAIL"
                    if c.get("status") == "NOT_CHECKED":
                        status_icon = "N/A"
                    enforcement = c.get("enforcement_level", "?")
                    check_impl = c.get("check_impl", "")
                    replayable = c.get("replayable", "")
                    print(f"    [{status_icon}] {c['check_id']:30s} enforcement={enforcement:4s}  impl={check_impl or 'none':30s}  replayable={replayable}")

                # Print receipt signature info
                sig_block = receipt.get("receipt_signature", {})
                if sig_block:
                    print(f"\n  Receipt signed:")
                    print(f"    key_id:    {sig_block.get('key_id', 'N/A')}")
                    print(f"    signed_by: {sig_block.get('signed_by', 'N/A')}")

                # 5. Verify provenance chain
                print(f"\n  Provenance verification:")

                # Verify constitution signature (full-document scheme)
                reloaded = load_constitution(str(signed_path))
                const_sig_valid = verify_constitution_full(reloaded, str(pub_path))
                print(f"    Constitution signature: {'valid' if const_sig_valid else 'INVALID'}")

                # Verify receipt signature
                receipt_sig_valid = verify_receipt_signature(receipt, str(pub_path))
                print(f"    Receipt signature:      {'valid' if receipt_sig_valid else 'INVALID'}")

                # Verify provenance bond
                const_ref = receipt.get("constitution_ref", {})
                bond_valid = const_ref.get("policy_hash") == signed.policy_hash
                print(f"    Provenance bond:        {'receipt.constitution_ref.policy_hash matches constitution' if bond_valid else 'MISMATCH'}")

                # Write receipt to output dir
                output_path = OUTPUT_DIR / f"{name}_receipt.json"
                with open(output_path, "w") as f:
                    json.dump(receipt, f, indent=2)
                print(f"\n  Receipt saved: {output_path}")

            print()

        # 6. Demonstrate wrong-key verification failure
        print("-" * 72)
        print("  Tamper Detection — wrong key verification")
        print("-" * 72)

        # Generate a second keypair (attacker/wrong key)
        wrong_priv, wrong_pub = generate_keypair(tmp / "wrong_keys")

        # Try to verify the last receipt with the wrong key
        last_receipt_path = OUTPUT_DIR / "research_assistant_receipt.json"
        with open(last_receipt_path) as f:
            last_receipt = json.load(f)

        wrong_key_valid = verify_receipt_signature(last_receipt, str(wrong_pub))
        print(f"  Verify receipt with wrong public key: {'valid' if wrong_key_valid else 'REJECTED (expected)'}")
        print()

        # Summary
        print("=" * 72)
        print("  SUMMARY: Same input, different constitutions, different behavior")
        print("=" * 72)
        print()
        print("  strict_financial_analyst:  5 checks, all at HALT  -> HALTED on C1")
        print("  permissive_support_agent:  2 checks at WARN + 1 custom -> WARNED")
        print("  research_assistant:        5 checks, C1=HALT rest=LOG -> HALTED on C1")
        print()
        print("  Every constitution signed with Ed25519.")
        print("  Every receipt signed with Ed25519.")
        print("  Provenance bond verified: receipt <-> constitution.")
        print("  Wrong key rejected.")
        print()
        print("  The constitution IS the control plane.")
        print()


if __name__ == "__main__":
    run_demo()
