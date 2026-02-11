"""
CLI entry points for sanna-generate and sanna-verify commands.
Legacy aliases: c3m-receipt, c3m-verify
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .receipt import generate_receipt, extract_trace_data, SannaReceipt, TOOL_VERSION
from .verify import verify_receipt, load_schema, VerificationResult


# =============================================================================
# RECEIPT CLI
# =============================================================================

def format_receipt_summary(receipt: SannaReceipt) -> str:
    """Format receipt as human-readable summary."""
    lines = [
        "=" * 60,
        "SANNA REASONING RECEIPT",
        "=" * 60,
        f"Tool:        v{receipt.tool_version}",
        f"Schema:      v{receipt.schema_version}",
        f"Checks:      v{receipt.checks_version}",
        f"Receipt ID:  {receipt.receipt_id}",
        f"Fingerprint: {receipt.receipt_fingerprint}",
        f"Trace ID:    {receipt.trace_id}",
        f"Generated:   {receipt.timestamp}",
        "",
        f"Status:      {receipt.coherence_status}",
        f"Passed:      {receipt.checks_passed}/{receipt.checks_passed + receipt.checks_failed}",
        "",
        "-" * 60,
        "CONSISTENCY CHECKS",
        "-" * 60,
    ]

    for check in receipt.checks:
        icon = "✓" if check["passed"] else "✗"
        lines.append(f"  [{icon}] {check['check_id']}: {check['name']}")
        if not check["passed"] and check.get("evidence"):
            lines.append(f"      └─ {check['evidence']}")

    prov = receipt.final_answer_provenance
    prov_str = prov.get("source", "unknown")
    if prov.get("span_name"):
        prov_str += f" ({prov['span_name']})"
    elif prov.get("field"):
        prov_str += f" [{prov['field']}]"

    lines.extend([
        "",
        "-" * 60,
        "PROVENANCE",
        "-" * 60,
        f"  Context Hash:  {receipt.context_hash}",
        f"  Output Hash:   {receipt.output_hash}",
        f"  Answer Source: {prov_str}",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def main_generate():
    """Entry point for sanna-generate command."""
    parser = argparse.ArgumentParser(
        description="Generate Sanna reasoning receipts from Langfuse traces"
    )
    parser.add_argument("trace_id", help="Langfuse trace ID to analyze")
    parser.add_argument("--format", choices=["summary", "json"], default="summary",
                       help="Output format (default: summary)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--version", action="version", version=f"sanna-generate {TOOL_VERSION}")

    args = parser.parse_args()

    # Import Langfuse only when needed
    try:
        from langfuse import Langfuse
    except ImportError:
        print("Error: langfuse not installed. Run: pip install sanna[langfuse]", file=sys.stderr)
        return 1

    # Connect to Langfuse
    lf = Langfuse()

    # Fetch trace
    print(f"Fetching trace {args.trace_id}...", file=sys.stderr)

    try:
        trace = lf.api.trace.get(args.trace_id)
    except Exception as e:
        print(f"Error fetching trace: {e}", file=sys.stderr)
        return 1

    # Extract data and generate receipt
    trace_data = extract_trace_data(trace)
    receipt = generate_receipt(trace_data)

    # Format output
    if args.format == "json":
        output = json.dumps(asdict(receipt), indent=2)
    else:
        output = format_receipt_summary(receipt)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Receipt written to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0 if receipt.coherence_status != "FAIL" else 1


# =============================================================================
# VERIFY CLI
# =============================================================================

VERIFIER_VERSION = "0.2.0"


def format_verify_summary(result: VerificationResult, receipt: dict) -> str:
    """Format verification result as human-readable summary."""
    # Check for extensions
    extensions = receipt.get("extensions")
    if extensions:
        ext_count = len(extensions)
        ext_status = f"✓ Present ({ext_count} keys, ignored)"
    else:
        ext_status = "None"

    lines = [
        "=" * 50,
        "SANNA RECEIPT VERIFICATION",
        "=" * 50,
        "",
        f"Status:      {'✓ VALID' if result.valid else '✗ INVALID'}",
        f"Schema:      v{receipt.get('schema_version', '?')}",
        f"Fingerprint: {'✓ Match' if result.computed_fingerprint == result.expected_fingerprint else '✗ Mismatch'}",
        f"Consistency: {'✓ OK' if result.computed_status == result.expected_status else '✗ Mismatch'}",
        f"Extensions:  {ext_status}",
    ]

    if result.errors:
        lines.extend([
            "",
            "-" * 50,
            "ERRORS",
            "-" * 50,
        ])
        for err in result.errors[:5]:  # Max 5 errors
            lines.append(f"  • {err}")

    if result.warnings:
        lines.extend([
            "",
            "-" * 50,
            "WARNINGS",
            "-" * 50,
        ])
        for warn in result.warnings[:3]:
            lines.append(f"  • {warn}")

    lines.extend(["", "=" * 50])
    return "\n".join(lines)


def format_verify_json(result: VerificationResult, receipt: dict) -> str:
    """Format verification result as JSON."""
    output = {
        "valid": result.valid,
        "exit_code": result.exit_code,
        "schema_version": receipt.get("schema_version"),
        "receipt_id": receipt.get("receipt_id"),
        "fingerprint_match": result.computed_fingerprint == result.expected_fingerprint,
        "status_match": result.computed_status == result.expected_status,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    return json.dumps(output, indent=2)


def main_verify():
    """Entry point for sanna-verify command."""
    parser = argparse.ArgumentParser(
        description="Verify Sanna reasoning receipts",
        epilog="Exit codes: 0=valid, 2=schema invalid, 3=fingerprint mismatch, 4=consistency error, 5=other"
    )
    parser.add_argument("receipt", help="Path to receipt JSON file")
    parser.add_argument("--format", choices=["summary", "json"], default="summary",
                       help="Output format (default: summary)")
    parser.add_argument("--schema", help="Path to schema file (optional, auto-detected)")
    parser.add_argument("--version", action="version", version=f"sanna-verify {VERIFIER_VERSION}")

    args = parser.parse_args()

    # Load receipt
    try:
        with open(args.receipt) as f:
            receipt = json.load(f)
    except FileNotFoundError:
        print(f"Error: Receipt file not found: {args.receipt}", file=sys.stderr)
        return 5
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in receipt: {e}", file=sys.stderr)
        return 5

    # Load schema
    try:
        schema = load_schema(args.schema)
    except FileNotFoundError as e:
        print(f"Error: Schema not found: {e}", file=sys.stderr)
        return 5

    # Verify
    result = verify_receipt(receipt, schema)

    # Output
    if args.format == "json":
        print(format_verify_json(result, receipt))
    else:
        print(format_verify_summary(result, receipt))

    return result.exit_code


# Legacy aliases
main_receipt = main_generate


if __name__ == "__main__":
    sys.exit(main_generate())
