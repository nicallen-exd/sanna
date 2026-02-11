"""
Sanna test suite — golden receipt verification, hashing, checks, and verifier.
"""

import json
import copy
import pytest
from pathlib import Path
from dataclasses import asdict

from sanna.hashing import hash_text, hash_obj, canonicalize_text, sha256_hex, canonical_json_bytes
from sanna.receipt import (
    generate_receipt, CheckResult, SannaReceipt,
    check_c1_context_contradiction, check_c2_unmarked_inference,
    check_c3_false_certainty, check_c4_conflict_collapse,
    check_c5_premature_compression, select_final_answer,
    extract_context, extract_query, find_snippet,
    TOOL_VERSION, SCHEMA_VERSION, CHECKS_VERSION,
)
from sanna.verify import (
    verify_receipt, load_schema, VerificationResult,
    verify_fingerprint, verify_status_consistency,
    verify_check_counts, verify_hash_format, verify_content_hashes,
)


# =============================================================================
# FIXTURES
# =============================================================================

GOLDEN_DIR = Path(__file__).parent.parent / "golden" / "receipts"
SCHEMA = load_schema()


def load_golden(name: str) -> dict:
    """Load a golden receipt by filename."""
    with open(GOLDEN_DIR / name) as f:
        return json.load(f)


def all_golden_receipts():
    """List all golden receipt files (excluding tampered)."""
    return sorted([f.name for f in GOLDEN_DIR.glob("*.json") if "tampered" not in f.name])


# =============================================================================
# HASHING TESTS
# =============================================================================

class TestHashing:
    def test_hash_text_deterministic(self):
        assert hash_text("hello world") == hash_text("hello world")

    def test_hash_text_different_inputs(self):
        assert hash_text("hello") != hash_text("world")

    def test_hash_text_16_hex_chars(self):
        h = hash_text("test")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_obj_deterministic(self):
        obj = {"b": 2, "a": 1}
        assert hash_obj(obj) == hash_obj({"a": 1, "b": 2})

    def test_hash_obj_key_order_independent(self):
        """Canonical JSON sorts keys, so order doesn't matter."""
        assert hash_obj({"z": 1, "a": 2}) == hash_obj({"a": 2, "z": 1})

    def test_canonicalize_text_nfc(self):
        # NFC normalization
        result = canonicalize_text("café")
        assert result == "café"

    def test_canonicalize_text_line_endings(self):
        assert canonicalize_text("a\r\nb") == canonicalize_text("a\nb")
        assert canonicalize_text("a\rb") == canonicalize_text("a\nb")

    def test_canonicalize_text_trailing_whitespace(self):
        assert canonicalize_text("hello   ") == "hello"
        assert canonicalize_text("line1   \nline2  ") == "line1\nline2"

    def test_canonicalize_text_none(self):
        assert canonicalize_text(None) == ""

    def test_sha256_hex_full(self):
        h = sha256_hex(b"test", truncate=0)
        assert len(h) == 64

    def test_sha256_hex_truncated(self):
        h = sha256_hex(b"test", truncate=16)
        assert len(h) == 16

    def test_canonical_json_bytes_no_spaces(self):
        result = canonical_json_bytes({"a": 1})
        assert result == b'{"a":1}'

    def test_canonical_json_bytes_sorted(self):
        result = canonical_json_bytes({"b": 2, "a": 1})
        assert result == b'{"a":1,"b":2}'


# =============================================================================
# C1-C5 CHECK TESTS
# =============================================================================

class TestC1ContextContradiction:
    def test_c1_pass_no_contradiction(self):
        result = check_c1_context_contradiction(
            "Paris is the capital of France.",
            "The capital of France is Paris."
        )
        assert result.passed
        assert result.check_id == "C1"

    def test_c1_fail_refund_contradiction(self):
        result = check_c1_context_contradiction(
            "Digital products are non-refundable once downloaded.",
            "You are eligible to request a refund for the software."
        )
        assert not result.passed
        assert result.severity == "critical"
        assert "non-refundable" in result.evidence.lower()

    def test_c1_pass_empty_context(self):
        result = check_c1_context_contradiction("", "some output")
        assert result.passed

    def test_c1_pass_empty_output(self):
        result = check_c1_context_contradiction("some context", "")
        assert result.passed


class TestC2UnmarkedInference:
    def test_c2_pass_hedged(self):
        result = check_c2_unmarked_inference(
            "Some context",
            "This may work, and it's possible that results could vary."
        )
        assert result.passed

    def test_c2_fail_definitive_no_hedging(self):
        result = check_c2_unmarked_inference(
            "Some context",
            "This will definitely work and is guaranteed to succeed."
        )
        assert not result.passed
        assert result.severity == "warning"

    def test_c2_pass_empty_output(self):
        result = check_c2_unmarked_inference("context", "")
        assert result.passed

    def test_c2_pass_definitive_with_hedging(self):
        result = check_c2_unmarked_inference(
            "context",
            "This is definitely the approach, though it may need adjustments."
        )
        assert result.passed


class TestC3FalseCertainty:
    def test_c3_pass_acknowledges_conditions(self):
        result = check_c3_false_certainty(
            "Available if usage is under 1000 requests. Requires registration.",
            "You can use the API, however this requires registration first."
        )
        assert result.passed

    def test_c3_fail_ignores_conditions(self):
        result = check_c3_false_certainty(
            "Available if usage is under 1000 requests. Requires registration.",
            "You can use the API. Go ahead and start making requests."
        )
        assert not result.passed
        assert result.severity == "warning"

    def test_c3_pass_empty_output(self):
        result = check_c3_false_certainty("context", "")
        assert result.passed


class TestC4ConflictCollapse:
    def test_c4_pass_acknowledges_tension(self):
        result = check_c4_conflict_collapse(
            "Users can access premium features. However, some features require admin approval.",
            "You can access premium features, however some require admin approval."
        )
        assert result.passed

    def test_c4_fail_collapses_tension(self):
        result = check_c4_conflict_collapse(
            "License transfers are permitted for enterprise. Individual licenses cannot be transferred.",
            "Yes, you can transfer your license."
        )
        assert not result.passed
        assert result.severity == "warning"

    def test_c4_pass_no_conflict(self):
        result = check_c4_conflict_collapse(
            "The sky is blue.",
            "The sky is blue on clear days."
        )
        assert result.passed


class TestC5PrematureCompression:
    def test_c5_pass_adequate_detail(self):
        result = check_c5_premature_compression(
            "Option A costs $10. Option B costs $20. Option C is free.",
            "There are three options: A at $10, B at $20, and C which is free. Each has different trade-offs."
        )
        assert result.passed

    def test_c5_fail_oversimplified(self):
        result = check_c5_premature_compression(
            "Deployment options include:\n- AWS\n- GCP\n- Azure\n- On-premise\nEach has different implications.",
            "Deploy to cloud"
        )
        assert not result.passed
        assert result.severity == "warning"

    def test_c5_pass_simple_context(self):
        result = check_c5_premature_compression(
            "The answer is 42.",
            "42"
        )
        assert result.passed


# =============================================================================
# RECEIPT GENERATION TESTS
# =============================================================================

class TestReceiptGeneration:
    def make_trace(self, context="Some context.", output="Some output.", query="question?"):
        return {
            "trace_id": "test-trace-001",
            "name": "test",
            "timestamp": "2026-01-01T00:00:00Z",
            "input": {"query": query},
            "output": {"final_answer": output},
            "metadata": {},
            "observations": [
                {
                    "id": "obs-ret",
                    "name": "retrieval",
                    "type": "SPAN",
                    "input": {"query": query},
                    "output": {"context": context},
                    "metadata": {},
                    "start_time": "2026-01-01T00:00:01Z",
                    "end_time": "2026-01-01T00:00:02Z",
                }
            ],
        }

    def test_receipt_has_required_fields(self):
        receipt = generate_receipt(self.make_trace())
        d = asdict(receipt)
        for field in ["schema_version", "tool_version", "checks_version",
                       "receipt_id", "receipt_fingerprint", "trace_id",
                       "timestamp", "inputs", "outputs", "context_hash",
                       "output_hash", "final_answer_provenance", "checks",
                       "checks_passed", "checks_failed", "coherence_status"]:
            assert field in d, f"Missing field: {field}"

    def test_receipt_versions(self):
        receipt = generate_receipt(self.make_trace())
        assert receipt.schema_version == SCHEMA_VERSION
        assert receipt.tool_version == TOOL_VERSION
        assert receipt.checks_version == CHECKS_VERSION

    def test_receipt_has_five_checks(self):
        receipt = generate_receipt(self.make_trace())
        assert len(receipt.checks) == 5

    def test_receipt_check_counts_consistent(self):
        receipt = generate_receipt(self.make_trace())
        assert receipt.checks_passed + receipt.checks_failed == 5

    def test_receipt_fingerprint_stable(self):
        """Same trace data should produce same fingerprint."""
        trace = self.make_trace()
        r1 = generate_receipt(trace)
        r2 = generate_receipt(trace)
        assert r1.receipt_fingerprint == r2.receipt_fingerprint

    def test_receipt_id_changes(self):
        """Receipt ID includes timestamp, should change between runs."""
        trace = self.make_trace()
        r1 = generate_receipt(trace)
        r2 = generate_receipt(trace)
        # May or may not differ (depends on timing), but both should be valid hex
        assert len(r1.receipt_id) == 16
        assert len(r2.receipt_id) == 16

    def test_hash_format(self):
        receipt = generate_receipt(self.make_trace())
        import re
        hex16 = re.compile(r"^[a-f0-9]{16}$")
        assert hex16.match(receipt.receipt_id)
        assert hex16.match(receipt.receipt_fingerprint)
        assert hex16.match(receipt.context_hash)
        assert hex16.match(receipt.output_hash)


# =============================================================================
# SELECT FINAL ANSWER TESTS
# =============================================================================

class TestSelectFinalAnswer:
    def test_trace_output_preferred(self):
        trace = {
            "trace_id": "test",
            "output": {"final_answer": "From trace"},
            "observations": [
                {
                    "id": "gen-1",
                    "name": "llm-generation",
                    "type": "GENERATION",
                    "output": {"response": "From span"},
                    "metadata": {},
                    "end_time": "2026-01-01T00:00:05Z",
                }
            ],
        }
        answer, prov = select_final_answer(trace)
        assert answer == "From trace"
        assert prov.source == "trace.output"

    def test_span_output_fallback(self):
        trace = {
            "trace_id": "test",
            "output": None,
            "observations": [
                {
                    "id": "gen-1",
                    "name": "llm-generation",
                    "type": "GENERATION",
                    "output": {"response": "From span"},
                    "metadata": {"model": "gpt-4"},
                    "end_time": "2026-01-01T00:00:05Z",
                }
            ],
        }
        answer, prov = select_final_answer(trace)
        assert answer == "From span"
        assert prov.source == "span.output"

    def test_no_answer_found(self):
        trace = {"trace_id": "test", "output": None, "observations": []}
        answer, prov = select_final_answer(trace)
        assert answer == ""
        assert prov.source == "none"


# =============================================================================
# EXTRACTION HELPERS
# =============================================================================

class TestExtractionHelpers:
    def test_extract_context(self):
        trace = {
            "observations": [
                {"name": "retrieval", "output": {"context": "Retrieved context"}}
            ]
        }
        assert extract_context(trace) == "Retrieved context"

    def test_extract_query(self):
        trace = {
            "observations": [
                {"name": "retrieval", "input": {"query": "User question"}}
            ]
        }
        assert extract_query(trace) == "User question"

    def test_find_snippet_found(self):
        result = find_snippet("This is a long text with the keyword here.", ["keyword"])
        assert "keyword" in result

    def test_find_snippet_not_found(self):
        result = find_snippet("Short text.", ["missing"])
        assert result == "Short text."

    def test_find_snippet_empty(self):
        assert find_snippet("", ["any"]) == ""


# =============================================================================
# VERIFIER TESTS
# =============================================================================

class TestVerifier:
    def test_valid_receipt(self):
        receipt = load_golden("002_pass_simple_qa.json")
        result = verify_receipt(receipt, SCHEMA)
        assert result.valid
        assert result.exit_code == 0

    def test_schema_invalid(self):
        receipt = {"not": "a valid receipt"}
        result = verify_receipt(receipt, SCHEMA)
        assert not result.valid
        assert result.exit_code == 2

    def test_fingerprint_mismatch(self):
        receipt = load_golden("002_pass_simple_qa.json")
        receipt["receipt_fingerprint"] = "0000000000000000"
        result = verify_receipt(receipt, SCHEMA)
        assert not result.valid
        assert result.exit_code == 3

    def test_content_tamper_detected(self):
        receipt = load_golden("002_pass_simple_qa.json")
        receipt["outputs"]["response"] = "TAMPERED CONTENT"
        result = verify_receipt(receipt, SCHEMA)
        assert not result.valid
        assert result.exit_code == 3
        assert any("tampered" in e.lower() for e in result.errors)

    def test_status_consistency_mismatch(self):
        receipt = load_golden("002_pass_simple_qa.json")
        receipt["coherence_status"] = "FAIL"  # Should be PASS
        result = verify_receipt(receipt, SCHEMA)
        assert not result.valid
        assert result.exit_code == 4

    def test_check_count_mismatch(self):
        receipt = load_golden("002_pass_simple_qa.json")
        receipt["checks_passed"] = 0  # Wrong
        result = verify_receipt(receipt, SCHEMA)
        assert not result.valid
        assert result.exit_code == 4


class TestVerifyFingerprint:
    def test_fingerprint_matches(self):
        receipt = load_golden("001_fail_c1_refund.json")
        match, computed, expected = verify_fingerprint(receipt)
        assert match
        assert computed == expected

    def test_fingerprint_mismatch(self):
        receipt = load_golden("001_fail_c1_refund.json")
        receipt["receipt_fingerprint"] = "0000000000000000"
        match, computed, expected = verify_fingerprint(receipt)
        assert not match


class TestVerifyStatusConsistency:
    def test_pass_status(self):
        receipt = {"checks": [{"passed": True, "severity": "info"}], "coherence_status": "PASS"}
        match, computed, expected = verify_status_consistency(receipt)
        assert match

    def test_warn_status(self):
        receipt = {"checks": [{"passed": False, "severity": "warning"}], "coherence_status": "WARN"}
        match, computed, expected = verify_status_consistency(receipt)
        assert match

    def test_fail_status(self):
        receipt = {"checks": [{"passed": False, "severity": "critical"}], "coherence_status": "FAIL"}
        match, computed, expected = verify_status_consistency(receipt)
        assert match


class TestVerifyHashFormat:
    def test_valid_hashes(self):
        receipt = {
            "receipt_id": "abcdef0123456789",
            "receipt_fingerprint": "0123456789abcdef",
            "context_hash": "fedcba9876543210",
            "output_hash": "1234567890abcdef",
        }
        assert verify_hash_format(receipt) == []

    def test_invalid_hash_length(self):
        receipt = {
            "receipt_id": "short",
            "receipt_fingerprint": "0123456789abcdef",
            "context_hash": "fedcba9876543210",
            "output_hash": "1234567890abcdef",
        }
        errors = verify_hash_format(receipt)
        assert len(errors) == 1
        assert "receipt_id" in errors[0]


# =============================================================================
# GOLDEN RECEIPT TESTS
# =============================================================================

class TestGoldenReceipts:
    """Verify all golden receipts pass the verifier."""

    @pytest.mark.parametrize("filename", all_golden_receipts())
    def test_golden_receipt_valid(self, filename):
        receipt = load_golden(filename)
        result = verify_receipt(receipt, SCHEMA)
        assert result.valid, f"{filename}: {result.errors}"
        assert result.exit_code == 0

    def test_tampered_receipt_detected(self):
        receipt = load_golden("999_tampered.json")
        result = verify_receipt(receipt, SCHEMA)
        assert not result.valid
        assert result.exit_code == 3

    def test_golden_receipt_count(self):
        """Ensure we have the expected number of golden receipts."""
        receipts = all_golden_receipts()
        assert len(receipts) >= 12, f"Expected at least 12 golden receipts, got {len(receipts)}"

    @pytest.mark.parametrize("filename,expected_status", [
        ("001_fail_c1_refund.json", "FAIL"),
        ("002_pass_simple_qa.json", "PASS"),
        ("005_warn_c2_unmarked_inference.json", "WARN"),
        ("008_fail_c1_factual.json", "FAIL"),
    ])
    def test_golden_expected_status(self, filename, expected_status):
        receipt = load_golden(filename)
        assert receipt["coherence_status"] == expected_status

    def test_golden_fail_c1_has_evidence(self):
        receipt = load_golden("001_fail_c1_refund.json")
        c1 = next(c for c in receipt["checks"] if c["check_id"] == "C1")
        assert not c1["passed"]
        assert c1["evidence"] is not None
        assert len(c1["evidence"]) > 0

    def test_golden_span_provenance(self):
        receipt = load_golden("011_pass_span_provenance.json")
        prov = receipt["final_answer_provenance"]
        assert prov["source"] == "span.output"
        assert prov["span_name"] is not None

    def test_golden_extensions_preserved(self):
        receipt = load_golden("012_pass_with_extensions.json")
        assert "extensions" in receipt
        assert receipt["extensions"]["vendor"] == "test-vendor"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_empty_trace(self):
        trace = {
            "trace_id": "empty",
            "name": "empty",
            "timestamp": None,
            "input": None,
            "output": None,
            "metadata": None,
            "observations": [],
        }
        receipt = generate_receipt(trace)
        assert receipt.coherence_status == "PASS"
        assert receipt.checks_passed == 5

    def test_unicode_content(self):
        trace = {
            "trace_id": "unicode-test",
            "name": "unicode",
            "timestamp": None,
            "input": None,
            "output": {"final_answer": "Ünïcödé résponse with émojis 🎉"},
            "metadata": None,
            "observations": [
                {
                    "id": "obs-ret",
                    "name": "retrieval",
                    "type": "SPAN",
                    "input": {"query": "unicode test"},
                    "output": {"context": "Cöntéxt with spëcial chàracters"},
                    "metadata": {},
                }
            ],
        }
        receipt = generate_receipt(trace)
        d = asdict(receipt)
        # Should be valid JSON
        json_str = json.dumps(d)
        reparsed = json.loads(json_str)
        assert reparsed["outputs"]["response"] == "Ünïcödé résponse with émojis 🎉"

    def test_very_long_content(self):
        long_context = "x" * 100000
        long_output = "y" * 100000
        trace = {
            "trace_id": "long-test",
            "name": "long",
            "timestamp": None,
            "input": None,
            "output": {"final_answer": long_output},
            "metadata": None,
            "observations": [
                {
                    "id": "obs-ret",
                    "name": "retrieval",
                    "type": "SPAN",
                    "input": {"query": "long"},
                    "output": {"context": long_context},
                    "metadata": {},
                }
            ],
        }
        receipt = generate_receipt(trace)
        assert receipt.context_hash is not None
        assert len(receipt.context_hash) == 16
