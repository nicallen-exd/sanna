"""
Sanna — Reasoning integrity infrastructure for AI agents.

Generates portable, offline-verifiable "reasoning receipts" that document
AI agent decisions with C1-C5 coherence checks and tamper-evident hashing.
"""

__version__ = "0.3.0"

from .hashing import hash_text, hash_obj, canonicalize_text
from .receipt import (
    generate_receipt,
    extract_trace_data,
    SannaReceipt,
    C3MReceipt,  # legacy alias
    CheckResult,
    FinalAnswerProvenance,
    TOOL_VERSION,
    SCHEMA_VERSION,
    CHECKS_VERSION,
)
from .verify import verify_receipt, load_schema, VerificationResult
from .middleware import sanna_observe, SannaResult, SannaHaltError

__all__ = [
    "__version__",
    "hash_text",
    "hash_obj",
    "canonicalize_text",
    "generate_receipt",
    "extract_trace_data",
    "SannaReceipt",
    "C3MReceipt",
    "CheckResult",
    "FinalAnswerProvenance",
    "verify_receipt",
    "load_schema",
    "VerificationResult",
    "TOOL_VERSION",
    "SCHEMA_VERSION",
    "CHECKS_VERSION",
    "sanna_observe",
    "SannaResult",
    "SannaHaltError",
]
