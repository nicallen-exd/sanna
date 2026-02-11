"""
Canonical hashing for deterministic receipts across platforms.
"""

import hashlib
import json
import unicodedata
from typing import Any


def canonicalize_text(s: str) -> str:
    """Normalize text for consistent hashing across platforms."""
    if s is None:
        return ""
    # Unicode normalization (NFC)
    s = unicodedata.normalize("NFC", s)
    # Normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace per line (prevents OS/editor diffs)
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.strip()


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize object to canonical JSON bytes."""
    canon = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),  # no spaces
        ensure_ascii=False,
    )
    return canon.encode("utf-8")


def sha256_hex(data: bytes, truncate: int = 16) -> str:
    """SHA256 hash, optionally truncated."""
    full_hash = hashlib.sha256(data).hexdigest()
    return full_hash[:truncate] if truncate else full_hash


def hash_text(s: str, truncate: int = 16) -> str:
    """Hash canonicalized text."""
    return sha256_hex(canonicalize_text(s).encode("utf-8"), truncate)


def hash_obj(obj: Any, truncate: int = 16) -> str:
    """Hash canonicalized JSON object."""
    return sha256_hex(canonical_json_bytes(obj), truncate)
