# Changelog

## [0.6.4] - 2026-02-13
### Fixed
- Schema validation enforced on enforcement paths (middleware, adapter) — typos in constitutions now produce clear errors
- CLI commands produce clean error messages instead of Python tracebacks for all common failure modes
- Chain verification checks constitution signature value equality (not just policy_hash)
- Float values in signed payloads caught at generation boundary with clear path information
- Private key files written with 0o600 permissions on POSIX systems

## [0.6.3] - 2026-02-13
### Fixed
- Receipt schema updated to allow signature fields in constitution_ref (signed receipts now pass schema validation)
- Constitution Ed25519 signature binds full document including provenance and signer metadata
- Receipt Ed25519 signature binds signer metadata (key_id, signed_by, signed_at)
- RFC 8785-style JSON canonicalization for cross-language verifier portability
- Float elimination from signed payloads (coverage_pct replaced with coverage_basis_points as integer)
- C4 contraction handling (can/can't no longer conflated)

### Added
- `policy_hash` replaces `document_hash` (semantic rename — hashes policy content only)
- `sanna-hash-constitution` CLI command for hash-only mode
- `sanna-sign-constitution` now requires `--private-key`
- Full chain verification: `sanna-verify --constitution --constitution-public-key`
- Signature scheme versioning (constitution_sig_v1, receipt_sig_v1)

## [0.6.2] - 2026-02-13
### Fixed
- Full SHA-256 key_id (64-char hex digest, was truncated to 16 chars)
- Demo rewritten with full Ed25519 provenance flow
- `sanna-keygen --signed-by` writes metadata file alongside keypair
- Schema patterns updated to ^[a-f0-9]{64}$

## [0.6.1] - 2026-02-13
### Added
- Ed25519 cryptographic signatures on constitutions and receipts
- Receipt-to-constitution provenance bond with offline verification
- Stable check IDs (sanna.* namespace, CHECK_REGISTRY)
- Replayable flag on check results
- PARTIAL status with evaluation_coverage block

### Fixed
- Removed auto-signing of unsigned constitutions (fail closed)
- Hash verification on constitution load
- C4 word-boundary fix ("can" no longer matches "cannot")
- C5 bullet-counting fix

## [0.6.0] - 2026-02-12
### Added
- Constitution enforcement drives check engine
- Invariant-to-check mapping
- Per-check enforcement levels (halt/warn/log)
- Three Constitutions Demo
- 290 tests
