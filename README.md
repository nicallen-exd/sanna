# Sanna

AI governance infrastructure that generates cryptographically signed "reasoning receipts" — portable JSON artifacts that document AI agent decisions and verify reasoning integrity offline. Constitutions define the rules. Checks enforce them. Receipts prove it happened. The name means "truth" in Swedish.

```bash
pip install sanna
```

## How It Works

### 1. Define a constitution

A constitution is a YAML file that declares which reasoning invariants your agent must uphold, and what happens when they're violated.

```yaml
# constitution.yaml
sanna_constitution: "1.0.0"

identity:
  agent_name: "support-agent"
  domain: "customer-service"

provenance:
  authored_by: "cs-team@company.com"
  approved_by:
    - "cs-director@company.com"
  approval_date: "2026-01-15"
  approval_method: "compliance-review"

invariants:
  - id: "INV_NO_FABRICATION"
    rule: "Do not claim facts absent from provided sources."
    enforcement: "halt"
  - id: "INV_MARK_INFERENCE"
    rule: "Clearly mark inferences and speculation as such."
    enforcement: "warn"
  - id: "INV_PRESERVE_TENSION"
    rule: "Do not collapse conflicting evidence."
    enforcement: "log"

boundaries:
  - id: "B001"
    description: "Only answer product and service questions"
    category: "scope"
    severity: "high"
```

Each invariant maps to a coherence check. The `enforcement` field controls what happens when that check fails — `halt` stops execution, `warn` emits a Python warning, `log` records silently.

### 2. Sign it

```bash
sanna-keygen --signed-by "your-name@company.com"
sanna-sign-constitution constitution.yaml --private-key sanna_ed25519.key
```

Constitutions are Ed25519-signed. The signature covers the full document — identity, provenance, invariants, and signer metadata. Tampering with any field invalidates the signature.

### 3. Wrap your agent function

```python
from sanna import sanna_observe, SannaHaltError

@sanna_observe(
    constitution_path="constitution.yaml",
    private_key_path="sanna_ed25519.key",
)
def support_agent(query: str, context: str) -> str:
    return llm.generate(query=query, context=context)

try:
    result = support_agent(
        query="Can I get a refund on my software?",
        context="Digital products are non-refundable once downloaded."
    )
    print(result.output)          # The agent's response
    print(result.receipt)         # The reasoning receipt (dict)
except SannaHaltError as e:
    print(f"HALTED: {e}")
    print(e.receipt)              # Receipt is still available
```

The constitution drives enforcement at runtime. Only invariants listed in the constitution are evaluated. Each check enforces independently — halt, warn, or log.

### 4. Verify offline

```bash
sanna-verify receipt.json
sanna-verify receipt.json --public-key sanna_ed25519.pub
sanna-verify receipt.json --constitution constitution.yaml --constitution-public-key sanna_ed25519.pub
```

No network. No API keys. No platform access. Full chain verification: receipt integrity, Ed25519 signature, receipt-to-constitution provenance bond, constitution signature.

Exit codes: 0=valid, 2=schema invalid, 3=fingerprint mismatch, 4=consistency error, 5=other error.

## Key Capabilities

- **Constitution-as-control-plane** — YAML governance rules with invariants that drive check behavior and enforcement levels (halt/warn/log)
- **Ed25519 cryptographic signatures** — on both constitutions and receipts, with full document binding (tampering with provenance or signer metadata invalidates the signature)
- **Receipt-to-constitution provenance bond** — offline verification of the complete chain from receipt to the specific signed constitution version that governed the execution
- **RFC 8785-style JSON canonicalization** — deterministic canonical bytes for cross-language verifier portability; floats rejected at signing boundary
- **Five coherence checks (C1-C5)** — context contradiction, unmarked inference, false certainty, conflict collapse, premature compression; stable IDs, replayable results
- **PARTIAL status with evaluation coverage** — custom invariants that can't yet be auto-evaluated are documented as NOT_CHECKED with integer basis-point coverage reporting
- **Full chain verification in one CLI command** — receipt schema, fingerprint, signature, constitution bond, constitution signature
- **Langfuse adapter** — generate receipts from Langfuse traces with constitution enforcement

## Coherence Checks

| Invariant | Check | What it catches |
|---|---|---|
| `INV_NO_FABRICATION` | C1 — Context Contradiction | Output contradicts explicit statements in the context |
| `INV_MARK_INFERENCE` | C2 — Mark Inferences | Definitive claims stated without hedging language |
| `INV_NO_FALSE_CERTAINTY` | C3 — No False Certainty | Confidence that exceeds what the evidence supports |
| `INV_PRESERVE_TENSION` | C4 — Preserve Tensions | Conflicting information collapsed into a single answer |
| `INV_NO_PREMATURE_COMPRESSION` | C5 — No Premature Compression | Complex, multi-faceted input reduced to a single sentence |

All checks are heuristic (pattern matching). They flag potential issues for human review. Custom invariants (any ID not in the built-in mapping) appear in the receipt as `NOT_CHECKED` — they document the policy but have no built-in evaluator.

## Three Constitutions Demo

Same agent. Same input. Same bad output. Three different constitutions. Three different outcomes.

| Constitution | Invariants | C1 Enforcement | Outcome |
|---|---|---|---|
| **Strict Financial Analyst** | All 5 at `halt` | halt | **HALTED** — execution stopped |
| **Permissive Support Agent** | 2 at `warn` + 1 custom | warn | **WARNED** — continued with warning |
| **Research Assistant** | C1 `halt`, rest `log` | halt | **HALTED** — only C1 can stop it |

```bash
python examples/three_constitutions_demo.py
```

## CLI Tools

| Command | Description |
|---|---|
| `sanna-verify` | Verify receipt integrity and full provenance chain |
| `sanna-generate` | Generate a receipt from a Langfuse trace |
| `sanna-keygen` | Generate Ed25519 keypair for signing |
| `sanna-sign-constitution` | Cryptographically sign a constitution with Ed25519 |
| `sanna-hash-constitution` | Compute policy hash without Ed25519 signing |
| `sanna-verify-constitution` | Verify a constitution's Ed25519 signature |
| `sanna-init-constitution` | Scaffold a new constitution YAML |

## Langfuse Integration

```bash
pip install sanna[langfuse]
```

```python
from sanna.adapters.langfuse import export_receipt

langfuse = Langfuse(...)
trace = langfuse.fetch_trace(trace_id)
receipt = export_receipt(trace.data, constitution_path="constitution.yaml")
```

## Install

```bash
pip install sanna                    # Core library
pip install sanna[langfuse]          # With Langfuse adapter
```

Development:

```bash
git clone https://github.com/nicallen-exd/sanna.git
cd sanna
pip install -e ".[dev]"
python -m pytest tests/ -q
```

427 tests. 0 failures.

## License

Apache 2.0

[PyPI](https://pypi.org/project/sanna/) · [GitHub](https://github.com/nicallen-exd/sanna)

---

*Sanna is Swedish for "truth."*
