# Sanna Langfuse Adapter

Converts Langfuse traces into Sanna reasoning receipts.

## Install

```bash
pip install sanna[langfuse]
```

## Usage

### Python — Generate a receipt from a Langfuse trace

```python
from langfuse import Langfuse
from sanna.adapters.langfuse import export_receipt

langfuse = Langfuse()
trace = langfuse.fetch_trace("your-trace-id")

receipt = export_receipt(trace.data)
print(receipt["coherence_status"])  # PASS, WARN, or FAIL
```

Add vendor metadata:

```python
receipt = export_receipt(trace.data, extensions={
    "environment": "production",
    "pipeline": "customer-support-v2",
})
```

### CLI — Generate from trace ID directly

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

sanna-generate <trace_id>                              # Human-readable
sanna-generate <trace_id> --format json -o receipt.json # JSON file
```

### Verify offline (no Langfuse needed)

```bash
sanna-verify receipt.json
```

Exit codes: `0` = valid, `2` = schema error, `3` = hash mismatch, `4` = consistency error.

## What the adapter does

1. Extracts query, context, and response from the Langfuse trace structure
2. Looks for context in retrieval span outputs (`documents`, `context`, `retrieved`, `chunks` keys)
3. Extracts the final answer using Sanna's answer selection logic (trace output > last LLM span)
4. Runs C1-C5 coherence checks
5. Returns a receipt dict with hashes, check results, and status

## Example receipt

See [example_receipt.json](./example_receipt.json) for a receipt where C1 (Context Contradiction) caught an agent claiming refund eligibility despite a non-refundable policy in context.
