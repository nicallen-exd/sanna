"""Sanna adapter for Langfuse.

Converts Langfuse traces to Sanna reasoning receipts.

Usage:
    from sanna.adapters.langfuse import export_receipt

    langfuse = Langfuse(...)
    trace = langfuse.fetch_trace(trace_id)
    receipt = export_receipt(trace.data)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


def langfuse_trace_to_trace_data(trace_data: Any) -> dict:
    """Convert a Langfuse trace object to the dict format generate_receipt expects.

    Args:
        trace_data: Langfuse trace data object (from fetch_trace().data).

    Returns:
        Dict ready for receipt generation.
    """
    trace_id = getattr(trace_data, "id", "unknown")

    # Extract context from observations
    context = ""
    observations_raw = getattr(trace_data, "observations", []) or []
    observations = []

    for obs in observations_raw:
        obs_output = getattr(obs, "output", None)
        if obs_output and isinstance(obs_output, dict):
            for key in ["documents", "context", "retrieved", "chunks"]:
                if key in obs_output:
                    ctx_val = obs_output[key]
                    if not context:
                        context = "\n".join(str(d) for d in ctx_val) if isinstance(ctx_val, list) else str(ctx_val)
                    break

        # Build observation dict for generate_receipt
        obs_dict = {
            "id": getattr(obs, "id", None),
            "name": getattr(obs, "name", None),
            "type": getattr(obs, "type", None),
            "input": getattr(obs, "input", None),
            "output": getattr(obs, "output", None),
            "metadata": getattr(obs, "metadata", None),
            "start_time": str(getattr(obs, "start_time", None)) if getattr(obs, "start_time", None) else None,
            "end_time": str(getattr(obs, "end_time", None)) if getattr(obs, "end_time", None) else None,
        }
        observations.append(obs_dict)

    # Extract query
    query = ""
    trace_input = getattr(trace_data, "input", None)
    if isinstance(trace_input, dict):
        query = trace_input.get("query", trace_input.get("question", trace_input.get("input", "")))
    elif isinstance(trace_input, str):
        query = trace_input

    # Build the trace dict in the format generate_receipt expects
    trace_output = getattr(trace_data, "output", None)

    # If no retrieval span was found but we have context from the trace input,
    # synthesize a retrieval observation so generate_receipt can find it
    if not context and isinstance(trace_input, dict):
        ctx_val = trace_input.get("context", "")
        if ctx_val:
            context = str(ctx_val)
            observations.insert(0, {
                "id": "synthetic-retrieval",
                "name": "retrieval",
                "type": "SPAN",
                "input": {"query": query},
                "output": {"context": context},
                "metadata": {},
                "start_time": None,
                "end_time": None,
            })

    return {
        "trace_id": trace_id,
        "name": getattr(trace_data, "name", None),
        "timestamp": str(getattr(trace_data, "timestamp", None)) if getattr(trace_data, "timestamp", None) else None,
        "input": trace_input if isinstance(trace_input, dict) else {"query": query},
        "output": trace_output if isinstance(trace_output, dict) else None,
        "metadata": getattr(trace_data, "metadata", None),
        "observations": observations,
    }


def export_receipt(
    trace_data: Any,
    extensions: dict[str, Any] | None = None,
    constitution: Any | None = None,
    constitution_path: str | None = None,
) -> dict[str, Any]:
    """Generate a Sanna receipt from a Langfuse trace.

    Args:
        trace_data: Langfuse trace data object.
        extensions: Optional vendor-specific metadata.
        constitution: Optional ConstitutionProvenance for governance tracking.
        constitution_path: Optional path to a constitution YAML/JSON file.
            When provided, the constitution's invariants drive which checks
            run and at what enforcement level (v0.6.0).

    Returns:
        Receipt as a JSON-serializable dict.
    """
    from sanna.receipt import generate_receipt

    trace_dict = langfuse_trace_to_trace_data(trace_data)

    if constitution_path is not None:
        # v0.6.1: Constitution must be signed before use
        from sanna.constitution import load_constitution, constitution_to_receipt_ref, SannaConstitutionError
        from sanna.enforcement import configure_checks
        from sanna.middleware import _generate_constitution_receipt, _generate_no_invariants_receipt

        loaded = load_constitution(constitution_path, validate=True)
        if not loaded.policy_hash:
            raise SannaConstitutionError(
                f"Constitution is not signed: {constitution_path}. "
                f"Run: sanna-sign-constitution {constitution_path}"
            )
        const_ref = constitution_to_receipt_ref(loaded)
        check_configs, custom_records = configure_checks(loaded)

        if check_configs or custom_records:
            receipt_dict = _generate_constitution_receipt(
                trace_dict,
                check_configs=check_configs,
                custom_records=custom_records,
                constitution_ref=const_ref,
                constitution_version=loaded.schema_version,
                extensions={"langfuse": extensions} if extensions else None,
            )
        else:
            receipt_dict = _generate_no_invariants_receipt(
                trace_dict,
                constitution_ref=const_ref,
                extensions={"langfuse": extensions} if extensions else None,
            )
        return receipt_dict

    # Legacy path: generate receipt without constitution enforcement
    receipt = generate_receipt(trace_dict, constitution=constitution)
    receipt_dict = asdict(receipt)

    if extensions:
        receipt_dict["extensions"] = extensions

    return receipt_dict
