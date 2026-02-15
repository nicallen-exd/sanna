"""Sanna gateway — MCP enforcement proxy (v0.10.0)."""


def main() -> None:
    """CLI entry point for ``sanna-gateway``."""
    from sanna.gateway.server import run_gateway

    run_gateway()
