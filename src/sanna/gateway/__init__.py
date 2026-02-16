"""Sanna gateway — MCP enforcement proxy (v0.11.1)."""


def main() -> None:
    """CLI entry point for ``sanna-gateway``.

    Dispatches to:
    - ``migrate`` subcommand: config migration wizard
    - Default (legacy): run the gateway proxy (``--config`` required)
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        from sanna.gateway.migrate import migrate_command

        sys.exit(migrate_command(sys.argv[2:]))
    else:
        from sanna.gateway.server import run_gateway

        run_gateway()
