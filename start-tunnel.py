"""Start OpenAI Secure MCP Tunnel for the local Chem_Agent MCP server.

The launcher intentionally keeps secrets in environment variables. The
``tunnel-client`` flags reference those variables, so keys are not echoed in the
process arguments or startup banner.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.error
import urllib.request
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_TUNNEL_CLIENT = Path(
    r"D:\myProjects\tunnel-client\tunnel-client-runtime-cloudflared.exe"
)
DEFAULT_MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_HEALTH_ADDR = "127.0.0.1:8080"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is missing in .env or the current environment")
    return value


def _build_command(
    tunnel_client: Path,
    tunnel_id: str,
    mcp_server_url: str,
    health_addr: str,
) -> list[str]:
    return [
        str(tunnel_client),
        "run",
        "--control-plane.api-key",
        "env:CONTROL_PLANE_API_KEY",
        "--control-plane.tunnel-id",
        tunnel_id,
        "--mcp.server-url",
        f"url={mcp_server_url}",
        "--mcp.extra-headers",
        "Authorization: env:MCP_AUTH_HEADER",
        "--health.listen-addr",
        health_addr,
    ]


def _probe_mcp(mcp_server_url: str, mcp_api_key: str) -> tuple[int | None, str]:
    req = urllib.request.Request(
        mcp_server_url,
        headers={
            "Authorization": f"Bearer {mcp_api_key}",
            "Accept": "text/event-stream",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", "")
    except OSError as exc:
        raise SystemExit(f"MCP server probe failed: {exc}") from exc


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate local configuration and MCP reachability without starting the tunnel",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="print the non-secret tunnel-client command and exit",
    )
    return parser


def _mode_label(check: bool, print_command: bool) -> str:
    if check:
        return "Checking Chem_Agent MCP Tunnel prerequisites"
    if print_command:
        return "Printing Chem_Agent MCP Tunnel command"
    return "Starting Chem_Agent MCP Tunnel"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)

    tunnel_client = Path(
        os.environ.get("TUNNEL_CLIENT_EXE", str(DEFAULT_TUNNEL_CLIENT))
    )
    if not tunnel_client.exists():
        raise SystemExit(f"tunnel-client executable not found: {tunnel_client}")

    tunnel_id = _required_env("CONTROL_PLANE_TUNNEL_ID")
    _required_env("CONTROL_PLANE_API_KEY")
    mcp_api_key = _required_env("MCP_API_KEY")

    mcp_server_url = os.environ.get("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL).strip()
    health_addr = os.environ.get("TUNNEL_HEALTH_ADDR", DEFAULT_HEALTH_ADDR).strip()

    env = os.environ.copy()
    env["MCP_AUTH_HEADER"] = f"Bearer {mcp_api_key}"

    cmd = _build_command(tunnel_client, tunnel_id, mcp_server_url, health_addr)

    print("========================================")
    print(_mode_label(args.check, args.print_command))
    print("========================================")
    print(f"MCP Server : {mcp_server_url}")
    print(f"Tunnel ID  : {tunnel_id}")
    print(f"Client     : {tunnel_client}")
    print(f"Health     : http://{health_addr}/healthz")
    print("")

    if args.print_command:
        print(subprocess.list2cmdline(cmd))
        return 0

    if args.check:
        status, content_type = _probe_mcp(mcp_server_url, mcp_api_key)
        print(f"MCP probe  : {status} {content_type}")
        if status != 200 or "text/event-stream" not in content_type:
            raise SystemExit("MCP server did not return the expected authenticated SSE response")
        print("Local tunnel prerequisites look OK.")
        return 0

    return subprocess.call(cmd, cwd=tunnel_client.parent, env=env)


if __name__ == "__main__":
    sys.exit(main())
