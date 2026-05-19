"""Tests for ``crossmem mcp ping`` — the post-install MCP self-test.

The subcommand spawns an ephemeral MCP server over stdio, sends an MCP
``initialize`` handshake followed by ``tools/list``, then kills the
process. On success it prints ``ok`` plus the tool names and exits 0;
on protocol error / crash / timeout it prints ``fail`` plus a reason
and exits 1.

The implementation talks JSON-RPC 2.0 directly so we do not need to
take a runtime dependency on the ``mcp`` client library. That also
makes these tests cheap: the fake server is a tiny Python script that
reads JSON-RPC from stdin and writes responses to stdout — no model
loading, no asyncio plumbing, runs in well under a second.
"""

from __future__ import annotations

import json
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from crossmem import cli, mcp_ping

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fake-server fixtures
# ---------------------------------------------------------------------------


_FAKE_SERVER_HEADER = """\
from __future__ import annotations

import json
import sys


def _send(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()


def main():
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req = json.loads(raw)
        method = req.get("method")
        req_id = req.get("id")
"""

_FAKE_SERVER_FOOTER = """

if __name__ == "__main__":
    main()
"""


def _write_fake_server(tmp_path: Path, body: str) -> Path:
    """Drop a single-file fake MCP server into ``tmp_path``.

    ``body`` is appended as the loop body that reacts to each incoming
    JSON-RPC request. The harness around it parses one request per
    line from stdin and writes one response per line to stdout,
    flushing after every write so the parent sees the bytes promptly.

    The body is dedented and re-indented to 8 spaces so it lands
    inside the ``for raw in sys.stdin:`` loop. Keep the supplied
    body's relative indentation consistent — tabs vs. spaces, mixed
    indent levels — and it will compose correctly.
    """
    indented = textwrap.indent(textwrap.dedent(body).strip("\n"), " " * 8)
    content = _FAKE_SERVER_HEADER + indented + _FAKE_SERVER_FOOTER
    server = tmp_path / "fake_server.py"
    server.write_text(content, encoding="utf-8")
    return server


@pytest.fixture
def ok_server(tmp_path: Path) -> list[str]:
    """A fake MCP server that answers ``initialize`` + ``tools/list`` cleanly."""
    body = textwrap.dedent(
        """
        if method == "initialize":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "0.0.0"},
                },
            })
        elif method == "notifications/initialized":
            # Notifications carry no id and require no response.
            continue
        elif method == "tools/list":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {"name": "query", "description": "x", "inputSchema": {}},
                        {"name": "store", "description": "y", "inputSchema": {}},
                    ],
                },
            })
        """
    )
    server = _write_fake_server(tmp_path, body)
    return [sys.executable, str(server)]


@pytest.fixture
def crashing_server(tmp_path: Path) -> list[str]:
    """A fake server that exits non-zero immediately."""
    server = tmp_path / "crash.py"
    server.write_text(
        "import sys\nsys.stderr.write('boom\\n')\nsys.exit(7)\n",
        encoding="utf-8",
    )
    return [sys.executable, str(server)]


@pytest.fixture
def wrong_protocol_server(tmp_path: Path) -> list[str]:
    """A server that returns a JSON-RPC error instead of a tools list."""
    body = textwrap.dedent(
        """
        if method == "initialize":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "0.0.0"},
                },
            })
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "method not found"},
            })
        """
    )
    server = _write_fake_server(tmp_path, body)
    return [sys.executable, str(server)]


@pytest.fixture
def silent_server(tmp_path: Path) -> list[str]:
    """A server that accepts the request but never replies."""
    server = tmp_path / "silent.py"
    server.write_text(
        textwrap.dedent(
            """
            import sys
            import time

            # Drain stdin so the parent's write doesn't block on a full pipe,
            # but never write back. The ping must hit its timeout.
            for _ in sys.stdin:
                pass
            time.sleep(10)
            """
        ),
        encoding="utf-8",
    )
    return [sys.executable, str(server)]


# ---------------------------------------------------------------------------
# ping() — programmatic API
# ---------------------------------------------------------------------------


def test_ping_ok_returns_tool_names(ok_server: list[str]) -> None:
    """A well-behaved server yields ``ok`` plus the advertised tool names."""
    result = mcp_ping.ping(ok_server, timeout=5.0)
    assert result.ok is True
    assert result.tools == ["query", "store"]
    assert result.error is None


def test_ping_crashed_server_reports_fail(crashing_server: list[str]) -> None:
    """A server that exits before answering surfaces as a clean failure."""
    result = mcp_ping.ping(crashing_server, timeout=5.0)
    assert result.ok is False
    assert result.tools == []
    assert result.error is not None
    # The non-zero exit code or stderr text should be referenced for debugging.
    combined = result.error.lower()
    assert "exit" in combined or "boom" in combined or "process" in combined


def test_ping_wrong_protocol_reports_fail(wrong_protocol_server: list[str]) -> None:
    """A JSON-RPC error response is treated as a failure with the error message."""
    result = mcp_ping.ping(wrong_protocol_server, timeout=5.0)
    assert result.ok is False
    assert result.tools == []
    assert result.error is not None
    assert "method not found" in result.error


def test_ping_silent_server_times_out(silent_server: list[str]) -> None:
    """A server that never responds must hit the timeout — and only the timeout."""
    result = mcp_ping.ping(silent_server, timeout=0.5)
    assert result.ok is False
    assert result.tools == []
    assert result.error is not None
    assert "timeout" in result.error.lower()


# ---------------------------------------------------------------------------
# crossmem mcp ping — CLI integration
# ---------------------------------------------------------------------------


def test_cli_mcp_ping_ok(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``crossmem mcp ping`` exits 0 and prints ``ok`` + tool names on success."""

    def fake_ping(cmd: list[str], timeout: float = 5.0) -> mcp_ping.PingResult:
        assert cmd  # the CLI must hand the ping function a non-empty command
        return mcp_ping.PingResult(ok=True, tools=["query", "store"], error=None)

    monkeypatch.setattr(cli, "ping", fake_ping)
    exit_code = cli.main(["mcp", "ping"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "ok" in out
    assert "query" in out
    assert "store" in out


def test_cli_mcp_ping_fail(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing ping surfaces as ``fail`` + reason on stderr/stdout with exit 1."""

    def fake_ping(cmd: list[str], timeout: float = 5.0) -> mcp_ping.PingResult:
        return mcp_ping.PingResult(ok=False, tools=[], error="timeout after 5.0s")

    monkeypatch.setattr(cli, "ping", fake_ping)
    exit_code = cli.main(["mcp", "ping"])
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert exit_code == 1
    assert "fail" in combined
    assert "timeout" in combined


def test_cli_mcp_subcommand_required(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bare ``crossmem mcp`` (no sub-subcommand) prints help and exits non-zero."""
    exit_code = cli.main(["mcp"])
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert exit_code != 0
    assert "ping" in combined  # help mentions the available sub-subcommand


def test_cli_mcp_ping_uses_default_server_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--command``, ping targets the in-package server module.

    The post-install probe must point at the *installed* crossmem server
    (``python -m crossmem.server``), not some ambient ``crossmem`` shim,
    so the test fixes the command and asserts the CLI forwards it.
    """
    seen: list[list[str]] = []

    def fake_ping(cmd: list[str], timeout: float = 5.0) -> mcp_ping.PingResult:
        seen.append(cmd)
        return mcp_ping.PingResult(ok=True, tools=["query"], error=None)

    monkeypatch.setattr(cli, "ping", fake_ping)
    cli.main(["mcp", "ping"])
    assert seen, "ping was never called"
    cmd = seen[0]
    # The first element is the interpreter; the rest must invoke the module.
    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert "crossmem.server" in cmd


# ---------------------------------------------------------------------------
# End-to-end through the real subprocess machinery
# ---------------------------------------------------------------------------


def test_ping_end_to_end_against_fake_server(
    ok_server: list[str],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the CLI through subprocess against the fake server (no mocks).

    This proves the pieces fit together: the CLI spawns the subprocess,
    the wire layer sends ``initialize`` + ``tools/list``, the response
    parser extracts the tool names, and ``_cmd_mcp_ping`` formats the
    output. The fake server stands in for ``crossmem.server`` so the
    test stays fast and offline.
    """
    captured_cmd: list[list[str]] = []
    real_ping = mcp_ping.ping

    def spy(cmd: list[str], timeout: float = 5.0) -> mcp_ping.PingResult:
        # Replace the default command with the fake server but keep the
        # real wire-protocol implementation under test.
        captured_cmd.append(cmd)
        return real_ping(ok_server, timeout=timeout)

    monkeypatch.setattr(cli, "ping", spy)
    exit_code = cli.main(["mcp", "ping"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "ok" in out
    assert "query" in out
    assert "store" in out


def test_ping_unknown_command_returns_clean_failure(tmp_path: Path) -> None:
    """A command that cannot be spawned at all is reported, not raised."""
    bogus = tmp_path / "does-not-exist-anywhere"
    result = mcp_ping.ping([str(bogus)], timeout=1.0)
    assert result.ok is False
    assert result.error is not None
    assert "could not start server" in result.error or "exit" in result.error


def test_ping_initialize_error_response(tmp_path: Path) -> None:
    """A JSON-RPC error on ``initialize`` aborts before sending tools/list."""
    body = textwrap.dedent(
        """
        if method == "initialize":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": "init refused"},
            })
        """
    )
    server = _write_fake_server(tmp_path, body)
    result = mcp_ping.ping([sys.executable, str(server)], timeout=2.0)
    assert result.ok is False
    assert result.error is not None
    assert "init refused" in result.error
    assert "initialize" in result.error


def test_ping_invalid_json_on_wire(tmp_path: Path) -> None:
    """Non-JSON output from the server surfaces as ``invalid JSON``."""
    server = tmp_path / "garbage.py"
    server.write_text(
        textwrap.dedent(
            """
            import sys

            sys.stdout.write("this is not json\\n")
            sys.stdout.flush()
            for _ in sys.stdin:
                pass
            """
        ),
        encoding="utf-8",
    )
    result = mcp_ping.ping([sys.executable, str(server)], timeout=2.0)
    assert result.ok is False
    assert result.error is not None
    assert "invalid JSON" in result.error


def test_ping_non_object_response(tmp_path: Path) -> None:
    """A JSON array (not an object) on the wire is rejected explicitly."""
    server = tmp_path / "array.py"
    server.write_text(
        textwrap.dedent(
            """
            import sys

            sys.stdout.write("[1, 2, 3]\\n")
            sys.stdout.flush()
            for _ in sys.stdin:
                pass
            """
        ),
        encoding="utf-8",
    )
    result = mcp_ping.ping([sys.executable, str(server)], timeout=2.0)
    assert result.ok is False
    assert result.error is not None
    assert "JSON object" in result.error


def test_ping_skips_notifications_before_response(tmp_path: Path) -> None:
    """Notifications (no ``id``) emitted before the response are tolerated."""
    body = textwrap.dedent(
        """
        if method == "initialize":
            # Emit a notification first to exercise the skip-and-keep-reading path.
            _send({"jsonrpc": "2.0", "method": "notifications/progress"})
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "0.0.0"},
                },
            })
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [{"name": "only", "description": "z", "inputSchema": {}}],
                },
            })
        """
    )
    server = _write_fake_server(tmp_path, body)
    result = mcp_ping.ping([sys.executable, str(server)], timeout=2.0)
    assert result.ok is True
    assert result.tools == ["only"]


def test_ping_tools_list_returns_non_list(tmp_path: Path) -> None:
    """A malformed ``tools/list`` payload (e.g. ``tools`` as a dict) fails clearly."""
    body = textwrap.dedent(
        """
        if method == "initialize":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "0.0.0"},
                },
            })
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": {"not": "a list"}},
            })
        """
    )
    server = _write_fake_server(tmp_path, body)
    result = mcp_ping.ping([sys.executable, str(server)], timeout=2.0)
    assert result.ok is False
    assert result.error is not None
    assert "non-list" in result.error


def test_ping_initialize_request_is_well_formed(
    tmp_path: Path,
) -> None:
    """The ``initialize`` request crossmem sends must include the MCP basics.

    Recording the raw bytes the ping writes to a server gives us a tight
    grip on the wire format without us having to inspect the JSON-RPC
    library itself. The server records each line it receives, then
    closes — the ping reports the failure (no ``tools/list`` answer),
    which is fine; the assertions live on the recorded transcript.
    """
    transcript = tmp_path / "transcript.jsonl"
    recorder = tmp_path / "recorder.py"
    recorder_src = (
        "import json\n"
        "import sys\n"
        "\n"
        f"path = {str(transcript)!r}\n"
        'with open(path, "w", encoding="utf-8") as fh:\n'
        "    for raw in sys.stdin:\n"
        "        raw = raw.strip()\n"
        "        if not raw:\n"
        "            continue\n"
        '        fh.write(raw + "\\n")\n'
        "        fh.flush()\n"
        "        req = json.loads(raw)\n"
        '        if req.get("method") == "initialize":\n'
        "            resp = {\n"
        '                "jsonrpc": "2.0",\n'
        '                "id": req["id"],\n'
        '                "result": {\n'
        '                    "protocolVersion": "2024-11-05",\n'
        '                    "capabilities": {"tools": {}},\n'
        '                    "serverInfo": {"name": "rec", "version": "0"},\n'
        "                },\n"
        "            }\n"
        '            sys.stdout.write(json.dumps(resp) + "\\n")\n'
        "            sys.stdout.flush()\n"
    )
    recorder.write_text(recorder_src, encoding="utf-8")
    cmd = [sys.executable, str(recorder)]
    mcp_ping.ping(cmd, timeout=1.0)
    lines = [
        json.loads(line)
        for line in transcript.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    methods = [msg.get("method") for msg in lines]
    assert "initialize" in methods
    assert "tools/list" in methods
    init = next(m for m in lines if m.get("method") == "initialize")
    assert init["jsonrpc"] == "2.0"
    assert "id" in init
    assert "params" in init
    assert "protocolVersion" in init["params"]
