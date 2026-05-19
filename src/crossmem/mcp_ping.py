"""Self-test probe that talks the MCP wire protocol over stdio.

Used by ``crossmem mcp ping`` to verify a freshly installed crossmem
MCP server is reachable and answers ``tools/list`` correctly. The post-
install path looks like this:

1. ``crossmem install`` writes the MCP-client config (e.g. the Goose
   YAML or the Claude Desktop JSON) and drops the DB file.
2. ``crossmem doctor`` confirms the Python environment is healthy.
3. ``crossmem mcp ping`` actually *spawns* the server and exchanges
   real JSON-RPC frames — the only step that proves the wiring works
   end-to-end before the user hands off to their LLM.

We talk the protocol directly (line-delimited JSON-RPC 2.0 on stdin/
stdout) rather than instantiate the upstream MCP client. That avoids
pulling another runtime dependency into the install probe, keeps the
fake-server tests trivial (a 10-line Python script), and matches the
wire format that every supported CLI's MCP client already speaks.

The ``initialize`` -> ``initialized`` -> ``tools/list`` handshake is
the smallest sequence that exercises the same code path the real
clients hit. Anything less (e.g. just spawning the server) would
mask protocol regressions.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["PingResult", "default_server_command", "ping"]

# MCP protocol version we advertise during ``initialize``. Servers are
# expected to negotiate down if they only speak an older spec; either
# way the response carries its own ``protocolVersion`` field so we
# don't need to track the latest revision aggressively here.
_PROTOCOL_VERSION = "2024-11-05"

# Client identity surfaced in the ``initialize`` params. Servers may
# log this; tests don't depend on the exact value.
_CLIENT_INFO = {"name": "crossmem-ping", "version": "1"}


@dataclass
class PingResult:
    """Outcome of a single MCP ping attempt.

    ``ok=True`` means the server answered ``tools/list`` and ``tools``
    holds the advertised names (in the server's order). ``ok=False``
    means something went wrong; ``error`` describes it in one short
    line suitable for both human output and log lines.
    """

    ok: bool
    tools: list[str] = field(default_factory=list)
    error: str | None = None


def default_server_command() -> list[str]:
    """Command that spawns the in-package MCP server over stdio.

    Using ``sys.executable -m crossmem.server`` (instead of the
    ``crossmem`` console script) guarantees we hit the same Python
    interpreter and the same installed package the user is running
    the CLI from — no PATH ambiguity, no shim drift.
    """
    return [sys.executable, "-m", "crossmem.server"]


def ping(command: Sequence[str], timeout: float = 5.0) -> PingResult:
    """Spawn ``command`` as an MCP server and exchange one handshake.

    Parameters
    ----------
    command:
        Argv used to start the server (e.g. ``["python", "-m",
        "crossmem.server"]``). The process is terminated after the
        handshake — successful or not.
    timeout:
        Maximum total time, in seconds, we wait for the server to
        answer both requests combined. A silent or slow server hits
        this and is reported as ``timeout`` rather than hanging.

    The server is always cleaned up before we return. On Windows
    ``terminate()`` is enough; on POSIX ``kill()`` is the fallback
    if the process is wedged.
    """
    try:
        proc = subprocess.Popen(  # noqa: S603 — command comes from us, not user input
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            encoding="utf-8",
        )
    except (OSError, FileNotFoundError) as exc:
        return PingResult(ok=False, error=f"could not start server: {exc}")

    # Reader thread queues every line the server emits on stdout. The
    # main thread polls with a timeout so we can give up cleanly when
    # the server stays silent — Popen.communicate would block forever.
    queue: Queue[str | None] = Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                queue.put(line)
        finally:
            queue.put(None)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    try:
        return _run_handshake(proc, queue, timeout)
    finally:
        # Best-effort shutdown. ``terminate`` is the polite signal;
        # ``kill`` handles servers that ignore it. We never raise from
        # the cleanup path so the caller always gets the PingResult.
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
        except OSError:
            pass


def _run_handshake(
    proc: subprocess.Popen[str],
    queue: Queue[str | None],
    timeout: float,
) -> PingResult:
    """Drive the ``initialize`` + ``tools/list`` exchange.

    Splitting this out of :func:`ping` keeps the resource-management
    try/finally clean and makes the protocol flow readable top-to-
    bottom: send, wait, validate, repeat.
    """
    # 1. Send ``initialize`` and wait for the response. Any JSON-RPC
    #    error here is fatal — the server simply isn't reachable.
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": _CLIENT_INFO,
        },
    }
    if not _send(proc, init_req):
        return _early_failure(proc, "failed to write initialize request")
    init_resp = _read_response(queue, proc, timeout)
    if init_resp.error is not None:
        return init_resp
    if "error" in init_resp.raw:
        msg = init_resp.raw["error"].get("message", "initialize failed")
        return PingResult(ok=False, error=f"initialize error: {msg}")

    # 2. Send the ``initialized`` notification (no response expected)
    #    and immediately follow with ``tools/list``. Per the MCP spec,
    #    clients must send the notification before any other request.
    notify = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    if not _send(proc, notify):
        return _early_failure(proc, "failed to write initialized notification")

    tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    if not _send(proc, tools_req):
        return _early_failure(proc, "failed to write tools/list request")
    tools_resp = _read_response(queue, proc, timeout)
    if tools_resp.error is not None:
        return tools_resp
    if "error" in tools_resp.raw:
        msg = tools_resp.raw["error"].get("message", "tools/list failed")
        return PingResult(ok=False, error=f"tools/list error: {msg}")

    tools = tools_resp.raw.get("result", {}).get("tools", [])
    if not isinstance(tools, list):
        return PingResult(ok=False, error="tools/list returned non-list 'tools'")
    names = [t.get("name", "") for t in tools if isinstance(t, dict)]
    return PingResult(ok=True, tools=names)


@dataclass
class _Response:
    """Internal helper: either a parsed JSON-RPC response or a failure."""

    raw: dict
    error: str | None = None


def _send(proc: subprocess.Popen[str], payload: dict) -> bool:
    """Write one JSON-RPC frame to the server. Returns False on broken pipe."""
    assert proc.stdin is not None
    try:
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        return False
    return True


def _read_response(
    queue: Queue[str | None],
    proc: subprocess.Popen[str],
    timeout: float,
) -> _Response | PingResult:
    """Pull the next non-blank line from the server, parsed as JSON.

    Returns a :class:`_Response` on success or a :class:`PingResult`
    when we already know the ping failed (timeout / server died /
    garbage on the wire). The two-return-shape keeps the caller's
    happy path linear.
    """
    while True:
        try:
            line = queue.get(timeout=timeout)
        except Empty:
            return PingResult(ok=False, error=f"timeout after {timeout}s")
        if line is None:
            # stdout closed before we got a reply — surface the exit
            # code and a snippet of stderr so the user can debug.
            return _early_failure(proc, "server closed stdout without responding")
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return PingResult(ok=False, error=f"invalid JSON from server: {exc}")
        if not isinstance(data, dict):
            return PingResult(ok=False, error="server response was not a JSON object")
        # Servers may emit notifications (no ``id``) ahead of responses
        # — skip them and keep waiting for the actual reply.
        if "id" not in data and "method" in data:
            continue
        return _Response(raw=data)


def _early_failure(proc: subprocess.Popen[str], detail: str) -> PingResult:
    """Build a fail result enriched with the server's exit status / stderr.

    Called from any code path where the server has clearly given up:
    process exited, stdout closed, write failed. Reading stderr is
    bounded by the already-finished process — no risk of hanging.
    """
    if proc.poll() is not None:
        proc.wait(timeout=1.0)
    code = proc.poll()
    stderr_tail = ""
    if proc.stderr is not None:
        try:
            stderr_tail = proc.stderr.read() or ""
        except (OSError, ValueError):
            stderr_tail = ""
    parts = [detail]
    if code is not None:
        parts.append(f"exit={code}")
    if stderr_tail.strip():
        # Keep the message compact — only the first line is useful in
        # a single-line CLI summary.
        first = stderr_tail.strip().splitlines()[0]
        parts.append(f"stderr={first}")
    return PingResult(ok=False, error="; ".join(parts))
