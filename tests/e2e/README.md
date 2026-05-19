# CrossMem End-to-End Test Suite

The E2E suite exercises CrossMem against the **published artefacts**
(the wheel, the installed `crossmem` console script, the MCP server)
inside disposable Docker containers. It complements — but does not
replace — the per-module unit tests under `tests/`. The unit tests
prove that individual functions behave correctly given mocked inputs;
the E2E suite proves that the installer, connectors and MCP server
hang together when a real CLI configuration lands on disk.

Running the suite is **manual** by design. Spinning up containers in
every CI job would dominate the runtime without proportional
information gain — the same container is rebuilt on every push only
to repeat the install/config validation that the unit tests already
cover for the relevant module. The suite is therefore opt-in:

```bash
bash tests/e2e/run_all.sh
```

The command above must exit 0 on a healthy tree. It writes a JSON
report to `tests/e2e/reports/<timestamp>.json` and a per-scenario log
file to `tests/e2e/reports/<timestamp>/<scenario>.log`.

## Prerequisites

* Docker daemon reachable from the shell (Docker Desktop on
  Windows/macOS, `docker.io` package on Debian/Ubuntu).
* `bash` 4+ on Linux/macOS, or PowerShell 7+ on Windows (for the
  Windows mirror introduced by task 27.2: `pwsh tests/e2e/run_all.ps1`).
* No network egress required at run time — the working-copy is
  baked into the image during `docker build`. `apt-get` and `pipx
  install` happen exactly once, at image-build time.

## Linux runner (task 27.1)

The Linux runner builds `tests/e2e/docker/Dockerfile.linux` with the
repo root as build context, then invokes each scenario script
sequentially inside a freshly-spawned container.

```bash
bash tests/e2e/run_all.sh
```

Override the image tag via the `CROSSMEM_E2E_LINUX_IMAGE` environment
variable if you need to run two branches side by side:

```bash
CROSSMEM_E2E_LINUX_IMAGE=crossmem-e2e:pr-123 bash tests/e2e/run_all.sh
```

## Windows runner (task 27.2)

The Windows runner is the byte-for-byte mirror of the Linux runner on
the `mcr.microsoft.com/windows/servercore:ltsc2022` base image. It
builds `tests/e2e/docker/Dockerfile.windows` with the repo root as
build context, then invokes each PowerShell scenario script
(`scenarios/*/*.ps1`) sequentially inside a freshly-spawned container.

```powershell
pwsh tests/e2e/run_all.ps1
```

Requirements:

* PowerShell 7+ (the cross-platform `pwsh`, not legacy Windows
  PowerShell 5.1). `Set-StrictMode -Version Latest` and
  `[ordered]@{...}` work on 5.1 too, but the manual command above is
  documented against the modern shell.
* Docker Desktop running in **Windows-containers** mode. The default
  Linux-containers mode cannot pull `mcr.microsoft.com/windows/*`
  images; switch via the Docker Desktop tray menu before invoking the
  runner.

Override the image tag via the `CROSSMEM_E2E_WINDOWS_IMAGE`
environment variable, mirroring the Linux convention:

```powershell
$env:CROSSMEM_E2E_WINDOWS_IMAGE = 'crossmem-e2e:pr-123'
pwsh tests/e2e/run_all.ps1
```

The Windows report uses the same schema as the Linux one, with
`runner: "windows"`; see the *Report schema* section below for the
field reference and a representative payload.

## Report schema

Every runner writes a report under `tests/e2e/reports/` with the
following shape (task 27.1 fixes this schema for tasks 27.2 and
27.4+):

```json
{
  "runner": "linux",
  "scenarios": [
    {
      "name": "scenarios/_smoke/hello.sh",
      "status": "pass",
      "duration_s": 0.420,
      "log_path": "reports/20260512T173045Z/hello.log"
    }
  ],
  "started_at": "2026-05-12T17:30:45Z",
  "finished_at": "2026-05-12T17:30:46Z"
}
```

The Windows runner produces the same shape with `runner: "windows"`
and `.ps1` scenario names:

```json
{
  "runner": "windows",
  "scenarios": [
    {
      "name": "scenarios/_smoke/hello.ps1",
      "status": "pass",
      "duration_s": 0.610,
      "log_path": "reports/20260512T173045Z/hello.log"
    }
  ],
  "started_at": "2026-05-12T17:30:45Z",
  "finished_at": "2026-05-12T17:30:46Z"
}
```

Fields:

* `runner` — one of `"linux"` or `"windows"`. Identifies which
  Dockerfile produced the run.
* `scenarios[]` — ordered list, one entry per executed scenario.
  Each entry carries `name` (path relative to `docker/`), `status`
  (`"pass"` or `"fail"`), `duration_s` (wall-clock seconds with
  ms precision), and `log_path` (location of the captured stdout +
  stderr, relative to `tests/e2e/`).
* `started_at` / `finished_at` — ISO-8601 UTC timestamps marking
  the wall-clock window of the run. Useful when correlating with
  CI artefacts.

## LLM-endpoint runners

Tasks 27.3a-c run the happy-path scenarios against a matrix of LLM
backends (`qwen`, `opus`, `mock`). Each runner lives in
`tests/e2e/runners/` and either executes the scenario or emits a
report fragment with `status: "skipped"` when its backend is not
configured. The harness keeps going either way.

### Matrix overview (task 27.3c)

The happy-path scenarios in `tests/e2e/scenarios/happy_path/` are
parametrised across `{qwen, opus, mock}` via the `runner` fixture in
`tests/e2e/conftest.py`. The fixture pulls the runner list from
`tests/e2e/matrix.py`, which is the single source of truth for the
matrix — add a runner there and every happy-path scenario picks it up
automatically.

Skip behaviour is per runner, not per scenario:

| Runner | Skip when... | Always-available |
|--------|--------------|------------------|
| `qwen` | `GET <url>/v1/models` is unreachable (connect error, timeout, non-2xx). | no — needs a reachable Qwen 3.5 endpoint |
| `opus` | `ANTHROPIC_API_KEY` is unset/blank, or `POST /v1/messages` probe fails. | no — needs a paid Anthropic key |
| `mock` | never. | **yes** — offline-CI guarantee |

The `mock` runner spins an in-process `MockLLMServer` on an ephemeral
loopback port for every scenario, exports the URL via
`CROSSMEM_E2E_MOCK_URL`, and stops the server afterwards. Scenarios
that need a deterministic LLM response use the fixtures under
`tests/e2e/mock_llm/fixtures/`; mock-only scenarios that want to
*break* the upstream use the fault fixtures under
`tests/e2e/mock_llm/fault_fixtures/` (see below).

### Mock runner & fault injection (task 27.3c)

`tests/e2e/runners/mock.py` is the offline column of the matrix. It
backs `tests/e2e/scenarios/fault_injection/`, which deliberately
exercises the four failure modes the mock LLM can simulate:

| Scenario module                                    | Fault                | Server knob          |
|----------------------------------------------------|----------------------|----------------------|
| `broken_json.py`                                   | malformed JSON body  | `raw_body`           |
| `http_5xx.py`                                      | HTTP 5xx response    | `status_code`        |
| `timeout_slow.py`                                  | upstream timeout     | `delay_s`            |
| `empty_tool_calls.py`                              | empty tool-call list | (none — plain fixture) |

The knobs are documented in `tests/e2e/mock_llm/server.py` and the
fault fixtures live in `tests/e2e/mock_llm/fault_fixtures/`. The
qwen and opus runners *cannot* exercise these failure modes — live
endpoints don't lie on demand — which is why fault-injection is mock-
only by design.

Quick check that the mock runner is wired up:

```bash
python -m tests.e2e.runners.mock --check-only
# stdout: {"name": "mock/connection-check", "status": "pass", ...}
# exit code: 0 (always — the mock has no skip path)
```

### Qwen runner (task 27.3a)

`tests/e2e/runners/qwen.py` targets a real Qwen 3.5 backend exposed
over an OpenAI-compatible REST endpoint (model id `qwen-3.5`).
Configuration lives in env vars — copy `tests/e2e/.env.example` to
`tests/e2e/.env` and override as needed:

| Variable                     | Default                          | Notes |
|------------------------------|----------------------------------|-------|
| `CROSSMEM_E2E_QWEN_URL`      | `http://192.168.178.45:8080`     | Empty / unset falls back to the default. Strip any trailing slash.

Before every scenario the runner probes `GET <url>/v1/models` with a
short timeout. Outcomes:

* **2xx response** — scenario runs normally. The report fragment uses
  `status: "pass"` or `"fail"` and the wall-clock duration.
* **Connect error, timeout, or non-2xx** — the runner *skips* the
  scenario. Fragment status is `"skipped"`, exit code is `2`, and the
  `reason` field spells out why (`endpoint unreachable: ConnectError`,
  `HTTP 503 from /v1/models`, etc.). CI hosts without a reachable Qwen
  see a clean skip instead of a hard failure.

Quick reachability probe (mirrors what `tests/e2e/run_all.sh` will do
once 27.3c wires the matrix together):

```bash
python -m tests.e2e.runners.qwen --check-only
# stdout: {"name": "qwen/connection-check", "status": "skipped", ...}
# exit code: 0 (reachable) or 2 (skipped)
```

### Opus runner (task 27.3b)

`tests/e2e/runners/opus.py` targets the official Anthropic Messages API
(model family `claude-opus`). The runner has no default endpoint URL —
it always points at `https://api.anthropic.com` — and authentication is
required via env var:

| Variable             | Default | Notes |
|----------------------|---------|-------|
| `ANTHROPIC_API_KEY`  | *(none)*| Required. Unset or blank values cause the runner to skip cleanly with `status: "skipped"` and `reason: "ANTHROPIC_API_KEY not set"`. Get a key at <https://console.anthropic.com/>.

When a key is present the runner sends a minimal probe call
(`POST /v1/messages`, `max_tokens=1`) before each scenario:

* **2xx response** — scenario runs normally; the fragment uses
  `status: "pass"` or `"fail"` and the wall-clock duration.
* **No key / blank key** — fragment status is `"skipped"`, exit code
  is `2`, and the `reason` field reads `ANTHROPIC_API_KEY not set`.
  The probe call is never sent, so cost stays at zero.
* **Auth error, rate limit, connect error, timeout, non-2xx** — the
  runner *skips* the scenario with an explanatory `reason` (e.g.
  `api unreachable: HTTP 401 from /v1/messages`). CI hosts without an
  Anthropic account see a clean skip instead of a hard failure.

Quick reachability probe:

```bash
python -m tests.e2e.runners.opus --check-only
# stdout: {"name": "opus/connection-check", "status": "skipped", ...}
# exit code: 0 (key valid, API reachable) or 2 (skipped)
```

## Layout

```
tests/e2e/
+-- README.md              # this file
+-- .env.example           # opt-in env vars per runner (task 27.3a+)
+-- run_all.sh             # Linux entry-point (task 27.1)
+-- run_all.ps1            # Windows entry-point (task 27.2)
+-- docker/
|   +-- Dockerfile.linux   # task 27.1
|   +-- Dockerfile.windows # task 27.2
|   +-- scenarios/
|       +-- _smoke/        # runner self-test ("does the image build at all?")
|       |   +-- hello.sh   # Linux canary
|       |   +-- hello.ps1  # Windows canary
|       +-- <cli>/         # per-CLI scenarios (tasks 27.4 - 27.15, TBD)
+-- runners/               # LLM-endpoint runners (task 27.3a+)
|   +-- qwen.py            # Qwen 3.5 over OpenAI-compatible REST (27.3a)
|   +-- opus.py            # Anthropic Claude Opus Messages API (27.3b)
+-- reports/               # JSON reports + per-scenario logs
    +-- .gitkeep
```

## Troubleshooting

* **`docker: command not found`** — install Docker Desktop (Windows /
  macOS) or `docker.io` (Debian/Ubuntu). The suite intentionally does
  not auto-install Docker on the host.
* **Image build pulls fresh layers every time** — `docker build`
  invalidates the cache when `pyproject.toml` or `src/` changes. To
  inspect what changed, run with `--progress=plain --no-cache=false`.
* **`reports/<timestamp>.json` missing after a failed run** — the
  script writes the report after **every** scenario completes, even
  on failure. A missing file means the runner itself crashed before
  the loop ended; check the script's stderr.
