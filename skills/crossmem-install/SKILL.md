---
name: crossmem-install
description: Install and register crossmem (portable MCP knowledge database) into the current AI coding CLI. Use this when the user asks to install crossmem, set up crossmem, register crossmem as an MCP server, add knowledge/memory to their CLI, or wire crossmem into Claude Code, Cursor, Cline, OpenCode, Pi, Kilo Code, Continue.dev, Gemini CLI, Goose, Windsurf, Amazon Q CLI, or Zed.
compatibility: Requires Python 3.10+ and pipx
---

# crossmem-install

This skill tells you, the LLM, how to install [crossmem](https://github.com/feanor5555/knowledge) into the CLI you are running inside. crossmem is a portable, offline-first, MCP-based knowledge database backed by a single SQLite file. After installation the user gains MCP tools such as `query`, `store`, `status`, `export`, and `import_data` inside this CLI.

Follow this skill end-to-end. Do **not** ask the user for permission for individual steps unless a destructive action is reached (see "Safety"). The whole point of crossmem is zero-config: you do the work, the user sees the result.

## When to use

Trigger on any of:

- "install crossmem", "set up crossmem", "add crossmem"
- "register crossmem as MCP server"
- "give Claude/Cursor/... long-term memory" combined with crossmem
- "configure crossmem for this CLI"
- The user mentions a missing `query`/`store` MCP tool and references crossmem

If the user is asking about crossmem **usage** (querying, storing, exporting data) and crossmem is already installed, this is the wrong skill — use the regular MCP tools directly.

## Progressive Disclosure: read only what you need

This file is the **master installer**. It contains:

1. The detect step — figure out which of the 12 supported CLIs you are running inside.
2. The shared prerequisites — Python, pipx, `crossmem` itself.
3. A pointer to the **per-CLI file** under `install/<cli>.md` that contains the exact JSON/YAML snippet, config path, and restart hint for the CLI you detected.

Do **not** load all 12 per-CLI files. Load **one** — the one matching the detected CLI. Each per-CLI file is self-contained: prerequisites, install command, MCP-config snippet, verify command, troubleshooting. That is the agent-skills.io Progressive Disclosure contract.

## Step 1: Detect the host CLI

Identify which CLI is invoking this skill. Use these signals, in order:

1. Environment / process — names like `claude`, `cursor`, `cline`, `opencode`, `pi`, `kilocode`, `continue`, `gemini`, `goose`, `windsurf`, `q`, `zed`.
2. Config file presence — see the per-CLI files for canonical paths.
3. If still ambiguous, ask the user exactly once: "Which CLI am I in? (Claude Code / Cursor / Cline / OpenCode / Pi / Kilo Code / Continue.dev / Gemini CLI / Goose / Windsurf / Amazon Q CLI / Zed)".

Supported CLIs and their per-CLI install files:

| CLI | File |
| --- | --- |
| Claude Code | `install/claude-code.md` |
| Cursor | `install/cursor.md` |
| Cline | `install/cline.md` |
| OpenCode | `install/opencode.md` |
| Pi | `install/pi.md` |
| Kilo Code | `install/kilo-code.md` |
| Continue.dev | `install/continue.md` |
| Gemini CLI | `install/gemini.md` |
| Goose | `install/goose.md` |
| Windsurf | `install/windsurf.md` |
| Amazon Q CLI | `install/amazon-q.md` |
| Zed | `install/zed.md` |

These files live in the crossmem repository under `install/`. If the user does not have the repo checked out locally, fetch the file you need from the repo's main branch, or run `crossmem docs install --cli <name>` (once installed) to print the same content.

If the detected CLI is **not** in the table, stop. Tell the user crossmem currently supports the 12 CLIs listed above and link to the project README.

## Step 2: Shared prerequisites

Run these checks before touching any CLI config. They are identical for every CLI; the per-CLI file assumes they are already green.

### 2.1 Python 3.10+

```bash
python --version
```

Expect `Python 3.10.x` or higher. On Windows you may need `py -3 --version`. If older, instruct the user to install a newer Python (do not auto-upgrade their system Python).

### 2.2 pipx

```bash
pipx --version
```

If pipx is missing:

- Linux/macOS: `python -m pip install --user pipx && python -m pipx ensurepath`
- Windows: `py -m pip install --user pipx && py -m pipx ensurepath`

Then ask the user to open a new shell so `pipx` is on PATH, and re-run this skill.

### 2.3 Install crossmem

```bash
pipx install crossmem
```

If `crossmem` is already installed, upgrade idempotently:

```bash
pipx upgrade crossmem
```

Either command is safe to run twice.

### 2.4 Preflight: `crossmem doctor`

```bash
crossmem doctor
```

This must exit `0`. It verifies the Python version, fastembed model availability, SQLite + sqlite-vec, and write access to `~/.crossmem/`. If `doctor` fails:

- `embedding model missing` — the first run downloads ~300 MB. Let it finish; do not interrupt.
- `~/.crossmem not writable` — check the user's home directory permissions.
- `sqlite-vec extension load failed` — confirm the user's Python is the one pipx used (`pipx environment --value PIPX_LOCAL_VENVS`).

Do not proceed to Step 3 until `doctor` is green.

## Step 3: Run `crossmem install`

```bash
crossmem install
```

This subcommand:

1. Auto-detects every supported CLI on the machine (not just the host CLI).
2. Backs up each detected CLI's config to `<config>.bak` before any write.
3. Writes the MCP server entry for crossmem into each detected CLI's config file in the correct shape (each CLI uses a slightly different key — `mcpServers`, `mcp.servers`, `context_servers`, `extensions`, etc.). The per-CLI file shows the exact resulting JSON/YAML.
4. Creates `~/.crossmem/knowledge.db` (SQLite + sqlite-vec + FTS5) on first run.

The command is idempotent: re-running it does not duplicate entries. If a backup already exists it is left untouched.

If the user wants to preview without writing, use `crossmem install --dry-run` (when available) to see the planned diff.

## Step 4: CLI-specific finalization

**Read the matching `install/<cli>.md` file now.** It contains:

- The exact path of the config file on Linux / macOS / Windows.
- The exact JSON or YAML block that `crossmem install` wrote into it.
- The restart instruction (GUI CLIs like Cursor, Cline, Kilo Code, Windsurf, Zed, and Continue.dev need a full restart; CLI-driven tools like Claude Code, OpenCode, Goose, Gemini, Amazon Q, and Pi reload on next invocation).
- A verify command and its expected output.
- CLI-specific troubleshooting.

Follow the file's "Verify" section verbatim. If verification fails, follow its "Troubleshooting" section.

## Step 5: MCP roundtrip self-test

After the per-CLI Verify section is green, prove the MCP wiring works from the host CLI itself. Issue an MCP `query` call through whatever interface the host CLI exposes for MCP tools:

- Expected behavior on a fresh install: `query` returns an empty result list (the database is empty), not an error. An empty result is a **pass** — it means the MCP roundtrip succeeded.
- If the host CLI exposes `crossmem mcp ping` (added in newer crossmem versions), run it; expect `ok` plus a tool list of at least `query`, `store`, `status`.

If `query` errors out:

- Re-check the per-CLI file's Verify section.
- Confirm the user restarted GUI CLIs.
- Run `crossmem doctor` again — if anything changed since Step 2.4 it will surface here.

## Step 6: Tell the user what they have

Once Step 5 is green, summarize for the user in one short paragraph:

- crossmem is installed and registered with `<detected CLI>`.
- `~/.crossmem/knowledge.db` is the single source of truth.
- MCP tools now available: `query`, `store`, `delete`, `cleanup`, `status`, `export`, `import_data`, `configure`.
- Optional backends (ChromaDB / Qdrant) are available via `pip install 'crossmem[chroma]'` or `pip install 'crossmem[qdrant]'` followed by `crossmem configure --backend <name>`. Do **not** install these unless the user asked for a remote backend.

## Versioning and upgrades

When the user upgrades crossmem later (`pipx upgrade crossmem`):

1. Re-run `crossmem doctor` — it surfaces schema migrations and missing assets.
2. Re-run `crossmem install` — idempotent; reapplies any new connector fields without duplicating entries.
3. If the upgrade changed the embedding model (rare, announced in release notes), `doctor` will instruct a full re-embed. Tell the user this is destructive of cached vectors but content is preserved.

Always re-run `doctor` after an upgrade. Never skip it just because "nothing looks broken".

## Safety

These are the only points where you must pause and confirm with the user:

- `crossmem delete --permanent` — bypasses the 30-day trash. Confirm intent before invoking.
- `crossmem configure --backend <remote>` with a non-local URL — sends data off-machine. Confirm intent and verify the URL is HTTPS.
- Manual edits to a CLI's config file outside `crossmem install` — prefer the installer; only edit by hand when the per-CLI Troubleshooting section says so.

Everything else in this skill (install, register, verify, re-run on upgrade) is non-destructive and idempotent. Do not ask permission for each step.

## If something goes wrong

In order:

1. `crossmem doctor` — the single source of truth for environment health. Read its full output.
2. The per-CLI `install/<cli>.md` Troubleshooting section.
3. The project issue tracker — link surfaced by `crossmem doctor --json` under the `support` key.

Do not silently retry failing commands in a loop. Surface the failure and diagnose.
