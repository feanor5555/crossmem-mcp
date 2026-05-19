# CrossMem

CrossMem is a knowledge store for AI coding CLIs. Install it once, then
talk to your LLM. No configuration, no commands to learn, no special
syntax. You keep coding, your LLM keeps the knowledge.

Your knowledge lives in a single file. Copy it to another machine and
everything moves with you. Works for one developer or a whole team.
Store as much as you want.

## How it feels

You manage knowledge by talking to your LLM. A few examples:

- "Add the knowledge from this file."
- "Store this link."
- "Remember the API docs at https://example.com/api."
- "Find what we know about retry logic."
- "Remove the docs for version 17, add the docs for version 18."
- "Forget everything tagged `legacy`."

That is the whole interface. Your LLM picks the right action and
CrossMem does it.

## Install

Tell your LLM in the CLI: "install crossmem". It handles the rest.
It may ask you one question. That is all.

If your LLM cannot find the package, share this link:
https://github.com/feanor5555/crossmem-mcp

## Supported

| Category | Options |
|---|---|
| CLIs | Claude Code, Cursor, Cline, OpenCode, Pi, Kilo Code, Continue.dev, Gemini CLI, Goose, Windsurf, Amazon Q CLI, Zed |
| Backends | SQLite (default), ChromaDB (optional), Qdrant (optional) |
| Sources | Web, GitHub, bring-your-own adapter |
| Languages | 50+ (multilingual search) |

## Why it works well

- Works with any MCP-capable coding CLI.
- Runs locally. No cloud account, no network calls during search.
- Combines full-text search and meaning-based search in one ranking,
  so you find things by the exact words you remember or by what they
  are about.
- Your whole knowledge base is one file in your home directory. Back
  it up, copy it, delete it — it is just a file.
- No vendor lock-in. Switch CLIs whenever you want; the knowledge
  stays.

## Documentation

- [`install/`](install/) — per-CLI install playbooks consumed by the
  LLM-driven zero-config install flow.
- [`skills/`](skills/) — Claude Code skill bundles shipped with the
  package.
