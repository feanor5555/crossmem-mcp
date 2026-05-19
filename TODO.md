# CrossMem — Open Tasks

Nur offene Tasks. Erledigtes steht im `git log`, Wahrheit steht im Code.

---

## 27. E2E Test Suite (Docker, Scripts)

<!-- 27.3 (mock-llm-server) wurde rescoped: Fault-Injection + Offline-Fallback werden Teil von 27.3c (e2e-llm-matrix). Originaler Mock-LLM-Task entfaellt. -->

- [ ] **27.19 mcp-direct-stdio**
  Pfad: `tests/e2e/docker/scenarios/common/mcp_direct.sh`.
  DoD: Skript spawnt `crossmem mcp serve` als Child-Process, kommuniziert via stdio-JSON-RPC und ruft jedes registrierte MCP-Tool genau einmal: `store`, `query`, `delete`, `cleanup` (mit `dry_run=True`), `status`, `export`, `import_data`. Zusaetzlich werden die Destructive-Gate-Cases verifiziert: (a) `cleanup` mit `dry_run=False` ohne `CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1` liefert Response mit `forced_dry_run=True`; (b) `empty_trash` ohne den Env-Switch liefert `{"removed": 0, "blocked": True, ...}`; (c) mit `CROSSMEM_ALLOW_DESTRUCTIVE_MCP=1` laufen beide destruktiven Pfade durch. Exit 0 wenn alle Tool-Responses dem Erwartungs-Schema entsprechen und das Gate korrekt blockt.

- [ ] **27.20 local-llm-vllm**
  Pfad: `tests/e2e/docker/scenarios/common/llm_check.sh`, Doku-Eintrag (z.B. `tests/e2e/README.md` oder `docs/e2e-llm.md`).
  DoD: User-Setup ist ein lokal laufender vLLM-Server mit OpenAI-kompatiblem Endpoint (Modell Qwen 3.5 7B/8B). Die E2E-Suite spawnt KEIN eigenes LLM — sie nutzt den vorhandenen vLLM-Endpoint. Konfiguration: `VLLM_BASE_URL` (z.B. `http://host.docker.internal:8000/v1`) und `VLLM_MODEL` (z.B. `Qwen/Qwen3.5-7B-Instruct`) in `tests/e2e/.env`; `.env.example` listet beide Variablen ohne Werte. Doku erklaert, wie der User seinen vLLM-Server adressiert (Host-IP/`host.docker.internal`, Port, Modell-Name) und welche Defaults gelten. Skript `llm_check.sh` macht (a) einen Chat-Completion-Probe-Call `POST {VLLM_BASE_URL}/chat/completions` mit dem Modell und einem trivialen Prompt, verifiziert die Antwortstruktur (`choices[0].message.content` nicht leer); (b) einen Tool-Call-Probe-Call mit einer Mini-Tool-Definition, verifiziert dass `choices[0].message.tool_calls` befuellt ist. Wenn `VLLM_BASE_URL` fehlt: exit `2 = skipped` (analog 27.23). Exit 0 wenn beide Probe-Calls liefern.

- [ ] **27.21 roundtrip-goose**
  Pfad: `tests/e2e/docker/scenarios/goose/roundtrip.sh`.
  DoD: Skript startet einen Goose-Container, konfiguriert ihn gegen den vLLM-OpenAI-Endpoint aus 27.20 (`VLLM_BASE_URL`/`VLLM_MODEL` aus `tests/e2e/.env`) und gegen den crossmem-MCP-Server, sendet einen Prompt, der das Modell zwingt, erst `crossmem.store` und dann `crossmem.query` aufzurufen. Verifiziert via DB-Inspektion, dass der gestorete Doc existiert, und via `query`-Response, dass er gefunden wird. Exit 0 bei OK. Skip-Verhalten wie 27.20 (exit `2`) wenn `VLLM_BASE_URL` fehlt.

- [ ] **27.22 roundtrip-opencode**
  Pfad: `tests/e2e/docker/scenarios/opencode/roundtrip.sh`.
  DoD: Wie 27.21, fuer OpenCode gegen den vLLM-OpenAI-Endpoint.

- [ ] **27.23 roundtrip-gemini-cli**
  Pfad: `tests/e2e/docker/scenarios/gemini-cli/roundtrip.sh`, `tests/e2e/.env.example`.
  DoD: Gemini CLI Roundtrip via OAuth-Free-Tier (Google-Account-Login, Free-Tier-Quota), kein API-Key. User fuehrt einmalig `gemini auth login` auf dem Host aus; das erzeugt eine Credentials-Datei (Pfad gemaess Gemini-CLI-Doku, typischerweise unter `~/.config/gemini/` bzw. `~/.gemini/` — exakter Pfad bei der Implementation aus der Gemini-CLI-Doku verifizieren). Das Test-Skript mountet diese Credentials-Datei readonly in den Container und startet den Gemini-CLI-Container gegen den crossmem-MCP-Server. Der Test sendet einen Prompt, der das Modell zwingt, erst `crossmem.store` und dann `crossmem.query` aufzurufen. Verifiziert via DB-Inspektion, dass der gestorete Doc existiert, und via `query`-Response, dass er gefunden wird. Wenn die Credentials-Datei auf dem Host fehlt: Skript exittet mit dediziertem Code `2 = skipped`, Report-Eintrag bekommt `status: "skipped"`. Kein API-Key in `.env` noetig. Exit 0 wenn beide Tool-Calls erfolgreich.

<!-- 27.24 entfaellt: roundtrip-amazon-q gestrichen. Nummer bleibt absichtlich frei. -->

<!-- 27.25 entfaellt: Claude Code laeuft im Docker nicht, da die Anthropic Max-Subscription keinen API-Key liefert. Manuelle Pflicht-Pruefung in der Release-Checkliste. Nummer bleibt absichtlich frei. -->


---

## 28. Additional Connectors (Kiro, Qwen CLI)

<!-- Sobald 28.1 und 28.2 gemergt sind, waechst die offiziell unterstuetzte CLI-Liste von 12 auf 14. README und CLAUDE.md (Connector-Liste, Vision-Abschnitt "12 CLIs") werden vom jeweiligen Implementation-Subagent im selben Worktree mit angepasst. -->

- [ ] **28.1 connector-kiro**
  Pfad: `src/crossmem/connectors/kiro.py`, `src/crossmem/connectors/__init__.py`, `tests/connectors/test_kiro.py`.
  DoD: Implementation abgeleitet von `CLIConnector` (siehe `src/crossmem/connectors/base.py`). Methoden: `name()`, `detect()`, `config_path()` (plattformspezifisch — Linux/macOS/Windows), `register(server_cmd)`, `unregister()`. Recherche zu Kiros MCP-Config-Format (AWS Kiro AI IDE) via @web-scraper, gefundenes Format im Source-Modul als Kommentar referenzieren. Tests analog zu bestehenden Connector-Tests in `tests/connectors/` (detect-Stub via Fake-Home, register/unregister-Roundtrip, Idempotenz, Backup-`.bak`-Pflicht). Eintrag in `src/crossmem/connectors/__init__.py`-Registry — `crossmem doctor` und `crossmem install` finden den Connector dadurch automatisch (Factory aus Task 26.20). README- und CLAUDE.md-Liste der unterstuetzten CLIs im selben Commit erweitern (12 -> 13). Coverage >=90% auf `kiro.py`.

- [ ] **28.2 connector-qwen-cli**
  Pfad: `src/crossmem/connectors/qwen_cli.py`, `src/crossmem/connectors/__init__.py`, `tests/connectors/test_qwen_cli.py`.
  DoD: Analog 28.1 fuer Qwen CLI (Alibaba `qwen-code`, GitHub `QwenLM/qwen-code`). Recherche zur Config-Datei der Qwen CLI via @web-scraper, Format im Source-Modul kommentieren. Tests + Registry-Eintrag wie 28.1. README- und CLAUDE.md-Liste auf 14 erweitern (sofern 28.1 bereits gemergt ist, sonst auf 13 -> 14 anpassen). Coverage >=90% auf `qwen_cli.py`.

- [ ] **28.3 install-doc-kiro**
  Pfad: `install/kiro.md`, `tests/skills/test_install_docs.py` (oder vergleichbarer Skill-Test).
  DoD: LLM-adressierte Install-Anleitung im selben Stil wie die uebrigen `install/<cli>.md` (siehe bestehende Files unter `install/`). Direkt ausfuehrbare Schritte mit fertigen JSON-Snippets und plattformspezifischen Pfaden (Linux/macOS/Windows), keine Platzhalter. Idempotent (Backup-`.bak`, kein doppelter Eintrag). Endet mit Verifikationssequenz `crossmem doctor` (`exit 0`) plus MCP-Roundtrip (`store` -> `query` mit Erwartungswert). Skill-Tests in `tests/skills/` decken die neue Datei ab (Existenz, Pflicht-Sektionen, kein Platzhalter-Token).

- [ ] **28.4 install-doc-qwen-cli**
  Pfad: `install/qwen_cli.md`, `tests/skills/test_install_docs.py` (oder vergleichbarer Skill-Test).
  DoD: Analog 28.3 fuer Qwen CLI.

- [ ] **28.5 cli-config-validate-kiro**
  Pfad: `tests/e2e/docker/scenarios/kiro/install-validate.sh`, `tests/e2e/docker/scenarios/kiro/install-validate.ps1`.
  DoD: Analog 27.4 (Tiefe-A-Pattern): Skripte (bash + PowerShell) seeden Fake-Home mit minimal valider Kiro-Config, rufen `crossmem install`, validieren die geschriebene Config gegen das in 28.1 recherchierte Kiro-MCP-Schema, pruefen `<config>.bak`, rufen `crossmem install` erneut (Idempotenz), und schliesslich `crossmem uninstall`. JSON-Report-Eintrag im 27.1-Schema. Exit 0 bei OK.

- [ ] **28.6 cli-config-validate-qwen-cli**
  Pfad: `tests/e2e/docker/scenarios/qwen-cli/install-validate.sh`, `tests/e2e/docker/scenarios/qwen-cli/install-validate.ps1`.
  DoD: Analog 28.5 fuer Qwen CLI.


---

## Release-Checkliste

Manueller Windows-Smoke vor jedem Release (kein Task): `pip install crossmem` in frischem `venv`, `crossmem doctor`, `crossmem install`, MCP-Config in Claude Code/Cursor pruefen, Query-Roundtrip.

Claude Code: MCP-Roundtrip manuell auf Host (Max-Login) verifizieren — da die Anthropic Max-Subscription keinen API-Key bereitstellt, ist Claude Code nicht Teil der Docker-E2E-Suite (vgl. weggefallener Task 27.25). Vorgehen: `crossmem install` auf dem Host, MCP-Eintrag in Claude Code pruefen, einen `crossmem.store`-Aufruf gefolgt von einem `crossmem.query`-Aufruf in einer Claude-Code-Session ausfuehren und die DB-/Response-Konsistenz verifizieren.
