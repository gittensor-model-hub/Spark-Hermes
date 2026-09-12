# Zenith workspace setup

Zenith coordinates long-running missions through persistent plans, ACP workers,
independent validation, and completion review. Use it for the Spark Hermes
engineering and research loop when a task needs investigation, multiple dependent
changes, repeated experiments, and evidence before promotion. It does not itself
implement Bittensor rewards or train a model.

## Installed components

- Source: `.local/tools/zenith`, revision `a8d9b5786f81e70e73cce7bb164f5f83da84a4b5`.
- Python package: `zenith-harness 0.1.0`, installed with `uv sync` in the nested `zenith` directory.
- Claude adapter: `@agentclientprotocol/claude-agent-acp 0.76.0`.
- Codex adapter: `@agentclientprotocol/codex-acp 1.11.0`.
- Hermes: existing installation, with its declared `agent-client-protocol==0.9.0` dependency added.
- `/zenith` skills: workspace and user skill directories for Codex, Claude Code,
  and Hermes, each referencing the corresponding workspace orchestrator prompt.
- Bundled playbooks and provider agent definitions are installed in their host
  directories. Codex playbooks also use `.agents/skills`; Hermes playbooks also
  use its user skill directory.

Python, uv, Node.js, and npm were already available. Spark Hermes's Python
training environment was not used for Zenith dependencies.

## Start a mission

Start a new harness session from `/home/speedy/Spark-Hermes` so newly installed
skills and MCP configuration are discovered. Invoke `/zenith` and append the full
mission. Codex also supports explicit skill invocation as `$zenith`.

For example:

```text
/zenith Build and validate Spark Hermes's continuous improvement pipeline:
SN74 miners submit GitHub PRs; reproducible evaluation scores contributions;
accepted changes improve both the agent stack and model-training pipeline.
Start with the 4B proof of concept. Complete CPU-verifiable engineering first;
do not launch GPU jobs. Preserve the full 27B target and identify prerequisites.
```

The skill loads `.codex/orchestrator_prompt.md`,
`.claude/orchestrator_prompt.md`, or `.hermes/orchestrator_prompt.md` for its host.
A new session is necessary if the current one has a fixed tool/skill catalog.
The three providers have separate persistent mission stores; use the same
provider to resume a mission.

## Services and restart

| Harness | Local MCP URL | Configuration | Mission store |
| --- | --- | --- | --- |
| Codex | `http://127.0.0.1:18741/mcp` | `~/.codex/config.toml` | `.local/zenith/codex` |
| Claude Code | `http://127.0.0.1:18742/mcp` | `.mcp.json`, approval in `.claude/settings.local.json` | `.local/zenith/claude` |
| Hermes | `http://127.0.0.1:18743/mcp` | `~/.hermes/config.yaml` | `.local/zenith/hermes` |

These services run as detached local processes. They survive the setup command,
but are not configured as boot services. After a host reboot, from this repository:

```bash
.local/tools/zenith/zenith/.venv/bin/python .local/tools/zenith-setup/service.py start all
.local/tools/zenith/zenith/.venv/bin/python .local/tools/zenith-setup/service.py status all
```

Replace `all` with `codex`, `claude`, or `hermes` to operate on one provider.
`stop` requests termination of the recorded service process group. Do not stop
services while mission workers are running. Logs and PID files are under
`.local/zenith/`. The start/status commands check live MCP tool discovery.

The upstream CLI is available as `zenith` or:

```bash
uv run --project .local/tools/zenith/zenith zenith --help
ZENITH_HOME="$PWD/.local/zenith/codex" zenith list-projects
```

`uv run zenith` is a CLI command, not a persistent server. The service manager
runs `uv run --frozen --project <package> zenith-server --mode orchestrator
--transport streamable-http --host 127.0.0.1 --port <provider-port>`.
Upstream also supports harness-managed stdio transport.

## Verification and limits

Installation checks cover:

- Skill validation and provider-specific prompt paths.
- Preservation of unrelated Codex and Hermes configuration values.
- Live MCP ping and all seven orchestration tools for each provider.
- Isolated project creation, inspection, cancellation, and inspection of the
  persisted cancellation for each running server.
- ACP protocol-v1 initialization for all three adapters, without model inference.
- Hermes ACP dependency check and Claude's connected MCP status.
- Zenith's server and CLI integration tests: 28 passed.

Local reports: `.local/zenith/verification.json` and
`.local/zenith/acp-verification.json`. Smoke projects are marked aborted because
cancellation is the tested lifecycle operation; they are not failed training runs.
No model-backed worker mission or GPU training was run by installation checks.
Provider authentication, model availability, and inference costs apply when a real
worker mission starts.

Upstream Zenith runs ACP workers in autonomous permission modes. Normal host
model and sandbox settings were preserved; installation did not grant unrelated
mission actions. Configuration backups are private files under
`.local/zenith/setup-backups/`.

The source checkout, dependencies, service scripts, and runtime state are local
installation files under ignored `.local/`; a fresh clone needs its own setup.
The installed skill paths and MCP ports are specific to this machine.

References: [upstream README](https://github.com/Intelligent-Internet/zenith)
and [technical report](https://github.com/Intelligent-Internet/zenith/blob/main/technical_report/Technical_Report.pdf).
The report supports adaptive planning and layered verification as design choices;
its benchmark results are not a guarantee for Spark Hermes.
