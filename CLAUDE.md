# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file autonomous agent ([arena_agent.py](arena_agent.py)) that competes in the "Agent Arena" — a leaderboard where the agent registers, fetches programming tasks, submits solutions (scored 0–100; ≥70 = LEVEL_UP), and skips when stuck. Built on **Google ADK** driving a **Gemini** model, with the four Arena actions exposed as a **remote MCP server** over Streamable HTTP.

## Run

```bash
venv/bin/python arena_agent.py        # Python 3.14, deps preinstalled in ./venv
```

There is no build, lint, or test setup — the whole program is one file run directly.

## Environment

- `GEMINI_API_KEY` and `TRACELOOP_API_KEY` are read from `os.environ` at import time; a missing key raises `KeyError` immediately.
- The code does **not** call `load_dotenv()`, so `.env` is not auto-loaded. Either export the vars in the shell or add a dotenv load before relying on `.env`.
- `ID_TOKEN` is a hardcoded Firebase JWT used to authenticate every MCP call. It is short-lived (~1h `exp`) and will need refreshing when calls start failing auth.

## Architecture

The control flow is a thin ADK agent loop wrapped around stateless MCP calls:

- **`RunState`** — single mutable object holding the run's `agent_id`, `task_id`, level, score, and `level_history`. It is captured by closure into the tool functions (not passed through ADK), so all four tools mutate the same state.
- **`make_arena_tools(state)`** — returns the four async tools (`register_agent`, `get_tasks`, `submit_task`, `skip_task`) that the LLM is allowed to call. Each parses the MCP text response and writes results back into `state` (e.g. regex-extracting `AGENT_ID:`, JSON-parsing the task `id`).
- **`mcp_call(tool, args, state)`** — opens a **fresh** `fastmcp` `Client` / `StreamableHttpTransport` session per call against `MCP_ENDPOINT`, invokes one tool, and flattens the response content blocks to text. Sessions are intentionally not reused.
- **`build_agent` → `run_turn` → `main`** — `main()` builds the `LlmAgent`, creates an `InMemorySessionService` session keyed by `run_id`, and kicks off a single turn with "Start now. Register and solve tasks." The LLM then autonomously chains tool calls. `MAX_TURNS` (20) is the task-attempt cap, enforced via the system prompt rather than in code.

The three tunables at the top — `AGENT_NAME`, `AGENT_STACK`, `MODEL` — are the intended edit points; `MCP_ENDPOINT`, `ID_TOKEN`, and `MAX_TURNS` below them are meant to stay fixed.

## Known rough edges

When editing, be aware the file currently has unfinished/buggy spots: `build_agent` references `LLMAgent` (import is `LlmAgent`), `LlmAgent`'s system-prompt argument is `instruction` (not `system_prompt`), and `main()` is defined but never invoked (no `asyncio.run(main())`). Confirm these against the installed `google-adk==2.3.0` API before assuming the program runs end-to-end.
