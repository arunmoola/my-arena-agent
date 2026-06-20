# my-arena-agent

An autonomous agent for [Agent Arena](https://tutorial.agent-arena.dev) — a
competitive arena where the agent registers, fetches progressively harder
programming tasks, solves them, submits answers for AI scoring (0–100), and
levels up on any score ≥ 70.

Built on **Google ADK** + **Gemini**, talking to the Arena's **FastMCP** server.
All communication is through four MCP tool calls (no direct REST):

| Tool | Purpose |
| --- | --- |
| `register_agent(name, stack)` | Register once at start; returns `AGENT_ID` + level. |
| `get_tasks(agent_id)` | Fetch the current sticky task (JSON: id, title, description, level, points). |
| `submit_task(agent_id, task_id, content)` | Submit an answer; scored 0–100, ≥ 70 → LEVEL_UP. |
| `skip_task(agent_id, task_id)` | Abandon the current task without penalty. |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY and ARENA_ID_TOKEN
```

`ARENA_ID_TOKEN` is a Firebase JWT from the Arena web app (DevTools →
Application → Storage). It expires in ~1 hour, so refresh it before each run.

## Run

```bash
python arena_agent.py
```

## Configuration

Edit the three identity lines at the top of [`arena_agent.py`](arena_agent.py):

```python
AGENT_NAME  = "manduke"                 # shown on the leaderboard
AGENT_STACK = "Python / ADK / Gemini"   # describe your stack
MODEL       = "gemini-2.5-flash"        # or gemini-2.5-pro-preview
```

Secrets (`GEMINI_API_KEY`, `TRACELOOP_API_KEY`, `ARENA_ID_TOKEN`) load from
`.env`, which is gitignored. Never commit real keys.
