import asyncio, base64, json, os, re, sys, time, uuid, logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# ── Google ADK ────────────────────
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# ── FastMCP ─────────────────
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

# ── Traceloop ─────────────────────────────────────────────────────────────────
from traceloop.sdk import Traceloop, set_association_properties
from traceloop.sdk.decorators import workflow
from traceloop.sdk.tracing import set_conversation_id

# ── OTel logging ──────────────────────────────────────────────────────────────
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor, ConsoleLogRecordExporter
from opentelemetry.sdk.resources import Resource

# ── Dynamic prompts ───────────────────────────────────────────────────────────
from prompts import build_task_prompt, detect_task_type

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# ── Change these three lines ──────────────────────────────
AGENT_NAME  = "manduke_v2"          # shown on the leaderboard
AGENT_STACK = "Python / ADK / Groq" # describe your stack
# Local Ollama (free) — needs `ollama serve` + the model pulled. Use the
# ollama_chat/ prefix for reliable tool-calling. Swap to a "gemini-2.5-*"
# string to go back to Gemini.
# MODEL = "ollama_chat/qwen2.5-coder:7b"
# MODEL = "groq/llama-3.3-70b-versatile"   # proven (reached L2), but tight 12k TPM
# MODEL = "groq/meta-llama/llama-4-scout-17b-16e-instruct"  # unreliable tool calls
# MODEL = "groq/qwen/qwen3.6-27b"
MODEL = "groq/openai/gpt-oss-120b"          # strong tool calling, roomier limits
# MODEL = "groq/meta-llama/llama-prompt-guard-2-86m" # no tool calling support
# MODEL = "groq/llama-3.1-8b-instant"
APP_NAME = "arena-adk-agent"
USER_ID  = AGENT_NAME

# ── Leave these as-is ─────────────────────────────────────
MCP_ENDPOINT   = "https://agent-arena-623774504237.asia-southeast1.run.app/mcp"
# Firebase JWT from the Arena web app (DevTools → Application → Storage).
# Expires in ~1 hour — keep it in .env (gitignored), never commit it.
ID_TOKEN       = os.environ["ARENA_ID_TOKEN"]
MAX_TURNS      = 20
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TRACELOOP_API_KEY=os.environ["TRACELOOP_API_KEY"]
GITHUB_URL="https://github.com/arunmoola/my-arena-agent"
LINKEDIN_URL="https://linkedin.com/in/arunprasad"

def resolve_model(model: str):
    """ADK speaks Gemini natively, so a bare "gemini-*" string is passed
    through. Anything else (groq/, openrouter/, mistral/, openai/, ollama_chat/,
    ...) is a LiteLLM provider route and gets wrapped in LiteLlm."""
    if model.startswith("gemini"):
        return model
    # num_retries=0: bare litellm.acompletion (what ADK's LiteLlm calls) treats
    # num_retries as a blanket count that retries EVERY exception — including
    # non-retryable 400s (tool_use_failed, reasoning_content) — because the
    # per-error RetryPolicy / _should_retry logic only runs via litellm.Router.
    # So we disable LiteLLM's retries and let run_turn classify + bail/retry.
    return LiteLlm(model=model, num_retries=0)

# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(tag: str, msg: str, level: str = "INFO") -> None:
    emoji = {
        "REGISTER": "📝", "FETCH": "📥", "SUBMIT": "📤",
        "SCORE": "🏆", "LEVEL": "🚀", "SKIP": "⏭️",
        "ERROR": "❌", "WARN": "⚠️", "DONE": "✅",
        "TASK": "📋", "LOOP": "🔄", "AGENT": "🤖",
        "TRACE": "📡", "RECOVER": "🔧",
    }.get(tag, "•")
    print(f"[{_ts()}] {emoji} [{tag}] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Run-scoped state
# ─────────────────────────────────────────────────────────────────────────────
class RunState:
    """Shared mutable state — passed into every tool via closure."""
    def __init__(self):
        self.run_id          = str(uuid.uuid4())
        self.execution_id    = str(uuid.uuid4())
        self.agent_id        = ""
        self.task_id         = ""
        self.conversation_id = ""

        self.current_level   = 1
        self.total_score     = 0
        self.tasks_attempted = 0
        self.tasks_passed    = 0
        self.level_history: list[dict] = []

        self.current_task: Optional[dict] = None

    def record(self, level, title, score, levelled_up):
        self.tasks_attempted += 1
        self.total_score     += score
        if levelled_up or score >= 70: self.tasks_passed += 1
        if levelled_up: self.current_level = level + 1
        self.level_history.append({
            "level": level, "task": title,
            "score": score, "up": levelled_up
        })
    
    def scoreboard(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  SCOREBOARD  (run {self.run_id[:8]})  model: {resolve_model(MODEL)}",
            f"{'─'*60}",
            f"  Current Level : {self.current_level}",
            f"  Total Score   : {self.total_score}",
            f"  Tasks Done    : {self.tasks_attempted}  (passed: {self.tasks_passed})",
            f"{'─'*60}",
        ]
        for entry in self.level_history:
            icon = "✅" if entry["up"] else ("🟡" if entry["score"] >= 70 else "❌")
            lines.append(
                f"  {icon} L{entry['level']}  {entry['task'][:40]:<40}  {entry['score']:>3}/100"
            )
        lines.append(f"{'─'*60}\n")
        return "\n".join(lines)


def check_token(token: str, min_seconds: int = 120) -> None:
    """Decode the Firebase JWT's exp and bail early with a friendly message if
    it's expired or about to expire — these tokens only last ~1 hour, so this
    saves a 40-line MCP AUTH_ERROR stack trace mid-run."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload))["exp"]
    except Exception:
        print("⚠️  ARENA_ID_TOKEN doesn't look like a valid JWT — re-copy it from "
              "the Arena web app (DevTools → Application → Storage).")
        sys.exit(1)
    left = exp - int(time.time())
    if left <= 0:
        print(f"⚠️  ARENA_ID_TOKEN expired {-left // 60} min ago. Refresh it in "
              ".env from the Arena web app (DevTools → Application → Storage).")
        sys.exit(1)
    if left < min_seconds:
        print(f"⚠️  ARENA_ID_TOKEN expires in {left}s — refresh it before running "
              "so it doesn't die mid-task.")
        sys.exit(1)
    print(f"✓ ARENA_ID_TOKEN valid for {left // 60} min.")

# ─────────────────────────────────────────────────────────────────────────────
# OTel / Traceloop logging
# ─────────────────────────────────────────────────────────────────────────────

class _OtelOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        tid = getattr(record, "otelTraceID", "0")
        return tid not in ("0", "00000000000000000000000000000000", None, "")


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))
    logger.addHandler(h)
    return logger


agent_logger = _make_logger("arena.agent")
task_logger  = _make_logger("arena.task")


def quiet_noisy_loggers() -> None:
    """Silence the full stack traces that LiteLLM (per-retry) and ADK's node
    runner dump to stderr *before* an exception propagates to our handlers.
    Our run_turn/main try-excepts catch the exception, but can't un-print what
    these libraries already logged on the way up — so we raise their log levels."""
    try:
        import litellm
        litellm.suppress_debug_info = True
    except Exception:
        pass
    for name in ("LiteLLM", "litellm", "google_adk", "google.adk", "httpx"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def init_tracing() -> None:
    quiet_noisy_loggers()
    Traceloop.init(
        app_name=APP_NAME,
        api_key=TRACELOOP_API_KEY or None,
        disable_batch=True,
        telemetry_enabled=False,
    )
    log_provider = LoggerProvider(resource=Resource.create({"service.name": APP_NAME}))
    exporter = ConsoleLogRecordExporter()
    if TRACELOOP_API_KEY:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        exporter = OTLPLogExporter(
            endpoint="https://api.traceloop.com/v1/logs",
            headers={"Authorization": f"Bearer {TRACELOOP_API_KEY}", "x-traceloop-sdk-version": "traceloop-sdk"},
        )
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    for logger in (agent_logger, task_logger):
        h = LoggingHandler(logger_provider=log_provider)
        h.setLevel(logging.INFO)
        h.addFilter(_OtelOnlyFilter())
        logger.addHandler(h)
    _log("TRACE", "Traceloop initialised.")

# ─────────────────────────────────────────────────────────────────────────────
# MCP helper
# ─────────────────────────────────────────────────────────────────────────────
async def _mcp_call(tool_name: str, arguments: dict, state: RunState) -> str:
    from fastmcp.exceptions import ToolError
    transport = StreamableHttpTransport(url=MCP_ENDPOINT)
    try:
        async with Client(transport=transport, name=APP_NAME) as client:
            set_association_properties({
                "execution.id": state.execution_id,
                "run.id":       state.run_id,
                "agent.id":     state.agent_id,
                "task.id":      state.task_id,
                "agent.name":   AGENT_NAME,
                "agent.stack":  AGENT_STACK,
            })
            if state.conversation_id:
                set_conversation_id(state.conversation_id)

            result = await client.call_tool(tool_name, arguments)
            if result is None:
                return f"ERROR: {tool_name} returned no response"
            return "\n".join(
                getattr(b, "text", "") for b in result.content if getattr(b, "text", None)
            )
    except ToolError as e:
        _log("ERROR", f"{tool_name}: {e}")
        return f"ERROR: {e}"
    except Exception as e:
        _log("ERROR", f"{tool_name}: {e}")
        return f"ERROR: {e}"

"""
async def do_register(state: RunState) -> str:
"""
"""Register the agent and capture the server-assigned AGENT_ID into state.
    Called directly from main() before the LLM loop so registration never
    depends on the model deciding to call it."""
"""
    result = await mcp_call("register_agent",
        {"idToken": ID_TOKEN, "name": AGENT_NAME, "stack": AGENT_STACK}, state)
    # Server returns text like "AGENT_ID: <id>. Level: 1. ...". Capture only
    # id-safe chars so the trailing sentence period isn't swallowed (\\S+ would
    # grab "2yBM...riky." with the dot, which then 404s as AGENT_NOT_FOUND).
    m = re.search(r"AGENT_ID:\\s*([A-Za-z0-9_-]+)", result)
    if m:
        state.agent_id = m.group(1)
    else:                               # tolerate a JSON-shaped response too
        try:
            d = json.loads(result)
            state.agent_id = d.get("agentId") or d.get("agent_id") or d.get("id") or state.agent_id
        except Exception:
            pass
    return result
"""

# ─────────────────────────────────────────────────────────────────────────────
# Tool factory
# ─────────────────────────────────────────────────────────────────────────────
def make_arena_tools(state: RunState) -> list:
    """Returns the four Arena tool functions with state captured via closure."""
    async def register_agent(name: str, stack: str) -> str:
        """Register this agent in the Agent Arena. Call once at the start."""
        result = await _mcp_call("register_agent", {
            "idToken":     ID_TOKEN,
            "name":        name,
            "stack":       stack,
            "linkedinUrl": LINKEDIN_URL,
            "githubUrl":   GITHUB_URL,
        }, state)

        match = re.search(r"AGENT_ID:\s*(\S+?)\.?(\s|$)", result)
        if match:
            state.agent_id = match.group(1)
            state.conversation_id = state.agent_id
            set_association_properties({"agent.id": state.agent_id, "run.id": state.run_id})
            set_conversation_id(state.agent_id)

        level_match = re.search(r"Level[:\s]+(\d+)", result)
        if level_match:
            state.current_level = int(level_match.group(1))

        agent_logger.info("Registered", extra={"agent_id": state.agent_id, "run_id": state.run_id})
        _log("REGISTER", f"agent_id={state.agent_id}  level={state.current_level}")
        return result

    async def get_tasks(agent_id: str) -> str:
        """Fetch the currently assigned task for this agent's level."""
        result = await _mcp_call("get_tasks", {
            "idToken": ID_TOKEN, "agentId": agent_id,
        }, state)

        try:
            data = json.loads(result)
            # Handle both dict and list responses
            task_obj = None
            if isinstance(data, dict) and "id" in data:
                task_obj = data
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "id" in data[0]:
                task_obj = data[0]

            if task_obj:
                state.task_id         = task_obj["id"]
                state.current_task    = task_obj
                state.conversation_id = f"{state.agent_id}-{state.task_id}"
                set_association_properties({"task.id": state.task_id, "execution.id": state.execution_id})
                set_conversation_id(state.conversation_id)
                _log("FETCH", f"task={state.task_id}  '{task_obj.get('title')}'  L{task_obj.get('level')}")
        except json.JSONDecodeError:
            pass
        return result

    async def submit_task(agent_id: str, task_id: str, content: str) -> str:
        """Submit the complete answer for the current task for AI evaluation."""
        new_exec = str(uuid.uuid4())
        state.execution_id = new_exec
        set_association_properties({
            "execution.id": new_exec,
            "task.id":      task_id,
            "agent.id":     agent_id,
        })

        task_logger.info("Submitting", extra={
            "agent_id": agent_id, "task_id": task_id, "execution_id": new_exec,
        })

        result = await _mcp_call("submit_task", {
            "idToken":     ID_TOKEN,
            "agentId":     agent_id,
            "taskId":      task_id,
            "executionId": new_exec,
            "content":     content,
            "metadata": {
                "agent_name": AGENT_NAME, "agent_stack": AGENT_STACK,
                "run_id": state.run_id, "execution_id": new_exec, "model": resolve_model(MODEL),
            },
        }, state)

        score_match = re.search(r"Score:\s*(\d+)/100", result)
        score       = int(score_match.group(1)) if score_match else -1
        levelled_up = "LEVEL_UP" in result

        task_title = state.current_task.get("title", state.task_id) if state.current_task else state.task_id
        state.record(state.current_level, task_title, score, levelled_up)

        lu_emoji = "🚀 LEVEL_UP!" if levelled_up else ""
        _log("SCORE", f"{score}/100  {lu_emoji}")
        print(state.scoreboard())

        task_logger.info("Submitted", extra={
            "agent_id": agent_id, "task_id": task_id,
            "score": score, "levelled_up": levelled_up,
        })
        return result

    async def skip_task(agent_id: str, task_id: str, reason: str = "") -> str:
        """Abandon the current task and allow get_tasks to return a new one."""
        _log("SKIP", f"skipping {task_id[:8]}  reason={reason[:50]}")
        return await _mcp_call("skip_task", {
            "idToken": ID_TOKEN, "agentId": agent_id,
            "taskId": task_id, "reason": reason,
        }, state)
    
    async def report_status() -> str:
        """Report the current agent status."""
        return (
            f"Agent: {AGENT_NAME}  ID: {state.agent_id}\n"
            f"Level: {state.current_level}  Total Score: {state.total_score}\n"
            f"Tasks attempted: {state.tasks_attempted}  Passed: {state.tasks_passed}\n"
            f"History: {json.dumps(state.level_history, indent=2)}"
        )

    return [register_agent, get_tasks, submit_task, skip_task, report_status]

SYSTEM_PROMPT = f"""
You are an expert autonomous agent competing in the Agent Arena evaluation system.
Your goal is to solve tasks with exceptional quality and advance through levels.

AVAILABLE TOOLS:
- register_agent(name, stack): Register once at the start.
- get_tasks(agent_id): Fetch the current task.
- skip_task(agent_id, task_id, reason): Skip an impossible task.
- submit_task(agent_id, task_id, content): Submit your final answer for evaluation.
- report_status(): Report progress before stopping.

RULES:
- Never submit the same task_id twice.
- Always use the task_id from the most recent get_tasks call.
- Do not ask for confirmation — act autonomously.
- When you receive a task with raw logs or questions, think through the solution step-by-step, determine the answer, and then use the native tool calling feature to invoke `submit_task`.
- Do not attempt to format or output raw JSON blocks or text code representations for tool calls manually. Let the system handle the function execution.

IDENTITY:
- Agent Name: {AGENT_NAME}
- Stack: {AGENT_STACK}
""".strip()

def build_agent(state: RunState) -> LlmAgent:
    return LlmAgent(
        name=AGENT_NAME,
        model=resolve_model(MODEL),
        tools=make_arena_tools(state),
        instruction=SYSTEM_PROMPT,
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Multi-turn runner
# ─────────────────────────────────────────────────────────────────────────────
class UnrecoverableRunError(Exception):
    """A run failure that won't fix itself on an immediate retry — auth
    failures, rate/quota limits (TPM/TPD), or bad requests (tool_use_failed,
    reasoning_content). Signals main() to stop and print the scoreboard."""


_UNRECOVERABLE_HINTS = (
    "auth_error", "authenticationerror", "invalid or expired",   # token
    "ratelimiterror", "rate_limit", "tokens per",                # TPM/TPD caps
    "badrequesterror", "tool_use_failed", "invalid_request_error",
)


def _is_unrecoverable(exc: Exception) -> bool:
    """True for errors where retrying immediately just hits the same wall."""
    blob = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in blob for hint in _UNRECOVERABLE_HINTS)


async def run_turn(
    runner:          Runner,
    session_service: InMemorySessionService,
    session_id:      str,
    message:         str,
) -> str:
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)],
    )

    final_text = ""
    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content,
        ):
            if not event.content or not event.content.parts:
                continue

            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args_str = str(dict(fc.args))
                    preview  = args_str[:120]
                    _log("AGENT", f"→ {fc.name}  {preview}{'...' if len(args_str) > 120 else ''}")

                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    resp_str = str(fr.response)[:150].replace("\n", " ")
                    _log("AGENT", f"← {fr.name}  {resp_str}{'...' if len(str(fr.response)) > 150 else ''}")

                elif hasattr(part, "text") and part.text and event.turn_complete:
                    final_text = part.text
    except Exception as e:
        # Log a single clean line instead of the full ADK/LiteLLM stack trace.
        # str(e) on LiteLLM errors still carries the useful detail (e.g. the
        # rate-limit "try again in Ns" message) on one line.
        msg = " ".join(str(e).split())
        _log("ERROR", f"run_turn aborted: {type(e).__name__}: {msg}")
        if _is_unrecoverable(e):
            # Don't let the caller retry into the same wall — surface a clean
            # typed signal so main() stops and shows the scoreboard.
            raise UnrecoverableRunError(f"{type(e).__name__}: {msg}") from None

    return final_text
        

# ─────────────────────────────────────────────────────────────────────────────
# Task driver
# ─────────────────────────────────────────────────────────────────────────────
async def _drive(
    state:           RunState,
    runner:          Runner,
    session_service: InMemorySessionService,
) -> None:
    """Bootstrap, run the task loop, and print the final report. Any
    unrecoverable failure surfaces from run_turn as UnrecoverableRunError, which
    main() catches to stop the run and show the scoreboard."""
    # ── Bootstrap: register and fetch first task ──────────────────────────────
    _log("REGISTER", "Bootstrapping — register then fetch first task...")
    await run_turn(
        runner, session_service, state.run_id,
        f"Call register_agent(name='{AGENT_NAME}', stack='{AGENT_STACK}') to register. "
        f"Then wait for the response and capture the agent_id."
        f"Then call get_tasks with your agent_id to fetch the first task. "
        f"Return ONLY a one-line summary: 'Task: <title> (Level <level>)'. "
        f"Do NOT solve or submit yet.",
    )

    if not state.current_task:
        _log("WARN", "No task after bootstrap. Attempting one more fetch...")
        await run_turn(
            runner, session_service, state.run_id,
            "Call get_tasks to fetch the first challenge.",
        )

    # ── Main task loop ────────────────────────────────────────────────────────
    for task_num in range(1, MAX_TURNS + 1):
        if not state.current_task or not state.task_id:
            _log("DONE", "No active task — stopping.")
            break

        task = state.current_task
        task_title = task.get("title", "Unknown")
        task_type  = detect_task_type(task_title, task.get("description", ""))
        desc       = task.get("description", "")[:600]

        print(f"\n{'━'*60}")
        _log("TASK", f"#{task_num} | {task_title}")
        _log("TASK", f"Type: {task_type.upper()} | Level: {task.get('level', '?')} | ID: {state.task_id[:8]}")
        _log("TASK", f"Desc: {desc}{'...' if len(task.get('description', '')) > 600 else ''}")
        print(f"{'━'*60}")

        # ── Single-turn solve ─────────────────────────────────────────────────
        # Create a dedicated clean session ID for the execution turn
        task_session_id = f"{state.run_id}-task-{task_num}"
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=task_session_id,
        )
        prev_attempted = state.tasks_attempted
        prompt = build_task_prompt(task, state.agent_id, state.task_id)
        _log("AGENT", "Solving task (analysis + solution + submit in one turn)...")
        await run_turn(runner, session_service, task_session_id, prompt)

        # ── Verify submission ─────────────────────────────────────────────────
        if state.tasks_attempted > prev_attempted:
            _log("SCORE", f"Task #{task_num} submitted successfully.")
        else:
            _log("WARN", f"Task #{task_num} was NOT submitted. Recovering...")
            recovery = await run_turn(
                runner, session_service, state.run_id,
                f"You have NOT submitted the current task yet. "
                f"Call submit_task(agent_id='{state.agent_id}', task_id='{state.task_id}', "
                f"content=<your complete final answer>) NOW. "
                f"If the task is impossible to solve, call skip_task with a reason, then get_tasks.",
            )
            if state.tasks_attempted == prev_attempted:
                _log("ERROR", f"Recovery failed for task #{task_num}. Moving on.")
                # Force skip so we don't get stuck on the same task
                await run_turn(
                    runner, session_service, state.run_id,
                    f"Call skip_task(agent_id='{state.agent_id}', task_id='{state.task_id}', "
                    f"reason='Agent failed to submit after recovery prompt.')",
                )

        # ── Prepare for next task ─────────────────────────────────────────────
        state.current_task = None
        state.task_id = ""

        _log("LOOP", "Fetching next task...")
        await run_turn(
            runner, session_service, state.run_id,
            "Call get_tasks to fetch the next challenge. "
            "If NO_TASKS is returned, call report_status() and stop. "
            "Otherwise, return a brief summary.",
        )

        if not state.current_task:
            _log("DONE", "No more tasks available.")
            break

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    _log("DONE", "Final status report")
    print(f"{'═'*60}")
    await run_turn(
        runner, session_service, state.run_id,
        "Call report_status() to summarize your full run.",
    )
    print(state.scoreboard())
    agent_logger.info("Run complete", extra={
        "run_id":          state.run_id,
        "total_score":     state.total_score,
        "tasks_attempted": state.tasks_attempted,
        "final_level":     state.current_level,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main workflow
# ─────────────────────────────────────────────────────────────────────────────

@workflow(name="arena_adk_run")
async def main() -> None:
    check_token(ID_TOKEN)
    state = RunState()

    print(f"\n{'═'*60}")
    print(f"  AGENT ARENA  —  {AGENT_NAME}  (model: {resolve_model(MODEL)})")
    print(f"{'═'*60}")
    _log("REGISTER", f"Agent: {AGENT_NAME}")
    _log("REGISTER", f"Run ID: {state.run_id}")
    _log("REGISTER", f"Max tasks: {MAX_TURNS}")
    print(f"{'═'*60}\n")

    set_association_properties({
        "run.id":       state.run_id,
        "execution.id": state.execution_id,
        "agent.name":   AGENT_NAME,
        "agent.stack":  AGENT_STACK,
    })

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=state.run_id,
    )

    agent  = build_agent(state)
    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name=APP_NAME,
    )

    # Drive the run; bail out cleanly (with scoreboard) on unrecoverable errors
    # like auth failures or rate/quota limits, instead of retrying into the wall.
    try:
        await _drive(state, runner, session_service)
    except UnrecoverableRunError as e:
        _log("ERROR", f"Unrecoverable error — stopping run: {e}")
        print(state.scoreboard())

if __name__ == "__main__":
    init_tracing()
    asyncio.run(main())