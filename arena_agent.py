import asyncio, json, os, re, uuid
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

load_dotenv()

# ── Change these three lines ──────────────────────────────
AGENT_NAME  = "manduke"          # shown on the leaderboard
AGENT_STACK = "Python / ADK / Gemini" # describe your stack
MODEL       = "gemini-2.5-flash"      # or gemini-2.5-pro-preview

# ── Leave these as-is ─────────────────────────────────────
MCP_ENDPOINT   = "https://agent-arena-623774504237.asia-southeast1.run.app/mcp"
# Firebase JWT from the Arena web app (DevTools → Application → Storage).
# Expires in ~1 hour — keep it in .env (gitignored), never commit it.
ID_TOKEN       = os.environ["ARENA_ID_TOKEN"]
MAX_TURNS      = 20
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TRACELOOP_API_KEY=os.environ["TRACELOOP_API_KEY"]
GITHUB_URL="github.com/arunprasad/my-arena-agent"
LINKED_IN_URL="linkedin.com/in/arunprasad"

SYSTEM_PROMPT = f"""
You are an autonomous agent that solves programming tasks in a competitive arena.
You are given a task description and you must write code to solve it.

TOOLS:
You have access to the following tools:
- register_agent(name, stack): Registers the agent with the arena. Call once at start.
- get_tasks(agent_id): Fetches the current task. Returns JSON with id, title, description.
- submit_task(agent_id, task_id, content): Submits your answer. Scored 0-100. Score >= 70 means LEVEL_UP.
- skip_task(agent_id, task_id): Skips a task. Call when stuck.
You must use the tools to register, fetch tasks, submit solutions, and skip tasks as needed.

RULES:
You must keep track of your current level, total score, and tasks attempted.
You must not attempt more than {MAX_TURNS} tasks in a single run.
You must not use any external resources or search engines. You must rely only on your own knowledge and the tools provided.
You must not ask for help or clarification. You must make your best effort to solve the tasks on your own.
You must not write any code that is malicious, harmful, or violates the rules of the arena.
You must not write any code that is unsafe, insecure, or violates best practices.
You must not write any code that is plagiarized or copied from external sources. You must write original code that you understand and can explain.
You must not write any code that is incomplete, incorrect, or does not solve the task. You must write code that is correct, complete, and solves the task as specified.
"""

class RunState:
    """Shared mutable state — passed into every tool via closure."""
    def __init__(self):
        self.run_id          = str(uuid.uuid4())
        self.agent_id        = ""
        self.task_id         = ""
        self.current_level   = 1
        self.total_score     = 0
        self.tasks_attempted = 0
        self.level_history   = []

    def record(self, level, title, score, levelled_up):
        self.tasks_attempted += 1
        self.total_score     += score
        if levelled_up: self.current_level = level + 1
        self.level_history.append({
            "level": level, "task": title,
            "score": score, "up": levelled_up
        })
        icon = "✓" if levelled_up else ("~" if score >= 70 else "✗")
        print(f"  {icon} L{level}  score={score}/100")


async def mcp_call(tool: str, args: dict, state: RunState) -> str:
    """Open a fresh MCP session, call one tool, return text result."""
    transport = StreamableHttpTransport(url=MCP_ENDPOINT)
    async with Client(transport, name="arena-agent") as c:
        result = await c.call_tool(tool, args)
    return "\n".join(
        getattr(b, "text", "")
        for b in result.content
        if getattr(b, "text", None)
    )

def make_arena_tools(state: RunState):
    """Returns the four Arena tool functions with state captured via closure."""
    async def register_agent(name: str, stack: str) -> str:
        """Register this agent. Call once at start. Returns AGENT_ID."""
        result = await mcp_call("register_agent",
            {"idToken": ID_TOKEN, "name": name, "stack": stack}, state)
        m = re.search(r"AGENT_ID:\s*(\S+)", result)
        if m: state.agent_id = m.group(1)
        return result

    async def get_tasks(agent_id: str) -> str:
        """Fetch the current task. Returns JSON with id, title, description."""
        result = await mcp_call("get_tasks",
            {"idToken": ID_TOKEN, "agentId": agent_id}, state)
        try:
            data = json.loads(result)
            if "id" in data: state.task_id = data["id"]
        except: pass
        return result

    async def submit_task(agent_id: str, task_id: str, content: str) -> str:
        """Submit your answer. Scored 0-100. Score >= 70 means LEVEL_UP."""
        result = await mcp_call("submit_task", {
            "idToken": ID_TOKEN, "agentId": agent_id,
            "taskId": task_id, "content": content,
            "metadata": {"agent_name": AGENT_NAME, "model": MODEL},
        }, state)
        return result

    async def skip_task(agent_id: str, task_id: str) -> str:
        """Skip a task. Call when stuck."""
        return await mcp_call("skip_task",
            {"idToken": ID_TOKEN, "agentId": agent_id, "taskId": task_id}, state)

    return [register_agent, get_tasks, submit_task, skip_task]

async def run_turn(runner, user_id, session_id, message):
    """Send one message; collect and return the agent's final text reply."""
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=message)])
    reply = None
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content):
        if event.content and event.content.parts and event.content.parts[0].text:
            reply = event.content.parts[0].text
    return reply
        
def build_agent(state: RunState) -> LlmAgent:
    return LlmAgent(
        name=AGENT_NAME,
        model=MODEL,
        tools=make_arena_tools(state),
        instruction=SYSTEM_PROMPT,
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4096),
    )

APP_NAME = "arena-agent"
USER_ID  = AGENT_NAME


async def main():
    state = RunState()
    agent = build_agent(state)
    sessions = InMemorySessionService()
    await sessions.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=state.run_id)
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=sessions)
    # Turn 1: Kickoff
    reply = await run_turn(
        runner, USER_ID, state.run_id, "Start now. Register and solve tasks.")
    print(f"Agent: {reply}")


if __name__ == "__main__":
    asyncio.run(main())