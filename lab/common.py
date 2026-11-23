"""Shared helpers for the lab: config + a tiny terminal runner for ADK agents.

Participants do not need to edit this file.
"""
from __future__ import annotations

import asyncio
import os
import uuid
import warnings

# Keep the terminal readable during the live session:
#  - ADK 2.x flags LoopAgent/SequentialAgent as deprecated in favour of the (pre-GA)
#    Workflow API. They still work and are the clearest teaching primitives, so we stay on them.
#  - google-auth warns when user ADC has no quota project; harmless here.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"google\.adk.*")
warnings.filterwarnings("ignore", message=".*deprecated in favor of Workflow.*")
warnings.filterwarnings("ignore", message=".*without a quota project.*")

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# --- Model choices -----------------------------------------------------------
# Flash is fast + cheap and good enough for tool-calling agents.
# Pro is used for the Critic in module 03 (better at judging).
FAST_MODEL = os.getenv("LAB_FAST_MODEL", "gemini-2.5-flash")
STRONG_MODEL = os.getenv("LAB_STRONG_MODEL", "gemini-2.5-pro")

APP_NAME = "ai4science_lab"
USER_ID = os.getenv("USER", "researcher")


def check_config() -> None:
    """Fail early with a readable message if Vertex AI config is missing."""
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")
    if use_vertex:
        missing = [k for k in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION") if not os.getenv(k)]
        if missing:
            raise SystemExit(
                f"Missing env vars {missing}. Copy .env.example to .env and fill it in, "
                "or export them in your shell."
            )
    elif not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit(
            "No credentials configured. Either set GOOGLE_GENAI_USE_VERTEXAI=TRUE + "
            "GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION (recommended, uses your Cloud "
            "Credits), or set GOOGLE_API_KEY for AI Studio."
        )


async def _run(agent, prompt: str, session_id: str | None = None, verbose: bool = True) -> str:
    session_service = InMemorySessionService()
    session_id = session_id or str(uuid.uuid4())[:8]
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    final_text = ""
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
        if verbose:
            _print_event(event)
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    if session and session.state.get("conclusion") and not final_text.strip():
        final_text = "CONCLUSION: " + session.state["conclusion"]
    if verbose and session and session.state:
        keys = [k for k in ("research_question", "hypothesis", "conclusion") if k in session.state]
        if keys:
            print("\n--- session state (what the next agent would see) ---")
            for k in keys:
                print(f"{k}: {_short(session.state[k], 300)}")
    return final_text


def _print_event(event) -> None:
    """Pretty-print what the agent is doing so the loop is visible on screen."""
    author = getattr(event, "author", "?")
    if not event.content or not event.content.parts:
        return
    for part in event.content.parts:
        if part.function_call:
            args = dict(part.function_call.args or {})
            print(f"  [{author}] -> tool {part.function_call.name}({_short(args)})")
        elif part.function_response:
            print(f"  [{author}] <- {part.function_response.name} returned {_short(part.function_response.response)}")
        elif part.executable_code:
            print(f"  [{author}] running code:\n{_indent(part.executable_code.code)}")
        elif part.code_execution_result:
            print(f"  [{author}] code output: {_short(part.code_execution_result.output)}")
        elif part.text and part.text.strip():
            print(f"  [{author}] {part.text.strip()}")


def _short(obj, n: int = 160) -> str:
    s = str(obj)
    return s if len(s) <= n else s[:n] + "..."


def _indent(code: str) -> str:
    return "\n".join("      " + line for line in code.strip().splitlines())


def run(agent, prompt: str, **kw) -> str:
    """Run an agent on one prompt and return the final text answer."""
    check_config()
    print(f"\n>>> {prompt}\n")
    answer = asyncio.run(_run(agent, prompt, **kw))
    print(f"\n=== Final answer ===\n{answer}\n")
    return answer


def chat(agent) -> None:
    """Interactive loop (used in module 04 to play with the HITL gate)."""
    check_config()
    session_id = str(uuid.uuid4())[:8]
    print("Type a message (or 'q' to quit).")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("q", "quit", "exit"):
            break
        if prompt:
            asyncio.run(_run(agent, prompt, session_id=session_id))
