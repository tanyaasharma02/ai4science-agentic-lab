"""Module 04 - human-in-the-loop gate + sharing the agent with your team.

Goal 1: nothing gets "committed" without a human saying yes.
Goal 2: run the same agent as a service your whole lab can hit
        (Vertex AI Agent Engine; Cloud Run alternative in the README).

Run locally (interactive):   python -m lab.modules.m04_hitl_and_sharing
Deploy:                      see deploy/agent_engine.py
"""
from google.adk.agents import Agent, SequentialAgent

from lab.common import FAST_MODEL, chat
from lab.modules.m03_multi_agent_loop import build_research_loop, build_setup_agent
from lab.tools.approval import commit_result, request_human_approval

# --- Reporter with a mandatory approval gate --------------------------------
reporter_agent = Agent(
    name="reporter_agent",
    model=FAST_MODEL,
    description="Writes up the accepted finding and gets human sign-off before committing it.",
    instruction="""
You finalise the research loop's output.

HYPOTHESIS: {hypothesis}
EXPERIMENT: {experiment}
CRITIQUE:   {critique}

Steps - do them in order and do not skip any:
1. Draft a title (<= 12 words) and a 3-sentence summary a PI can read in 20 s.
2. Call `request_human_approval` with the title, summary, and a dict of the
   key numbers as evidence.
3. If approved=True: call `commit_result` with the title and a payload holding
   the parameters, metrics and plot paths; then tell the user it was committed.
   If approved=False: do NOT commit. Summarise the reviewer_comment and
   suggest what the next round of the loop should change.
""",
    tools=[request_human_approval, commit_result],
)

# Solution to TODO(7): gate "expensive" simulation calls BEFORE they run.
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool


def guard_expensive_runs(tool: BaseTool, args: dict, tool_context, *a, **kw):
    """before_tool_callback: ask a human before any run with drive_amplitude > 1.0.
    Returning a dict short-circuits the tool call and feeds that dict back to the model."""
    if tool.name == "run_simulation" and float(args.get("drive_amplitude", 0.5)) > 1.0:
        verdict = request_human_approval(
            proposal_title="Expensive simulation run",
            proposal_summary=f"run_simulation with {args}",
            evidence={"drive_amplitude": args.get("drive_amplitude")},
        )
        if not verdict["approved"]:
            return {"error": "run rejected by human reviewer: " + verdict["reviewer_comment"]}
    return None  # proceed normally


research_loop = build_research_loop()
for sub in research_loop.sub_agents:
    if sub.name == "experiment_agent":
        sub.before_tool_callback = guard_expensive_runs

root_agent = SequentialAgent(
    name="co_scientist_with_hitl",
    sub_agents=[build_setup_agent(), research_loop, reporter_agent],
)

# TODO(7): make the gate stricter - e.g. require approval BEFORE each
# experiment round when drive_amplitude > 1.0 ("expensive run"). Hint: add a
# before_tool_callback on experiment_agent that calls request_human_approval.

if __name__ == "__main__":
    print("Ask a research question; you will be asked to approve the result at the end.")
    print("Tip: export LAB_AUTO_APPROVE=1 to skip the prompt when recording a demo.\n")
    chat(root_agent)
