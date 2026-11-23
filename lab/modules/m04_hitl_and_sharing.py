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

root_agent = SequentialAgent(
    name="co_scientist_with_hitl",
    sub_agents=[build_setup_agent(), build_research_loop(), reporter_agent],
)

# TODO(7): make the gate stricter - e.g. require approval BEFORE each
# experiment round when drive_amplitude > 1.0 ("expensive run"). Hint: add a
# before_tool_callback on experiment_agent that calls request_human_approval.

if __name__ == "__main__":
    print("Ask a research question; you will be asked to approve the result at the end.")
    print("Tip: export LAB_AUTO_APPROVE=1 to skip the prompt when recording a demo.\n")
    chat(root_agent)
