"""Loop-control tool for the critic.

ADK ships `exit_loop`, but a critic that just stops gives you no record of WHY.
This variant stores the conclusion in session state (so later agents and the
caller can read it) and then escalates to end the LoopAgent.
"""
from google.adk.tools.tool_context import ToolContext


def accept(conclusion: str, tool_context: ToolContext) -> dict:
    """Accept the current hypothesis/experiment and stop the research loop.

    Call this ONLY when the evidence clearly confirms or refutes the hypothesis
    and the experimental plan was adequate. Do not call it to request changes.

    Args:
      conclusion: 2-4 sentences: the verdict (confirmed / refuted), the key
        numbers that justify it, and one caveat.

    Returns:
      dict acknowledging the stop.
    """
    tool_context.state["conclusion"] = conclusion
    tool_context.actions.escalate = True  # tells the enclosing LoopAgent to stop
    return {"status": "loop stopped", "conclusion": conclusion}
