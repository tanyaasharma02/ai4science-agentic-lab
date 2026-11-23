"""Module 01 - one agent, two tools.

Goal: see an LLM agent decide WHEN and HOW to call your code.

  run_simulation()  -> your numerical code wrapped as a tool
  analyst (AgentTool) -> a second agent that can write & run Python for ad-hoc maths

Run:   python -m lab.modules.m01_single_agent_tools
"""
from google.adk.agents import Agent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools.agent_tool import AgentTool

from lab.common import FAST_MODEL, run
from lab.tools.simulation import run_simulation

# ---------------------------------------------------------------------------
# TODO(1): read the docstring of run_simulation() in lab/tools/simulation.py.
# That docstring IS the tool spec the model sees. Good docstrings = reliable calls.
# ---------------------------------------------------------------------------

# A specialist that only does maths. Gemini's built-in code execution cannot be
# combined with function tools in the same agent on Vertex AI, so we give it its
# own agent and expose it as a tool (AgentTool). This "one agent per built-in
# capability" pattern is the normal way to compose search / code / your tools.
analyst_agent = Agent(
    name="analyst",
    model=FAST_MODEL,
    description="Runs Python to fit curves, compute statistics or interpolate from numbers you give it.",
    instruction="""
You are a numerical analyst. You receive numbers in the request and must
answer by WRITING AND EXECUTING Python (numpy is available). Show the key
result with 3 significant figures and one sentence of interpretation.
Never estimate by eye when you can compute.
""",
    code_executor=BuiltInCodeExecutor(),
)

experiment_agent = Agent(
    name="experiment_agent",
    model=FAST_MODEL,
    description="Runs oscillator simulations and analyses the results.",
    instruction="""
You are a computational physicist's assistant.

You have ONE experimental apparatus: the `run_simulation` tool. Use it to
answer questions about the damped driven oscillator. Rules:
- Always call the tool; never guess numbers.
- When a question needs several runs (a sweep), call the tool several times
  with different parameters, then compare.
- Keep plots on (make_plot=True) so a human can check your work later.
- Report: the parameters you used, the numbers you got, and a one-paragraph
  interpretation. Be explicit about uncertainty (measurement noise is on).
""",
    tools=[
        run_simulation,
        # TODO(2): uncomment to let the agent delegate maths (e.g. locate the
        # resonance peak by fitting) to the analyst. Then re-run the sweep prompt
        # and watch the extra tool call appear.
        AgentTool(agent=analyst_agent),
    ],
)

if __name__ == "__main__":
    # Prompt A - single call
    run(experiment_agent, "What is the steady-state amplitude with default parameters?")

    # Prompt B - the agent must plan a sweep (several tool calls)
    run(experiment_agent,
        "Sweep drive_frequency over 0.4, 0.7, 1.0, 1.3, 1.6 with stiffness=1.0 and damping=0.1. "
        "Where is the resonance peak and how sharp is it? Fit a curve rather than eyeballing.")

