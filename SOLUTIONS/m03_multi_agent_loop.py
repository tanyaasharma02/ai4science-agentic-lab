"""Module 03 - the mini co-scientist: Hypothesis -> Experiment -> Critic loop.

Goal: three specialised agents coordinate through shared session state, with
the Critic deciding when the loop is done. This is the orchestration pattern
behind lab-in-the-loop / simulation-in-the-loop systems.

  LoopAgent (max 3 rounds)
    +-- hypothesis_agent   writes a falsifiable hypothesis + the runs to test it   -> state["hypothesis"]
    +-- experiment_agent   executes the runs with run_simulation (+ plot checks)    -> state["experiment"]
    +-- critic_agent       accepts (calls accept -> state["conclusion"]) or asks for another round

Run:   python -m lab.modules.m03_multi_agent_loop
"""
from google.adk.agents import Agent, LoopAgent, SequentialAgent

from lab.common import FAST_MODEL, STRONG_MODEL, run
from lab.tools.imaging import analyse_image
from lab.tools.loop_control import accept
from lab.tools.simulation import run_simulation

# ADK agents can have only ONE parent, so we build fresh instances via functions.
# (Module 04 reuses these builders to compose a bigger pipeline.)


def build_hypothesis_agent() -> Agent:
  return Agent(
      name="hypothesis_agent",
      model=FAST_MODEL,
      description="Proposes a falsifiable hypothesis and a minimal experiment plan.",
      instruction="""
  You are the principal investigator's planning brain.

  Research question: {research_question}
  Previous critique (empty on the first round): {critique?}

  Write ONE falsifiable hypothesis about the damped driven oscillator and a
  minimal experiment plan to test it with the `run_simulation` apparatus
  (available parameters: damping, stiffness, drive_amplitude, drive_frequency,
  noise_std, duration_s). If there is a previous critique, address it directly:
  change the plan, do not repeat it.

  Output format (plain text):
  HYPOTHESIS: <one sentence, quantitative if possible>
  PLAN: <numbered list of runs with explicit parameter values, max 6 runs>
  SUCCESS_CRITERION: <what result would confirm / refute it>
  """,
      output_key="hypothesis",
  )

def build_experiment_agent() -> Agent:
  return Agent(
      name="experiment_agent",
      model=FAST_MODEL,
      description="Executes the experiment plan with the simulation and vision tools.",
      instruction="""
  You are the lab automation system. Execute EXACTLY the plan below, nothing more.

  {hypothesis}

  Rules:
  - One `run_simulation` call per planned run, with the stated parameters.
  - For at least one run, call `analyse_image` on the plot_path with the question
    "Has the oscillation reached a steady state? Estimate the steady-state amplitude."
    to sanity-check the numeric amplitude.
  - Then report a compact table of run -> parameters -> steady_state_amplitude
    (and resonance_ratio), followed by 2-3 sentences on what the data show
    relative to SUCCESS_CRITERION. No speculation beyond the data.
  """,
      tools=[run_simulation, analyse_image],
      output_key="experiment",
  )

def build_critic_agent() -> Agent:
  return Agent(
      name="critic_agent",
      model=STRONG_MODEL,
      description="Reviews the evidence; accepts or requests another round.",
      instruction="""
  You are a sceptical reviewer. You are given a hypothesis and the experiment
  that tested it.

  HYPOTHESIS & PLAN:
  {hypothesis}

  EXPERIMENT RESULTS:
  {experiment}

  Decide:
  - If the data clearly confirm or refute the hypothesis AND the plan was
    adequate (enough points, sensible ranges, steady state reached), call the
    `accept` tool with a 2-4 sentence conclusion (verdict, key numbers, one caveat).
  - Otherwise do NOT call accept. Write "REVISE:" followed by specific,
    actionable critique (e.g. "noise dominates at amplitude < 0.1, increase
    drive_amplitude", "sweep was too coarse near resonance, add 0.9 and 1.1").

  Be concrete. Never accept on round one unless the evidence is genuinely strong.
  """,
      tools=[accept],
      output_key="critique",
  )

# --- 4. Wire it up ------------------------------------------------------------
def build_research_loop(max_iterations: int = 3) -> LoopAgent:
  # TODO(4): what happens with max_iterations=1? with 5? Watch cost vs quality.
  return LoopAgent(
      name="research_loop",
      sub_agents=[build_hypothesis_agent(), build_experiment_agent(), build_critic_agent()],
      max_iterations=max_iterations,
  )


def build_setup_agent() -> Agent:
  # A tiny first step that stores the question in session state for the others.
  return Agent(
      name="setup_agent",
      model=FAST_MODEL,
      instruction="Restate the user's research question in one sentence. Output only that sentence.",
      output_key="research_question",
  )


def build_reporter_agent() -> Agent:
  # Solution to TODO(6): a reporter that turns the loop's state into a methods paragraph.
  return Agent(
      name="reporter_agent",
      model=FAST_MODEL,
      instruction="""
Write a short 'Methods and Results' paragraph (max 120 words) suitable for a lab
notebook, based only on:
HYPOTHESIS: {hypothesis}
EXPERIMENT: {experiment}
CONCLUSION: {conclusion?}
State the parameters swept, the key numbers, the verdict and one caveat.
""",
      output_key="report",
  )


mini_co_scientist = SequentialAgent(
    name="mini_co_scientist",
    sub_agents=[build_setup_agent(), build_research_loop(), build_reporter_agent()],
)

if __name__ == "__main__":
    run(mini_co_scientist,
        "How does the resonance peak amplitude of the oscillator depend on damping? "
        "I care about damping between 0.02 and 0.5.")

    # TODO(5): swap the Critic to FAST_MODEL. Does it accept too early?
    # TODO(6): add a 4th agent 'reporter' after the loop that writes a short methods paragraph.
