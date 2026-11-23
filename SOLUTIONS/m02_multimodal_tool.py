"""Module 02 - add a multimodal tool.

Goal: the agent can now LOOK at the plot it produced (or at a micrograph) and
get structured numbers back, via a Gemini-as-VLM tool.

Run:   python -m lab.modules.m02_multimodal_tool
"""
from google.adk.agents import Agent

from lab.common import FAST_MODEL, run
from lab.tools.imaging import analyse_image, make_synthetic_micrograph
from lab.tools.simulation import run_simulation

experiment_agent = Agent(
    name="experiment_agent",
    model=FAST_MODEL,
    description="Runs simulations, inspects the resulting plots, and analyses images.",
    instruction="""
You are a computational scientist's assistant with two kinds of tools:
- `run_simulation`: run the oscillator model; it returns metrics AND a plot_path.
- `analyse_image`: ask a vision model a focused question about an image file.
  Always pass a precise question (what to measure), never a vague one.

Workflow for simulation questions:
1. run the simulation,
2. call analyse_image on the returned plot_path to independently check whether
   the signal reached steady state and whether the amplitude you computed is
   consistent with what is visible,
3. report both the numeric result and the visual sanity check. If they disagree,
   say so and propose a longer duration_s.

For micrograph questions use `make_synthetic_micrograph` then `analyse_image`,
and compare the vision estimate with the ground truth the generator returns.
""",
    tools=[run_simulation, analyse_image, make_synthetic_micrograph],
)

if __name__ == "__main__":
    run(experiment_agent,
        "Run the oscillator with damping=0.02, drive_frequency=1.0 and duration_s=20. "
        "Did it reach steady state? Check the plot, don't just trust the number.")

    # TODO(3): try the imaging path - how far off is the VLM on particle count / size?
    run(experiment_agent,
        "Generate a synthetic micrograph with 30 particles of mean radius 10 px, "
        "then estimate the particle count and mean radius from the image and compare to ground truth.")
