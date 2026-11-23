"""Module 05 - swap in YOUR code (homework template).

Replace `my_model_step` with whatever you already run: a JAX ESM step, a GENE
surrogate, a BoTorch acquisition, an MPRA scoring function, a Walrus forecast.
Keep the three rules and the rest of the lab works unchanged:

  1. typed arguments + a docstring that says WHEN to use it and what ranges are valid
  2. return a small JSON-serialisable dict (numbers, strings, short lists, file paths)
  3. do expensive work inside; the agent should only see summaries, never raw arrays

Then point module 03's experiment_agent at [my_model_step, analyse_image] and
rewrite the hypothesis_agent's parameter list. That is the whole migration.
"""
from __future__ import annotations

import os
import time

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def my_model_step(
    param_a: float,
    param_b: float = 1.0,
    n_samples: int = 1000,
    save_artifact: bool = True,
) -> dict:
    """<ONE LINE: what this computes, in domain language>.

    <WHEN the agent should use it, and what question it answers.>

    Args:
      param_a: <meaning, units, valid range>.
      param_b: <meaning, units, valid range>.
      n_samples: <cost knob - say how runtime scales>.
      save_artifact: if True, write a plot/npz and return its path.

    Returns:
      dict with <metric_1>, <metric_2>, artifact_path, and the params used.
    """
    # ---- your real code goes here --------------------------------------
    # e.g.  result = jax.jit(step)(params)   /   subprocess.run(["gene", ...])
    rng = np.random.default_rng()
    samples = rng.normal(param_a, param_b, size=n_samples)
    metric_1 = float(samples.mean())
    metric_2 = float(samples.std())
    # --------------------------------------------------------------------

    artifact_path = None
    if save_artifact:
        os.makedirs(OUT_DIR, exist_ok=True)
        artifact_path = os.path.abspath(os.path.join(OUT_DIR, f"my_model_{int(time.time()*1000)}.npz"))
        np.savez(artifact_path, samples=samples)

    return {
        "metric_1": round(metric_1, 4),
        "metric_2": round(metric_2, 4),
        "artifact_path": artifact_path,
        "params": {"param_a": param_a, "param_b": param_b, "n_samples": n_samples},
    }


# Optional: a guard-rail wrapper. Reject out-of-range calls BEFORE running
# anything expensive, and tell the model why so it can self-correct.
def my_model_step_guarded(param_a: float, param_b: float = 1.0, n_samples: int = 1000) -> dict:
    """Same as my_model_step but validates inputs first. Prefer this one."""
    if not (-10 <= param_a <= 10):
        return {"error": f"param_a={param_a} out of range [-10, 10]"}
    if n_samples > 100_000:
        return {"error": "n_samples > 100k needs human approval; call request_human_approval first"}
    return my_model_step(param_a, param_b, n_samples)
