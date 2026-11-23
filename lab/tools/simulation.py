"""Tool 1: a stand-in for YOUR simulation / analysis code.

In the lab this is a damped, driven oscillator with measurement noise.
In your project this is whatever you already have: a JAX model step, a
GENE/GS2 wrapper, a BoTorch acquisition, an MPRA scoring function...

The only contract the agent cares about:
  * typed arguments with a docstring (the LLM reads this to decide how to call it)
  * a JSON-serialisable return value (dict of numbers/strings/lists)
"""
from __future__ import annotations

import os
import time

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def run_simulation(
    damping: float = 0.1,
    stiffness: float = 1.0,
    drive_amplitude: float = 0.5,
    drive_frequency: float = 0.8,
    noise_std: float = 0.05,
    duration_s: float = 40.0,
    make_plot: bool = True,
) -> dict:
    """Simulate a damped, driven harmonic oscillator and return summary metrics.

    Use this to test a hypothesis about how the steady-state amplitude depends
    on the parameters. The system is  x'' + 2*damping*x' + stiffness*x =
    drive_amplitude*cos(drive_frequency*t), integrated from rest.

    Args:
      damping: damping ratio (0.01 - 2.0). Larger = energy lost faster.
      stiffness: spring constant k (0.1 - 5.0). Natural frequency = sqrt(k).
      drive_amplitude: amplitude of the external drive (0 - 2).
      drive_frequency: angular frequency of the external drive (0.1 - 3.0).
      noise_std: std-dev of Gaussian measurement noise added to x(t).
      duration_s: how long to simulate, in seconds.
      make_plot: if True, save a PNG of x(t) and return its path.

    Returns:
      dict with steady_state_amplitude, natural_frequency, resonance_ratio
      (drive_frequency / natural_frequency), n_steps, plot_path (or None),
      and the parameters used.
    """
    dt = 0.01
    n = int(duration_s / dt)
    t = np.arange(n) * dt
    x = np.zeros(n)
    v = 0.0
    for i in range(1, n):
        a = drive_amplitude * np.cos(drive_frequency * t[i - 1]) - 2 * damping * v - stiffness * x[i - 1]
        v += a * dt
        x[i] = x[i - 1] + v * dt
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    x_meas = x + rng.normal(0.0, noise_std, size=n)

    tail = x_meas[int(0.75 * n):]  # last quarter ~ steady state
    amplitude = float((tail.max() - tail.min()) / 2)
    natural = float(np.sqrt(stiffness))

    plot_path = None
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(OUT_DIR, exist_ok=True)
        plot_path = os.path.abspath(os.path.join(OUT_DIR, f"sim_{int(time.time()*1000)}.png"))
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(t, x_meas, lw=0.8)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("x(t)")
        ax.set_title(f"damping={damping}, k={stiffness}, drive A={drive_amplitude}, w={drive_frequency}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=110)
        plt.close(fig)

    return {
        "steady_state_amplitude": round(amplitude, 4),
        "natural_frequency": round(natural, 4),
        "resonance_ratio": round(drive_frequency / natural, 4),
        "n_steps": n,
        "plot_path": plot_path,
        "params": {
            "damping": damping,
            "stiffness": stiffness,
            "drive_amplitude": drive_amplitude,
            "drive_frequency": drive_frequency,
            "noise_std": noise_std,
        },
    }


if __name__ == "__main__":
    print(run_simulation())
