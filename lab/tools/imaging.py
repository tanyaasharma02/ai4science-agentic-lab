"""Tool 2: multimodal interpretation of an image / plot with Gemini.

Pattern: the *agent* does not look at pixels. It calls a tool that asks Gemini
(as a VLM) a focused question and returns STRUCTURED JSON. That keeps the
orchestration loop cheap and the vision step testable and swappable
(e.g. replace with your own segmentation model later).

Also includes a tiny synthetic "micrograph" generator so the lab has images
without downloading anything.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from google import genai
from google.genai import types

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        from dotenv import load_dotenv
        load_dotenv()  # so the tool also works when imported outside lab.common
        _client = genai.Client()  # picks up GOOGLE_GENAI_USE_VERTEXAI / project / location from env
    return _client


def analyse_image(image_path: str, question: str = "Describe what you see and extract any quantitative features.") -> dict:
    """Interpret a scientific image or plot with Gemini and return structured findings.

    Works for simulation plots (from run_simulation) and for micrographs
    (see make_synthetic_micrograph). Returns JSON so downstream agents can
    reason over numbers, not prose.

    Args:
      image_path: absolute path to a PNG/JPG file.
      question: what to look for, e.g. "Is the oscillation at steady state?" or
        "Count the particles and estimate their mean diameter in pixels".

    Returns:
      dict with keys: summary (str), features (dict of name -> number/str),
      confidence (0-1), caveats (list[str]).
    """
    if not os.path.exists(image_path):
        return {"error": f"file not found: {image_path}"}
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

    # Keep the schema simple: Vertex rejects union types and additionalProperties.
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                    },
                    "required": ["name", "value", "unit"],
                },
            },
            "confidence": {"type": "number"},
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "features", "confidence", "caveats"],
    }
    resp = _get_client().models.generate_content(
        model=os.getenv("LAB_FAST_MODEL", "gemini-2.5-flash"),
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=mime),
            f"You are a careful lab scientist. {question} "
            "Report numbers you can actually read off the image; if you must estimate, say so in caveats.",
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        ),
    )
    try:
        return json.loads(resp.text)
    except Exception:  # keep the loop alive even if the model misbehaves
        return {"summary": resp.text, "features": {}, "confidence": 0.0, "caveats": ["non-JSON response"]}


def make_synthetic_micrograph(n_particles: int = 40, mean_radius_px: int = 12, seed: int = 0) -> dict:
    """Generate a synthetic SEM-like image of round particles (for practice only).

    Args:
      n_particles: how many particles to draw.
      mean_radius_px: mean particle radius in pixels (log-normal spread).
      seed: RNG seed for reproducibility.

    Returns:
      dict with image_path and the ground-truth n_particles / mean_radius_px so
      you can score the vision tool's answer.
    """
    from PIL import Image, ImageDraw, ImageFilter

    rng = np.random.default_rng(seed)
    size = 512
    img = Image.new("L", (size, size), color=40)
    draw = ImageDraw.Draw(img)
    radii = []
    for _ in range(n_particles):
        r = max(3, int(rng.lognormal(np.log(mean_radius_px), 0.25)))
        x, y = rng.integers(r, size - r, size=2)
        shade = int(rng.integers(150, 230))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=shade)
        radii.append(r)
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    noise = rng.normal(0, 12, size=(size, size))
    arr = np.clip(np.asarray(img, dtype=float) + noise, 0, 255).astype("uint8")
    out = Image.fromarray(arr)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.abspath(os.path.join(DATA_DIR, f"micrograph_seed{seed}.png"))
    out.save(path)
    return {"image_path": path, "n_particles": n_particles, "mean_radius_px": float(np.mean(radii))}


if __name__ == "__main__":
    print(make_synthetic_micrograph())
