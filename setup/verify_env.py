"""Pre-flight check. Run this BEFORE the session:  python setup/verify_env.py

Prints a ✅/❌ line per requirement and a one-line fix for each ❌.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

OK, BAD = "\u2705", "\u274c"
problems = 0


def check(label: str, ok: bool, fix: str = "") -> None:
    global problems
    print(f"{OK if ok else BAD} {label}" + ("" if ok else f"\n     fix: {fix}"))
    problems += 0 if ok else 1


# 1. Python
v = sys.version_info
check(f"Python {v.major}.{v.minor}", (v.major, v.minor) >= (3, 10), "install Python 3.10+ (3.11 recommended)")

# 2. Packages
for pkg, pipname in [("google.adk", "google-adk"), ("google.genai", "google-genai"),
                     ("numpy", "numpy"), ("matplotlib", "matplotlib"), ("PIL", "pillow"), ("dotenv", "python-dotenv")]:
    try:
        importlib.import_module(pkg)
        check(f"package {pipname}", True)
    except Exception:
        check(f"package {pipname}", False, "pip install -r requirements.txt")

# 3. Credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")
api_key = os.getenv("GOOGLE_API_KEY")
if use_vertex:
    proj, loc = os.getenv("GOOGLE_CLOUD_PROJECT"), os.getenv("GOOGLE_CLOUD_LOCATION")
    check(f"GOOGLE_CLOUD_PROJECT={proj}", bool(proj), "export GOOGLE_CLOUD_PROJECT=<your-project-id>")
    check(f"GOOGLE_CLOUD_LOCATION={loc}", bool(loc), "export GOOGLE_CLOUD_LOCATION=us-central1")
    check("gcloud CLI on PATH", shutil.which("gcloud") is not None, "https://cloud.google.com/sdk/docs/install")
    adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    check("Application Default Credentials", os.path.exists(adc) or bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
          "gcloud auth application-default login")
    if proj and shutil.which("gcloud"):
        r = subprocess.run(["gcloud", "services", "list", "--enabled", "--project", proj,
                            "--filter=name:aiplatform.googleapis.com", "--format=value(name)"],
                           capture_output=True, text=True)
        check("Vertex AI API enabled", "aiplatform" in r.stdout,
              f"gcloud services enable aiplatform.googleapis.com --project {proj}")
elif api_key:
    check("GOOGLE_API_KEY set (AI Studio mode)", True)
else:
    check("credentials", False, "copy .env.example to .env and fill in Vertex AI project, or set GOOGLE_API_KEY")

# 4. Live model call (the real test)
if problems == 0:
    try:
        from google import genai
        client = genai.Client()
        r = client.models.generate_content(model=os.getenv("LAB_FAST_MODEL", "gemini-2.5-flash"),
                                           contents="Reply with the single word: ready")
        check(f"Gemini call -> {r.text.strip()!r}", "ready" in r.text.lower(), "check project/location/quota")
    except Exception as e:  # noqa: BLE001
        check("Gemini call", False, f"{type(e).__name__}: {str(e)[:200]}")

print("\nAll good - see you in the session!" if problems == 0 else f"\n{problems} issue(s) above. Fix them, re-run, and ping us with a screenshot if stuck.")
sys.exit(1 if problems else 0)
