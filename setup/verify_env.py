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
        if r.returncode != 0:
            print(f"\u2139\ufe0f  could not query enabled APIs via gcloud (not logged in?) - relying on the live call below")
        elif "aiplatform" not in r.stdout:
            print(f"\u26a0\ufe0f  Vertex AI API not listed as enabled - if the live call below fails, run:\n"
                  f"     gcloud services enable aiplatform.googleapis.com --project {proj}")
        else:
            check("Vertex AI API enabled", True)
elif api_key:
    check("GOOGLE_API_KEY set (AI Studio mode)", True)
else:
    check("credentials", False, "copy .env.example to .env and fill in Vertex AI project, or set GOOGLE_API_KEY")

# 4. Live model call (the real test) - run whenever credentials are configured.
#    Model availability differs per project / region / account age (e.g. new
#    accounts may not get gemini-2.5-*), so if the configured model fails we
#    probe a short fallback list and print the exact .env line to use.
FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-3-pro-preview", "gemini-2.5-pro"]


def _ping(client, model: str) -> tuple[bool, str]:
    try:
        r = client.models.generate_content(model=model, contents="Reply with the single word: ready")
        return "ready" in (r.text or "").lower(), (r.text or "").strip()
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:160]}"


if use_vertex or api_key:
    try:
        from google import genai
        client = genai.Client()
        fast = os.getenv("LAB_FAST_MODEL", "gemini-2.5-flash")
        strong = os.getenv("LAB_STRONG_MODEL", "gemini-3-flash-preview")
        ok, info = _ping(client, fast)
        if ok:
            check(f"Gemini call ({fast}) -> {info!r}", True)
        else:
            print(f"\u26a0\ufe0f  LAB_FAST_MODEL={fast} failed: {info}")
            working = [m for m in FALLBACK_MODELS if m != fast and _ping(client, m)[0]]
            if working:
                check(f"Gemini call ({working[0]})", True)
                print(f"\u2139\ufe0f  {fast} is not available on this project/account. Add this line to your .env and re-run:\n"
                      f"     LAB_FAST_MODEL={working[0]}\n"
                      f"     (models that answered here: {', '.join(working)})")
                if strong not in working:
                    print(f"     LAB_STRONG_MODEL={working[-1] if len(working) > 1 else working[0]}")
            else:
                check("Gemini call", False,
                      "none of the known models answered - check project ID, GOOGLE_CLOUD_LOCATION=global, "
                      "that the Vertex AI API is enabled, and that your account has roles/aiplatform.user")
        if ok and strong != fast:
            s_ok, s_info = _ping(client, strong)
            if not s_ok:
                print(f"\u26a0\ufe0f  LAB_STRONG_MODEL={strong} failed: {s_info}\n"
                      f"     module 03/04 use it for the critic - set LAB_STRONG_MODEL={fast} in .env if this persists")
    except Exception as e:  # noqa: BLE001
        check("Gemini call", False, f"{type(e).__name__}: {str(e)[:200]}")

print("\nAll good - see you in the session!" if problems == 0 else f"\n{problems} issue(s) above. Fix them, re-run, and ping us with a screenshot if stuck.")
sys.exit(1 if problems else 0)
