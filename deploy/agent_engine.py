"""Deploy the HITL co-scientist to Vertex AI Agent Engine so the whole lab shares one endpoint.

Usage:
  pip install "google-cloud-aiplatform[adk,agent_engines]"
  python deploy/agent_engine.py            # deploy
  python deploy/agent_engine.py --query "How does resonance amplitude depend on damping?"

Notes for the session:
  * Deployment takes ~3-6 min. Pre-deploy before the demo; show the command, then hit the live one.
  * The terminal-based approval tool will auto-approve when LAB_AUTO_APPROVE=1 is set in env_vars
    (a remote service has no stdin). For real HITL in production, replace it with a long-running
    tool or a Chat/Slack approval - see README "Human-in-the-loop in production".
  * Cloud Run alternative:  adk deploy cloud_run --project $GOOGLE_CLOUD_PROJECT \
        --region $GOOGLE_CLOUD_LOCATION --service_name ai4science-lab lab/
"""
from __future__ import annotations

import argparse
import os

import vertexai
from vertexai import agent_engines
from vertexai.preview import reasoning_engines

from lab.modules.m04_hitl_and_sharing import root_agent

PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
STAGING_BUCKET = os.environ.get("STAGING_BUCKET", f"gs://{PROJECT}-agent-engine-staging")
DISPLAY_NAME = "ai4science-mini-co-scientist"


def deploy():
    vertexai.init(project=PROJECT, location=LOCATION, staging_bucket=STAGING_BUCKET)
    app = reasoning_engines.AdkApp(agent=root_agent, enable_tracing=True)
    remote = agent_engines.create(
        agent_engine=app,
        display_name=DISPLAY_NAME,
        requirements=["google-cloud-aiplatform[adk,agent_engines]", "google-adk", "google-genai",
                      "numpy", "matplotlib", "pillow", "python-dotenv"],
        extra_packages=["lab"],
        env_vars={"LAB_AUTO_APPROVE": "1", "GOOGLE_GENAI_USE_VERTEXAI": "TRUE"},
    )
    print("Deployed:", remote.resource_name)
    print("Share this resource name with your team; they can query it with --resource.")
    return remote


def query(resource_name: str, text: str):
    vertexai.init(project=PROJECT, location=LOCATION)
    remote = agent_engines.get(resource_name)
    session = remote.create_session(user_id=os.getenv("USER", "teammate"))
    for event in remote.stream_query(user_id=session["user_id"], session_id=session["id"], message=text):
        for part in event.get("content", {}).get("parts", []):
            if "text" in part:
                print(part["text"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="send a question to an already deployed agent")
    ap.add_argument("--resource", help="projects/.../reasoningEngines/... (from deploy output)")
    a = ap.parse_args()
    if a.query:
        if not a.resource:
            raise SystemExit("--resource is required with --query")
        query(a.resource, a.query)
    else:
        deploy()
