"""Tool 3: the human-in-the-loop gate.

The simplest possible HITL: a tool the agent MUST call before "committing" a
result. In the lab it prompts in the terminal. In production you swap the body
for a Slack/Chat message, a ticket, or a long-running ADK tool that pauses the
run until a human approves (see module 04 notes).
"""
from __future__ import annotations

import json
import os
import time

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def request_human_approval(proposal_title: str, proposal_summary: str, evidence: dict) -> dict:
    """Ask a human to approve or reject a proposal before it is committed.

    Call this exactly once, at the end, when the critic is satisfied. Do not
    commit or report a result as final until this tool returns approved=True.

    Args:
      proposal_title: one-line title of the finding or next experiment.
      proposal_summary: 2-4 sentences a busy PI can read in 20 seconds.
      evidence: the key numbers that support it (e.g. amplitudes, ratios).

    Returns:
      dict with approved (bool) and reviewer_comment (str).
    """
    print("\n" + "=" * 70)
    print("HUMAN APPROVAL REQUESTED")
    print("=" * 70)
    print(f"Title   : {proposal_title}")
    print(f"Summary : {proposal_summary}")
    print(f"Evidence: {json.dumps(evidence, indent=2)}")
    print("-" * 70)
    if os.getenv("LAB_AUTO_APPROVE", "").lower() in ("1", "true"):
        decision, comment = "y", "auto-approved (LAB_AUTO_APPROVE)"
    else:
        decision = input("Approve? [y/N] ").strip().lower()
        comment = input("Comment (optional): ").strip()
    approved = decision == "y"
    print("=" * 70 + "\n")
    return {"approved": approved, "reviewer_comment": comment}


def commit_result(title: str, payload: dict) -> dict:
    """Persist an approved result to the shared results log (outputs/results.jsonl).

    Only call this after request_human_approval returned approved=True.

    Args:
      title: short title of the result.
      payload: anything JSON-serialisable worth keeping (params, metrics, plot paths).

    Returns:
      dict with the path written and a timestamp.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.abspath(os.path.join(OUT_DIR, "results.jsonl"))
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "title": title, "payload": payload}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"written_to": path, "ts": record["ts"]}
