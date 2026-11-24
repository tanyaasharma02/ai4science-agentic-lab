# Agentic research workflows on Google Cloud

**Hands-on lab · Google.org AI for Science Accelerator · 24 Nov 2026**
*Multi-agent coordination, tool calling & human-in-the-loop with ADK + Gemini*

You will build, in ~45 minutes, the orchestration layer of a lab-in-the-loop system:

```
hypothesis agent  ->  experiment agent  ->  critic agent  ->  (loop)  ->  human approval  ->  commit
                          |  tool calls
                          +-- run_simulation()   <- your code goes here
                          +-- analyse_image()    <- Gemini as a vision model
```

Everything runs against **Vertex AI** with your Cloud Credits. No datasets to download.

**Session recording (24 Nov 2026):** [video](https://drive.google.com/file/d/1cooYGMUfUua1X4e0xs1YIlOy9kJnBMH7/view) · [notes + transcript](https://docs.google.com/document/d/18W6xm5ovlbuymGx0EaF9Dmb85vqqIFxC5iRhlZvsTjg/edit) *(accelerator participants)*

---

## 0. Setup (do this BEFORE the session, ~5 min)

```bash
git clone https://github.com/tanyaasharma02/ai4science-agentic-lab.git && cd ai4science-agentic-lab
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then edit: project id + location
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID
python setup/verify_env.py                               # all green? you're ready
```

No GCP project yet? Put an AI Studio key in `.env` as `GOOGLE_API_KEY` instead; everything except module 04's deploy works.

**Using Antigravity (recommended):** open this folder in Antigravity, run the commands above in its terminal, and when you hit a `TODO(n)` in a module, ask Antigravity to help - that *is* part of the exercise.

**Using Colab instead:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tanyaasharma02/ai4science-agentic-lab/blob/main/notebooks/lab.ipynb) — same modules as cells.

---

## 1. The modules

| # | File | You learn | Run |
|---|------|-----------|-----|
| 01 | `lab/modules/m01_single_agent_tools.py` | A docstring is a tool spec. Agent plans a parameter sweep on its own. Optional code executor for ad-hoc maths. | `python -m lab.modules.m01_single_agent_tools` |
| 02 | `lab/modules/m02_multimodal_tool.py` | Vision as a *tool* returning structured JSON; agent sanity-checks its own plot; synthetic micrograph test. | `python -m lab.modules.m02_multimodal_tool` |
| 03 | `lab/modules/m03_multi_agent_loop.py` | `LoopAgent` of Hypothesis -> Experiment -> Critic sharing state via `output_key`; critic ends the loop with `exit_loop`. **The mini co-scientist.** | `python -m lab.modules.m03_multi_agent_loop` |
| 04 | `lab/modules/m04_hitl_and_sharing.py` | A mandatory human-approval tool before `commit_result`; deploy the same agent to **Agent Engine** for your whole team. | `python -m lab.modules.m04_hitl_and_sharing` |
| 05 | `lab/modules/m05_swap_in_your_code.py` | Template + three rules for wrapping *your* JAX/PyTorch/HPC code as a tool. Homework. | - |

Each module has small `TODO(n)` markers. `SOLUTIONS/` has finished versions if you fall behind (`python -m SOLUTIONS.m03_multi_agent_loop`).

> **Model availability:** the lab defaults to `gemini-2.5-flash` (workers) and `gemini-2.5-pro` (critic). If your project returns a 404 for Pro, set `LAB_STRONG_MODEL=gemini-2.5-flash` (or `gemini-3-flash-preview` with `GOOGLE_CLOUD_LOCATION=global`) in `.env`.

---

## 2. Patterns worth stealing (what the modules demonstrate)

* **Agent priming** = `instruction` + what you inject from state (`{hypothesis}`, `{critique?}`). Roles stay narrow; the orchestration (not the prompt) carries the workflow.
* **Reliable tool calling**: typed args, precise docstrings, small JSON returns, validate inputs *inside* the tool and return `{"error": ...}` so the model can self-correct (see `my_model_step_guarded`).
* **Vision as a tool, not a vibe**: `analyse_image` forces a JSON schema; downstream agents reason over numbers.
* **Cheap model for workers, strong model for the judge**: Flash for hypothesis/experiment, Pro for the critic. Swap and watch acceptance quality change (`TODO(5)`).
* **Human gate as a tool the model must call**: easiest HITL that actually works; the agent cannot "commit" around it.
* **Shared environment**: one deployed agent, many users, sessions per user; traces on in Cloud Trace.

---

## 3. Sharing the agent with your lab (module 04 part 2)

```bash
pip install "google-cloud-aiplatform[adk,agent_engines]"
export STAGING_BUCKET=gs://$GOOGLE_CLOUD_PROJECT-agent-engine-staging   # create it once
gsutil mb -l $GOOGLE_CLOUD_LOCATION $STAGING_BUCKET
python deploy/agent_engine.py                        # ~3-6 min, prints a resource name
python deploy/agent_engine.py --resource <name> --query "How does resonance amplitude depend on damping?"
```

Cloud Run alternative (if your lab already runs containers): `adk deploy cloud_run --project $GOOGLE_CLOUD_PROJECT --region $GOOGLE_CLOUD_LOCATION --service_name ai4science-lab lab/`

### Human-in-the-loop in production
A deployed service has no terminal, so the lab's `input()` gate is replaced by `LAB_AUTO_APPROVE=1` in the deploy. Real options, in increasing effort:
1. **Notify + hold**: tool posts to Google Chat/Slack and polls a Firestore doc / Sheet for the decision (minutes).
2. **ADK long-running tool**: mark the approval tool as long-running; the run pauses and resumes when your UI calls back with the decision.
3. **ADK 2.0 Workflow interrupts** (pre-GA): first-class HITL nodes in a graph workflow.

---

## 4. Where this helps (and where it doesn't)

* Helps: orchestrating dataset-generation sweeps and simulation campaigns; closing design-build-test loops with a reviewer step; routing cheap vs. strong models per step to control inference cost; giving a whole lab one shared, traceable agent.
* Doesn't: replace compute. Your TPU/GPU hours are still the bottleneck; agents just stop you wasting them on badly planned runs.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `404 model not found` / model "not available" for your account | Model availability differs per project, region and account age (new accounts may not get `gemini-2.5-*`). Run `python setup/verify_env.py` - it probes the alternatives and prints the exact `LAB_FAST_MODEL=` / `LAB_STRONG_MODEL=` lines for your `.env`. Also try `GOOGLE_CLOUD_LOCATION=global`. |
| `403 PERMISSION_DENIED aiplatform` | `gcloud services enable aiplatform.googleapis.com`; check billing / credits attached. |
| `Missing env vars` | `.env` not in the repo root, or run from another directory. |
| Agent never calls the tool | Read the docstring the model sees - make the "when to use" sentence explicit. |
| Critic accepts round 1 every time | Use `STRONG_MODEL` for the critic; tighten the "never accept on round one unless..." rule. |

Questions after the session: open an issue on this repo or ping Tanya.
