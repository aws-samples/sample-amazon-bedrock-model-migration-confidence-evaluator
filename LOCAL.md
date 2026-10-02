# Running ModelShift locally

A self-contained local run of the ModelShift web app. One command starts a
FastAPI server + the single-page UI on `http://127.0.0.1:8971`.

There is **no build step and no container needed** for local use - it is a plain
Python app. (Container / ECS Fargate deployment files are not part of this
package; ask the ModelShift maintainer for the `deploy/` bundle.)

**New here?** Open the app and click **Guide** in the left rail: an interactive
walk-through of a run (logs -> candidates -> replay -> scores -> results ->
fixes -> report), with a "Start here" checklist for your first run.

---

## What you can do

- **Ingest** a LiteLLM log export (upload) or an S3 prefix; filter by team/user and cap the record count.
- **Replay** every request against candidate models on Bedrock Runtime, Bedrock Mantle, or your own LiteLLM proxy (its models are discovered automatically).
- **Score** each answer on six checks with an LLM judge you can steer; get a Migration Confidence score and verdict per candidate.
- **Investigate** any evaluation: original vs candidate response with the differences highlighted.
- **Fix and re-test**: edit a suggested prompt or reasoning-effort change, re-test it with a regression check, and copy the patch.
- **Share** a PDF report (dark, or print-friendly light).

---

## Prerequisites

- **Python 3.11+** on your PATH (`python3 --version`).
- For **live** mode: **AWS credentials** with Bedrock access in your shell
  (`aws configure` / `aws sso login`, or `AWS_PROFILE=...`). The app signs the
  Bedrock `/openai/v1` calls with your local credentials (SigV4) - no API key
  needed. For the **demo** mode below, no AWS at all is required.

---

## Quick start

### macOS / Linux
```bash
cd modelshift
./run.sh --demo     # offline: seeds a sample run so the UI has data, no AWS
# then open http://127.0.0.1:8971
```

### Windows
```bat
cd modelshift
run.bat --demo
REM then open http://127.0.0.1:8971
```

The launcher creates a local `.venv/`, installs `requirements.txt`, and starts
the server. First run takes ~30-60s to install dependencies; subsequent runs are
instant. Press **Ctrl-C** to stop.

---

## Demo vs. Live

| Mode | Command | AWS needed | What it does |
|------|---------|-----------|--------------|
| **Demo** | `./run.sh --demo` | No | Offline transports + a seeded "member-services" run so you can click through the whole UI immediately. |
| **Live** | `./run.sh` (or `--live`) | Yes | Real replays against Bedrock (Luna/Terra/Sol/GPT-5.4, Claude) using your local AWS credentials. Upload your own LiteLLM logs to evaluate. |

Live mode is the default when you omit the flag.

---

## Configuration (all optional)

Set via environment variable before launching, or pass a flag through the launcher.

| Env var | Flag | Default | Purpose |
|---------|------|---------|---------|
| `PORT` | `--port` | `8971` | HTTP port |
| `HOST` | `--host` | `127.0.0.1` | Bind address (keep loopback for local) |
| `REGION` | `--region` | `us-east-1` | AWS region for Bedrock |
| `MODELSHIFT_HOME` | - | `~/.modelshift` | Where run state + settings are stored |
| `MODELSHIFT_LITELLM_BASE` | `--litellm-base` | (unset) | Route candidates through a LiteLLM proxy instead of Bedrock direct |
| `MODELSHIFT_BEDROCK_KEY` | `--bedrock-key` | (unset) | Use a Bedrock API key (Bearer) instead of SigV4 |

Examples:
```bash
PORT=9000 ./run.sh --demo
REGION=us-west-2 ./run.sh
./run.sh --litellm-base http://127.0.0.1:4000
```

---

## Where local state lives

Runs and settings persist under **`~/.modelshift/`** by default:
- `~/.modelshift/runs/` - one JSON per evaluation run (survives restarts)
- `~/.modelshift/settings.json` - UI-editable settings (judge models, prices, thresholds, LiteLLM base/key)

Relocate both with a single env var: `MODELSHIFT_HOME=/path/to/state ./run.sh`.
Delete `~/.modelshift/` to reset to a clean state.

---

## Verifying it's up

```bash
curl http://127.0.0.1:8971/api/v1/health     # {"status":"ok","build_id":"0.1.0+..."}
curl http://127.0.0.1:8971/api/v1/version    # full build detail
```

The left rail of the UI also shows the running **build id** and start time - if it
doesn't match your source, you're looking at a stale server.

---

## Stopping / restarting

- **Foreground:** Ctrl-C in the terminal running `run.sh`.
- **Find a stray server:** `ps -eo pid,cmd | grep '[m]odelshift.server'` then `kill <pid>`.
- On this dev host there's also `restart.sh`, which force-kills stale servers and
  verifies the running build matches source. `run.sh` is the portable equivalent
  for a fresh machine (it doesn't assume any pre-existing venv).

---

## Running the tests (optional)

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python -m flake8
```
