#!/usr/bin/env bash
# ModelShift - portable local launcher (macOS / Linux).
#
# One command to run the app on any machine: creates a local virtualenv,
# installs dependencies, and starts the server. No hardcoded paths - it
# resolves everything relative to this script, so it works wherever the
# folder is unzipped.
#
# Usage:
#   ./run.sh                 # LIVE mode (real Bedrock via your local AWS creds), :8971
#   ./run.sh --demo          # OFFLINE demo (seeded sample run, no AWS needed)
#   PORT=9000 ./run.sh       # override the port
#   REGION=us-west-2 ./run.sh --live
#   ./run.sh --bedrock-key sk-...     # use a Bedrock API key instead of SigV4
#
# Any flags after the mode are passed straight through to `python -m modelshift.server`.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV="${VENV:-$REPO_DIR/.venv}"
PORT="${PORT:-8971}"
HOST="${HOST:-127.0.0.1}"
REGION="${REGION:-us-east-1}"

# Pick a Python 3.11+ interpreter.
PY_BOOT="${PYTHON:-python3}"
if ! command -v "$PY_BOOT" >/dev/null 2>&1; then
  PY_BOOT=python
fi
if ! command -v "$PY_BOOT" >/dev/null 2>&1; then
  echo "!! No python3 found on PATH. Install Python 3.11+ and retry." >&2
  exit 1
fi

# Create the venv on first run.
if [ ! -x "$VENV/bin/python" ]; then
  echo "==> Creating virtualenv at $VENV ..."
  "$PY_BOOT" -m venv "$VENV"
fi
PY="$VENV/bin/python"

# Install / refresh dependencies (fast no-op once satisfied).
echo "==> Installing dependencies (requirements.txt) ..."
"$PY" -m pip install --quiet --upgrade pip >/dev/null
"$PY" -m pip install --quiet -r "$REPO_DIR/requirements.txt"

# Default mode is live; first --demo/--live flag overrides.
MODE="--live"
PASS_ARGS=()
for a in "$@"; do
  case "$a" in
    --demo) MODE="--demo" ;;
    --live) MODE="--live" ;;
    *)      PASS_ARGS+=("$a") ;;
  esac
done

echo "==> Starting ModelShift ($MODE) on http://$HOST:$PORT  (region=$REGION)"
if [ "$MODE" = "--live" ]; then
  echo "    LIVE mode uses your local AWS credentials (SigV4) for Bedrock."
  echo "    Run './run.sh --demo' for an offline seeded demo with no AWS."
fi
echo "    Open http://$HOST:$PORT in your browser. Ctrl-C to stop."
echo

exec "$PY" -m modelshift.server "$MODE" \
  --region "$REGION" --host "$HOST" --port "$PORT" "${PASS_ARGS[@]}"
