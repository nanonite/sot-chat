#!/usr/bin/env bash
# Register the `sot` conda environment if needed and launch the sot-chat app.
# Run with --help for options and environment knobs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run_sot_chat.sh [--host HOST] [--port PORT] [--context-mode MODE] [extra sot-chat args...]

Creates or activates the conda environment if needed, then launches the SoT chat server.

Options:
  -h, --help                  Show this help and exit
  --host HOST                 Bind address (default 127.0.0.1)
  --port PORT                 Bind port (default 8787)
  --context-mode MODE         inject | native (default native)
                              native: resume the harness session and use a text handoff only
                                      when no native session exists.
                              inject: send the running summary plus recent turns on every
                                      reply and never resume a native harness session.

Environment knobs:
  SOT_ENV_NAME                Conda env name (default sot)
  SOT_PYTHON_VERSION          Python version for the env (default 3.10)
  SOT_CHAT_HOST               Bind address (default 127.0.0.1)
  SOT_CHAT_PORT               Bind port (default 8787)
  SOT_CONTEXT_MODE            inject | native (default native)
  SOT_CONTEXT_BUDGET_CHARS    Max working-context characters (default 24000)
  SOT_CONTEXT_TAIL_CHARS      Verbatim recent-turn window (default 8000)
  SOT_CONTEXT_SUMMARY_CHARS   Max running-summary length (default 4000)
  SOT_WORKDIR                 Working directory for harness calls
  SOT_PROVIDER_TIMEOUT        Harness timeout in seconds (default 900)
  SOT_UPLOAD_MAX_FILES        Max uploaded files per conversation (default 50)
  SOT_UPLOAD_MAX_FILE_BYTES   Max bytes per uploaded file (default 1000000)
  SOT_UPLOAD_MAX_TOTAL_BYTES  Max uploaded bytes per conversation (default 6000000)
  SOT_UPLOAD_MAX_CONTEXT_CHARS Max uploaded-context characters (default 120000)
  SOT_CHAT_QUIET              Set to 1 to silence HTTP request logs
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
  esac
done

ENV_NAME="${SOT_ENV_NAME:-sot}"
PYTHON_VERSION="${SOT_PYTHON_VERSION:-3.10}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- locate conda -----------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  for candidate in "$HOME/anaconda3/etc/profile.d/conda.sh" \
                   "$HOME/miniconda3/etc/profile.d/conda.sh" \
                   "$HOME/miniforge3/etc/profile.d/conda.sh"; do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found on PATH" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- create the environment if it does not exist ----------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> creating conda environment '$ENV_NAME' (python $PYTHON_VERSION)"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
echo "==> using $CONDA_PREFIX ($(python --version 2>&1))"

# --- install the package (editable) if the entry point is absent ------------
if ! command -v sot-chat >/dev/null 2>&1; then
  echo "==> installing dependencies and package (editable)"
  python -m pip install --upgrade pip
  python -m pip install -r "$REPO_DIR/requirements.txt"
  python -m pip install -e "$REPO_DIR"
fi

# --- launch -----------------------------------------------------------------
echo "==> launching sot-chat (context mode: ${SOT_CONTEXT_MODE:-native})"
exec sot-chat "$@"
