#!/usr/bin/env bash
# NixOS-native launcher for the sot-chat app.
# Uses uv (no conda) to provision Python 3.10 and run the editable package.
# The upstream conda launcher (run_sot_chat.sh) is left untouched.
# Run with --help for options and environment knobs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run_sot_chat_nix.sh [--host HOST] [--port PORT] [--context-mode MODE] [extra sot-chat args...]

Creates or reuses the uv virtualenv in .venv, then launches the SoT chat server.
This is the NixOS launcher; it does not require conda.

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
  SOT_PYTHON_VERSION          Python version when creating .venv (default 3.10)
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

The harness CLIs (opencode, codex, claude) must be on PATH, or be installed via
the NixOS configuration, for the "native" context mode to resume sessions.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
  esac
done

PYTHON_VERSION="${SOT_PYTHON_VERSION:-3.10}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

# --- locate uv --------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [[ -x "$candidate" ]]; then
      PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
      break
    fi
  done
fi
if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: uv not found on PATH
hint: install it on NixOS with one of
  nix profile install nixpkgs#uv
  nix shell nixpkgs#uv -c ./run_sot_chat_nix.sh
EOF
  exit 1
fi

# --- provision the venv -----------------------------------------------------
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "==> creating uv virtualenv in $VENV_DIR (python $PYTHON_VERSION)"
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

# --- install the package (editable) if the entry point is absent ------------
if [ ! -x "$VENV_DIR/bin/sot-chat" ]; then
  echo "==> installing dependencies and package (editable)"
  uv pip install --python "$VENV_DIR/bin/python" -r "$REPO_DIR/requirements.txt"
  uv pip install --python "$VENV_DIR/bin/python" -e "$REPO_DIR"
fi

# --- launch -----------------------------------------------------------------
echo "==> using $VENV_DIR ($("$VENV_DIR/bin/python" --version 2>&1))"
echo "==> launching sot-chat (context mode: ${SOT_CONTEXT_MODE:-native})"
export PATH="$VENV_DIR/bin:$PATH"
exec "$VENV_DIR/bin/sot-chat" "$@"
