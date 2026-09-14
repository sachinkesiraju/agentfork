#!/bin/sh
# Install agentfork: a venv + the `agentfork` executable on PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/sachinkesiraju/agentfork/main/install.sh | sh
#
# Env:
#   AGENTFORK_REF     git ref to install (default: main)
#   AGENTFORK_VENV    where to create the venv (default: ~/.local/share/agentfork)
#   AGENTFORK_BIN     bin dir to symlink into (default: ~/.local/bin)
set -eu

REF="${AGENTFORK_REF:-main}"
VENV="${AGENTFORK_VENV:-$HOME/.local/share/agentfork}"
BIN="${AGENTFORK_BIN:-$HOME/.local/bin}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "error: $1 is required" >&2; exit 1; }; }
need python3
need git

python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "error: Python >= 3.10 is required" >&2; exit 1; }

echo "installing agentfork ($REF) into $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q \
  "git+https://github.com/sachinkesiraju/agentfork.git@$REF"

mkdir -p "$BIN"
ln -sf "$VENV/bin/agentfork" "$BIN/agentfork"

echo "installed: $BIN/agentfork"
case ":$PATH:" in
  *":$BIN:"*) : ;;
  *) echo "note: $BIN is not on PATH — add it, then run:" ;;
esac
echo "  agentfork up"
