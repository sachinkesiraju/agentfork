#!/usr/bin/env bash
# Prepare a patched SGLang for agentfork's tree-cache branching.
#
# agentfork's SGLang backends (SGLangKVBackend, SGLangHTTPBackend) talk to a
# SGLang inference server that has agentfork's patches applied. Vanilla SGLang
# has no branch/tree forking; the patches add the /tree_cache and
# /tree_generate endpoints those backends use.
#
# This script does the agentfork-specific part: clone SGLang at the commit the
# patches target and apply them. Installing and launching the server on a GPU
# host is SGLang's own job (its instructions apply); the commands to do that
# are printed at the end.
#
# Usage: tools/setup_sglang.sh [dest_dir]   (default: ./sglang)
set -euo pipefail

# the patches are pinned to this SGLang commit; they will not apply to latest
SGLANG_COMMIT=40517b593b23870cf351a05a1d53e930cea6a58d
DEST="${1:-sglang}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [ -e "$DEST" ]; then
  echo "error: '$DEST' already exists; pass another path or remove it" >&2
  exit 1
fi
command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }

echo "cloning SGLang into ./$DEST and checking out the pinned commit..."
git clone --quiet https://github.com/sgl-project/sglang "$DEST"
git -C "$DEST" checkout --quiet "$SGLANG_COMMIT"

for patch in "$REPO"/patches/0001-*.patch \
             "$REPO"/patches/0002-*.patch \
             "$REPO"/patches/0003-*.patch; do
  echo "applying $(basename "$patch")"
  git -C "$DEST" apply "$patch"
done

cat <<EOF

patched SGLang is ready in ./$DEST

Next, on a GPU host (see SGLang's own docs for install details and flags):

  pip install -e "$DEST/python[all]"
  python -m sglang.launch_server --model-path <hf-model> \\
      --admin-api-key <secret> --port 30000

Then point agentfork at it:

  from agentfork import ForkOrchestrator, SGLangHTTPBackend
  kv = SGLangHTTPBackend("http://<host>:30000", admin_api_key="<secret>")
EOF
