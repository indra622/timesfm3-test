#!/usr/bin/env bash
# Clone the pinned third-party sources this project builds on.
# They are not committed here (size and separate licenses); run once after cloning.
set -euo pipefail
cd "$(dirname "$0")/.."
fetch() {
  local dir=$1 url=$2 commit=$3
  if [ ! -d "$dir/.git" ]; then git clone --quiet "$url" "$dir"; fi
  git -C "$dir" fetch --quiet origin "$commit" || true
  git -C "$dir" checkout --quiet "$commit"
  echo "$dir @ $(git -C "$dir" rev-parse --short HEAD)"
}
fetch upstream-timesfm    https://github.com/google-research/timesfm.git   7360853c4f8ea28bb1b3eaf5b7af2d8e6b8fcf05
fetch upstream-fev        https://github.com/autogluon/fev.git             ae3e1a35762e0019f3a0a9094a0475cada76491a
fetch upstream-dcrnn      https://github.com/liyaguang/DCRNN.git           602afd9d767d3aa1c9b3eac51710d6aeee12c227
fetch upstream-staeformer https://github.com/XDZhelheim/STAEformer.git     fc49d39b2f1a8e3cf37b6289d7240680e1690f3f
fetch upstream-torch-mts  https://github.com/XDZhelheim/Torch-MTS.git      2db4de371584067160f9a37f1ae59495699b4a0a
