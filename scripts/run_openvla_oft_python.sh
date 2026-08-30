#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
oft_python="${OFT_PYTHON:-$repo_root/.conda/envs/openvla-oft/bin/python}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SCRIPT [ARG ...]" >&2
  exit 2
fi
if [[ ! -x "$oft_python" ]]; then
  echo "OpenVLA-OFT Python is not executable: $oft_python" >&2
  echo "Override it with OFT_PYTHON=/absolute/path/to/python if needed." >&2
  exit 2
fi

"$oft_python" - <<'PY'
import sys
import imageio
import imageio.v2

major, minor = (int(value) for value in imageio.__version__.split(".")[:2])
if (major, minor) < (2, 10):
    raise RuntimeError(f"imageio>=2.10 is required; found {imageio.__version__}")
print(f"OpenVLA-OFT runtime: {sys.executable}; imageio={imageio.__version__}", flush=True)
PY

cd "$repo_root"
exec "$oft_python" "$@"
