#!/usr/bin/env bash
# Draw the main-text figures from the archived figure inputs.
#   FLEXMOPEX_DATA_ROOT   data archive (default: F:/ExpData/flexmopex-revision1 or /mnt/f/...)
#   PY                    interpreter (default: the dmg-research venv)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
PY=${PY:-/home/jingxin/code/dmg-research/.venv/bin/python}
"$PY" "$here/render_main_figures.py" "$@"
