#!/usr/bin/env bash
# Draw the supplementary figures.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
data=${DATA:-$root/inputs/supplementary}
out=${OUT:?set OUT to a new output directory}
py=${PY:-$(command -v python || command -v python3)}
"$py" "$here/fig_attributes.py" --data "$data" --out "$out" --name attributes
"$py" "$here/fig_refit.py" --data "$data" --out "$out" --name refit
