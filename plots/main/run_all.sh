#!/usr/bin/env bash
# Draw Figures 3-10. One module per figure; shared style/geometry helpers live in _common.py.
#   FLEXMOPEX_DATA_ROOT   data archive (default: F:/ExpData/flexmopex-revision1 or /mnt/f/...)
#   PY                    interpreter (default: the dmg-research venv)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/jingxin/code/dmg-research/.venv/bin/python}
OUT=${FIG_OUT:-}
args=(); [ -n "$OUT" ] && args=(--out "$OUT")
for mod in fig03_study_area fig04_prediction fig05_detectability_recovery fig06_transfer \
           fig07_evidence_maps fig08_attribute_relationships fig09_regional_exclusion \
           fig10_process_removal; do
  echo "--- ${mod}"
  "$PY" "$here/${mod}.py" "${args[@]}"
done
