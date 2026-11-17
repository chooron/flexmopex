#!/usr/bin/env bash
# Render the Supporting Information figures from the canonical CSV tables.
#   FLEXMOPEX_DATA_ROOT   data archive (default: F:/ExpData/flexmopex-revision1 or /mnt/f/...)
#   PY                    interpreter (default: the dmg-research venv)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/jingxin/code/dmg-research/.venv/bin/python}
OUT=${SI_FIG_OUT:-}
args=(); [ -n "$OUT" ] && args=(--out "$OUT")
for mod in fig_s1_synthetic_assignments fig_s2_attribute_matrix fig_s3_refit_consequences \
           fig_s4_flow_regime fig_s5_metric_sensitivity fig_s6_component_multiplicity \
           fig_s7_lambda_sensitivity; do
  echo "--- ${mod}"
  "$PY" "$here/${mod}.py" "${args[@]}"
done
