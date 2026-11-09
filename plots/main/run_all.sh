#!/usr/bin/env bash
# Draw the main-text figures. Figures 5 and 10 are unapproved previews and are
# skipped unless PREVIEWS=1.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
data=${DATA:-$root/inputs/figures}
out=${OUT:?set OUT to a new output directory}
regions=${REGIONS:-$root/inputs/figures/regions.gpkg}
states=${STATES:-$root/inputs/figures/states.gpkg}
field=${FIELD:-loro_regio}
py=${PY:-$(command -v python || command -v python3)}
"$py" "$here/check_data.py" "$data"
"$py" "$here/fig3_study_area.py" --data "$data" --out "$out" --regions "$regions" --states "$states" --region-field "$field"
"$py" "$here/fig4_prediction.py" --data "$data" --out "$out"
"$py" "$here/fig6_transfer.py" --data "$data" --out "$out"
"$py" "$here/fig7_evidence.py" --data "$data" --out "$out" --regions "$regions" --states "$states" --region-field "$field"
"$py" "$here/fig8_attributes.py" --data "$data" --out "$out"
"$py" "$here/fig9_regional.py" --data "$data" --out "$out"
if [ "${PREVIEWS:-0}" = "1" ]; then
  "$py" "$here/fig5_detectability.py" --data "$data" --out "$out"
  "$py" "$here/fig10_removal.py" --data "$data" --out "$out"
fi
