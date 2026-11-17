#!/usr/bin/env python3
"""Write RUN_INDEX.csv and MANIFEST.csv for the external data archive.

RUN_INDEX  one row per training run: canonical run_id, experiment family, setting, seed,
           selected epoch, checkpoint presence, file count and size.
MANIFEST   one row per file with its size (and sha256 on request, --sha).

Usage: python tools/build_data_manifest.py [--sha]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import paths  # noqa: E402

FAMILIES = [("normal", "normal"), ("loro", "loro"), ("lopo", "lopo"),
            ("ablation_nmul", "ablation_nmul"), ("ablation_lambda", "ablation_lambda"),
            ("synthetic", "synthetic"), ("teacher_e2_s0", "teacher")]


def run_id(rel: Path) -> str:
    parts = rel.parts
    fam = parts[0]
    if fam == "normal":
        return f"normal_{parts[1]}_ref_s{parts[3].split('_')[1]}"
    if fam == "loro":
        return f"loro_{parts[1]}_{parts[2]}_s{parts[3].split('_')[1]}"
    if fam == "lopo":
        return f"lopo_flex_{parts[2]}_s{parts[3].split('_')[1]}"
    if fam == "ablation_nmul":
        return f"ablation_nmul_{parts[1]}_{parts[2]}_s{parts[3].split('_')[1]}"
    if fam == "ablation_lambda":
        return f"ablation_lambda_{parts[2]}_s{parts[3].split('_')[1]}"
    if fam == "synthetic":
        return "synthetic_" + "_".join(parts[1:]).replace("/", "_")
    return "_".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", action="store_true", help="also hash every file (slow)")
    a = ap.parse_args()

    rows, files = [], []
    for fam, _label in FAMILIES:
        root = paths.EXPERIMENTS / fam
        if not root.is_dir():
            continue
        for seed_dir in sorted(root.rglob("seed_*")):
            if not seed_dir.is_dir():
                continue
            rel = seed_dir.relative_to(paths.EXPERIMENTS)
            es = seed_dir / "early_stopping.json"
            sel = json.loads(es.read_text()).get("best_epoch") if es.is_file() else None
            size = sum(f.stat().st_size for f in seed_dir.rglob("*") if f.is_file())
            n = sum(1 for f in seed_dir.rglob("*") if f.is_file())
            rows.append(dict(run_id=run_id(rel), experiment=fam, seed=int(seed_dir.name.split("_")[1]),
                             selected_epoch=sel,
                             has_best_checkpoint=(seed_dir / "model/best_checkpoint.pt").is_file(),
                             n_files=n, bytes=size,
                             run_dir=str(rel)))
    import pandas as pd
    pd.DataFrame(rows).sort_values("run_id").to_csv(paths.DATA_ROOT / "RUN_INDEX.csv", index=False)

    for p in sorted(paths.DATA_ROOT.rglob("*")):
        if p.is_file() and p.name not in ("MANIFEST.csv",):
            rec = dict(path=str(p.relative_to(paths.DATA_ROOT)), bytes=p.stat().st_size)
            if a.sha:
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for b in iter(lambda: f.read(1 << 20), b""):
                        h.update(b)
                rec["sha256"] = h.hexdigest()
            files.append(rec)
    pd.DataFrame(files).to_csv(paths.DATA_ROOT / "MANIFEST.csv", index=False)
    print(f"RUN_INDEX.csv: {len(rows)} runs | MANIFEST.csv: {len(files)} files "
          f"({sum(f['bytes'] for f in files) / 1e9:.1f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
