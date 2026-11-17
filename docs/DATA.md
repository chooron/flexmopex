# Data archive

Root: `FLEXMOPEX_DATA_ROOT` (default `F:\ExpData\flexmopex-revision1`, WSL `/mnt/f/...`).
Everything the code reads or writes outside the repository lives here; `paths.py` is the only
place that resolves it.

```text
flexmopex-revision1/
├── RUN_INDEX.csv                 one row per training run (184)
├── MANIFEST.csv                  one row per file (29,013 files, 12.3 GB)
├── experiments/
│   ├── normal/{flex,full}/ref/seed_{42,43,44}
│   ├── loro/{flex,full}/region{0..6}/seed_{42,43,44}
│   ├── lopo/flex/remove_w_{snow,sub,phen,int}/seed_{42,43,44}
│   ├── ablation_nmul/{flex/nmul{1,2,8,16,32},full/nmul{1,2}}/seed_{42,43,44}
│   ├── ablation_lambda/flex/lam{0.0005 … 0.1}/seed_{42,43,44}
│   ├── synthetic/{e1,e2,e3,e4,e5,mdf,sdf}/<scenario>/seed_{42,43,44}
│   │                holdout_eval/  nse_tables/  dataset/
│   └── teacher_e2_s0/seed_42      archived E-S0 teacher (config + checkpoint)
├── analysis/
│   ├── tables/                   30 canonical CSVs (the manuscript/SI tables)
│   ├── derived/{evidence,evaluation,ablation_tables,seasonal_task3,run_summaries,counterfactual}
│   ├── static/                   attribute matrices (camels_attributes.npy, caravan_671_attributes.csv)
│   └── figures_inputs/           figure-ready CSV/JSON/gpkg exports
├── figures/{main,supplementary}/ rendered PNG (+PDF for the SI)
└── archive/                      frozen SI evidence zip and earlier statistics bundles
```

## Run-id scheme

`RUN_INDEX.csv` gives every run a self-describing id; `paths.py` exposes the matching accessors.

| Family | run_id pattern | accessor |
|---|---|---|
| normal | `normal_{model}_ref_s{seed}` | `paths.normal(model, seed)` |
| loro | `loro_{model}_region{r}_s{seed}` | `paths.loro(model, region, seed)` |
| lopo | `lopo_flex_remove_w_{process}_s{seed}` | `paths.lopo(process, seed)` |
| ablation n_mul | `ablation_nmul_{model}_nmul{m}_s{seed}` | `paths.ablation_nmul(m, seed, model)` |
| ablation lambda | `ablation_lambda_lam{value}_s{seed}` | `paths.ablation_lambda(lam, seed)` |
| synthetic | `synthetic_{family}_{scenario}_s{seed}` | `paths.synthetic_run(family, scenario, seed)` |
| teacher | `teacher_e2_s0` | `paths.TEACHER_E2` |

## What a run directory contains

`config_snapshot.json`, `source_config.yaml`, `manifest.json`, `provenance.json`,
`early_stopping.json` (selected epoch and reason), `train_loss_history.json`,
`structural_diagnostics.jsonl` (per-epoch gates, BCE and support fractions; Flex only),
`model/{best,final,last}_checkpoint.pt` plus per-epoch checkpoints,
`test1995-2010_Ep{sel}/` and `train1980-1995_Ep{sel}/` (metrics, streamflow, gates),
`diagnostics/` (per-epoch hard gates), `train_worker.log`.

## Derived tables

`analysis/derived/` holds the per-basin exports used by the analyses: three-seed counterfactual
`ΔJ` for the normal and LORO runs, fixed-off and LOPO losses, the ablation summaries, the Task-3
seasonal tables, and `run_summaries/` (per-run NSE/KGE and gates for 57 runs).

`analysis/tables/` is the canonical table set; each file's provenance and manuscript/SI mapping is
documented in `analysis/tables/README_manifest.md`. The figure scripts read those CSVs directly,
so a figure cannot disagree with its table.

## Rebuilding

```bash
python tools/build_data_manifest.py          # refresh RUN_INDEX.csv and MANIFEST.csv
python tools/build_data_manifest.py --sha    # add sha256 per file (slow, ~12 GB)
```

The archive was assembled from the training server's result tree (RF_S0 V2 runs) and the local
`results/ablation` re-run; the frozen evidence bundle in `archive/` remains the immutable source
record for the SI numbers.
