# Inputs

## Shipped tables (small, read-only)

| Path | Content | Used by |
|---|---|---|
| `figures/` | per-figure plotting tables: `fig3_catchment_nse.csv`, `fig3_summary.json`, `fig4_scatter.csv`, `fig4_recall.csv`, `fig4_meta.json`, `fig5_recovery.csv`, `fig6_evidence.csv`, `fig7_regional.csv`, `fig8_scatter.csv`, `fig8_contrast.csv`, `fig8_meta.json`, `attr_corr.csv`, `catchments.csv`, `regions.gpkg`, `states.gpkg` | `plots/main/fig3…fig10` |
| `evidence/` | three-seed counterfactual `DeltaJ` per basin and process (`rf_counterfactual_training_s{42,43,44}.csv`), seven-region held-out runs (`rf_counterfactual_loro_r{0..6}_s{42,43,44}.csv`), derived evidence classes (`rf_evidence_classes_comparison.csv`, `rf_loro_classes_fullref_verified.csv`), three-seed mean gates (`v6_weights_seedwise.csv`) | `evaluate/regional*.py`, `evaluate/attribute_auc.py`, `evaluate/attribute_correlations.py` |
| `evaluation/` | fixed-off per-basin losses for three seeds (`fixed_off_per_basin_s{42,43,44}.csv`), leave-one-process-out losses (`rf_lopo_basinwise.csv`), evidence classes | `evaluate/removal.py` |
| `supplementary/` | attribute matrix input (`si_attr_corr.csv`) and refit tables (`si_refit_diff.csv`, `si_refit_exceed.csv`) | `plots/supplementary/` |
| `caravan_671_attributes.csv` | prepared Caravan attributes (671 rows) for `rho_w` | `evaluate/attribute_correlations.py` |
| `camels_attributes.npy` | the (671, 35) raw CAMELS attribute matrix used for `rho_q` (extracted from `camels_dataset`, same order as the config list) | `evaluate/attribute_correlations.py` |
| `teacher_s0/` | archived candidate E-S0 teacher: `config_snapshot.json`, `model/best_checkpoint.pt`, `model/normalization_statistics.json`; see `SOURCE.txt` | `train/synthetic/generate.py --teacher-aprime` |

## External runs (not shipped, GB-scale)

`inputs/trained/` must contain the archived run tree:

```
trained/
├── normal/full/ref/seed_{42,43,44}/      Full checkpoints, metrics, scored windows
├── normal/flex/ref/seed_{42,43,44}/      Flex checkpoints, metrics, scored windows
├── lopo/flex/remove_w_{snow,sub,phen,int}/seed_{42,43,44}/
├── loro/flex/region{0..6}/seed_{42,43,44}/
├── synthetic/synthetic_e1/scen{A,B,C,D}/seed_{42,43,44}/p_struct_best_train.npy
└── synthetic_eval/rfs0.synthetic_e1.E1.scen{A,B,C,D}.nmul4.s{42,43,44}/p_struct_best_holdout.npy
```

`inputs/synthetic/` must contain `scenario_{A,B,C,D,A_E2,A_low,A_high}/dataset.pkl`
and `evaluator_archive.npz` written by the generator.

Synthetic learner checkpoints are read from `inputs/synthetic_checkpoints/`
(override with `FLEXMOPEX_SYNTHETIC_CKPTS`); the layout is
`<family>/<scenario>/seed_<s>/best_checkpoint.pt` plus `summary.json`.

Any of these locations can be replaced by a symlink or a mount; set
`FLEXMOPEX_TEACHER_DIR`, `FLEXMOPEX_DATA_DIR` or `FLEXMOPEX_SYNTHETIC_CKPTS`
instead of editing code.
