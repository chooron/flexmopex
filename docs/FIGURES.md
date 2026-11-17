# Figures

All figures are 7.0 in wide at 600 dpi. Main-text figures are PNG; the Supporting Information set
ships PNG **and** vector PDF. No canvas carries a "Figure S…" string — numbering lives in the
captions.

```bash
bash plots/main/run_all.sh            # Figures 3-10  -> $FLEXMOPEX_DATA_ROOT/figures/main
bash plots/supplementary/run_all.sh   # Figures S1-S7 -> $FLEXMOPEX_DATA_ROOT/figures/supplementary
```

## Main text (`plots/main/`)

| Figure | Script (one module per figure) | Data |
|---|---|---|
| 3 study area | `plots/main/fig03_study_area.py` | `analysis/figures_inputs/{catchments.csv,regions.gpkg,states.gpkg}` |
| 4 prediction | `plots/main/fig04_prediction.py` | `fig3_catchment_nse.csv`, `fig3_summary.json` |
| 5 detectability & recovery | `plots/main/fig05_detectability_recovery.py` | `fig4_scatter.csv`, `fig4_recall.csv`, `fig4_meta.json` |
| 6 scenario recovery & transfer | `plots/main/fig06_transfer.py` | `fig5_recovery.csv` |
| 7 evidence maps | `plots/main/fig07_evidence_maps.py` | `fig6_evidence.csv` + GIS layers |
| 8 attribute associations | `plots/main/fig08_attribute_relationships.py` | `attr_corr.csv` (= `analysis/tables/Figure8_and_S2_*.csv`) |
| 9 regional exclusion | `plots/main/fig09_regional_exclusion.py` | `fig7_regional.csv` |
| 10 process removal | `plots/main/fig10_process_removal.py` | `fig8_scatter.csv`, `fig8_contrast.csv`, `fig8_meta.json` |

Shared style and geometry helpers: `plots/figstyle.py` and `plots/main/_common.py`. Panel geometry is set at print size, so the font sizes in the
style sheet are the sizes that appear on the page.

## Supporting Information (`plots/supplementary/`)

| Figure | Size (in) | Script | Table it reads |
|---|---|---|---|
| S1 synthetic assignments | 7.00 × 4.80 | `fig_s1_synthetic_assignments.py` | `tables/FigureS1_synthetic_assignment_points.csv` |
| S2 attribute matrix | 7.00 × 6.07 | `fig_s2_attribute_matrix.py` | `tables/Figure8_and_S2_attribute_spearman_correlations.csv` |
| S3 refitting consequences | 7.00 × 2.70 | `fig_s3_refit_consequences.py` | `tables/FigureS3_perbasin_refit_diff.csv`, `FigureS3_refit_exceedance_by_threshold.csv` |
| S4 flow-regime contributions | 7.00 × 4.10 | `fig_s4_flow_regime.py` | `tables/TableS8_flow_regime_contributions.csv` |
| S5 metric sensitivity | 7.00 × 4.20 | `fig_s5_metric_sensitivity.py` | `tables/TableS9a_metric_sensitivity_all671.csv`, `TableS9b_metric_rank_reversals.csv` |
| S6 component multiplicity | 7.00 × 2.80 | `fig_s6_component_multiplicity.py` | `tables/TableS11a_nmul_checkpoints_and_skill.csv`, `TableS11b_nmul_gates_and_evidence.csv` |
| S7 lambda sensitivity | 7.00 × 2.90 | `fig_s7_lambda_sensitivity.py` | `tables/TableS12a_lambda_checkpoints_and_skill.csv`, `TableS12b_lambda_sparsity_snowfree_evidence.csv` |

Shared style: `plots/wrrstyle.py` (canonical process palette `#1f5a99` / `#18987b` / `#993366` /
`#d4923a`, 9.5 pt bold left-aligned panel titles, 8.75 pt axis labels, 7.75 pt ticks).

## Single source of truth

Every SI figure reads its summary statistics straight from `analysis/tables/*.csv`; there is no
intermediate copy. `analysis/tables/README_manifest.md` documents the provenance of each table and
its mapping to a manuscript/SI table number.

Two known, documented deviations:

* `FigureS2` is frozen at the version approved for submission. The module regenerates the same
  design and the same data (the canonical CSV equals the frozen plotted table at 1e-16) but differs
  from the archived PNG at the few-percent level inside the heat-map cells, so the archived PNG
  remains the shipping artifact.
* `FigureS6`/`FigureS7` are also written as `fig_task1_nmul.*` / `fig_task2_lambda.*`; the two name
  pairs were byte-identical in the delivered package, and the repository keeps the `FigureS6/S7`
  names only.
