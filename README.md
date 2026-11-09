# Flex-MOPEX

Reproduction package for the Flex-MOPEX experiments: a differentiable daily
MOPEX model with learned structural gates, its real-basin and synthetic training
code, the analysis scripts behind the paper's results, and one script per figure.

## Layout

```text
train/                     training code
  run_model.py             real-basin entry (Basic / Full / Fixed / Flex, LORO)
  my_trainer.py            epoch loop, checkpointing, early stopping
  model_builder.py         physics and network registry
  local_model_handler.py   dPL wrapper (physics + parameter/gate networks)
  models/mopex.py          daily water-balance step and the Flex model class
  models/base_mopex.py     forcings, parameter descaling, state loop, UH routing
  models/fixed_weight_mopex.py   fixed-gate endpoints
  models/structure_net.py  attribute-only gate encoder
  models/cf_trainer.py     counterfactual supervision (BCE) and target refresh
  models/nse_dyn_aic_batch_loss.py   fit loss + complexity charge
  models/early_stopping.py, parameter_nets.py, pub_trainer.py, pub_sampler.py
  synthetic/generate.py    synthetic dataset generation (teacher forward + noise)
  synthetic/support.py     teacher loading and batched teacher simulation
  synthetic/truth.py       gate allocation, spatial split, loader dataset writer
  synthetic/teacher.py     E2 teacher physics (first submission and E-S0); teacher only
  synthetic/train.py       synthetic learner entry
  synthetic/strict/        train-only fold, sampler and strict trainer

configs/                   full.yaml, flex.yaml, regional.yaml, test_cpu.yaml
inputs/                    verified tables, E2 teacher, and documented external data
evaluate/                  analysis scripts, one per result block
  scorer.py                frozen per-basin scoring module used by rescore_synthetic.py
plots/                     figstyle / data_io / geo / map_frame + per-figure scripts
tests/                     simulator, training-graph and checkpoint tests
```

## Environment

`pyproject.toml` declares the package and its dependencies (PyTorch CPU/CUDA
index included); nothing is installed by this repository.

Python 3.10 with PyTorch 2.9, NumPy, pandas, SciPy, scikit-learn, Hydra,
OmegaConf, PyYAML, tqdm, `dmg` 1.4.1 and `hydrodl2` 1.3.4; plotting adds
Matplotlib, GeoPandas, Shapely and PyProj (`requirements.txt`). No code imports
the original research tree.

## Data

The package ships the small verified tables it needs (`inputs/figures`,
`inputs/evidence`, `inputs/evaluation`, `inputs/supplementary`,
`inputs/caravan_671_attributes.csv`, the E2 teacher in `inputs/teacher_s0`) and
the small arrays `gage_id`/`camels_dates`/`basin_groups` under `data/`.

Two things are not shipped because they are large:

- `data/camels_dataset` — the prepared CAMELS-US bundle, a pickle of
  `(forcings (671, 12418, 3), streamflow (671, 12418, 1), attributes (671, 35))`
  with forcing order `prcp, tmean, pet`, plus `gage_id.npy` and
  `camels_dates.npy`.
- the archived runs under `inputs/trained/` and the scenario datasets under
  `inputs/synthetic/` (checkpoints, scored windows, `dataset.pkl`,
  `evaluator_archive.npz`).

`data/README.md` and `inputs/README.md` give the exact schema, shapes and layout
each of them must have, and which script reads it. Point the code at another
location with `FLEXMOPEX_DATA_DIR`, `FLEXMOPEX_TEACHER_DIR` or
`FLEXMOPEX_SYNTHETIC_CKPTS` instead of editing paths.

## Model

One daily step per catchment component; storages in mm, fluxes in mm/day,
temperature in °C.

1. **Phenology.** `GSI = clip((T - tmin)/(tmax - tmin), 0, 1)`,
   `PET_eff = PET·(1 - w_phen + w_phen·GSI)`.
2. **Partition.** `f_r = sigmoid(T - tcrit)`, `P_r = P·f_r`, `P_s = P - P_r`.
3. **Interception (liquid only).**
   `season = ½(1 + κ·cos(2π(doy - φ)/365.25))`,
   `E_i = w_int·min(P_r·season, P_r, PET_eff)`, `P_r,net = P_r - E_i`,
   `PET_soil = max(PET_eff - E_i, 0)`.
4. **Snow.** `P_snow = P_s·w_snow`,
   `Q_N = min(sigmoid(T-tcrit)·softplus(T-tcrit)·ddf, S_n)`,
   `S_n ← S_n + P_snow - Q_N`, `P_eff = P_r,net + P_s(1-w_snow) + Q_N`.
5. **Upper store.** overflow `q1f = max(S1 + P_eff - Sb1, 0)`, then
   `ET1 = min(PET_soil·min(S1/Sb1, 1), S1)`, then
   `q_w = S1(1 - e^{-1/tw})`.
6. **Lower store.** `q2f = w_sub·max(S2 - Sb2, 0)`, `q2u = S2(1 - e^{-1/tu})`,
   `ET2 = min((PET_soil - ET1)⁺·min(S2/Se, 1), S2)`.
7. **Routing.** two stores with `r_c = 1 - e^{-1/tc}`; `Q = q_f + q_s`.
8. **Outlet.** four components are averaged and convolved with a normalized
   15-ordinate gamma unit hydrograph.

Twelve parameters describe each component (`Sb1, tw, tu, Se, tc, ddf, tcrit,
Sb2, κ, φ, tmin, tmax`, the last two shared conceptually through the phenology
term) plus two routing controls, giving 50,754 trainable values at four
components. Bounds: `Sb1 0.01-50`, `tw 0.01-5`, `tu 1-2000`, `Se 1-1000`,
`tc 0.1-30`, `ddf 0-20`, `tcrit -3-3`, `Sb2 1-1500`, `κ 0-1`, `φ 0-365`,
`tmin -10-5`, `tmax 5-30`, `rout_a 0-2.9`, `rout_b 0-6.5`.

Structural gates: a 35 → 128 → 128 → 8 network maps standardized catchment
attributes to one OFF/ON logit pair per process. Training samples gates with a
Gumbel-softmax switch; evaluation uses the deterministic softmax probability, so
reported gates are posterior probabilities, not sampled decisions. A gate of 0
removes the optional part of its process only: with `w_snow = 0` melt of an
existing snowpack still occurs, and with `w_sub = 0` slow drainage and lower-store
evaporation remain.

## Training

**Real basins.** 671 CAMELS-US catchments; training window 1980-10-01 to
1995-09-30, test window 1995-10-01 to 2010-09-30, 365-day warm-up, batch size
100, Adadelta at learning rate 1.0, at most 100 epochs with a minimum of 50, and
early stopping on the training loss (patience 20, `min_delta` 1e-4). The selected
checkpoint is the epoch-mean minimum, written together with `best`, `last` and
`final` roles plus an `early_stopping.json` record. Three seeds (42/43/44) are
used for every arm.

**Objectives.** The fit term is the per-catchment variance-normalized squared
error of discharge. The complexity charge is
`λ · Σ_p k_p · mean_{b,t} w_{b,p}` with `λ = 0.007` and
`k = (2, 2, 2, 1)` for phenology, interception, snow and subsurface. Flex adds
counterfactual supervision: at the start of every epoch each process is evaluated
ON and OFF, `ΔJ = J_OFF - J_ON` (with the complexity term subtracted),
`τ_p = median|ΔJ|`, `q = sigmoid(ΔJ/τ_p)`, `c = 2|q - 0.5|`, and a
confidence-weighted BCE trains the gate logits. The two gradient paths are
isolated: fit and complexity terms cannot reach the gate network, and the BCE
term cannot reach the parameter backbone.

**Endpoints.** Full fixes all gates at 1; Basic at 0; Fixed at user-supplied
values. Flex learns them. Regional runs hold out one of seven regions and refit.
The release refuses to train into a non-empty directory or inside `inputs/`.

**Synthetic learners.** Scenario datasets contain 536 training and 135 holdout
basins with teacher-generated noisy discharge. `train/synthetic/train.py` accepts
the five method variants used in the paper (`Canonical_BCE`, `Plain_BCE`,
`Direct_Fit`, `Matched_Direct_Fit`, `Strict_Direct_Fit`) with the same budget as
the real-basin runs; folds are built from training basins only.

## Analysis

| Result | Script | Method |
|---|---|---|
| 3.1 | `evaluate/skill.py` | Per-seed cross-basin medians and four alternative aggregations; paired Flex−Full difference per seed, basin mean over seeds, median with 2,000 basin resamples (seed 20261106). |
| 3.2 | `evaluate/detectability.py` | True-ON basins split into high (`ΔJ > P_p`), weak (`0 ≤ ΔJ ≤ P_p`) and negative (`ΔJ < 0`) strata; recall of `w > 0.5` per stratum. |
| 3.3 | `evaluate/detectability.py` | Scenario contrast (rule minus random scenarios) for balanced accuracy and AUC, per seed, averaged over seeds, with 2,000 basin resamples over the 135 holdout basins. |
| 3.4 | `evaluate/evidence.py`, `regional.py`, `regional_intervals.py` | Three-seed evidence classes (`decisive_open` / `decisive_closed` / `uncertain`) from the sign of all seeds and `|median ΔJ| > SD`; LORO classes scored against the Full-data class as reference with balanced accuracy; region-stratified basin bootstrap (1,000 draws, seed 20260516). |
| 3.4 | `evaluate/attribute_auc.py` | Held-out gate as a continuous predictor of the Full-data decisive class, AUC with 1,000 stratified resamples (seed 20260521); single-class regions stay NaN. |
| 3.5 | `evaluate/fixed_off.py` | Full checkpoints replayed with one gate set to 0 and the other three at 1, parameters and routing frozen; per-basin NSE/KGE losses. |
| 3.5 | `evaluate/removal.py` | Fixed-off versus independently refitted LOPO: three-seed basin means, group medians, ON−OFF difference of medians, and the paired Fixed−LOPO difference, each with 2,000 basin resamples (seed 20261106). |

None of these scripts forward the model except `fixed_off.py`; the rest read
frozen artefacts. Every script writes to a new path and refuses to overwrite.

## Figures

`plots/main/fig3_study_area.py` … `fig10_removal.py` cover the current candidate
numbering (study area = Figure 3, process removal = Figure 10); the study-area
number is not yet confirmed against the manuscript;
`plots/supplementary/fig_attributes.py` and `fig_refit.py` to the two
supplementary candidates. Each script is standalone:

```sh
python plots/main/fig4_prediction.py --data inputs/figures --out /path/to/figures
```

`run_all.sh` in each folder draws everything. Figures 5 and 10 remain unapproved
previews and are skipped unless `PREVIEWS=1`; Figure 7 reproduces the verified
classes but not the candidate image pixel for pixel.

## Usage

```sh
# real basins
python train/run_model.py --config configs/flex.yaml --mode train_test \
    --model-type flex --seed 42 --output-root /path/to/results --run-name flex_seed42

# synthetic dataset (teacher forward; scenarios A-D, add --scenario-tag _low/_high for the noise variants)
python train/synthetic/generate.py --out /path/to/synthetic-data --scenarios A,B,C,D

# E2 variant: first-submission S0 teacher from inputs/teacher_s0
FLEXMOPEX_TEACHER_DIR=inputs/teacher_s0 python train/synthetic/generate.py \
    --out /path/to/synthetic-data --scenarios A --scenario-tag _E2 \
    --teacher-aprime --run-label E2_teacher_aprime

# synthetic learner
python train/synthetic/train.py --run_id E1_A_42 --scenario A \
    --method Canonical_BCE --seed 42 --output_dir /path/to/synthetic-run

# scenario-correct NSE/KGE re-scoring of saved learner checkpoints (CPU)
FLEXMOPEX_SYNTHETIC_CKPTS=/path/to/synthetic/checkpoints \
    python evaluate/rescore_synthetic.py --out /path/to/rescore --scenario A

# analysis
python evaluate/skill.py --out /path/to/3_1.csv
python evaluate/detectability.py --out /path/to/tables
python evaluate/attribute_auc.py --out /path/to/attributes
python evaluate/removal.py --output /path/to/removal
python evaluate/regional.py --output /path/to/regional
python evaluate/regional_intervals.py --output /path/to/intervals.csv
python evaluate/fixed_off.py --seed 42 --device cuda:0 \
    --output /path/to/fixed_off.csv --diag-output /path/to/replay.csv

# figures
OUT=/path/to/figures plots/main/run_all.sh
OUT=/path/to/supplementary plots/supplementary/run_all.sh
```

Configurations declare `nmul: 16`; every formal run passes `--nmul 4`
(50,754 parameters), which is also the default here.

## Verification

Verified on this machine, one job at a time:

- Training entries: Full, Flex, Basic, Fixed, regional hold-out and all five
  synthetic methods reach forward, loss and backward on real data; a 4-basin
  1-epoch Flex run completes with checkpoints and early-stopping metadata, and
  the production batch size of 100 basins passes preflight on GPU.
- Simulator tests 11/11, training-graph checks 7/7, checkpoint-selection checks
  12/12.
- Results 3.1, 3.2, 3.3 and the 3.4 attribute AUC regenerate the archived tables
  with identical values; the 3.4 evidence and regional tables and the 3.5
  fixed-off/LOPO tables regenerate byte for byte, including 2,684 rows per seed
  for three seeds.
- Attribute associations regenerate exactly: `rho_w` and `rho_q` differ from the
  verified table by at most 1e-16.
- The dataset generator rebuilds Scenario A deterministically (two independent runs
  are byte-identical). Against the archived dataset the truth gates, basin axis,
  dates, forcings, attributes and noise multipliers match exactly, while teacher
  discharge differs by at most 9.1e-4 mm/day from the archived file.
- Figures 4, 6, 8, 9 and both supplementary candidates regenerate byte for byte.

## Known limitations

- The dataset generator reproduces A-D, low/high noise and E2, and the
  scenario-specific rescoring reproduces the archived per-run NSE. Regenerated
  truth gates, basin axis, dates, forcings, attributes and noise match the
  archived datasets exactly; teacher discharge differs by <=1e-3 mm/day from the
  archived files, so regenerated datasets are equivalent but not byte-identical.
- `rho_w` uses the prepared Caravan attribute table and `rho_q` the raw CAMELS
  table, so the two correlations have different masks and sample sizes.
- Figure 7 uses the verified class table but its raster differs from the
  candidate image; the cause is unresolved. Figures 5 and 10 await approval.
- Full 100-epoch training runs were not executed here.
- Replaying a frozen Full checkpoint reproduces the archived per-basin metrics
  exactly, but its daily hydrograph differs from the stored prediction arrays by
  up to ~0.0027 mm/day; the cause is unresolved and is reported with the Figure 10
  and fixed-off results.
