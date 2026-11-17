# Flex-MOPEX

**Reproducible experiments for a differentiable daily MOPEX model with learned structural gates.**

This repository contains real-basin and synthetic training workflows, the evaluation scripts used for the paper's results, and standalone figure-generation scripts. It also documents the required external datasets, archived outputs, verification status, and known reproduction limits.

| Component | Scope |
| :-- | :-- |
| Hydrological model | Daily, four-component MOPEX with optional processes controlled by structural gates |
| Real-basin experiments | 671 CAMELS-US catchments; Basic, Full, Fixed, Flex, and regional hold-out runs |
| Synthetic experiments | 536 training and 135 holdout basins; five learner variants |
| Reproducibility | Result-specific analyses, figure scripts, tests, and archived-output comparisons |

**Contents:** [Repository](#repository-structure) · [Environment and data](#environment-and-data) · [Model](#model-formulation) · [Training](#training-methodology) · [Reproduction commands](#reproduction-commands) · [Analysis and figures](#analysis-and-figures) · [Verification](#verification) · [Known limitations](#known-limitations)

## Repository structure

The layout separates model implementation, experiment entry points, evaluation, and plotting. Paths below are relative to the repository root.

```text
./
├── train/                             Training and simulation
│   ├── run_model.py                   Real-basin entry: Basic / Full / Fixed / Flex / LORO
│   ├── my_trainer.py                  Epoch loop, checkpoints, early stopping
│   ├── model_builder.py               Physics and network registry
│   ├── local_model_handler.py         dPL wrapper for physics and learned networks
│   ├── models/
│   │   ├── mopex.py                   Daily water balance and Flex model
│   │   ├── base_mopex.py              Forcings, parameter scaling, state loop, UH routing
│   │   ├── fixed_weight_mopex.py      Fixed-gate model endpoints
│   │   ├── structure_net.py           Attribute-only structural gate encoder
│   │   ├── cf_trainer.py              Counterfactual BCE and target refresh
│   │   ├── nse_dyn_aic_batch_loss.py  Fit loss and complexity charge
│   │   ├── early_stopping.py
│   │   ├── parameter_nets.py
│   │   ├── pub_trainer.py
│   │   └── pub_sampler.py
│   └── synthetic/
│       ├── generate.py                Teacher simulation and noise generation
│       ├── support.py                 Teacher loading and batched simulation
│       ├── truth.py                   Gate allocation, spatial split, dataset writer
│       ├── teacher.py                 E2 / E-S0 teacher physics (teacher only)
│       ├── train.py                   Synthetic learner entry
│       └── strict/                    Train-only folds, sampling, strict training
├── configs/                           full.yaml, flex.yaml, regional.yaml, test_cpu.yaml
├── data/                              Small metadata arrays (gage ids, basin groups, dates)
├── paths.py                           Resolves the external data archive (FLEXMOPEX_DATA_ROOT)
├── docs/                              DATA.md, TRAINING.md, FIGURES.md
├── tools/                             build_data_manifest.py (RUN_INDEX + MANIFEST)
├── evaluate/                          Result-specific analysis modules
│   └── scorer.py                      Frozen per-basin scorer for rescore_synthetic.py
├── plots/                             Plotting helpers and figure-specific scripts
│   ├── main/
│   └── supplementary/
├── tests/                             Simulator, training graph, checkpoint selection
├── pyproject.toml                     Package metadata and dependency declarations
└── requirements.txt                   Additional dependency specification
```

The `plots/` package carries the shared styles (`figstyle.py`, `wrrstyle.py`), `geo.py`, and one
module per figure. `paths.py` is the single place that resolves the external data archive:
`FLEXMOPEX_DATA_ROOT`, default `F:\ExpData\flexmopex-revision1` (WSL `/mnt/f/...`); no other file
hard-codes a data path. Archive layout, run-id scheme and manifest tooling are documented in
[`docs/DATA.md`](docs/DATA.md); training results in [`docs/TRAINING.md`](docs/TRAINING.md) and the
figure inventory in [`docs/FIGURES.md`](docs/FIGURES.md).

## Environment and data

### Environment

The documented environment uses **Python 3.10** and **PyTorch 2.9**, together with:

- **Scientific and training:** NumPy, pandas, SciPy, scikit-learn, Hydra, OmegaConf, PyYAML, tqdm, `dmg` 1.4.1, and `hydrodl2` 1.3.4.
- **Plotting:** Matplotlib, GeoPandas, Shapely, and PyProj.

Dependency declarations, including the PyTorch CPU/CUDA index, are provided in `pyproject.toml` and `requirements.txt`. The repository itself does not install packages, and no code imports the original research tree.

### Included and external data

**Included in the repository**

- Code only: `train/`, `evaluate/`, `plots/`, `tools/`, `configs/`, `tests/`, and the small arrays
  under `data/` (`gage_id`, `camels_dates`, `basin_groups`).

**Held in the external data archive** (`FLEXMOPEX_DATA_ROOT`)

| Archive path | Contents |
| :-- | :-- |
| `experiments/` | 184 training runs in six families — `normal`, `loro`, `lopo`, `ablation_nmul`, `ablation_lambda`, `synthetic` — plus the archived E2 teacher |
| `analysis/tables/` | 30 canonical CSV tables; every manuscript/SI number and every SI figure reads these |
| `analysis/derived/` | per-basin derived tables: counterfactual `ΔJ`, fixed-off and LOPO losses, Task-3 seasonal tables, per-run summaries |
| `analysis/static/` | attribute matrices (`camels_attributes.npy`, `caravan_671_attributes.csv`) |
| `analysis/figures_inputs/` | figure-ready CSV/JSON/GPKG exports |
| `figures/{main,supplementary}/` | rendered figures (600 dpi PNG, plus vector PDF for the SI) |
| `archive/` | frozen SI evidence bundle |

The archive also ships `RUN_INDEX.csv` (one row per run: id, family, seed, selected epoch,
checkpoint presence) and `MANIFEST.csv` (one row per file); both are regenerated with
`python tools/build_data_manifest.py`.

The prepared CAMELS-US bundle contains:

```text
forcings    (671, 12418, 3)   order: prcp, tmean, pet
streamflow  (671, 12418, 1)
attributes  (671,    35)
```

Exact schemas, shapes and archive layouts are documented in [`docs/DATA.md`](docs/DATA.md). Paths are configured without editing source code:

| Environment variable | External resource |
| :-- | :-- |
| `FLEXMOPEX_DATA_ROOT` | root of the data archive (runs, tables, figures) |
| `FLEXMOPEX_TEACHER_DIR` | teacher checkpoint override (defaults into the archive) |
| `FLEXMOPEX_SYNTHETIC_CKPTS` | synthetic learner checkpoints override |

## Model formulation

### Daily process overview

The simulator advances each of four catchment components daily. **Storage** is measured in mm, **fluxes** in mm/day, and **temperature** in °C.

```text
Temperature + precipitation + PET
                  │
                  ├── Phenology ─────────────── effective PET
                  │
                  └── Rain / snow partition
                           │
                  ┌────────┴────────┐
                  │                 │
             Interception      Snow storage / melt
                  │                 │
                  └────────┬────────┘
                           │
                   Effective input
                           │
                     Upper store
                           │
                     Lower store
                           │
                    Two-store routing
                           │
           Four-component mean + gamma UH
                           │
                    Outlet discharge
```

The schematic is an overview, not a replacement for the equations or the implementation's state-update order.

### 1. Phenology and effective evapotranspiration

The phenology gate changes the fraction of potential evapotranspiration available to the model:

$$
\begin{aligned}
\mathrm{GSI}
  &= \mathrm{clip}\!\left(\frac{T-t_{\min}}{t_{\max}-t_{\min}},\ 0,\ 1\right),\\[4pt]
\mathrm{PET}_{\mathrm{eff}}
  &= \mathrm{PET}\left(1-w_{\mathrm{phen}}+w_{\mathrm{phen}}\,\mathrm{GSI}\right).
\end{aligned}
$$

Here, $w_{\mathrm{phen}}$ controls the optional phenology adjustment.

### 2. Rain–snow partition

Precipitation is separated using a temperature-dependent rainfall fraction:

$$
\begin{aligned}
f_r &= \sigma(T-t_{\mathrm{crit}}),\\
P_r &= P\,f_r,\\
P_s &= P-P_r.
\end{aligned}
$$

$P_r$ and $P_s$ denote liquid and solid precipitation, respectively; $\sigma$ is the sigmoid function.

### 3. Interception

Interception acts on **liquid precipitation only**. The seasonal factor and interception flux are:

$$
\begin{aligned}
s_{\mathrm{season}}
  &= \frac{1}{2}\left[1+\kappa\cos\!\left(\frac{2\pi(\mathrm{doy}-\phi)}{365.25}\right)\right],\\[4pt]
E_i
  &= w_{\mathrm{int}}\min\!\left(P_r s_{\mathrm{season}},\ P_r,\ \mathrm{PET}_{\mathrm{eff}}\right),\\
P_{r,\mathrm{net}} &= P_r-E_i,\\
\mathrm{PET}_{\mathrm{soil}}
  &= \max\!\left(\mathrm{PET}_{\mathrm{eff}}-E_i,\ 0\right).
\end{aligned}
$$

The interception gate $w_{\mathrm{int}}$ scales the optional liquid-water loss.

### 4. Snow accumulation and melt

$$
\begin{aligned}
P_{\mathrm{snow}} &= w_{\mathrm{snow}}\,P_s,\\
Q_N &= \min\!\left[
  \sigma(T-t_{\mathrm{crit}})\,
  \mathrm{softplus}(T-t_{\mathrm{crit}})\,\mathrm{ddf},\ S_n
\right],\\[3pt]
S_n &\leftarrow S_n+P_{\mathrm{snow}}-Q_N,\\
P_{\mathrm{eff}} &= P_{r,\mathrm{net}}+P_s(1-w_{\mathrm{snow}})+Q_N.
\end{aligned}
$$

Setting $w_{\mathrm{snow}}=0$ stops **new snow accumulation through this gate**, but an existing snowpack can still melt.

### 5. Upper soil store

The implementation computes overflow, evapotranspiration, and a release term in that order:

$$
\begin{aligned}
q_{1f} &= \max(S_1+P_{\mathrm{eff}}-S_{b1},\ 0),\\
\mathrm{ET}_1 &= \min\!\left(\mathrm{PET}_{\mathrm{soil}}\min\!\left(\frac{S_1}{S_{b1}},1\right),\ S_1\right),\\
q_w &= S_1\left(1-e^{-1/t_w}\right).
\end{aligned}
$$

### 6. Lower soil store

$$
\begin{aligned}
q_{2f} &= w_{\mathrm{sub}}\max(S_2-S_{b2},\ 0),\\
q_{2u} &= S_2\left(1-e^{-1/t_u}\right),\\
\mathrm{ET}_2 &= \min\!\left[
  (\mathrm{PET}_{\mathrm{soil}}-\mathrm{ET}_1)^{+}
  \min\!\left(\frac{S_2}{S_e},1\right),\ S_2
\right].
\end{aligned}
$$

The subsurface gate $w_{\mathrm{sub}}$ controls the optional $q_{2f}$ term. At $w_{\mathrm{sub}}=0$, slow drainage $q_{2u}$ and lower-store evaporation $\mathrm{ET}_2$ remain active.

### 7. Routing and outlet

The routing system uses two stores, with a release coefficient and total discharge defined as:

$$
\begin{aligned}
r_c &= 1-e^{-1/t_c},\\
Q &= q_f+q_s.
\end{aligned}
$$

The four component outputs are averaged and convolved with a **normalized, 15-ordinate gamma unit hydrograph** at the outlet.

### Parameters and bounds

Each component is described by 12 parameters: $S_{b1}$, $t_w$, $t_u$, $S_e$, $t_c$, $\mathrm{ddf}$, $t_{\mathrm{crit}}$, $S_{b2}$, $\kappa$, $\phi$, $t_{\min}$, and $t_{\max}$. The last two are conceptually shared through the phenology term. Two further controls specify routing. The documented four-component model has **50,754 trainable values**.

| Parameter | Bound | Parameter | Bound |
| :-- | :-- | :-- | :-- |
| $S_{b1}$ | 0.01–50 | $S_{b2}$ | 1–1500 |
| $t_w$ | 0.01–5 | $t_u$ | 1–2000 |
| $S_e$ | 1–1000 | $t_c$ | 0.1–30 |
| $\mathrm{ddf}$ | 0–20 | $t_{\mathrm{crit}}$ | −3–3 |
| $\kappa$ | 0–1 | $\phi$ | 0–365 |
| $t_{\min}$ | −10–5 | $t_{\max}$ | 5–30 |
| `rout_a` | 0–2.9 | `rout_b` | 0–6.5 |

### Structural gates

The gate network uses **35 standardized catchment attributes** to produce one OFF/ON logit pair for each of four optional processes:

$$
\underbrace{35}_{\text{attributes}}
\;\longrightarrow\;128\;\longrightarrow\;128
\;\longrightarrow\;
\underbrace{8}_{4\times(\text{OFF, ON})\text{ logits}}.
$$

| Phase | Gate interpretation |
| :-- | :-- |
| Training | Gumbel-softmax samples a structural switch. |
| Evaluation | Deterministic softmax gives the reported ON probability. |

Reported gates are **posterior probabilities, not sampled binary decisions**. Setting a gate to zero removes only the corresponding optional process contribution, as illustrated by the snow and subsurface equations above.

## Training methodology

### Real-basin configuration

| Setting | Value |
| :-- | :-- |
| Catchments | 671 CAMELS-US basins |
| Training | 1980-10-01 to 1995-09-30 |
| Testing | 1995-10-01 to 2010-09-30 |
| Warm-up | 365 days |
| Optimizer | Adadelta, learning rate 1.0 |
| Batch size | 100 basins |
| Epoch budget | Minimum 50, maximum 100 |
| Early stopping | Training loss; patience 20; `min_delta=1e-4` |
| Random seeds | 42, 43, 44 for each arm |
| Checkpoint selection | Minimum epoch-mean loss; `best`, `last`, and `final` roles plus `early_stopping.json` |

**Multiplier note:** Some configuration files declare `nmul: 16`. Formal runs explicitly pass `--nmul 4` (50,754 trainable values), which is also the default in this release.

### Optimization objectives

**Fit objective.** Discharge prediction is trained using per-catchment variance-normalized squared error.

**Complexity charge.** Optional processes are penalized according to their gate weights:

$$
\boxed{
\mathcal{L}_{\mathrm{complexity}}
=\lambda\sum_{p}k_p\,\mathrm{mean}_{b,t}(w_{b,p})
}
$$

$$
\lambda=0.007,
\qquad
(k_{\mathrm{phen}},k_{\mathrm{int}},k_{\mathrm{snow}},k_{\mathrm{sub}})
=(2,2,2,1).
$$

**Counterfactual gate supervision (Flex).** At the beginning of every epoch, each optional process is evaluated in its ON and OFF states. The complexity contribution is subtracted from the counterfactual comparison before training targets are constructed:

$$
\begin{aligned}
\Delta J_p &= J_{\mathrm{OFF},p}-J_{\mathrm{ON},p},\\[3pt]
\tau_p &= \mathrm{median}\!\left(|\Delta J_p|\right),\\[3pt]
q_p &= \sigma\!\left(\frac{\Delta J_p}{\tau_p}\right),\\[3pt]
c_p &= 2\,|q_p-0.5|.
\end{aligned}
$$

Here, $q_p$ is the soft ON target and $c_p$ is its confidence weight. A **confidence-weighted binary cross-entropy (BCE)** trains the gate logits.

The two gradient paths are isolated:

| Objective | Parameter backbone | Gate network |
| :-- | :--: | :--: |
| Fit + complexity | Receives gradients | **Blocked** |
| Counterfactual BCE | **Blocked** | Receives gradients |

### Experiment variants

| Variant | Structural gates |
| :-- | :-- |
| `Basic` | All fixed to 0 |
| `Full` | All fixed to 1 |
| `Fixed` | User-supplied fixed values |
| `Flex` | Learned from catchment attributes |
| Regional / LORO | One of seven regions is held out; models are refitted |

Training refuses a non-empty output directory and refuses to write runs inside the repository's
`data/` tree; run directories belong in the archive (`$FLEXMOPEX_DATA_ROOT/experiments/...`).

### Synthetic experiments

Synthetic scenario datasets consist of **536 training** and **135 holdout** basins, with noisy teacher-generated discharge. `train/synthetic/train.py` supports five learner methods:

`Canonical_BCE` · `Plain_BCE` · `Direct_Fit` · `Matched_Direct_Fit` · `Strict_Direct_Fit`

They use the same training budget as the real-basin experiments. Folds are constructed exclusively from training basins.

## Reproduction commands

Run these commands from the repository root after configuring the required dependencies and external data. Replace placeholder output paths with new destinations.

### Real-basin training and evaluation

```bash
python train/run_model.py \
    --config configs/flex.yaml \
    --mode train_test \
    --model-type flex \
    --seed 42 \
    --output-root /path/to/results \
    --run-name flex_seed42
```

### Synthetic data generation

Generate the A–D scenarios. For noise variants, add `--scenario-tag _low` or `--scenario-tag _high`.

```bash
python train/synthetic/generate.py \
    --out /path/to/synthetic-data \
    --scenarios A,B,C,D
```

Generate the E2 variant with the first-submission S0 teacher:

```bash
FLEXMOPEX_TEACHER_DIR="$FLEXMOPEX_DATA_ROOT/experiments/teacher_e2_s0/seed_42" \
python train/synthetic/generate.py \
    --out /path/to/synthetic-data \
    --scenarios A \
    --scenario-tag _E2 \
    --teacher-aprime \
    --run-label E2_teacher_aprime
```

### Synthetic learner and rescoring

```bash
python train/synthetic/train.py \
    --run_id E1_A_42 \
    --scenario A \
    --method Canonical_BCE \
    --seed 42 \
    --output_dir /path/to/synthetic-run
```

Recompute scenario-correct NSE/KGE from saved learner checkpoints on CPU:

```bash
FLEXMOPEX_SYNTHETIC_CKPTS=/path/to/synthetic/checkpoints \
python evaluate/rescore_synthetic.py \
    --out /path/to/rescore \
    --scenario A
```

### Analysis scripts

```bash
python evaluate/skill_normal.py          --out /path/to/3_1.csv
python evaluate/evidence_normal.py       --out /path/to/evidence
python evaluate/synthetic_detectability.py --out /path/to/tables
python evaluate/attributes_synthetic.py  --out /path/to/attributes
python evaluate/attributes_normal.py     --out /path/to/attributes-normal
python evaluate/removal_lopo.py          --output /path/to/removal
python evaluate/regional_loro.py         --output /path/to/regional
python evaluate/regional_loro_intervals.py --output /path/to/intervals.csv

python evaluate/removal_fixed_off.py \
    --seed 42 \
    --device cuda:0 \
    --output /path/to/fixed_off.csv \
    --diag-output /path/to/replay.csv
```

### Figures

```bash
# Main figures 3-10 (reads analysis/figures_inputs, writes figures/main)
bash plots/main/run_all.sh
python plots/main/render_main_figures.py --only attr

# Supporting Information figures S1-S7 (reads analysis/tables)
bash plots/supplementary/run_all.sh
python plots/supplementary/fig_s4_flow_regime.py      # one figure

# Refresh the archive manifests (RUN_INDEX.csv + MANIFEST.csv)
python tools/build_data_manifest.py
```

Figures 5 and 10 are unapproved previews. They are skipped in the batch commands unless `PREVIEWS=1` is set.

## Analysis and figures

### Analysis-to-result map

Each result block has a dedicated analysis script. The table preserves the paper's section numbering and the reported aggregation / uncertainty procedures.

| Paper result | Script | Calculation and reference |
| :-- | :-- | :-- |
| **3.1 — Skill** | `evaluate/skill_normal.py` | Cross-basin medians for each seed and four alternative aggregations. Paired Flex−Full comparison: mean over seeds per basin, median over basins, 2,000 basin bootstrap draws (seed `20261106`). |
| **3.2 — Detectability** | `evaluate/synthetic_detectability.py` | Among true-ON basins, high ($\Delta J>P_p$), weak ($0\le\Delta J\le P_p$), and negative ($\Delta J<0$) strata; recall for $w>0.5$. |
| **3.3 — Scenario contrast** | `evaluate/synthetic_detectability.py` | Rule minus random scenario: balanced accuracy and AUC per seed, averaged over seeds; 2,000 basin resamples of 135 holdout basins. |
| **3.4 — Evidence / regional** | `evaluate/evidence_normal.py`, `regional_loro.py`, `regional_loro_intervals.py` | Three-seed classes `decisive_open`, `decisive_closed`, and `uncertain` require consistent seed signs and $|\mathrm{median}(\Delta J)|>\mathrm{SD}$. LORO versus Full-data class: balanced accuracy; region-stratified bootstrap, 1,000 draws (seed `20260516`). |
| **3.4 — Attribute AUC** | `evaluate/attributes_synthetic.py` | Held-out gate predicts the Full-data decisive class as a continuous score; 1,000 stratified resamples (seed `20260521`); single-class regions produce `NaN`. |
| **3.5 — Fixed-off** | `evaluate/removal_fixed_off.py` | Replay frozen Full checkpoints, set one gate to 0 and the other three to 1, keep parameters and routing fixed; compute per-basin NSE/KGE losses. |
| **3.5 — Process removal** | `evaluate/removal_lopo.py` | Compare fixed-off and independently refitted LOPO using three-seed basin means, group medians, ON−OFF differences of medians, and paired Fixed−LOPO differences; 2,000 basin resamples (seed `20261106`). |

**Evaluation boundary:** Among the listed analyses, only `evaluate/removal_fixed_off.py` forwards the model; the other scripts read frozen artifacts. All scripts write to new destinations and refuse overwrites.

### Figure scripts and status

- **Main figures:** `plots/main/render_main_figures.py` draws Figures 3-10 in one pass (or one at a
  time with `--only <key>`); the study-area panel uses the archived region and state layers.
- **Supplementary candidates:** `plots/supplementary/fig_attributes.py` and `plots/supplementary/fig_refit.py`.
- **Outstanding figure checks:** Figures 5 and 10 are awaiting approval; Figure 7 reproduces verified classes but not the candidate raster pixel for pixel.

## Verification

The following checks were reported for the packaged implementation **on the original verification machine**, with one job run at a time. They do not imply that a full training campaign was rerun when preparing this README.

| Check | Reported outcome |
| :-- | :-- |
| Entry-point smoke tests | Full, Flex, Basic, Fixed, regional hold-out, and all five synthetic methods passed forward/loss/backward on real data. |
| Short Flex run | Four basins, one epoch; checkpoints and early-stopping metadata produced. |
| GPU preflight | Production batch size of 100 basins passed. |
| Simulator tests | **11 / 11** passed. |
| Training-graph tests | **7 / 7** passed. |
| Checkpoint-selection tests | **12 / 12** passed. |
| Numerical tables | Results 3.1–3.3 and 3.4 attribute AUC matched archived values; 3.4 evidence/regional and 3.5 fixed-off/LOPO tables matched byte for byte. |
| Fixed-off/LOPO volume | **2,684 rows per seed** across three seeds. |
| Attribute associations | `rho_w` and `rho_q` differed from verified tables by at most $10^{-16}$. |
| Synthetic Scenario A | Two generated runs were byte-identical; truth gates, basin axis, dates, forcings, attributes, and noise multipliers matched the archived dataset exactly. |
| Teacher discharge | Maximum difference from the archived Scenario A file: $9.1\times10^{-4}$ mm/day. |
| Figure regeneration | Figures 4, 6, 8, 9 and both supplementary candidates matched byte for byte. |
