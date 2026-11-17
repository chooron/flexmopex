# Training runs and the numbers they support

184 runs in six families plus the archived teacher. Every family is RF_S0 (rain-snow partitioning
before liquid-rainfall interception with seasonal κ), three seeds (42/43/44) unless noted, and uses
the same split: training 1980-10-01 … 1995-09-30, test 1995-10-01 … 2010-09-30, 671 CAMELS basins,
365-day warm-up.

| Family | Runs | Entry / config | What it supports |
|---|---|---|---|
| `normal` | 6 (flex, full × 3 seeds) | `train/run_model.py`, `configs/{flex,full}.yaml` | main skill table, paired Flex−Full contrast, evidence census, attribute correlations |
| `loro` | 42 (flex, full × 7 regions × 3 seeds) | `configs/regional.yaml` | regional exclusion (LORO) recovery, regional skill |
| `lopo` | 12 (4 processes × 3 seeds) | `configs/flex.yaml --removed-process` | removal gaps and refitting consequences |
| `ablation_nmul` | 15 + 2 controls | as flex, `--nmul m` | component multiplicity (m = 1,2,4,8,16,32) |
| `ablation_lambda` | 36 (12 λ × 3 seeds) | as flex, `--alpha λ` | complexity-weight sensitivity, sparsity and snow-free screening |
| `synthetic` | learners over scenarios A–D for E1–E5, MDF, SDF | `train/synthetic/train.py`, `train/synthetic/generate.py` | assignment recovery and transfer (Table 1, Table S4) |
| `teacher_e2_s0` | 1 | archived checkpoint | alternative-teacher (E2) block, Table S10 |

## Checkpoint selection

Selection monitors the training loss with `patience = 20` and `min_epochs = 50`; the selected epoch
and stop reason are recorded in `early_stopping.json`. Reference selected epochs
(`normal/flex/ref`): 85 / 91 / 100; `normal/full/ref`: 93 / 99 / 83.
LOPO selected epochs: snow 95/84/50, subsurface 93/94/83, phenology 78/58/87, interception 99/76/85.

## Headline numbers (all reproduce from `analysis/tables/`)

Real basins, 671 catchments, three seeds:

* Flex test NSE: 0.6670 (mean of seed medians) / **0.6675** (median of basin three-seed means) /
  0.5746 (population mean); Full: 0.6656 / **0.6678** / 0.5771.
* Paired Flex − Full: ΔNSE **−0.0025 [−0.0048, −0.0005]**, ΔKGE **+0.0028 [−0.0018, +0.0082]**.
* Seasonal climatology benchmark: median 0.0637, mean 0.1313; Flex beats it in **625 / 671** basins,
  paired median gain **+0.5514**.
* Evidence census (Eq. S6, all three seeds agree in sign and |median ΔJ| > SD):
  snow 347/257/67, subsurface 509/47/115, phenology 205/257/209, interception 20/581/70
  (ON/OFF/UNC of 671).
* Removal gaps, ON−OFF median: fixed-off 0.1943 / 0.1908 / 0.0515 / 0.0261; LOPO
  0.1488 / 0.0572 / 0.0125 / 0.0137; paired Fixed − LOPO +0.0047 / +0.0991 / +0.0128 / −0.0019.

Ablations:

* n_mul: test NSE 0.5003 (m=1) → 0.6287 (m=2) → **0.6675** (m=4) → 0.6770 (m=8) → 0.6847 (m=16)
  → 0.6899 (m=32); m = 1 is degenerate (subsurface decisive ON = 0).
* λ: test NSE 0.66 ± 0.01 across the grid with the reference 0.007 at 0.6675; the unpenalized
  training error stays 0.317–0.335 until λ ≥ 0.01; basins with frac_snow < 0.01 carrying a snow
  gate above 0.5 fall 55 → 0 between λ = 0.0005 and 0.005.

## Reproducing

```bash
python train/run_model.py --config configs/flex.yaml --mode train_test --nmul 4 --alpha 0.007 --seed 42
bash   train/synthetic/run_e1.sh          # synthetic family E1 (see the script header)
```

Analysis modules are one per result block; each writes into `analysis/derived/` and prints the
table it produced. See `docs/FIGURES.md` for the figure side.
