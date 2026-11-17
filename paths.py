"""Single place that resolves the external data root.

The repository ships code only. Every run, derived table and figure export lives under one
external directory whose location is *not* hard-coded:

    export FLEXMOPEX_DATA_ROOT=/mnt/f/ExpData/flexmopex-revision1     # Linux / WSL
    set    FLEXMOPEX_DATA_ROOT=F:\\ExpData\\flexmopex-revision1        # Windows

Default (this workstation): F:/ExpData/flexmopex-revision1

Layout
    experiments/<family>/...        training runs, grouped by experiment
    analysis/tables/                canonical manuscript and SI CSV tables
    analysis/derived/               per-basin derived tables (evidence, evaluation, ablation, seasonal)
    analysis/static/                attribute matrices used by the attribute analyses
    analysis/figures_inputs/        figure-ready CSV/JSON exports
    figures/{main,supplementary}/   rendered figures
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
# The archive lives on the Windows data drive: F:\ExpData\flexmopex-revision1.
# Under WSL the same directory is mounted at /mnt/f, so resolve accordingly unless the
# environment variable pins an explicit location.
_WINDOWS_DEFAULT = r"F:\ExpData\flexmopex-revision1"
_POSIX_DEFAULT = "/mnt/f/ExpData/flexmopex-revision1"
_DEFAULT = _WINDOWS_DEFAULT if os.name == "nt" else _POSIX_DEFAULT
DATA_ROOT = Path(os.environ.get("FLEXMOPEX_DATA_ROOT", _DEFAULT))

EXPERIMENTS = DATA_ROOT / "experiments"
ANALYSIS = DATA_ROOT / "analysis"
TABLES = ANALYSIS / "tables"
DERIVED = ANALYSIS / "derived"
STATIC = ANALYSIS / "static"
FIG_INPUTS = ANALYSIS / "figures_inputs"
FIGURES = DATA_ROOT / "figures"

# ---- experiment families (see docs/DATA.md for the run_id naming scheme) ----
NORMAL = EXPERIMENTS / "normal"
LORO = EXPERIMENTS / "loro"
LOPO = EXPERIMENTS / "lopo"
ABLATION_NMUL = EXPERIMENTS / "ablation_nmul"
ABLATION_LAMBDA = EXPERIMENTS / "ablation_lambda"
SYNTHETIC = EXPERIMENTS / "synthetic"
TEACHER_E2 = EXPERIMENTS / "teacher_e2_s0"


def normal(model: str = "flex", seed: int = 42) -> Path:
    """Normal-basin run, e.g. normal/flex/ref/seed_42."""
    return NORMAL / model / "ref" / f"seed_{seed}"


def loro(model: str, region: int, seed: int = 42) -> Path:
    """Leave-one-region-out run, e.g. normal->loro/flex/region3/seed_42."""
    return LORO / model / f"region{region}" / f"seed_{seed}"


def lopo(process: str, seed: int = 42) -> Path:
    """Leave-one-process-out run, process in {snow, sub, phen, int}."""
    return LOPO / "flex" / f"remove_w_{process}" / f"seed_{seed}"


def ablation_nmul(m: int, seed: int = 42, model: str = "flex") -> Path:
    return ABLATION_NMUL / model / f"nmul{m}" / f"seed_{seed}"


def ablation_lambda(lam: float, seed: int = 42) -> Path:
    return ABLATION_LAMBDA / "flex" / f"lam{lam:.6f}" / f"seed_{seed}"


def synthetic_run(family: str, scenario: str, seed: int = 42) -> Path:
    """Synthetic learner run, e.g. synthetic/e1/E1/scenA/seed_42 (family e1..e5, mdf, sdf)."""
    tag = {"e1": "E1", "e2": "E2", "e3": "E3", "e4": "E4",
           "mdf": "MDF", "sdf": "SDF"}.get(family, family.upper())
    return SYNTHETIC / family / tag / scenario / f"seed_{seed}"


def synthetic_dir(family: str) -> Path:
    """Family directory holding scenario datasets (generate.py output)."""
    return SYNTHETIC / family
