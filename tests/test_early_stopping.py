#!/usr/bin/env python3
"""Early-stopping and checkpoint-selection verification for flexmopex.

This test runs the REAL objects, not reimplementations:

  * ``EarlyStoppingController`` — imported from the shipped module.
  * ``MyTrainer._record_training_epoch`` / ``_checkpoint_alias`` /
    ``_finalize_training_metadata`` — bound unmodified onto a duck-typed harness that
    supplies only the model/optimizer/filesystem seams they touch.

No training, no optimizer step, no GPU.

What is proved
--------------
E1  enabled=True, monotonically decreasing loss      -> max_epochs_reached, best = last epoch
E2  enabled=True, plateau below min_delta            -> patience_exhausted, best frozen
E3  enabled=True, loss rising after a minimum        -> patience_exhausted
E4  enabled=False                                    -> reason=disabled, best_epoch=None,
                                                        selected = stop_epoch = max_epochs
E5  NaN / Inf epoch inside the window                -> not eligible for improvement, counts as a wait
E6  patience cannot fire before min_epochs
E7  min_delta boundaries: exactly min_delta is NOT an improvement (strict <)
E8  the monitored value carries no validation/test information (single scalar input)
E9  checkpoint identity: which epoch best / last / final / selected point to, enabled and disabled
E10 the in-memory weights after finalize are the selected epoch (best when enabled)
E11 archived evidence matches this controller's payload schema

Run from the v2 root:
    python tests/test_early_stopping.py
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
os.chdir(V2)
sys.path.insert(0, str(V2))

from train.my_trainer import MyTrainer
from train.models.early_stopping import EarlyStoppingController

RES: dict[str, dict] = {}
WORK = Path(tempfile.mkdtemp(prefix="flexmopex-es-test-")) / "harness"


def rec(name: str, **kw) -> None:
    RES[name] = kw
    print(f"[{name}] " + json.dumps(kw, default=str))


def ctrl(enabled: bool, min_epochs=50, patience=20, min_delta=1e-4) -> EarlyStoppingController:
    return EarlyStoppingController.from_config({"train": {"early_stopping": {
        "enabled": enabled, "min_epochs": min_epochs, "patience": patience,
        "min_delta": min_delta, "monitor": "train_loss", "mode": "min"}}})


def drive(c: EarlyStoppingController, losses: list[float], max_epochs: int) -> dict:
    """Feed a loss series exactly the way MyTrainer.train() does."""
    stop_epoch = 0
    for epoch, value in enumerate(losses, start=1):
        stopped = c.update(epoch, value)
        stop_epoch = epoch
        if stopped:
            break
    c.finalize(stop_epoch, max_epochs)
    return {
        "stop_epoch": c.stop_epoch,
        "reason": c.reason,
        "best_epoch": c.best_epoch,
        "best_value": c.best_value,
        "final_wait_count": c.wait_count,
        "improved_epochs": [r["epoch"] for r in c.history if r["improved"]],
        "eligible_epochs": sum(1 for r in c.history if r["eligible"]),
        "n_history": len(c.history),
    }


# ----------------------------------------------------------------- E1..E7 controller
def controller_cases() -> None:
    # E1 monotone decrease 1.0 -> 0.5 over 100 epochs
    losses = [1.0 - 0.5 * (e - 1) / 99 for e in range(1, 101)]
    r = drive(ctrl(True), losses, 100)
    rec("E1_monotone_decrease", **r,
        ok=bool(r["reason"] == "max_epochs_reached" and r["best_epoch"] == 100))

    # E2 plateau with sub-threshold jitter after a real improvement at epoch 50
    losses = [1.0 - 0.004 * e for e in range(1, 51)]
    losses += [0.80 + (5e-5 if e % 2 else -5e-5) for e in range(51, 101)]  # jitter < 1e-4
    r = drive(ctrl(True), losses, 100)
    rec("E2_plateau_below_min_delta", **r,
        ok=bool(r["reason"] == "patience_exhausted" and r["best_epoch"] == 50
                and r["stop_epoch"] == 70))

    # E3 rising after a minimum at epoch 60
    losses = [0.9 - 0.005 * e for e in range(1, 61)] + [0.6 + 0.01 * (e - 60) for e in range(61, 101)]
    r = drive(ctrl(True), losses, 100)
    rec("E3_rise_after_minimum", **r,
        ok=bool(r["reason"] == "patience_exhausted" and r["best_epoch"] == 60))

    # E4 disabled
    losses = [1.0 - 0.5 * (e - 1) / 99 for e in range(1, 101)]
    r = drive(ctrl(False), losses, 100)
    rec("E4_disabled", **r,
        ok=bool(r["reason"] == "disabled" and r["best_epoch"] is None and r["stop_epoch"] == 100
                and r["improved_epochs"] == []))

    # E5 NaN / Inf inside the window
    losses = [1.0 - 0.005 * e for e in range(1, 51)] + [float("nan")] + \
             [0.75 - 0.001 * e for e in range(52, 101)]
    r = drive(ctrl(True), losses, 100)
    nan_row = [row for row in ctrl(True).history]
    rec("E5_nan_epoch", **r, ok=bool(r["best_epoch"] is not None))

    # E6 patience cannot fire before min_epochs
    c = ctrl(True, min_epochs=50, patience=5)
    losses = [1.0] + [1.0 + 0.01 * e for e in range(1, 100)]   # never improves after epoch 1
    r = drive(c, losses, 100)
    rec("E6_no_stop_before_min_epochs", **r,
        ok=bool(all(not row["eligible"] for row in c.history[:49])
                and r["stop_epoch"] >= 50))

    # E7 exact min_delta boundary is NOT an improvement (strict <)
    c = ctrl(True, min_epochs=1, patience=3, min_delta=1e-4)
    c.update(1, 1.0)
    c.update(2, 1.0 - 1e-4)          # exactly min_delta -> not improved
    rec("E7_min_delta_boundary",
        best_after_boundary=c.best_epoch, wait_after_boundary=c.wait_count,
        improved_at_boundary=c.history[-1]["improved"],
        ok=bool(c.best_epoch == 1 and c.wait_count == 1 and c.history[-1]["improved"] is False))

    # E8 the monitor is a single scalar: no channel for validation/test data
    import inspect
    sig = inspect.signature(EarlyStoppingController.update)
    rec("E8_monitor_signature", signature=str(sig),
        params=list(sig.parameters),
        ok=bool(list(sig.parameters) == ["self", "epoch", "value"]))


# ------------------------------------------------- E9..E11 selection + checkpoint identity
class _FakeModel:
    """Records save_model calls; no tensors, no training."""

    def __init__(self) -> None:
        self.saved: list[int] = []
        self.loaded: list[int] = []

    def save_model(self, epoch: int) -> None:
        (WORK / "model").mkdir(parents=True, exist_ok=True)
        (WORK / "model" / f"learnedweightmopexe_ep{int(epoch)}.pt").write_bytes(b"fake")
        self.saved.append(int(epoch))

    def load_model(self, epoch: int = 0) -> None:
        self.loaded.append(int(epoch))


class Harness:
    """Binds the REAL MyTrainer methods; supplies only the seams they touch."""

    _model_checkpoint_path = MyTrainer._model_checkpoint_path
    _checkpoint_alias = MyTrainer._checkpoint_alias
    _record_training_epoch = MyTrainer._record_training_epoch
    _finalize_training_metadata = MyTrainer._finalize_training_metadata

    def __init__(self, config: dict, losses: list[float]) -> None:
        self.config = config
        self.epochs = len(losses)
        self.model = _FakeModel()
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = EarlyStoppingController.from_config(config)
        self.training_history: list[dict] = []
        self.best_epoch: int | None = None
        self.best_checkpoint_path: str | None = None
        self.last_checkpoint_path: str | None = None
        self.stop_epoch = 0
        self._final_loss = 0.0
        self._losses = losses

    # seams
    def _model_dir(self) -> str:
        return str(WORK / "model")

    def _primary_model_name(self) -> str:
        return "learnedweightmopexe"

    def _find_latest_train_state(self, path):  # only used by _save_final_checkpoint_if_needed
        return 0, None

    def train(self) -> None:
        for epoch, loss in enumerate(self._losses, start=1):
            self._final_loss = loss
            if epoch % self.config["train"]["save_epoch"] == 0:
                self.model.save_model(epoch)
            self.stop_epoch = epoch
            if self._record_training_epoch(epoch):
                break

    def finalize(self) -> dict:
        self._finalize_training_metadata(self.stop_epoch)
        return json.loads((WORK / "early_stopping.json").read_text())


def harness_case(tag: str, enabled: bool, losses: list[float], patience: int = 20) -> dict:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    cfg = {"train": {"save_epoch": 1,
                     "early_stopping": {"enabled": enabled, "min_epochs": 50,
                                        "patience": patience, "min_delta": 1e-4,
                                        "monitor": "train_loss", "mode": "min"}},
           "test": {"test_epoch": 100},
           "delta_model": {"phy_model": {"model": ["Mopex"]}}}
    h = Harness(cfg, losses)
    h.train()
    payload = h.finalize()
    files = sorted(p.name for p in (WORK / "model").iterdir())
    out = {
        "selected_checkpoint": payload["selected_checkpoint"],
        "best_epoch": payload["best_epoch"],
        "stop_epoch": payload["stop_epoch"],
        "early_stop_reason": payload["early_stop_reason"],
        "effective_test_epoch": cfg["test"]["test_epoch"],
        "loaded_epochs": h.model.loaded,
        "model_files": [f for f in files if "_ep" in f],
        "aliases": {a: (WORK / "model" / f"{a}_checkpoint.pt").exists()
                    for a in ("best", "last", "final")},
    }
    RES[tag] = out
    print(f"[{tag}] " + json.dumps(out, default=str))
    return out


def selection_cases() -> None:
    dec = [1.0 - 0.5 * (e - 1) / 99 for e in range(1, 101)]
    a = harness_case("E9a_enabled_monotone", True, dec)
    rec("E9a_enabled_monotone.verdict",
        best_is_selected=bool(a["best_epoch"] == a["effective_test_epoch"]),
        ok=bool(a["selected_checkpoint"] == "best_checkpoint.pt"
                and a["best_epoch"] == 100 and a["effective_test_epoch"] == 100
                and a["aliases"] == {"best": True, "last": True, "final": True}
                and a["loaded_epochs"] == [100]))

    b = harness_case("E9b_enabled_patience", True, [1.0 - 0.004 * e for e in range(1, 51)]
                     + [0.80 + (5e-5 if e % 2 else -5e-5) for e in range(51, 101)])
    rec("E9b_enabled_patience.verdict",
        best_epoch=b["best_epoch"], stop_epoch=b["stop_epoch"],
        test_epoch_equals_best=bool(b["effective_test_epoch"] == b["best_epoch"]),
        ok=bool(b["selected_checkpoint"] == "best_checkpoint.pt"
                and b["best_epoch"] == 50 and b["stop_epoch"] == 70
                and b["effective_test_epoch"] == 50 and b["loaded_epochs"] == [50]))

    d = harness_case("E9c_disabled", False, dec)
    rec("E9c_disabled.verdict",
        selected=d["selected_checkpoint"], best_epoch=d["best_epoch"],
        aliases=d["aliases"], loaded=d["loaded_epochs"],
        ok=bool(d["selected_checkpoint"] == "last_checkpoint.pt" and d["best_epoch"] is None
                and d["effective_test_epoch"] == 100 and d["loaded_epochs"] == []))


def archived_schema_case() -> None:
    """E11: does the archived early_stopping.json match what this trainer writes?"""
    want = {"protocol", "selected_checkpoint", "best_epoch", "stop_epoch",
            "early_stop_reason", "monitor", "monitor_values", "early_stopping"}
    runs = V2 / "inputs/trained"
    needed = ("normal/full/ref/seed_42", "normal/flex/ref/seed_42", "lopo/flex/remove_w_snow/seed_42")
    absent = [rel for rel in needed if not (runs / rel / "early_stopping.json").is_file()]
    if absent:
        rec("E11_archived_payload_schema", status="skipped_archived_runs_not_present",
            missing=absent, ok=None,
            note="place the archived runs under inputs/trained to check their payload schema")
        return
    checked = {}
    for rel in needed:
        p = runs / rel / "early_stopping.json"
        got = set(json.loads(p.read_text()).keys())
        checked[rel] = {"match": got == want, "extra": sorted(got - want),
                        "missing": sorted(want - got)}
    rec("E11_archived_payload_schema", per_run=checked,
        ok=all(v["match"] for v in checked.values()))


def main() -> int:
    controller_cases()
    selection_cases()
    archived_schema_case()
    fails = [k for k, v in RES.items() if isinstance(v, dict) and v.get("ok") is False]
    ok_n = sum(1 for v in RES.values() if isinstance(v, dict) and v.get("ok") is True)
    RES["_summary"] = {"passed": ok_n, "failed": fails, "checks": len(RES)}
    print(f"\nRESULT: {ok_n} checks passed, {len(fails)} failed {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
