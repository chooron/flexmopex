#!/usr/bin/env python3
"""Build reduced code/figure and selected-training-result ZIPs; never upload/publish."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import paths  # noqa: E402

TRAINING_FILES = {
    "best_checkpoint.pt", "config_snapshot.json", "source_config.yaml",
    "manifest.json", "provenance.json", "early_stopping.json",
    "train_loss_history.json", "loss_history.json", "normalization_statistics.json",
    "metrics.json", "metrics_agg.json", "launch_provenance.json", "summary.json",
}
CODE_DIRS = {"configs", "data", "docs", "evaluate", "plots", "tests", "tools", "train"}
CODE_ROOT = {"README.md", "paths.py", "pyproject.toml", "requirements.txt", ".gitignore",
             "LICENSE", "LICENSE.md", "LICENSE.txt"}
SKIP_DIRS = {".git", ".pi", "__pycache__", ".venv", "diagnostics"}
SECRET = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)"
    r"[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_./+\-]{16,})"
)
TEXT_EXTS = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".toml", ".sh", ".csv"}
FIGURE = re.compile(r"(?:Figure(?:0[3-9]|10)_.*\.png|FigureS[1-7]_.*\.(?:png|pdf))$")
TABLE = re.compile(r"Table(?:[12]|S[1-8])_.*\.csv$")


def data_allowed(rel: Path) -> bool:
    p = rel.parts
    return (
        len(p) >= 3 and p[0] == "experiments" and rel.name in TRAINING_FILES
        and not any(x in SKIP_DIRS for x in p)
        and not (p[1] == "ablation_nmul" and p[2] == "full")
        and not (p[1] == "synthetic" and p[2] in {"dataset", "holdout_eval", "nse_tables"})
    )


def collect_data() -> dict[str, Path]:
    selected = {}
    epochs = defaultdict(set)
    root = paths.DATA_ROOT
    for directory, dirs, names in os.walk(root / "experiments", followlinks=False):
        parent = Path(directory)
        rel_parent = parent.relative_to(root)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (parent / d).is_symlink())
        if str(rel_parent).replace(os.sep, "/") == "experiments/ablation_nmul":
            dirs[:] = [d for d in dirs if d != "full"]
        if str(rel_parent).replace(os.sep, "/") == "experiments/synthetic":
            dirs[:] = [d for d in dirs if d not in {"dataset", "holdout_eval", "nse_tables"}]
        for name in sorted(names):
            rel = rel_parent / name
            if not data_allowed(rel):
                continue
            source = root / rel
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Unexpected selected file: {source}")
            selected[rel.as_posix()] = source
            if name in {"metrics.json", "metrics_agg.json"}:
                for i, part in enumerate(rel.parts):
                    match = re.fullmatch(r"(?:train|test).*_Ep(\d+)", part)
                    if match:
                        epochs[Path(*rel.parts[:i]).as_posix()].add(int(match.group(1)))
    ambiguous = [run for run, values in epochs.items() if len(values) > 1]
    if ambiguous:
        raise ValueError(f"Multiple metric epochs need explicit selection: {ambiguous}")
    if not any(p.endswith("/best_checkpoint.pt") for p in selected):
        raise ValueError("No selected model weights found")
    return selected


def collect_code() -> dict[str, Path]:
    output = subprocess.check_output(
        ["git", "-C", str(REPO), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    selected = {}
    for raw in output.split(b"\0"):
        if not raw:
            continue
        rel = Path(os.fsdecode(raw))
        if rel.parts[0] not in CODE_DIRS and rel.as_posix() not in CODE_ROOT:
            continue
        if any(p in SKIP_DIRS for p in rel.parts) or rel.suffix in {".pyc", ".pyo"}:
            continue
        source = REPO / rel
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Unexpected code file: {source}")
        selected[rel.as_posix()] = source
    figures = [p for p in paths.FIGURES.glob("*/*") if p.is_file() and FIGURE.fullmatch(p.name)]
    if len(figures) != 21:
        raise ValueError(f"Expected 8 main PNGs, 7 SI PNGs, and 6 SI PDFs; found {len(figures)}")
    for source in figures:
        selected[(Path("figures") / source.relative_to(paths.FIGURES)).as_posix()] = source
    tables = [p for p in paths.TABLES.glob("*.csv") if TABLE.fullmatch(p.name)]
    if len(tables) != 10:
        raise ValueError(f"Expected 2 main and 8 SI publication tables; found {len(tables)}")
    for source in tables + [paths.TABLES / "SI_Tables_S1_to_S8_formatted.md"]:
        selected[(Path("tables") / source.name).as_posix()] = source
    return selected


def check_secrets(source: Path) -> None:
    if source.suffix not in TEXT_EXTS:
        return
    with source.open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            match = SECRET.search(line)
            value = match.group(1).upper() if match else ""
            placeholder = value.startswith(("YOUR_", "EXAMPLE", "PLACEHOLDER"))
            private_key = re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", line)
            if private_key or (match and not placeholder):
                raise ValueError(f"Possible credential; inspect privately: {source}, line {number}")


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def verify_bundle(target: Path, kind: str) -> int:
    prefix = f"flexmopex-{kind}/"
    with zipfile.ZipFile(target) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        records = list(csv.DictReader(io.StringIO(bundle.read(prefix + "MANIFEST.csv").decode())))
        if set(names) != {prefix + r["path"] for r in records} | {prefix + "MANIFEST.csv"}:
            raise ValueError("ZIP manifest membership mismatch")
        for row in records:
            rel = Path(row["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe archive path")
            if kind == "data" and rel.name not in {"ARCHIVE_README.md", "CHECKPOINT_INDEX.csv"}:
                if not data_allowed(rel):
                    raise ValueError(f"Disallowed data artifact: {rel}")
            h, size = hashlib.sha256(), 0
            with bundle.open(prefix + row["path"]) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    h.update(chunk)
                    size += len(chunk)
            if h.hexdigest() != row["sha256"] or size != int(row["bytes"]):
                raise ValueError(f"Content verification failed: {rel}")
    return len(names)


def write_bundle(out: Path, kind: str, files: dict[str, Path], readme: str) -> Path:
    target = out / f"flexmopex-{kind}.zip"
    partial = target.with_suffix(".zip.partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite an existing archive: {target}")
    records = []
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        def add_text(name: str, content: bytes) -> None:
            bundle.writestr(f"flexmopex-{kind}/{name}", content)
            records.append(dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))

        add_text("ARCHIVE_README.md", readme.encode("utf-8"))
        for rel, source in sorted(files.items()):
            before = source.stat()
            h, size = hashlib.sha256(), 0
            info = zipfile.ZipInfo.from_file(source, f"flexmopex-{kind}/{rel}")
            info.compress_type = zipfile.ZIP_DEFLATED
            with source.open("rb") as src, bundle.open(info, "w", force_zip64=True) as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed while archiving: {source}")
            records.append(dict(path=rel, bytes=size, sha256=h.hexdigest()))
        if kind == "data":
            weights = [r for r in records if r["path"].endswith("/best_checkpoint.pt")]
            add_text("CHECKPOINT_INDEX.csv", csv_bytes(weights, ["path", "bytes", "sha256"]))
        bundle.writestr(f"flexmopex-{kind}/MANIFEST.csv", csv_bytes(records, ["path", "bytes", "sha256"]))
    count = verify_bundle(partial, kind)
    if target.exists():
        raise FileExistsError(f"Target appeared during packaging: {target}")
    partial.rename(target)
    print(f"VERIFIED {target}: {count} ZIP members, {target.stat().st_size} bytes", flush=True)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=paths.DATA_ROOT / "archive")
    ap.add_argument("--plan", action="store_true", help="print selection without writing archives")
    args = ap.parse_args()
    selections = {"code": collect_code(), "data": collect_data()}
    for kind, files in selections.items():
        size = sum(p.stat().st_size for p in files.values())
        weights = sum(p.name == "best_checkpoint.pt" for p in files.values())
        print(f"PLAN {kind}: {len(files)} source files, {size} bytes, {weights} selected checkpoints", flush=True)
    if args.plan:
        return 0
    if not args.out.is_dir():
        raise NotADirectoryError(args.out)
    for kind in selections:
        target = args.out / f"flexmopex-{kind}.zip"
        if target.exists() or target.with_suffix(".zip.partial").exists():
            raise FileExistsError(target)
    for files in selections.values():
        for source in files.values():
            check_secrets(source)
    print("PASS: no credential-like literals detected in selected text files", flush=True)
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    common = (
        f"Source code base commit: {head}. Local working-tree edits are included; MANIFEST.csv "
        "records the exact delivered contents, excluding the manifest itself.\n\n"
        "This is an RF_S0 V2 archive. Rain/snow partitioning precedes liquid-rainfall interception; "
        "the reference is m=4 and lambda=0.007, not selected by test-set NSE. Historical synthetic "
        "E2/teacher records, where present, are provenance rather than a single-factor causal ablation.\n\n"
    )
    readmes = {
        "code": "# Flex-MOPEX code, final figures, and publication tables\n\n" + common + (
            "Contains training/evaluation/plotting code, configuration and small metadata files, "
            "canonical Figures 3–10 and S1–S7, and 2 main plus 8 SI publication tables. "
            "SI figures have PNG/PDF pairs except approved Figure S2 (PNG only). "
            "Figures 1–2 are not included because editable sources were unavailable.\n\n"
            "The latest Figure 8 notation/spacing, S3 negative tick, and S6 centered single-row "
            "legend/equal-height panels are included. S2 is the approved frozen PNG; its renderer "
            "has a documented non-byte-identical reproduction limitation. Publication tables are "
            "archived exports, not newly audited manuscript text.\n\n"
            "The companion data ZIP intentionally contains only selected training results and weights. "
            "Raw inputs, figure-backing analyses and intermediate calculations are not uploaded. "
            "Training or rerendering requires separately obtained original inputs and a correctly "
            "configured FLEXMOPEX_DATA_ROOT. See README.md and docs/ for the full-workspace layout.\n"
        ),
        "data": "# Flex-MOPEX selected training results and model weights\n\n" + common + (
            "Contains only best_checkpoint.pt weights, training configurations, selected-epoch "
            "records, loss histories, normalization metadata, final training/test metrics, and "
            "training provenance. CHECKPOINT_INDEX.csv lists every included model checkpoint. "
            "Original training-result files are preserved without recalculating metrics.\n\n"
            "Excluded: simulated/observed daily runoff arrays; daily gate arrays; raw CAMELS and "
            "synthetic datasets; analysis/derived, counterfactual and figure-input exports; "
            "per-epoch diagnostics/checkpoints; redundant final/last checkpoints; worker logs; "
            "old ZIPs; unused matched-Full multiplicity controls. Full-MOPEX m=4 is retained "
            "in the normal/reference runs.\n\n"
            "Extract alongside flexmopex-code and set FLEXMOPEX_DATA_ROOT to this directory when "
            "accessing these archived training results. This is not a full input/analysis archive; "
            "end-to-end retraining, counterfactual analysis, and figure rendering require separately "
            "obtained inputs that were intentionally omitted.\n"
        ),
    }
    targets = [write_bundle(args.out, kind, files, readmes[kind]) for kind, files in selections.items()]
    lines = []
    for target in targets:
        h = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {target.name}\n")
    checksums = args.out / "flexmopex-checksums.sha256"
    with checksums.open("x", encoding="utf-8") as stream:
        stream.writelines(lines)
    print("SHA256 checksums:", checksums, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
