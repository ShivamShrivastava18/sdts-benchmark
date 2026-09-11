"""python -m sdts.doctor: pre-flight checks that must pass before any
experiment run.

Checks:

1. ``gpu``            torch imports and sees an accelerator (CUDA or MPS).
2. ``pins``           every ``name==version`` in pyproject.toml matches the
                      installed distribution.
3. ``dataset_cache``  every cached raw file in ``data/cache/<id>/`` matches
                      the sha256 of its source entry in ``configs/datasets.yaml``.
                      A dataset with nothing cached is a warning (the loader
                      downloads on demand); a mismatch is a failure.
4. ``disk``           free space at the repo root is above 20 GB.

Writes a JSON report (default ``runs/doctor_report.json``) and exits 0 only
if no check failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
DATASETS_YAML = REPO_ROOT / "configs" / "datasets.yaml"
DATA_CACHE = REPO_ROOT / "data" / "cache"
DEFAULT_REPORT = REPO_ROOT / "runs" / "doctor_report.json"
MIN_FREE_GB = 20.0

_PIN_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(\[[^\]]*\])?\s*==\s*([^\s;]+)")


def _check(name: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, **extra}


def check_gpu(allow_cpu: bool = False) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure is the finding
        return _check("gpu", "fail", f"torch import failed: {exc!r}")
    info: dict[str, Any] = {"torch": torch.__version__}
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        return _check("gpu", "pass", f"cuda: {info['device_name']}", **info)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        info["device"] = "mps"
        info["device_name"] = platform.processor() or platform.machine()
        return _check("gpu", "pass", f"mps: {info['device_name']}", **info)
    info["device"] = "cpu"
    status = "warn" if allow_cpu else "fail"
    return _check("gpu", status, "torch sees no CUDA or MPS device", **info)


def _pinned_requirements(pyproject: Path) -> list[tuple[str, str, bool]]:
    """Return (name, pinned version, required). Packages from an optional
    dependency group are not required: they are only checked if installed."""
    data = tomllib.loads(pyproject.read_text())
    project = data.get("project", {})
    reqs = [(r, True) for r in project.get("dependencies", [])]
    for group in project.get("optional-dependencies", {}).values():
        reqs.extend((r, False) for r in group)
    pins: list[tuple[str, str, bool]] = []
    for req, required in reqs:
        m = _PIN_RE.match(req)
        if m:
            pins.append((m.group(1), m.group(3), required))
    return pins


def check_pins(pyproject: Path = PYPROJECT) -> dict[str, Any]:
    if not pyproject.exists():
        return _check("pins", "fail", f"{pyproject} not found")
    pins = _pinned_requirements(pyproject)
    if not pins:
        return _check("pins", "fail", "no ==pinned dependencies found in pyproject.toml")
    mismatches: list[str] = []
    skipped: list[str] = []
    checked: dict[str, str] = {}
    for name, want, required in pins:
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            if required:
                mismatches.append(f"{name}: pinned {want}, not installed")
            else:
                skipped.append(name)
            continue
        checked[name] = have
        if have != want:
            mismatches.append(f"{name}: pinned {want}, installed {have}")
    if mismatches:
        return _check("pins", "fail", "; ".join(mismatches), installed=checked, optional_absent=skipped)
    detail = f"{len(checked)} pinned packages match"
    if skipped:
        detail += f"; optional not installed: {', '.join(skipped)}"
    return _check("pins", "pass", detail, installed=checked, optional_absent=skipped)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_dataset_cache(
    datasets_yaml: Path = DATASETS_YAML, cache_dir: Path = DATA_CACHE
) -> dict[str, Any]:
    if not datasets_yaml.exists():
        return _check("dataset_cache", "warn", f"{datasets_yaml} not found; no datasets configured")
    import yaml

    cfg = yaml.safe_load(datasets_yaml.read_text()) or {}
    datasets: dict[str, dict[str, Any]] = cfg.get("datasets") or {}
    if not datasets:
        return _check("dataset_cache", "pass", "no datasets configured yet")
    missing: list[str] = []
    bad: list[str] = []
    ok = 0
    for ds_id, spec in datasets.items():
        sources = spec.get("sources") or []
        cached, seen = [], set()
        for src in sources:
            raw = cache_dir / ds_id / str(src.get("filename"))
            if raw.exists() and raw not in seen:  # fallbacks may share a filename
                cached.append((src, raw))
                seen.add(raw)
        if not cached:
            missing.append(ds_id)
            continue
        for src, raw in cached:
            want = str(src.get("sha256", "")).lower()
            have = _sha256(raw)
            if have != want:
                bad.append(f"{ds_id}/{raw.name}: expected {want[:12]}.., got {have[:12]}..")
            else:
                ok += 1
    if bad:
        return _check("dataset_cache", "fail", "checksum mismatch: " + "; ".join(bad),
                      ok=ok, missing=missing)
    if missing:
        return _check("dataset_cache", "warn",
                      f"{ok} verified, {len(missing)} not cached yet: {', '.join(missing)}",
                      ok=ok, missing=missing)
    return _check("dataset_cache", "pass", f"{ok} cached datasets verified", ok=ok)


def check_disk(root: Path = REPO_ROOT, min_free_gb: float = MIN_FREE_GB) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    free_gb = usage.free / 1e9
    status = "pass" if free_gb >= min_free_gb else "fail"
    return _check("disk", status, f"{free_gb:.1f} GB free at {root} (need {min_free_gb:g} GB)",
                  free_gb=round(free_gb, 2), min_free_gb=min_free_gb)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def run(allow_cpu: bool = False, min_free_gb: float = MIN_FREE_GB) -> dict[str, Any]:
    checks = [
        check_gpu(allow_cpu=allow_cpu),
        check_pins(),
        check_dataset_cache(),
        check_disk(min_free_gb=min_free_gb),
    ]
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "checks": checks,
        "ok": all(c["status"] != "fail" for c in checks),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdts.doctor", description=__doc__)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="downgrade a missing accelerator from fail to warn")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    args = ap.parse_args(argv)

    report = run(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    for c in report["checks"]:
        print(f"[{c['status'].upper():4}] {c['name']:14} {c['detail']}")
    print(f"report: {args.report}")
    print("doctor:", "OK" if report["ok"] else "FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
