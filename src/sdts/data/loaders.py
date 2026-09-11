"""Dataset loaders.

``load(dataset_id)`` returns the canonical frame for a dataset:

* every source in ``configs/datasets.yaml`` is tried in order; the raw file
  is downloaded once into ``data/cache/<id>/`` and its sha256 verified
  against the registry. A checksum mismatch raises and nothing proceeds.
  A source that cannot be downloaded falls through to the next one.
* the raw file is parsed with the per-source ``read`` spec (csv, possibly
  inside nested zips, or xls), then the dataset-level cleaning is applied:
  column renames, dropped columns, dedupe, target binarisation, and the
  missing-value marking policy (``na_tokens`` and ``zero_as_missing``).
* categorical NaNs become the literal level ``"missing"``. Numeric NaNs are
  left as NaN; ``sdts.data.splits`` imputes them with training-split medians
  so that nothing leaks across splits.
* dtypes are cast through ``sdts.data.schema.Schema`` so every method and
  metric sees exactly the same types, and the result is cached as parquet
  keyed by a hash of the dataset's registry entry.

Kaggle sources need a ``~/.kaggle/kaggle.json`` token and the ``kaggle``
package (``pip install -e ".[kaggle]"``).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_YAML = REPO_ROOT / "configs" / "datasets.yaml"
CACHE_DIR = REPO_ROOT / "data" / "cache"
MISSING_LEVEL = "missing"


class ChecksumMismatch(RuntimeError):
    pass


class DownloadFailed(RuntimeError):
    pass


# --------------------------------------------------------------------------- registry
def registry(path: Path = DATASETS_YAML) -> dict[str, dict[str, Any]]:
    cfg = yaml.safe_load(path.read_text()) or {}
    return dict(cfg.get("datasets") or {})


def dataset_ids(arm: str | None = None) -> list[str]:
    reg = registry()
    return [k for k, v in reg.items() if arm is None or v.get("arm") == arm]


def spec_hash(spec: dict[str, Any]) -> str:
    """Stable short hash of a registry entry, used to key parquet and schema caches."""
    blob = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# --------------------------------------------------------------------------- download
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_url(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "sdts-benchmark/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)


def _download_kaggle(dataset: str, filename: str, dest: Path) -> None:
    exe = Path(sys.executable).with_name("kaggle")
    cmd = [str(exe) if exe.exists() else "kaggle", "datasets", "download",
           "-d", dataset, "-f", filename, "-p"]
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(cmd + [td], check=True, capture_output=True, text=True)
        got = list(Path(td).iterdir())
        if not got:
            raise DownloadFailed(f"kaggle returned no file for {dataset}/{filename}")
        src = got[0]
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as zf:
                member = filename if filename in zf.namelist() else zf.namelist()[0]
                with zf.open(member) as fh, dest.open("wb") as out:
                    shutil.copyfileobj(fh, out)
        else:
            shutil.copy(src, dest)


def fetch_raw(dataset_id: str, spec: dict[str, Any], cache_dir: Path = CACHE_DIR) -> tuple[Path, dict]:
    """Return (path to verified raw file, the source spec that produced it).

    An already-cached file that matches its recorded sha256 is used without
    any network access. A cached file that does not match raises.
    """
    ddir = cache_dir / dataset_id
    ddir.mkdir(parents=True, exist_ok=True)
    sources = spec["sources"]

    for src in sources:
        dest = ddir / src["filename"]
        if dest.exists():
            have = sha256_of(dest)
            if have != src["sha256"].lower():
                raise ChecksumMismatch(
                    f"{dataset_id}: cached {dest.name} has sha256 {have}, "
                    f"registry says {src['sha256']}. Refusing to proceed; delete the "
                    "file to re-download or fix the registry."
                )
            return dest, src

    errors: list[str] = []
    for src in sources:
        dest = ddir / src["filename"]
        kind = src.get("kind", "url")
        try:
            if kind == "url":
                log.info("downloading %s from %s", dataset_id, src["url"])
                _download_url(src["url"], dest)
            elif kind == "kaggle":
                log.info("downloading %s from kaggle:%s", dataset_id, src["dataset"])
                _download_kaggle(src["dataset"], src["file"], dest)
            elif kind == "manual":
                raise DownloadFailed(
                    f"manual source: obtain {src['filename']} from {src['url']} and place it at {dest}"
                )
            else:
                raise DownloadFailed(f"unknown source kind {kind!r}")
        except ChecksumMismatch:
            raise
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, then we fall through
            errors.append(f"{kind}:{src.get('url') or src.get('dataset')} -> {exc!r}")
            if dest.exists():
                dest.unlink()
            continue
        have = sha256_of(dest)
        if have != src["sha256"].lower():
            dest.unlink()
            raise ChecksumMismatch(
                f"{dataset_id}: downloaded {src.get('url') or src.get('dataset')} has sha256 "
                f"{have}, registry says {src['sha256']}. Refusing to proceed."
            )
        return dest, src
    raise DownloadFailed(f"{dataset_id}: every source failed:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------- parse
def _open_member_chain(path: Path, members: list[str]) -> bytes:
    data = path.read_bytes()
    for member in members:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            data = zf.read(member)
    return data


def parse_raw(path: Path, read: dict[str, Any]) -> pd.DataFrame:
    read = dict(read)
    fmt = read.pop("format", "csv")
    members = read.pop("members", [])
    payload = _open_member_chain(path, members) if members else path.read_bytes()
    buf = io.BytesIO(payload)
    if fmt == "csv":
        if "header" in read and read["header"] is None:
            read["header"] = None
        return pd.read_csv(buf, **read)
    if fmt == "xls":
        return pd.read_excel(buf, **read)
    raise ValueError(f"unknown format {fmt!r}")


# --------------------------------------------------------------------------- clean
def _binarise_target(s: pd.Series, positive: list[Any]) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        pos = set()
        for p in positive:
            try:
                pos.add(float(p))
            except (TypeError, ValueError):
                continue  # a string label meant for a text-coded source
        return s.astype(float).isin(pos).astype("int64")
    pos_str = {str(p).strip() for p in positive}
    return s.astype(str).str.strip().isin(pos_str).astype("int64")


def clean(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Apply the registry's dataset-level cleaning to a parsed raw frame."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if rename := spec.get("rename"):
        df = df.rename(columns=rename)

    # strip whitespace in string cells (UCI files are padded)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().where(df[c].notna(), np.nan)

    if dd := spec.get("dedupe_by"):
        df = df.sort_values(dd["order"]).drop_duplicates(dd["key"], keep="first")

    missing = spec.get("missing") or {}
    for tok in missing.get("na_tokens", []):
        df = df.replace(str(tok), np.nan)
    for c in missing.get("zero_as_missing", []):
        df[c] = pd.to_numeric(df[c], errors="coerce").replace(0, np.nan)

    for c in spec.get("drop_columns", []):
        if c in df.columns:
            df = df.drop(columns=c)

    target = spec["target"]
    df[target] = _binarise_target(df[target], spec["positive"])

    df = df.reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- public
def load_raw_clean(dataset_id: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Cleaned frame before schema casting; used by schema inference."""
    spec = registry()[dataset_id]
    path, src = fetch_raw(dataset_id, spec, cache_dir)
    df = parse_raw(path, src.get("read", {}))
    return clean(df, spec)


def load(dataset_id: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Canonical, schema-cast frame for ``dataset_id``. Cached as parquet."""
    from sdts.data.schema import get_schema  # local import: schema imports loaders

    reg = registry()
    if dataset_id not in reg:
        raise KeyError(f"unknown dataset {dataset_id!r}; known: {sorted(reg)}")
    spec = reg[dataset_id]
    pq = cache_dir / dataset_id / f"{dataset_id}.{spec_hash(spec)}.parquet"
    schema = get_schema(dataset_id)
    if pq.exists():
        return schema.cast(pd.read_parquet(pq))
    df = schema.cast(load_raw_clean(dataset_id, cache_dir))
    pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq, index=False)
    return df
