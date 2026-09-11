"""Generator worker: fit and sample one cell in an isolated process.

Invoked by ``run_cell`` as ``python -m sdts.runner.worker spec.json``.
Reads the spec (dataset, n, method, seed, hparams), loads and splits the
data, subsamples the training split, fits the generator, samples n rows,
and writes ``<out>/synthetic.parquet`` and ``<out>/worker.json`` (effective
hparams, cost, extra result fields). On any exception it writes
``worker.json`` with the traceback and exits 1.

The parent never uses torch; this process never uses xgboost. That
separation is deliberate: the two libraries bundle separate OpenMP runtimes
and deadlock when their parallel regions share a process on macOS.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    spec_path = Path(argv[0])
    spec = json.loads(spec_path.read_text())
    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"status": "ok", "error": None}
    try:
        from sdts.data import splits
        from sdts.eval.cost import track
        from sdts.methods.base import get_method, seed_everything

        # This process never runs xgboost, so torch could use several threads;
        # measured 2026-09-07: no speed-up for these small nets (tabddpm 19 s at
        # 1, 4 and 12 threads; ctgan slower at 12), so the default stays 1,
        # identical to the in-process tests. Override with SDTS_TORCH_THREADS.
        import os

        import torch

        threads = int(os.environ.get("SDTS_TORCH_THREADS", 1))
        torch.set_num_threads(threads)
        report["torch_threads"] = threads

        seed = int(spec["seed"])
        seed_everything(seed)
        sp = splits.split(spec["dataset"])
        n = len(sp.train) if spec["n"] == "full" else int(spec["n"])
        train = splits.ladder_subsample(sp.train, n, seed, sp.schema.target)
        cls = get_method(spec["method"])
        if not cls.enabled:
            raise RuntimeError(f"method {spec['method']!r} is registered but disabled")
        method = cls(sp.schema, spec.get("hparams") or {}, seed)
        with track() as t_fit:
            method.fit(train)
        with track() as t_sample:
            synth = method.sample(n)
        sp.schema.validate(synth)
        synth.to_parquet(out / "synthetic.parquet", index=False)
        train.to_parquet(out / "train.parquet", index=False)
        report.update({
            "n_train": n,
            "hparams": method.hparams,
            "cost": {"fit_seconds": t_fit.seconds, "sample_seconds": t_sample.seconds,
                     "peak_rss_mb": t_sample.peak_rss_mb},
            "extra": method.extra_result_fields,
        })
    except MemoryError:
        report.update({"status": "oom", "error": traceback.format_exc()})
    except Exception:  # noqa: BLE001 - recorded, never dropped
        report.update({"status": "failed", "error": traceback.format_exc()})
    (out / "worker.json").write_text(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
