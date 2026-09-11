"""python -m sdts.data.report: one table row per dataset.

Columns: arm, rows, features, numeric, categorical, positive rate, train
rows, and the max ladder rung. Prints markdown and, with ``--write-readme``,
replaces the block between the README markers.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from sdts.data import loaders
from sdts.data.splits import split

README = loaders.REPO_ROOT / "README.md"
START, END = "<!-- dataset-table:start -->", "<!-- dataset-table:end -->"


def table(dataset_ids: list[str] | None = None) -> str:
    reg = loaders.registry()
    ids = dataset_ids or list(reg)
    rows = ["| dataset | arm | rows | features | numeric | categorical | positive rate | train rows | max rung |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in ids:
        sp = split(d)
        sc = sp.schema
        n = len(sp.train) + len(sp.val) + len(sp.test)
        pos = (sp.train[sc.target].mean() * len(sp.train) + sp.val[sc.target].sum() + sp.test[sc.target].sum()) / n
        rung = sp.max_rung if reg[d].get("arm") == "ladder" else "n/a (true size)"
        rows.append(f"| {d} | {reg[d].get('arm')} | {n} | {len(sc.features)} | {len(sc.numeric)} | "
                    f"{len(sc.categorical)} | {pos:.3f} | {len(sp.train)} | {rung} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdts.data.report", description=__doc__)
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--write-readme", action="store_true")
    args = ap.parse_args(argv)
    md = table(args.datasets or None)
    print(md)
    if args.write_readme:
        text = README.read_text()
        if START not in text:
            raise SystemExit(f"README lacks {START} marker")
        new = re.sub(re.escape(START) + ".*?" + re.escape(END),
                     f"{START}\n{md}\n{END}", text, flags=re.S)
        README.write_text(new)
        print(f"wrote table into {README}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
