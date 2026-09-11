# sdts: a small-data benchmark for tabular synthesis

Below what training-set size do deep tabular generative models (CTGAN,
TVAE, TabDDPM) stop beating trivial statistical baselines (independent
marginals, Gaussian copula, SMOTE), and is the ranking between them
stable in that regime?

This repository is the preregistered benchmark that answers it: the
harness, the frozen protocol, and every result file it produced. The
deliverable is the numbers in `results/`, one JSON per
(dataset, n, method, seed, tuning budget). Nothing is recomputed by hand.

## Results at a glance

| | |
|---|---|
| cells, all status `ok` | 2,220 |
| (dataset, deep model) pairs with no crossover | 23 of 24 |
| ladder cells won by the best trivial baseline | 40 of 49 |
| cells where the top-two gap is inside seed noise | 81% |
| deep-model ranking stability below 5,000 rows | Kendall tau 0.806 |
| ranking agreement, natively small vs subsampled | Kendall tau 0.57 to 0.64 |

The preregistered hypothesis holds on its main clause: no deep model beats
the best trivial baseline by more than seed noise at any rung we measured,
on 23 of 24 dataset-and-model pairs. It is falsified on its
ranking-stability clause: the deep models' ranking is stable at small
sizes, not unstable, and stability *decays* as data grows. Both outcomes
are reported as found.

## Design

- **Ladder arm**, 8 datasets subsampled to 200, 500, 1000, 2000, 5000,
  10000 and 20000 training rows, capped at each dataset's training split:
  adult, bank, credit_default, diabetes130, support2, cardio, stroke,
  cdc_diabetes.
- **Small arm**, 4 natively small clinical datasets at true size, never
  subsampled: pima, breast_cancer_wisconsin, heart_cleveland,
  heart_failure. These exist to test whether subsampling a large dataset
  is a valid proxy for a genuinely small one. It is only moderately so.
- **Methods**: marginals, Gaussian copula, SMOTE, unconditional SMOTE,
  CTGAN, TVAE, TabDDPM, plus a disjoint sample of real rows as the
  reference ceiling. All behind one interface; the runner special-cases
  none of them.
- **Metric**: TSTR AUROC, the mean AUROC of logistic regression, random
  forest and XGBoost trained on synthetic data and evaluated on the full
  real test split. Fidelity (C2ST, KS, TVD, association distance),
  privacy (DCR, NNDR, membership inference) and cost are secondary.
- **Protocol**: 20 Optuna TPE trials per (dataset, n, method) against the
  validation split, then 5 evaluation seeds on the winner. The test split
  is never read during tuning and the code raises if it is.

## Setup

Requires Python 3.12. Every dependency is pinned exactly.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

The two Kaggle-hosted datasets (cardio, stroke) additionally need a Kaggle
API token and `pip install -e ".[kaggle]"`.

## Reproducing the analysis

Every figure, table and number is regenerated from the committed result
files by one command. It needs no network and no dataset download,
because nothing is recomputed from raw data:

```bash
make figures
```

That writes `results/aggregate/` (long and summary tables),
`results/analysis/` (the five preregistered analyses),
`results/figures/` and `results/tables/`.

## Reproducing the experiments

This does need the datasets, which are downloaded once and checksummed:

```bash
python -m sdts.doctor                                    # must exit 0
python -m sdts.runner.run_cell --smoke                   # end to end, under 120 s
python -m sdts.runner.run_grid --grid stage1 --confirm-full
python -m sdts.runner.run_grid --grid stage2 --confirm-full
```

Both grids resume: a cell with a valid result file is skipped, so a run
survives interruption. Stage 1 is 1,960 cells and took about 51 hours of
single-thread CPU on an Apple M-series laptop. Logs, manifests and the
budget ledger are written to `runs/`, which is not tracked.

## Layout

```
configs/      datasets.yaml, methods.yaml, grid.yaml, schemas/
src/sdts/
  data/       loaders with checksums, schema inference, splits and the ladder
  methods/    one file per generator, all behind base.Method
  eval/       utility (TSTR), fidelity, privacy, cost
  tune/       frozen search spaces and the Optuna TPE driver
  runner/     run_cell.py, run_grid.py, worker.py
  analysis/   aggregate.py, primary.py, figures.py, tables.py, exploratory/
  doctor.py   pre-flight checks
third_party/  vendored TabDDPM (MIT), unmodified
results/      committed result JSONs and everything derived from them
```

## Preregistration

`PREREGISTRATION.md` fixes the hypothesis, the primary metric, the
dataset list, the size ladder, the method list, the seed count, the
search spaces, the five analyses and the falsification rule. It was
committed after a pilot and before any tuned cell ran. Changes after that
point are appended to its Section 7 with a date, never edited in place.

Three rules govern everything here, and they are worth stating because
they explain the repository's shape:

1. No number in any output is typed by hand or computed in a notebook;
   everything traces to a committed JSON.
2. Failed cells are recorded with their traceback, never dropped. Dropping
   them would inflate the survivors.
3. No single-seed number is reported anywhere. Mean and standard
   deviation, or nothing.

## Data and licences

No raw data is redistributed. `configs/datasets.yaml` records, per
dataset, a primary and a fallback source, the sha256 of the raw file, the
target definition, every cleaning decision and the licence as stated by
the host. The loader verifies the checksum and refuses to proceed on a
mismatch.

Two datasets (cardio, stroke) are Kaggle-hosted with unclear or
educational-use-only terms; they are used here on that basis and their
terms are recorded verbatim in the registry.

## Licence

MIT for the code in this repository; see `LICENSE`. The vendored TabDDPM
implementation under `third_party/` keeps its own MIT licence and
copyright notice. Dataset licences are the hosts'.
