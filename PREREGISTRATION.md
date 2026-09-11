# Preregistration

Filled at Phase 6, after the Stage 0 pilot and before the full grid.
Committed, sha256 recorded in the README, not edited afterwards. Anything
that must change later is appended under Section 7 with a date and a
reason; the original text stays.

Author: Shivam Shrivastava
Date filled: 2026-09-07
Stage 0 report this is based on: `runs/STAGE0_REPORT.md` at commit `cb973e6`

---

## 1. Hypothesis

> Below 5,000 training rows, no deep tabular generative model (CTGAN,
> TVAE, TabDDPM), even after a 20-trial tuning budget, achieves a higher
> TSTR AUROC than the best of four trivial baselines (independent
> marginals, Gaussian copula, SMOTE, ucSMOTE), and the ranking among the
> deep models at those sizes is not stable between adjacent rungs of the
> size ladder.

N = 5,000 was set by the pilot: with default hyper-parameters the best
baseline (a SMOTE variant) beat every deep model at every pilot rung
(adult n = 200, 1,000, 5,000; pima at its true size of 460), with the
margin larger than the pooled seed std of the two compared methods in
every cell. The pilot did not test above 5,000, so the hypothesis makes
no claim there. The direction was the one expected before the pilot.

## 2. Primary metric

**TSTR AUROC**: train logistic regression, random forest and XGBoost
(fixed constants in `configs/methods.yaml`, never tuned) on the synthetic
frame, evaluate on the full real test split, average the three AUROCs.
Implemented once as `sdts.eval.utility.PRIMARY_METRIC =
"tstr_auroc_mean"`.

One primary metric. Everything else is secondary and is labelled so in
the paper: TSTR accuracy and F1, TRTR (real n rows) as the ceiling and
majority-class AUROC 0.5 as the floor; C2ST AUC (primary *fidelity*
metric, still secondary to utility), mean KS, mean TVD, association-matrix
distance; DCR median, DCR rate, NNDR, membership-inference AUC; fit,
sample and evaluation seconds, peak RSS.

## 3. Design

- Ladder datasets (8): adult, bank, credit_default, diabetes130,
  support2, cardio, stroke, cdc_diabetes. Sources, checksums, targets and
  cleaning are fixed in `configs/datasets.yaml`; schemas in
  `configs/schemas/`.
- Small-arm datasets (4), used at true size and never subsampled: pima
  (460 training rows), breast_cancer_wisconsin (341), heart_cleveland
  (181), heart_failure (179).
- Size ladder: n in {200, 500, 1000, 2000, 5000, 10000, 20000}, capped at
  each dataset's training-split size (48 (dataset, n) ladder cells in
  total: adult 5 rungs, bank 7, credit_default 6, diabetes130 7,
  support2 5, cardio 7, stroke 4, cdc_diabetes 7).
- Methods (8 active): baselines marginals, copula, smote, ucsmote; deep
  ctgan, tvae, tabddpm; reference real_subsample (real rows, the ceiling,
  never a competitor in any ranking). tabsyn is registered and disabled.
  Implementations are frozen at the pinned versions in `pyproject.toml`
  and the vendored TabDDPM commit in `third_party/`.
- Splits: 60/20/20 stratified, seeded by dataset id; validation and test
  at full size at every rung; training-split median imputation.
- Evaluation seeds: 5 (0..4). Every reported number is a mean and a
  standard deviation (ddof = 1) over the seeds that finished with status
  ok; a cell with fewer than 2 ok seeds reports no std and is flagged.
- Tuning: Optuna TPE (`TPESampler(seed=0)`), 20 trials per (dataset, n,
  method), objective = primary metric computed on the **validation**
  split (synthetic rows as training data). The test split is never read
  during tuning; the code asserts this and a test proves the assertion
  fires. One tuning run per (dataset, n, method) with the fixed tuning
  seed; the best trial's configuration is then run with the 5 evaluation
  seeds. A trial that fails or exceeds the cell timeout scores minus
  infinity and is recorded. Methods without a search space skip tuning
  and record `tuning_budget: 0, tuning_skipped: "no hyperparameters"`:
  marginals, real_subsample, copula.
- Search spaces: transcribed from the *reduced* spaces of Kindji et al.,
  arXiv 2406.12945, Appendix A (Table A.6 TVAE, A.7 CTGAN, A.9 TabDDPM,
  A.10 SMOTE), with these compute-bound adaptations declared now, before
  any tuned cell is run, because the study runs on a single-thread CPU
  laptop (measured 2026-09-07: the published TabDDPM upper end costs 23
  minutes per trial at 10k steps; the adapted upper end 3.6 minutes):
  - TabDDPM: layers {2, 4} (not 6), layer width {128, 256, 512} (not
    1024), batch {256, 1024} (not 4096), training iterations fixed at
    10,000 (not 20,000), 1,000 timesteps and dropout 0 as published, lr
    qLogUniform(3.5e-4, 9.2e-4) as published.
  - CTGAN: as published (reduced column) with epochs fixed at the SDV
    default of 300 (Kindji: 400) and the numerical encoder fixed at SDV's
    cluster-based normaliser (Kindji's CDF/PLE encoders are not in SDV).
  - TVAE: as published (reduced column) with the same two fixes; the
    learning rate is not exposed by SDV 1.38.3 and stays at the library
    default.
  - SMOTE and ucSMOTE: k in [2, 20], grid, one trial per value (19).
  The exact spaces are in `src/sdts/tune/spaces.py`; they may not change
  after this file is committed.
- Per-cell timeout: 3,600 s per fit-and-sample (worker) for every trial
  and every evaluation cell; a timeout is a recorded result.
- Stage 1 = the ladder grid with tuning; Stage 2 = the four small-arm
  datasets at true size with the same tuning and seeds. Both run after
  human sign-off on this document.

## 4. Primary analyses

Definitions shared by all analyses. "Mean" is over ok seeds. The *pair
std* of two methods in a cell is sqrt((s1^2 + s2^2) / 2) of their seed
stds. The *best baseline* in a cell is the baseline with the highest
mean primary metric. Rankings are over the 7 competing methods (deep and
baseline; real_subsample excluded) by mean primary metric; a method with
no ok seed in a cell ranks last and the count of such cases is reported.
Kendall tau is tau-b as implemented in `scipy.stats.kendalltau`.

1. **Crossover point.** For each (ladder dataset, deep method): the
   smallest rung n at which the deep method's mean exceeds the best
   baseline's mean by more than their pair std. Undefined if it never
   does, reported as "none up to n_max(dataset)" rather than dropped.
   Marked on Figure 1.
2. **Headline number.** Per rung: the fraction of ladder datasets in which
   the best baseline's mean exceeds every deep model's mean. Reported
   twice: any margin, and margin larger than the pair std of the best
   baseline and the best deep model. Figure 2.
3. **Ranking stability.** Per ladder dataset and each pair of adjacent
   rungs: Kendall tau between the two rankings of the 7 competing
   methods, and separately between the rankings of the 3 deep methods.
   Reported per dataset and as the mean over datasets per rung pair.
   Figure 3.
4. **Effect size against noise.** Per (dataset, n) cell: the gap between
   the top two competing methods divided by their pair std. The fraction
   of cells below 1.0 is reported, and the cells are plotted against the
   diagonal. Figure 4.
5. **Subsample validity.** For each small-arm dataset: Kendall tau
   between its ranking (true size) and the ranking of every ladder
   dataset at the rung nearest to the small dataset's training size
   (pima 460 -> 500; breast_cancer_wisconsin 341 -> 200; heart_cleveland
   181 -> 200; heart_failure 179 -> 200). Reported per pair and as the
   mean per small dataset.

Utility and privacy are shown jointly (primary metric against DCR rate
and membership-inference AUC, one point per (dataset, n, method)) as
Figure 5; this is descriptive and secondary, no hypothesis attaches to
it.

## 5. Falsification

> The hypothesis is falsified if, at n <= 500, a tuned deep model beats
> the best tuned baseline on the primary metric by more than their pair
> std in more than half of the ladder datasets (5 or more of 8). The
> ranking-stability clause is falsified if the mean Kendall tau between
> the deep-model rankings at adjacent rungs below 5,000 is at least 0.8.

Either way the result is reported as found. Nothing in Section 3 or 4
changes in response to it.

## 6. Declared exploratory

Anything below may be run and reported only under an explicit
`exploratory` label in the results directory, the figure caption and the
paper text:

- Per-classifier breakdown of TSTR (LR, RF, XGB separately)
- Categorical-heavy versus continuous-heavy dataset splits
- Any post-hoc subgroup analysis, including by class imbalance
- Cost and energy comparisons beyond the recorded seconds and RSS
- Sensitivity of the headline number to the noise measure (pair std
  versus pooled std over all methods, as in the Stage 0 report)
- Results of the disabled tabsyn stub, if it is ever enabled

## 7. Deviations

Empty at freeze. Every later deviation is appended here with a date, the
reason, and which results were affected. A deviation with no entry here
is a research-integrity failure, not a documentation lapse.

### 2026-09-07, before Stage 1 started: rung count corrected

Section 3 states 48 ladder cells with adult at 5 rungs. The training
split of adult has 19,536 rows, so the ladder rule already written in the
same sentence ("capped at each dataset's training-split size") gives adult
6 rungs (200 to 10,000) and 49 ladder cells. The rule is unchanged; the
arithmetic in the prose was wrong. `python -m sdts.runner.run_grid --grid
stage1 --dry-run` is the authoritative cell list. No result was affected;
none existed.
