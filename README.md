# Verifiable Parameterization of Bayesian Networks from Scientific Literature

Companion code for the paper _Verifiable Parameterization of Bayesian Networks from Scientific Literature: Unlocking Unstructured Empirical Evidence_.

The repository reconstructs Conditional Probability Tables (CPTs) of Bayesian Networks directly from aggregated statistical summaries (t-tests, correlations, ANOVA, χ², means/variances) rather than from raw tabular data. An LLM is used **only** to extract explicit numeric constraints from PDFs (Chain-of-Thought prompting + Pydantic validation); CPTs are then reconstructed by transparent mathematical optimization, so every learned parameter is traceable to a reported statistic.

## Key Contributions

- Five CPT reconstruction strategies from JSON statistical summaries, including two novel **copula-based** approaches.
- A controlled **synthetic benchmark** over mixed-type BNs (binary, ordinal, nominal, continuous) with configurable iteration budgets.
- A **real-world end-to-end evaluation** on three published medical studies (paired PDF + released raw data as proxy ground truth).
- **LLM baselines** (`gpt-4o-mini`, `gpt-5.1`, `o4-mini`) for direct LLM parameterization.
- A **sensitivity suite** over extraction noise, conditional-independence loss, network size, and test-query budget.

## Project Structure

```
src/
├── run_synthetic.py        # Synthetic benchmark: five strategies × iteration grid
├── run_real_world.py       # End-to-end real-world pipeline
├── run_llm_baseline.py     # Direct LLM parameterization baselines
├── run_sensitivity.py      # Sensitivity sweeps
├── plot_*.py               # Plotting
├── approaches/             # copula_optimization, synthetic_em, constraint_optimization, ipf, mcmc
├── extractor/              # CoT extraction pipeline + Pydantic schemas + per-study data
└── utils/                  # eval managers, baselines, converters, pgmpy patches

paper/
├── results/                # Synthetic (by iteration/type), LLM, and real-world results + plots
└── sensitivity/            # CSVs from run_sensitivity.py
```

## Methods

**Reconstruction strategies.** (1) **Direct MCMC** over the CPT simplex with a simulation-based discrepancy as energy. (2) **Constrained Entropy Maximization** via SLSQP, turning reports into equality constraints on marginals. (3) **Iterative Proportional Fitting** against synthesized proxy target marginals. (4) **EM on Synthetic Data**, stacking pairwise Gaussian-copula samples and imputing with a modified pgmpy EM estimator. (5) **Copula-Based Structural Optimization**, optimizing global copula edge correlations by greedy hill-climbing, then sampling and quantile-binning.

**Baselines.** Random Guess (Dirichlet, α = 1) as a lower bound, Baseline MLE on full raw data as an oracle upper bound, and the three LLM baselines.

## Real-World Evaluation

End-to-end on three peer-reviewed studies that publish both a PDF and the underlying raw data. Raw data is used **only** to compute proxy ground-truth CPTs (BIC hill-climbing + MLE) and sanity-check extractions; the strategies see only the JSON summaries extracted from the PDF.

| DOI | Study | Domain |
| --- | --- | --- |
| [10.1186/s13104-019-4632-2](https://doi.org/10.1186/s13104-019-4632-2) | Magai et al. 2019, _BMC Research Notes_ — neonatal hyperbilirubinemia RCT | Neonatology |
| [10.1161/JAHA.118.011771](https://doi.org/10.1161/JAHA.118.011771) | Etyang et al. 2019, _JAHA_ — malaria exposure and blood pressure (MR) | Cardiovascular epidemiology |
| [10.1371/journal.pmed.1002785](https://doi.org/10.1371/journal.pmed.1002785) | Xu et al. 2019, _PLOS Medicine_ — LEAN cluster RCT for schizophrenia | Psychiatry |

Variable type maps and target columns are defined in `DATASET_CONFIGS` in `src/run_real_world.py`. Data lives under `src/extractor/real_world/<DOI>/`; results are written to `paper/results/real/`.

## Evaluation Framework

Quality is assessed at three levels: **parameter-level** (weighted Hellinger, weighted KL, TVD vs. ground-truth CPTs, plus a JSON-level TVD on reproduced summary statistics), **inference-level** (log-loss, posterior Hellinger, accuracy over held-out queries), and **decision-level** (regret and decision accuracy on a synthetic intervention-selection task).

## Setup

Requires Python 3.13+ and [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry shell
```

Add a `.env` in the repo root with `OPENAI_API_KEY=<your_key>` for the LLM baselines and the extractor.

## Usage

```bash
python src/run_synthetic.py      # synthetic benchmark
python src/run_real_world.py     # real-world pipeline
python src/run_llm_baseline.py   # LLM baselines
python src/run_sensitivity.py    # sensitivity suite
python src/plot_synthetic.py     # plots
python src/plot_sensitivity.py
```

Key synthetic settings (`src/run_synthetic.py`): `N_NODES=5`, `EDGE_PROB=0.3`, `N_SEEDS=5`, `N_SAMPLES=10000`, `ITERATION_LEVELS=[500, 1000, 2000, 5000]`, and `TYPE_CONFIGS` (mixed / discrete_only / binary_only).

## Key Results

- Copula-based methods (Synthetic EM and Copula Optimization) are the strongest pipelines overall.
- Copula Optimization approaches the MLE upper bound on inference accuracy and decision regret without accessing raw data.
- Binary-only settings are easiest; mixed-type settings remain hardest.
- Real-world constraints are noisier and many edges lack explicit reported statistics, leaving parts of the parameter space weakly constrained.

## Limitations

- **Non-identifiability:** summary statistics constrain, but do not uniquely identify, a CPT.
- **Reporting error** in source studies propagates into reconstructed CPTs.
- **Discrete approximation:** quantile-based discretization can introduce artefacts in mixed-type networks.
- **Single-paper scope:** semantically equivalent variables across studies are not yet harmonized.

## Acknowledgements

This work is supported by the German Federal Ministry of Education and Research (BMBF) under grant 16IS23069 _Software Campus 3.0_ (TU München).
