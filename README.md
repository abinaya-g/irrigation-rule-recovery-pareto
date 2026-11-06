# Irrigation-need classification, label-rule recovery and water–yield-risk Pareto analysis

Code, notebook and complete results for the manuscript

> **Irrigation-Need Classification, Label-Rule Recovery, and an Assumption-Explicit Water–Yield-Risk Pareto Analysis for Precision Irrigation: A Computational Demonstration on a Public Dataset**
> P. Sundaravadivel and G. Abinaya, Saveetha Engineering College, Chennai, India.

Everything runs on a **CPU only** (Kaggle CPU session, Google Colab, or a laptop). No GPU is required.

## Contents

| Path | Description |
|---|---|
| `irrigation_analysis.py` | Full pipeline: dataset audit and rule recovery, classification, explainability, derived layer and optimization |
| `Irrigation_Analysis_Kaggle_CPU.ipynb` | The same code as a notebook, organized stage by stage, for Kaggle/Colab |
| `scripts/make_figs.py` | Builds the model-comparison, confusion-matrix and global SHAP figures plus the supplementary workbook from `outputs/` |
| `scripts/make_fig1.py` | Draws the framework diagram (Figure 1) |
| `scripts/results.py` | Helper that loads all result files |
| `outputs/tables/` | Every result table as CSV, including per-fold, per-scenario and per-run results |
| `outputs/figures/` | Every figure of the manuscript (PNG, 300 dpi) |
| `outputs/json/` | Summaries, selected hyperparameters, confusion matrices and software versions |
| `outputs/Supplementary_Tables_S1-S13.xlsx` | Supplementary Tables S1–S13 |

## Data

The analysis uses the public Kaggle dataset **“Irrigation Water Requirement Prediction Dataset”** (MIT licence):
<https://www.kaggle.com/datasets/miadul/irrigation-water-requirement-prediction-dataset>

The dataset is not redistributed here. To obtain it:

* **Kaggle notebook:** *Add data* → search for `miadul/irrigation-water-requirement-prediction-dataset`. The code finds the file under `/kaggle/input` automatically.
* **Locally:** `kaggle datasets download -d miadul/irrigation-water-requirement-prediction-dataset -p data --unzip`, or set the environment variable `IRR_DATA` to the path of `irrigation_prediction.csv`.

## How to run

```bash
pip install -r requirements.txt
python irrigation_analysis.py --stage all          # audit, clf, explain, moo, orig
python irrigation_analysis.py --stage audit        # or run a single stage
python scripts/make_fig1.py
python scripts/make_figs.py
```

| Stage | What it produces | Manuscript |
|---|---|---|
| `audit` | Marginal-distribution audit, univariate tests with Holm/BH correction, exact label-rule recovery, shallow-tree rule recovery | §4.1–4.3, Tables 7–8, Figure 6 |
| `clf` | Library-default and equal-budget Optuna tuning of nine classifiers, cross-validation-only model selection, single test evaluation with bootstrap CIs, per-class metrics, QWK, calibration, 5 × 10 repeated CV, corrected resampled t-test and Wilcoxon tests with Holm correction, partition stability | §4.4–4.7, §4.10, Tables 9–12, 15, Figures 7–9 |
| `explain` | Exact TreeSHAP, beeswarm and partial-dependence plots, gain, permutation and drop-column importance, non-nested feature configurations | §4.8–4.9, Tables 13–14, Figures 10–13 |
| `moo` | FAO-56/FAO-33 derived layer, exact Pareto sets for 150 scenarios, classifier-coupling test, bounds sensitivity, NSGA-II vs MOPSO-CD benchmark (10 seeds, 2,400 evaluations), joint Monte-Carlo uncertainty | §4.11–4.14, Tables 16–18, Figures 14–16 |
| `orig` | Benchmark under a common alternative protocol (three unnormalized objectives, union reference set) | §4.12 |

Runtime on a 12-core CPU is about 2.5 h for the complete pipeline, most of it hyperparameter tuning. Set the environment variable `NJOBS` to control parallelism; `--trials` sets the number of Optuna trials per model (default 30).

## Reproducibility

* Random seeds: 42 for data splitting, tuning and optimization; 42–51 for the optimizer benchmark; 1000–1009 for the partition-stability analysis.
* Software versions used for the reported results (also in `outputs/json/versions.json`): Python 3.10.11, NumPy 2.2.6, pandas 2.2.3, SciPy 1.14.1, scikit-learn 1.6.0, XGBoost 3.1.2, LightGBM 4.6.0, CatBoost 1.2.10, Optuna 5.0.0, SHAP 0.48.0, pymoo 0.6.2, Matplotlib 3.10.6.
* Optuna runs three trials in parallel (`TRIAL_JOBS=3`), so the selected hyperparameters can vary slightly between machines. The values used in the manuscript are in `outputs/json/C_best_params.json`.

## Key result

The label `Irrigation_Need` is reproduced exactly (10,000 / 10,000 records) by an additive score:
2 points each for `Soil_Moisture < 25`, `Rainfall_mm < 300` and a Flowering/Vegetative growth stage, and 1 point each for `Temperature_C > 30`, `Wind_Speed_kmh > 10` and no mulching. A score of 0–3 gives Low, 4–6 Medium and 7–9 High. Classification accuracy on this dataset therefore measures rule recovery, not field performance.

## Licence

MIT – see `LICENSE`.
