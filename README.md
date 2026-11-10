# CO2 backfill ML

Code and data for the article *Learning the optimal CO2 injection for gold tailings-based cement paste backfill
from small laboratory data* (submitted to the Journal of Cleaner Production), plus a small web app for engineers.

Gaussian process models (with and without a quadratic trend in injection time) and seven benchmark models
predict the porosity of CO2-injected gold-tailings cement paste backfill and locate the injection time that
gives the lowest porosity, with a 95% interval.

> **Preliminary results.** The data are a preliminary file derived from the published results of
> Hussain et al. (2025), not specimen-level laboratory measurements (see [DATA.md](DATA.md)).
> The GP's 95% bands are too narrow on these data (63-77% coverage instead of 95%).

## Reproduce the paper's results

Needs Python 3.11 (tested with 3.11.1 on Windows 10). From the repository root:

```bash
pip install -r requirements.txt
```

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Rebuild the training data from the raw file | `python paper/clean_data.py` | seconds | checks it equals `co2_backfill_train.csv` |
| 2. All results: tables and Figs. res1-res4 | `python paper/reproduce_results.py` | ~15 min | `paper/outputs/results.xlsx`, `paper/outputs/res*.pdf` |
| 3. Choice of the GP smoothness limits | `python paper/compare_gp_settings.py` | ~45 min (4 cores) | `paper/outputs/gp_settings_comparison.csv` |

Where each paper item comes from:

| Paper item | Produced by | Where |
|---|---|---|
| Table "Pre-processing" and the dataset numbers in the results section | `paper/clean_data.py` | console output |
| Fig. res1 (dataset) | `paper/reproduce_results.py` | `res1_dataset.pdf` |
| Fig. res2 (R2, random split vs leave-one-time-out) | `paper/reproduce_results.py` | `res2_scores.pdf`; sheets `1_Random_split`, `2_Leave_one_time_out` |
| Table "Leave-one-time-out performance" | `paper/reproduce_results.py` | sheet `2_Leave_one_time_out` |
| Fig. res3 and peak-test numbers | `paper/reproduce_results.py` | `res3_peak_test.pdf`; sheet `3_Peak_test` |
| Fig. res4 and Table "Optimum injection time" | `paper/reproduce_results.py` | `res4_final_model.pdf`; sheets `4_Final_optimum`, `5_Final_curves` |
| GP limits (time length scale >= 5 min, noise >= 0.05), section "Gaussian process surrogate" | `paper/compare_gp_settings.py` | `gp_settings_comparison.csv` |

The pre-computed outputs are included in `paper/outputs/`, so results can be checked without running anything.
`CO2_backfill_ML_Colab.ipynb` is a step-by-step walkthrough of the same analysis that also runs in Google Colab.
All random seeds are fixed. The library versions in `requirements.txt` are the ones used for the paper;
other versions can change the last digit of some results.

## Repository contents

| Path | What it is |
|---|---|
| `backfill_model.py` | All model code: input scaling, the 8 models, scores, optimum search. Used by the scripts, the notebook's logic and the app. |
| `paper/` | Scripts that reproduce the paper's results, and their outputs. |
| `co2_backfill_train.csv`, `data/` | Training data and the raw file it was built from; see [DATA.md](DATA.md). |
| `CO2_backfill_ML_Colab.ipynb` | Walkthrough notebook (Colab or Jupyter). |
| `app.py`, `.streamlit/`, `Dockerfile`, `package/`, `make_package.py` | The web app and its installers. |

## Web app

A web app for lab engineers: enter a mix and a CO2 injection time to see the predicted porosity with a 95%
interval, the best CO2 time for that recipe, all 8 models side by side, and how well the model predicts new lab results.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. The first start trains the models (about 1 minute). It makes no calls to
outside services (no Google Fonts, no CDNs, usage statistics off) and only accepts connections from the same computer.
In mainland China, if `pip` is slow, add `-i https://pypi.tuna.tsinghua.edu.cn/simple`.

| Tab | What it does |
|---|---|
| Predict | Predicted porosity and 95% interval; porosity vs CO2 time curve with the band and the best time (95% range from 2,000 model curves). |
| Compare models | All 8 models for the same inputs, with their leave-one-time-out scores. |
| Check new lab data | Upload a CSV or type values; shows predicted vs measured, error, and whether each result is inside the 95% band. |
| Retrain (optional) | Adds the new rows and refits; shows leave-one-recipe-out and leave-one-time-out scores before and after. Only for the current browser tab; the data files are never changed. |
| Next tests | The CO2 times the lab should test next (6, 7, 9, 11, 12, 13 min). |
| About and limits | Data, model and caveats. |

New lab data: a CSV with columns `co2_time_min,cement_fraction,stirring_rpm,co2_flow_L_min,curing_days,porosity_pct`
(`cement_fraction = 1 / (1 + ratio)`: 1:4 = 0.2, 1:6 = 0.1429, 1:8 = 0.1111). The app offers a template.

**Docker:** `docker build -t co2-backfill .` then `docker run -p 8501:8501 co2-backfill`
(in China add `--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` to the build).

**Sending the app to someone:** `python make_package.py` builds `CO2_backfill_app.zip` with double-click start
files for Windows and Mac and a plain-language `START_HERE.txt`.

**More measured properties later (UCS, CO2 uptake, slump, ...):** add the column to the CSV and one entry to
`TARGETS` in `backfill_model.py`, e.g.
`"ucs_mpa": dict(label="UCS", unit="MPa", best="max", meaning="Higher UCS means a stronger backfill.")`.
A "Property to predict" selector then appears in the app.

On the Retrain tab, the GP's smoothness settings are learned once and kept fixed in each fold, so the scores
take seconds rather than half an hour; they are slightly optimistic compared with the paper's tests.

## Archiving

For a permanent, citable copy, link the GitHub repository to [Zenodo](https://zenodo.org) (log in with GitHub,
switch the repository on under *GitHub*), then create a GitHub release. Zenodo archives the release and gives it a DOI
to cite in the article's data and code availability statement.

## Citation and licence

See [CITATION.cff](CITATION.cff). Code: MIT licence ([LICENSE](LICENSE)). Data: see [DATA.md](DATA.md).
