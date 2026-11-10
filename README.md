# CO2 backfill predictor (web app)

A small web app for the CO2-injected gold-tailings cement paste backfill (CGCPB) ML study.
Enter a mix and a CO2 injection time to see the predicted porosity with a 95% interval, the best
CO2 time for that recipe, all 8 models side by side, and how well the model predicts new lab results.

> **Preliminary results.** The training data look derived from published results (Hussain et al., JCP 2025),
> not raw lab data, and the 95% bands are too narrow (63-77% coverage instead of 95%).

## Run it on your computer (one command)

Needs Python 3.11 or newer.

```bash
pip install -r requirements.txt && streamlit run app.py
```

The app opens at http://localhost:8501. The first start trains the models (about 1 minute); after that it is fast.

In mainland China, if `pip` is slow, use a mirror:

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

The app itself makes no calls to outside services (no Google Fonts, no CDNs, usage statistics turned off),
so it works on a closed network once installed.

## Sending the app to someone

Run `python make_package.py`. It builds `CO2_backfill_app.zip` with double-click start files for Windows and Mac
and a plain-language `START_HERE.txt` (sources in `package/`). Rebuild it after any change to the app.

## Run it with Docker

```bash
docker build -t co2-backfill .
docker run -p 8501:8501 co2-backfill
```

In China, add `--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` to the build command.
If Docker Hub is blocked, build the image elsewhere and share it as a file:
`docker save co2-backfill -o co2-backfill.tar`, then `docker load -i co2-backfill.tar` on the other machine.

## What is in the app

| Tab | What it does |
|---|---|
| Predict | Predicted porosity and 95% interval; porosity vs CO2 time curve with the band and the best time (95% range from 2,000 model curves). |
| Compare models | All 8 models (quadratic fit, linear, random forest, XGBoost, SVR, MLP, GP, GP + quadratic trend) for the same inputs, with their honest test scores. |
| Check new lab data | Upload a CSV or type values; shows predicted vs measured, error, and whether each result is inside the 95% band. |
| Retrain (optional) | Adds the new rows and refits; shows leave-one-recipe-out and leave-one-time-out scores before and after. Only for the current browser tab. The original CSV is never changed. |
| Next tests | The CO2 times the lab should test next (6, 7, 9, 11, 12, 13 min) with the model's predictions. |
| About and limits | Data, model and caveats. |

Every table can be downloaded as CSV. The app warns when an input is outside the tested range.

### New lab data format

CSV with these columns (download a template from the app):

```
co2_time_min,cement_fraction,stirring_rpm,co2_flow_L_min,curing_days,porosity_pct
```

`cement_fraction = 1 / (1 + ratio)`, so 1:4 = 0.2, 1:6 = 0.1429, 1:8 = 0.1111.

## Files

- `app.py`: the web interface (Streamlit).
- `backfill_model.py`: the model code, copied from `CO2_backfill_ML_Colab.ipynb`.
- `co2_backfill_train.csv`: training data (read only).
- `requirements.txt`, `Dockerfile`, `.streamlit/config.toml`: setup.

## Adding more measured properties later (UCS, CO2 uptake, slump, ...)

Add the new column to the CSV and one entry to `TARGETS` in `backfill_model.py`, for example

```python
"ucs_mpa": dict(label="UCS", unit="MPa", best="max", meaning="Higher UCS means a stronger backfill."),
```

A "Property to predict" selector then appears in the sidebar. `best="max"` makes the app look for the
CO2 time with the highest value instead of the lowest. Rows missing a property will need to be dropped
for that property's models (not handled yet).

## Notes on the numbers

- The optimum ranges match the notebook closely but not exactly (e.g. 7 days: 7.8-13.7 min here vs
  7.6-13.6 in the notebook), because of newer library versions.
- On the Retrain tab, the GP's smoothness settings are learned once and kept fixed in each fold so the
  scores take seconds, not half an hour. They are slightly optimistic compared with the notebook.
