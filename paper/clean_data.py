"""Rebuild the training data (co2_backfill_train.csv) from the raw preliminary file.

Steps (paper, Table "Pre-processing of the preliminary dataset"):
  1. Repair the cement-to-tailings ratio (the spreadsheet stored 1:4 as the time 01:04:00);
     the numeric column Cement_Ratio (4, 6, 8) is used instead.
  2. Drop the 14 engineered columns (log, sqrt, square, inverse, products, z-score, outlier flag).
  3. Keep only the 10 injection times tested in the source programme (0-20 min); drop the
     interpolated times 2.5, 3, 12, 14, 18 and the extrapolated times 22, 25, 28, 30 min.
  4. Combine porosity at 3, 7 and 14 days into one response with curing age as an input.

Run from the repository root:  python paper/clean_data.py
It writes paper/outputs/co2_backfill_train_rebuilt.csv and checks it against co2_backfill_train.csv.
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw_preliminary_dataset.xlsx")
TESTED_TIMES = [0, 0.5, 1, 1.5, 2, 5, 8, 10, 15, 20]       # times tested in Hussain et al. (2025)
AGES = {"Porosity_3_days": 3, "Porosity_7_days": 7, "Porosity_14_days": 14}

def load_raw(path=RAW):
    return pd.read_excel(path, sheet_name="Cleaned_Data")

def clean(raw):
    d = raw[["CO2_injection_time", "Cement_Ratio", "Stirring_rate", "CO2_flow_rate", *AGES]]   # steps 1-2
    d = d[d.CO2_injection_time.isin(TESTED_TIMES)]                                              # step 3
    d = d.melt(id_vars=["CO2_injection_time", "Cement_Ratio", "Stirring_rate", "CO2_flow_rate"],
               var_name="age", value_name="porosity_pct")                                       # step 4
    out = pd.DataFrame({
        "cement_ratio": "1:" + d.Cement_Ratio.astype(int).astype(str),
        "cement_fraction": 1 / (1 + d.Cement_Ratio),
        "stirring_rpm": d.Stirring_rate, "co2_flow_L_min": d.CO2_flow_rate,
        "curing_days": d.age.map(AGES), "co2_time_min": d.CO2_injection_time.astype(float),
        "porosity_pct": d.porosity_pct})
    out.insert(0, "recipe", [f"{r}_{s}rpm_{f}Lmin_{a}d" for r, s, f, a in
                             zip(out.cement_ratio, out.stirring_rpm, out.co2_flow_L_min, out.curing_days)])
    out = out.sort_values(["cement_fraction", "stirring_rpm", "co2_flow_L_min", "curing_days", "co2_time_min"],
                          ascending=[False, True, True, True, True])
    assert not out.duplicated(["recipe", "co2_time_min"]).any() and not out.isna().any().any()
    return out.reset_index(drop=True)

if __name__ == "__main__":
    raw = load_raw()
    new = clean(raw)
    os.makedirs(os.path.join(ROOT, "paper", "outputs"), exist_ok=True)
    new.to_csv(os.path.join(ROOT, "paper", "outputs", "co2_backfill_train_rebuilt.csv"), index=False)
    print(f"raw: {len(raw)} records, {raw.CO2_injection_time.nunique()} injection times")
    print(f"clean: {len(new)} records, {new.recipe.nunique()} recipes, "
          f"{len(new) // 3} conditions x 3 curing ages, times {sorted(new.co2_time_min.unique().tolist())}")
    old = pd.read_csv(os.path.join(ROOT, "co2_backfill_train.csv"))
    same = old.columns.equals(new.columns) and len(old) == len(new) and \
        (old.select_dtypes(exclude="number") == new.select_dtypes(exclude="number")).all().all() and \
        np.allclose(old.select_dtypes("number"), new.select_dtypes("number"), rtol=0, atol=1e-9)
    print("matches co2_backfill_train.csv:", same)
