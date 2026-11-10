"""Why limit the GP's smoothness (time length scale >= 5 min, noise >= 0.05)? This compares the options.

For each GP variant and each pair of limits it reports:
  * leave-one-time-out R2, RMSE, error at the held-out 10 min and 95% band coverage;
  * the peak test with the 10 min records hidden (median optimum found, share of recipes within 2 min);
  * the width of the 95% interval for the optimum (1:6, 800 rpm, 1 L/min; mean over 3, 7 and 14 days),
    and the fitted time length scale and noise on all data.
Run from the repository root:  python paper/compare_gp_settings.py   (about 30-60 minutes; uses all cores)
Writes paper/outputs/gp_settings_comparison.csv.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # one thread per worker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import backfill_model as bm
from clean_data import ROOT

T = "porosity_pct"
TIME_SCALES = [1, 2.5, 5, 10]   # lower bound on the time length scale (min); 1 min = the bound of the other inputs
NOISES = [1e-5, 0.05]           # lower bound on the noise (standardised); 1e-5 = scikit-learn's default, i.e. no floor

def evaluate(trend, ts, noise):
    df = bm.load_data(os.path.join(ROOT, bm.CSV))
    make = lambda: bm.GP(T, trend=trend, min_time_scale=ts, min_noise=noise)
    # leave-one-time-out
    p, inside = np.zeros(len(df)), np.zeros(len(df), bool)
    for t in sorted(df.co2_time_min.unique()):
        te = (df.co2_time_min == t).values
        mu, lo, hi = bm.predict_band(make().fit(df[~te]), df[te])
        p[te] = mu; inside[te] = (df[T].values[te] >= lo) & (df[T].values[te] <= hi)
    t10 = (df.co2_time_min == 10).values
    # peak test, 10 min hidden
    base = df.drop_duplicates("recipe").set_index("recipe")
    G = pd.concat([pd.DataFrame({"recipe": r, "co2_time_min": bm.GRID, **{c: base.loc[r, c] for c in bm.MIX}})
                   for r in base.index], ignore_index=True)
    G["pred"] = make().fit(df[df.co2_time_min != 10]).predict(G)
    found = G.loc[G.groupby("recipe").pred.idxmin()].co2_time_min
    # optimum interval on all data
    full, rng, widths = make().fit(df), np.random.default_rng(42), []
    for days in (3, 7, 14):
        _, t_opt = bm.curve_and_optimum(full, dict(cement_fraction=1 / 7, stirring_rpm=800, co2_flow_L_min=1, curing_days=days), rng=rng)
        widths.append(np.percentile(t_opt, 97.5) - np.percentile(t_opt, 2.5))
    k = full.gp.kernel_
    s = bm.scores(df[T], p)
    return {"model": "GP + quadratic trend" if trend else "GP", "min time scale (min)": ts, "min noise": noise,
            "LOTO R2": s["R2"], "LOTO RMSE": s["RMSE"], "LOTO MAE at 10 min": np.abs(df[T].values[t10] - p[t10]).mean(),
            "LOTO 95% coverage": inside.mean(),
            "10 min hidden: optimum found (median, min)": found.median(),
            "10 min hidden: within 2 min (%)": 100 * (np.abs(found - 10) <= 2).mean(),
            "optimum 95% interval width (min)": np.mean(widths),
            "fitted time length scale (min)": k.k1.k2.length_scale[0] * 20, "fitted noise": k.k2.noise_level}

if __name__ == "__main__":
    jobs = [(tr, ts, nz) for tr in (False, True) for ts in TIME_SCALES for nz in NOISES]
    with ProcessPoolExecutor() as ex:
        res = pd.DataFrame(list(ex.map(evaluate, *zip(*jobs))))
    res.to_csv(os.path.join(ROOT, "paper", "outputs", "gp_settings_comparison.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(res.round(3).to_string(index=False))
