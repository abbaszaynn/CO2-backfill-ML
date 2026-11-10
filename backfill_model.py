"""Model code for the CO2 backfill app.

The model classes, input scaling and scores are copied from CO2_backfill_ML_Colab.ipynb
(section 3). Changes from the notebook:
  * the target column is a parameter, so more targets (UCS, CO2 uptake, ...) can be added later;
  * QuadFit returns NaN for recipes it has never seen instead of failing;
  * GP accepts a fixed kernel, used to make cross-validation fast in the app.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
try:
    from xgboost import XGBRegressor
except Exception:                       # e.g. missing libomp on a Mac: run without XGBoost
    XGBRegressor = None
warnings.filterwarnings("ignore")      # GP bound warnings are expected (see notebook)

CSV = "co2_backfill_train.csv"
INPUTS = ["co2_time_min", "cement_fraction", "stirring_rpm", "co2_flow_L_min", "curing_days"]
MIX = INPUTS[1:]

# One entry per measured response. Add UCS, CO2 uptake, slump, ... here when the raw data arrives.
# best: "min" if a lower value is better, "max" if higher is better.
TARGETS = {
    "porosity_pct": dict(label="Porosity", unit="%", best="min",
                         meaning="Lower porosity means a denser, stronger backfill."),
}

GRID = np.round(np.arange(0, 20.01, 0.1), 2)       # CO2 times for curves (min)
NEXT_TIMES = [6, 7, 9, 11, 12, 13]                  # times the lab should test next

# Leave-one-time-out results from the notebook run on 23 Oct 2026 (the honest test).
NOTEBOOK_LOTO = pd.DataFrame({
    "model": ["Quadratic fit", "Linear regression", "Random forest", "XGBoost", "SVR", "MLP", "GP", "GP + quadratic trend"],
    "R2": [0.405, -0.483, 0.595, 0.592, 0.748, 0.789, 0.684, 0.450],
    "RMSE (%)": [1.021, 1.612, 0.842, 0.845, 0.664, 0.608, 0.744, 0.982],
    "optimum found with 10 min hidden (min)": [10.5, 20.0, 6.6, 8.0, 8.2, 8.1, 9.5, 9.8],
}).set_index("model")


# ---------- from the notebook ----------
def X_of(d):
    # scale inputs to similar ranges
    return np.c_[d.co2_time_min / 20, (d.cement_fraction - 0.1) / 0.1, (d.stirring_rpm - 500) / 300,
                 (d.co2_flow_L_min - 1) / 2, (d.curing_days - 3) / 11]

def trend_of(d):
    # U-shaped trend in time that can change with curing age, plus the mix inputs
    t, a = d.co2_time_min / 20, (d.curing_days - 3) / 11
    return np.c_[X_of(d), t**2, t * a, t**2 * a]

class Scaled:
    # wraps an sklearn model and standardises the target
    def __init__(self, make, target): self.make, self.target = make, target
    def fit(self, d):
        self.sc = StandardScaler().fit(d[[self.target]]); self.m = self.make()
        self.m.fit(X_of(d), self.sc.transform(d[[self.target]]).ravel()); return self
    def predict(self, d): return self.m.predict(X_of(d)) * self.sc.scale_[0] + self.sc.mean_[0]

# GP smoothness settings (chosen by comparing options with the leave-one-time-out test):
MIN_TIME_SCALE_MIN = 5      # the GP may not wiggle faster than about 5 minutes of injection time
MIN_NOISE = 0.05            # treats small jumps between neighbouring times as lab scatter, not signal

class GP:
    # Gaussian process; trend=True adds the U-shaped trend first; kernel= reuses fitted settings;
    # min_time_scale (min) and min_noise override the smoothness settings above (used in paper/compare_gp_settings.py)
    def __init__(self, target, trend=False, kernel=None, min_time_scale=MIN_TIME_SCALE_MIN, min_noise=MIN_NOISE):
        self.target, self.trend, self.kernel = target, trend, kernel
        self.min_time_scale, self.min_noise = min_time_scale, min_noise
    def fit(self, d):
        y = d[self.target].values
        self.lr = LinearRegression().fit(trend_of(d), y) if self.trend else None
        r = y - (self.lr.predict(trend_of(d)) if self.trend else 0)
        self.sc = StandardScaler().fit(r.reshape(-1, 1))
        bounds = [(self.min_time_scale / 20, 1e3)] + [(5e-2, 1e3)] * 4
        kernel = self.kernel if self.kernel is not None else ConstantKernel(1.0, (1e-2, 1e1)) * Matern(np.ones(5), nu=2.5, length_scale_bounds=bounds) \
                 + WhiteKernel(self.min_noise, (self.min_noise, 1.0))
        self.gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=0, random_state=0,
                                           optimizer=None if self.kernel is not None else "fmin_l_bfgs_b")
        self.gp.fit(X_of(d), self.sc.transform(r.reshape(-1, 1)).ravel()); return self
    def predict(self, d, return_std=False):
        m, s = self.gp.predict(X_of(d), return_std=True)
        m = m * self.sc.scale_[0] + self.sc.mean_[0] + (self.lr.predict(trend_of(d)) if self.trend else 0)
        return (m, s * self.sc.scale_[0]) if return_std else m

class QuadFit:
    # one quadratic curve in time per recipe (NaN for recipes not in the training data)
    def __init__(self, target): self.target = target
    def fit(self, d): self.d = d; return self
    def predict(self, d):
        out = np.full(len(d), np.nan)
        for rid in d.recipe.unique():
            g = self.d[self.d.recipe == rid]; idx = (d.recipe == rid).values
            if g.co2_time_min.nunique() >= 3:
                out[idx] = np.polyval(np.polyfit(g.co2_time_min, g[self.target], 2), d[idx].co2_time_min)
        return out

MODELS = {
    "Quadratic fit":        lambda t: QuadFit(t),
    "Linear regression":    lambda t: Scaled(LinearRegression, t),
    "Random forest":        lambda t: Scaled(lambda: RandomForestRegressor(500, min_samples_leaf=2, random_state=42, n_jobs=-1), t),
    "XGBoost":              lambda t: Scaled(lambda: XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=4, random_state=42), t),
    "SVR":                  lambda t: Scaled(lambda: SVR(C=10, epsilon=0.01), t),
    "MLP":                  lambda t: Scaled(lambda: MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=4000, random_state=42), t),
    "GP":                   lambda t: GP(t, trend=False),
    "GP + quadratic trend": lambda t: GP(t, trend=True),
}
if XGBRegressor is None: del MODELS["XGBoost"]
BEST = "GP + quadratic trend"

def scores(y, p):
    y, p = np.asarray(y), np.asarray(p); yb = y.mean()
    return {"R2": 1 - np.sum((y - p)**2) / np.sum((y - yb)**2),
            "RMSE": np.sqrt(np.mean((y - p)**2)), "MAE": np.mean(np.abs(y - p)),
            "IA": 1 - np.sum((y - p)**2) / np.sum((np.abs(p - yb) + np.abs(y - yb))**2)}


# ---------- helpers for the app ----------
def recipe_key(d):
    # same format as the CSV's recipe column, e.g. "1:6_800rpm_1Lmin_14d"
    ratio = (1 / d.cement_fraction - 1).round(2)
    return [f"1:{r:g}_{s:g}rpm_{f:g}Lmin_{a:g}d" for r, s, f, a in
            zip(ratio, d.stirring_rpm, d.co2_flow_L_min, d.curing_days)]

def load_data(path=CSV):
    return pd.read_csv(path)

def input_ranges(df):
    return {c: (df[c].min(), df[c].max(), sorted(df[c].unique())) for c in INPUTS}

def range_issues(row, ranges):
    """Return (outside, between) lists of input names for one input row (a dict)."""
    outside, between = [], []
    for c in INPUTS:
        lo, hi, levels = ranges[c]
        if not lo - 1e-9 <= row[c] <= hi + 1e-9: outside.append(c)
        elif c != "co2_time_min" and not np.isclose(levels, row[c]).any(): between.append(c)
    return outside, between

def clean_new_data(raw, target):
    """Check an uploaded/entered table. Returns (clean DataFrame, list of messages)."""
    missing = [c for c in INPUTS + [target] if c not in raw.columns]
    if missing: return None, [f"Missing columns: {', '.join(missing)}"]
    d = raw[INPUTS + [target]]
    d = d[d[target].notna() & (d[target].astype(str).str.strip() != "")]   # rows without a measurement are ignored quietly
    d = d.apply(pd.to_numeric, errors="coerce")
    bad = d.isna().any(axis=1)
    msgs = [f"{bad.sum()} row(s) skipped because of empty or non-numeric values."] if bad.any() else []
    d = d[~bad].reset_index(drop=True)
    d.insert(0, "recipe", recipe_key(d))
    return d, msgs

def fit_all(df, target):
    return {name: make(target).fit(df) for name, make in MODELS.items()}

def predict_band(gp, d):
    mu, sd = gp.predict(d, return_std=True)
    return mu, mu - 1.96 * sd, mu + 1.96 * sd

def curve_and_optimum(gp, mix, best="min", n_draws=2000, rng=42):
    """Mean curve with 95% band over GRID, and the optimum time from n_draws posterior curves
    (same method as notebook section 7). rng is a seed or a numpy Generator shared between calls."""
    Xg = pd.DataFrame({"co2_time_min": GRID, **mix})
    mean_s, cov_s = gp.gp.predict(X_of(Xg), return_cov=True)
    s, trend = gp.sc.scale_[0], gp.lr.predict(trend_of(Xg)) if gp.trend else 0
    draws = np.random.default_rng(rng).multivariate_normal(mean_s, cov_s + 1e-9 * np.eye(len(GRID)), size=n_draws) \
            * s + gp.sc.mean_[0] + trend
    t_opt = GRID[draws.argmin(axis=1) if best == "min" else draws.argmax(axis=1)]
    mu = mean_s * s + gp.sc.mean_[0] + trend; sd = np.sqrt(np.clip(np.diag(cov_s), 0, None)) * s
    curve = pd.DataFrame({"co2_time_min": GRID, "predicted": mu, "lower_95": mu - 1.96 * sd, "upper_95": mu + 1.96 * sd})
    return curve, t_opt

def group_cv(df, target, groups, kernel):
    """Cross-validation of the GP + quadratic trend, holding out one group at a time.
    The kernel settings are fixed (learned once on all data) so it runs in seconds."""
    p = np.zeros(len(df)); inside = np.zeros(len(df), bool)
    for g in pd.unique(groups):
        te = (groups == g).values
        mu, lo, hi = predict_band(GP(target, trend=True, kernel=kernel).fit(df[~te]), df[te])
        p[te] = mu; inside[te] = (df[target].values[te] >= lo) & (df[target].values[te] <= hi)
    return {**scores(df[target], p), "95% band coverage": inside.mean()}
