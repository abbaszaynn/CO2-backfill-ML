"""Reproduce every number, table and figure of the paper's results section (preliminary data).

Run from the repository root:  python paper/reproduce_results.py      (about 10-15 minutes)

Outputs in paper/outputs/:
  results.xlsx           tables: random split, leave-one-time-out, peak test, optimum, final curves, data
  res1_dataset.pdf/png   Fig. res1: preliminary data, retained and removed injection times
  res2_scores.pdf/png    Fig. res2: R2 under random split and leave-one-time-out
  res3_peak_test.pdf/png Fig. res3: locating the optimum with 10 (and 8+10) min held out
  res4_final_model.pdf/png Fig. res4: final GP + quadratic trend and optimum intervals
The model code is backfill_model.py (the same code the web app uses). Seeds are fixed throughout.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import backfill_model as bm
from clean_data import ROOT, TESTED_TIMES, load_raw

T = "porosity_pct"
OUT = os.path.join(ROOT, "paper", "outputs")
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#6a4fd6"
AGE_COL = {3: BLUE, 7: ORANGE, 14: GREEN}
EXAMPLE = dict(cement_ratio="1:6", stirring_rpm=800, co2_flow_L_min=1)       # recipe shown in the figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})

def example(d):
    return d[(d.cement_ratio == EXAMPLE["cement_ratio"]) & (d.stirring_rpm == EXAMPLE["stirring_rpm"])
             & (d.co2_flow_L_min == EXAMPLE["co2_flow_L_min"])]

def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

def log(msg, t0=[time.time()]): print(f"[{time.time() - t0[0]:5.0f}s] {msg}", flush=True)


# ---------- Test 1: random 5-fold split (misleading on purpose) ----------
def random_split(df):
    rows, kf = [], KFold(5, shuffle=True, random_state=42)
    for name, make in bm.MODELS.items():
        if name == "Quadratic fit": continue                       # needs whole curves, not random rows
        p = np.zeros(len(df))
        for tr, te in kf.split(df): p[te] = make(T).fit(df.iloc[tr]).predict(df.iloc[te])
        rows.append({"model": name, **bm.scores(df[T], p)})
    return pd.DataFrame(rows).set_index("model")

# ---------- Test 2: leave-one-time-out ----------
def leave_one_time_out(df):
    rows, t10 = [], (df.co2_time_min == 10).values
    for name, make in bm.MODELS.items():
        p, inside = np.zeros(len(df)), []
        for t in sorted(df.co2_time_min.unique()):
            te = (df.co2_time_min == t).values
            m = make(T).fit(df[~te])
            if isinstance(m, bm.GP):
                mu, lo, hi = bm.predict_band(m, df[te]); p[te] = mu
                inside += list((df[T].values[te] >= lo) & (df[T].values[te] <= hi))
            else:
                p[te] = m.predict(df[te])
        rows.append({"model": name, **bm.scores(df[T], p),
                     "MAE at 10 min": np.mean(np.abs(df[T].values[t10] - p[t10])),
                     "95% band coverage": np.mean(inside) if inside else np.nan})
        log(f"leave-one-time-out: {name}")
    return pd.DataFrame(rows).set_index("model")

# ---------- Test 3: can a model find an optimum it has not seen? ----------
def peak_test(df):
    true_opt = df.loc[df.groupby("recipe")[T].idxmin()].set_index("recipe").co2_time_min
    base = df.drop_duplicates("recipe").set_index("recipe")
    G = pd.concat([pd.DataFrame({"recipe": r, "co2_time_min": bm.GRID, **{c: base.loc[r, c] for c in bm.MIX}})
                   for r in base.index], ignore_index=True)
    rows, curves = [], {}
    for name, make in bm.MODELS.items():
        r = {"model": name}
        for label, hidden in [("hide 10", [10]), ("hide 8+10", [8, 10])]:
            G["pred"] = make(T).fit(df[~df.co2_time_min.isin(hidden)]).predict(G)
            found = G.loc[G.groupby("recipe").pred.idxmin()].set_index("recipe").co2_time_min
            err = (found - true_opt.loc[found.index]).abs()
            r[f"{label}: optimum found (median, min)"] = found.median()
            r[f"{label}: within 2 min (%)"] = 100 * (err <= 2).mean()
            if label == "hide 10": curves[name] = G[["recipe", "co2_time_min", "pred"]].copy()
        rows.append(r); log(f"peak test: {name}")
    return pd.DataFrame(rows).set_index("model"), curves

# ---------- Final model: optimum with 95% interval from 2,000 posterior draws ----------
def final_optimum(df):
    final = bm.GP(T, trend=True).fit(df)
    rng = np.random.default_rng(42)                                 # one generator for all nine cases, in this order
    rows, curves = [], []
    for ratio, frac in [("1:4", 0.2), ("1:6", 1 / 7), ("1:8", 1 / 9)]:
        for days in (3, 7, 14):
            mix = dict(cement_fraction=frac, stirring_rpm=800, co2_flow_L_min=1, curing_days=days)
            curve, t_opt = bm.curve_and_optimum(final, mix, rng=rng)
            obs = df[(df.cement_ratio == ratio) & (df.stirring_rpm == 800) & (df.co2_flow_L_min == 1) & (df.curing_days == days)]
            rows.append({"cement_ratio": ratio, "curing_days": days, "optimum_median_min": np.median(t_opt),
                         "range_2.5pct_min": np.percentile(t_opt, 2.5), "range_97.5pct_min": np.percentile(t_opt, 97.5),
                         "predicted_lowest_porosity_pct": curve.predicted.min(),
                         "measured_lowest_porosity_pct": obs[T].min(),
                         "measured_best_time_min": obs.loc[obs[T].idxmin(), "co2_time_min"]})
            curves.append(curve.rename(columns={"predicted": "pred_mean", "lower_95": "band_low", "upper_95": "band_high"})
                          .assign(cement_ratio=ratio, curing_days=days))
    curves = pd.concat(curves)[["cement_ratio", "curing_days", "co2_time_min", "pred_mean", "band_low", "band_high"]]
    log("final model")
    return pd.DataFrame(rows), curves


# ---------- figures ----------
def fig_dataset(raw):
    d = raw[(raw.Cement_Ratio == 6) & (raw.Stirring_rate == 800) & (raw.CO2_flow_rate == 1)].sort_values("CO2_injection_time")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.axvspan(20, 30.5, color="#f0ede6", lw=0, zorder=0)
    ax.text(25.2, 14.1, "outside tested range", ha="center", color="#555")
    kept = d.CO2_injection_time.isin(TESTED_TIMES)
    for col, days in [("Porosity_3_days", 3), ("Porosity_7_days", 7), ("Porosity_14_days", 14)]:
        c = AGE_COL[days]
        ax.plot(d.CO2_injection_time, d[col], color=c, lw=1.5, label=f"{days} days")
        ax.scatter(d.CO2_injection_time[kept], d[col][kept], color=c, edgecolor="white", s=28, zorder=3)
        ax.scatter(d.CO2_injection_time[~kept], d[col][~kept], color="white", edgecolor=c, lw=1.3, s=28, zorder=3)
    h, _ = ax.get_legend_handles_labels()
    marks = [plt.scatter([], [], color="#555", s=28), plt.scatter([], [], color="white", edgecolor="#555", lw=1.3, s=28)]
    ax.legend(h + marks, ["3 days", "7 days", "14 days", "tested time (retained)", "filled-in time (removed)"],
              ncol=2, loc="upper left")
    ax.set(xlabel="CO$_2$ injection time (min)", ylabel="Porosity (%)")
    save(fig, "res1_dataset")

def fig_scores(rs, loto):
    names = list(rs.index)[::-1]; y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for vals, off, c, lab, tc in [(rs.R2, 0.18, BLUE, "random 5-fold split", "#555"),
                                   (loto.R2.reindex(rs.index), -0.18, ORANGE, "leave-one-time-out", "black")]:
        v = vals.loc[names].values
        ax.barh(y + off, np.maximum(v, -0.25), height=0.34, color=c, label=lab)
        for yi, vi in zip(y + off, v): ax.text(max(vi, 0) + 0.015, yi, f"{vi:.2f}", va="center", fontsize=8, color=tc)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set(yticks=y, yticklabels=names, xlim=(-0.3, 1.12), xlabel="$R^2$ on held-out data (bars below −0.25 truncated)")
    ax.grid(axis="y", visible=False); ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1))
    save(fig, "res2_scores")

def fig_peak(df, peak, curves):
    rid = "1:6_800rpm_1Lmin_14d"; obs = df[df.recipe == rid]; h = obs.co2_time_min == 10
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.4), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.55})
    for name, c in [("GP + quadratic trend", BLUE), ("Quadratic fit", ORANGE), ("Random forest", GREEN)]:
        cc = curves[name][curves[name].recipe == rid]; a.plot(cc.co2_time_min, cc.pred, color=c, lw=1.6, label=name)
    a.scatter(obs[~h].co2_time_min, obs[~h][T], color="black", s=22, zorder=3, label="training data")
    a.scatter(obs[h].co2_time_min, obs[h][T], color="white", edgecolor="black", lw=1.3, s=36, zorder=3, label="held-out 10 min")
    a.set(xlabel="CO$_2$ injection time (min)", ylabel="Porosity (%)"); a.legend(fontsize=7.5)
    names = list(peak.index)[::-1]; y = np.arange(len(names))
    for col, off, c, lab in [("hide 10: within 2 min (%)", 0.18, BLUE, "10 min held out"),
                             ("hide 8+10: within 2 min (%)", -0.18, PURPLE, "8 and 10 min held out")]:
        v = peak[col].loc[names].values
        b.barh(y + off, v, height=0.34, color=c, label=lab)
        for yi, vi in zip(y + off, v): b.text(vi + 1.5, yi, f"{vi:.0f}", va="center", fontsize=7.5)
    b.set(yticks=y, yticklabels=names, xlim=(0, 115), xlabel=r"Recipes with $|\Delta t^*| \leq 2$ min (%)")
    b.grid(axis="y", visible=False); b.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1))
    for ax, lab in [(a, "(a)"), (b, "(b)")]: ax.set_title(lab, loc="left", fontweight="bold", x=-0.15)
    save(fig, "res3_peak_test")

def fig_final(df, opt, fcurves):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.4), gridspec_kw={"wspace": 0.35})
    for days, c in AGE_COL.items():
        cc = fcurves[(fcurves.cement_ratio == EXAMPLE["cement_ratio"]) & (fcurves.curing_days == days)]
        a.fill_between(cc.co2_time_min, cc.band_low, cc.band_high, color=c, alpha=0.15, lw=0)
        a.plot(cc.co2_time_min, cc.pred_mean, color=c, lw=1.6, label=f"{days} days")
        o = example(df[df.curing_days == days]); a.scatter(o.co2_time_min, o[T], color=c, edgecolor="white", s=22, zorder=3)
    a.set(xlabel="CO$_2$ injection time (min)", ylabel="Porosity (%)"); a.legend(title="Curing age", fontsize=7.5)
    y = np.arange(len(opt))[::-1]
    for yi, (_, r) in zip(y, opt.iterrows()):
        c = {"1:4": BLUE, "1:6": ORANGE, "1:8": GREEN}[r.cement_ratio]
        b.plot([r["range_2.5pct_min"], r["range_97.5pct_min"]], [yi, yi], color=c, alpha=0.35, lw=5, solid_capstyle="round")
        b.scatter(r.optimum_median_min, yi, color=c, edgecolor="white", s=36, zorder=3)
    b.axvline(10, color="#555", ls="--", lw=0.8)
    b.set(yticks=y, yticklabels=[f"{r.cement_ratio}, {r.curing_days} d" for _, r in opt.iterrows()], xlim=(6, 15),
          xlabel="$t^*$ (min): median and 95% interval")
    b.grid(axis="y", visible=False)
    for ax, lab in [(a, "(a)"), (b, "(b)")]: ax.set_title(lab, loc="left", fontweight="bold", x=-0.12)
    save(fig, "res4_final_model")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df, raw = bm.load_data(os.path.join(ROOT, bm.CSV)), load_raw()
    log(f"data: {len(df)} records, {df.recipe.nunique()} recipes")
    rs = random_split(df); log("random split done")
    loto = leave_one_time_out(df)
    peak, curves = peak_test(df)
    opt, fcurves = final_optimum(df)

    with pd.ExcelWriter(os.path.join(OUT, "results.xlsx")) as xw:
        for name, t in [("1_Random_split", rs), ("2_Leave_one_time_out", loto), ("3_Peak_test", peak)]:
            t.round(3).to_excel(xw, sheet_name=name)
        opt.round(3).to_excel(xw, sheet_name="4_Final_optimum", index=False)
        fcurves.round(3).to_excel(xw, sheet_name="5_Final_curves", index=False)
        df.to_excel(xw, sheet_name="6_Train_data", index=False)
    fig_dataset(raw); fig_scores(rs, loto); fig_peak(df, peak, curves); fig_final(df, opt, fcurves)

    pd.set_option("display.width", 200)
    for title, t in [("Random 5-fold split", rs), ("Leave-one-time-out (Table loto)", loto),
                     ("Peak test", peak), ("Optimum (Table optimum)", opt.set_index(["cement_ratio", "curing_days"]))]:
        print(f"\n== {title}\n{t.round(3).to_string()}")
    log(f"done; outputs in {OUT}")
