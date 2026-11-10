"""Web front end for the CO2 backfill ML model. Run with:  streamlit run app.py"""
import altair as alt, numpy as np, pandas as pd, streamlit as st
import backfill_model as bm

st.set_page_config(page_title="CO2 backfill predictor", layout="wide")

LABELS = {"co2_time_min": "CO2 injection time (min)", "cement_fraction": "Cement fraction",
          "stirring_rpm": "Stirring speed (rpm)", "co2_flow_L_min": "CO2 flow rate (L/min)",
          "curing_days": "Curing age (days)"}
BLUE, GREEN, RED, ORANGE = "#2a78d6", "#1baf7a", "#d63a2a", "#eb8a34"


@st.cache_resource(show_spinner="Training the 8 models on first start (about 1 minute)...")
def base_models(target):
    df = bm.load_data()
    return df, bm.fit_all(df, target)

@st.cache_data(show_spinner="Scoring the original model (about 30 s)...")
def cv_table(_df, _kernel, target, key):
    # key identifies the data set, so results are cached per data set
    return pd.DataFrame({"Leave one recipe out": bm.group_cv(_df, target, _df.recipe, _kernel),
                         "Leave one CO2 time out": bm.group_cv(_df, target, _df.co2_time_min, _kernel)}).T

def csv_bytes(d): return d.to_csv(index=False).encode("utf-8")

def meaning(text): st.caption(f"**What this means:** {text}")


# ---------- banner, target, model in use ----------
st.warning("**Preliminary results.** The data appear to be derived from published results, not raw lab "
           "measurements, and the 95% bands are too narrow (they held 63-77% of hidden test points, not 95%). "
           "Use the numbers to plan tests, not as final values.")

target = st.sidebar.selectbox("Property to predict", list(bm.TARGETS), format_func=lambda t: bm.TARGETS[t]["label"]) \
         if len(bm.TARGETS) > 1 else next(iter(bm.TARGETS))
T = bm.TARGETS[target]; unit, best = T["unit"], T["best"]
base_df, base_m = base_models(target)
ss = st.session_state
ss.setdefault("retrained", None)
use_new = ss.retrained is not None and ss.get("use_retrained", False)
df, models = (ss.retrained["df"], ss.retrained["models"]) if use_new else (base_df, base_m)
gp = models[bm.BEST]
ranges = bm.input_ranges(df)

# ---------- sidebar inputs ----------
st.sidebar.header("Mix and test settings")
def tested(c, fmt="{:g}"): return "Tested: " + ", ".join(fmt.format(v) for v in ranges[c][2])
time = st.sidebar.number_input(LABELS["co2_time_min"], 0.0, 60.0, 10.0, 0.5, help=tested("co2_time_min"))
ratio = st.sidebar.number_input("Cement : tailings ratio, 1 :", 1.0, 30.0, 6.0, 0.5,
                                help="Tested: 1:4, 1:6, 1:8. Cement fraction = 1 / (1 + ratio).")
rpm = st.sidebar.number_input(LABELS["stirring_rpm"], 0.0, 3000.0, 800.0, 50.0, help=tested("stirring_rpm"))
flow = st.sidebar.number_input(LABELS["co2_flow_L_min"], 0.0, 20.0, 1.0, 0.5, help=tested("co2_flow_L_min"))
days = st.sidebar.number_input(LABELS["curing_days"], 1.0, 365.0, 14.0, 1.0, help=tested("curing_days"))
mix = dict(cement_fraction=1 / (1 + ratio), stirring_rpm=rpm, co2_flow_L_min=flow, curing_days=days)
st.sidebar.caption(f"Cement fraction = {mix['cement_fraction']:.3f}")
if use_new: st.sidebar.info("Using the **retrained** model (original data + your new rows).")

point = pd.DataFrame([{"co2_time_min": time, **mix}])
point.insert(0, "recipe", bm.recipe_key(point))
recipe = point.recipe[0]
outside, between = bm.range_issues(point.iloc[0], ranges)
if outside:
    st.error("**Outside the training range:** " + ", ".join(
        f"Cement : tailings ratio (tested 1:{1 / ranges[c][1] - 1:g} to 1:{1 / ranges[c][0] - 1:g})"
        if c == "cement_fraction" else f"{LABELS[c]} (tested {ranges[c][0]:g} to {ranges[c][1]:g})" for c in outside)
        + ". The model has no data here, so its predictions are unreliable.")
if between:
    st.info("**Between tested values:** " + ", ".join(LABELS[c] for c in between)
            + ". The model is interpolating between tested levels; treat the result with extra care.")

tabs = st.tabs(["Predict", "Compare models", "Check new lab data", "Retrain (optional)", "Next tests", "About and limits"])

# ---------- 1. predict + curve ----------
with tabs[0]:
    mu, lo, hi = (v[0] for v in bm.predict_band(gp, point))
    c1, c2 = st.columns(2)
    c1.metric(f"Predicted {T['label'].lower()} at {time:g} min", f"{mu:.2f} {unit}")
    c2.metric("95% interval", f"{lo:.2f} to {hi:.2f} {unit}")
    meaning(f"for this mix, the model expects {T['label'].lower()} of about {mu:.2f} {unit}, most likely between "
            f"{lo:.2f} and {hi:.2f} {unit}. {T['meaning']} The range is currently too narrow, so some lab "
            "results will fall outside it.")

    curve, t_opt = bm.curve_and_optimum(gp, mix, best)
    o_med, o_lo, o_hi = np.median(t_opt), np.percentile(t_opt, 2.5), np.percentile(t_opt, 97.5)
    y_best = curve.predicted.min() if best == "min" else curve.predicted.max()
    st.subheader(f"{T['label']} vs CO2 injection time ({recipe.replace('_', ', ')})")
    xt, yt = LABELS["co2_time_min"], f"{T['label']} ({unit})"
    x = alt.X("co2_time_min:Q", title=xt)
    y = alt.Y("predicted:Q", title=yt, scale=alt.Scale(zero=False))
    layers = [
        alt.Chart(pd.DataFrame({"a": [o_lo], "b": [o_hi]})).mark_rect(color=GREEN, opacity=0.12).encode(
            alt.X("a:Q", title=xt), alt.X2("b:Q")),
        alt.Chart(curve).mark_area(color=BLUE, opacity=0.2).encode(x, alt.Y("lower_95:Q", title=yt), alt.Y2("upper_95:Q")),
        alt.Chart(curve).mark_line(color=BLUE, strokeWidth=2.5).encode(x, y, tooltip=[
            alt.Tooltip("co2_time_min:Q", title="time (min)"), alt.Tooltip("predicted:Q", format=".2f"),
            alt.Tooltip("lower_95:Q", format=".2f"), alt.Tooltip("upper_95:Q", format=".2f")]),
        alt.Chart(pd.DataFrame({"t": [o_med]})).mark_rule(color=GREEN, strokeDash=[5, 3], strokeWidth=2).encode(
            alt.X("t:Q", title=xt)),
        alt.Chart(pd.DataFrame({"co2_time_min": [time], "predicted": [mu]})).mark_point(
            color=RED, filled=True, size=120).encode(x, y),
    ]
    obs = df[df.recipe == recipe]
    if len(obs):
        layers.append(alt.Chart(obs).mark_circle(color=ORANGE, size=50, opacity=1).encode(
            x, alt.Y(f"{target}:Q", title=yt), tooltip=["co2_time_min", target]))
    new = ss.get("new_data")
    if new is not None and (new.recipe == recipe).any():
        layers.append(alt.Chart(new[new.recipe == recipe]).mark_point(color=RED, shape="diamond", size=90).encode(
            x, alt.Y(f"{target}:Q", title=yt), tooltip=["co2_time_min", target]))
    st.altair_chart(alt.layer(*layers).properties(height=380), width="stretch")
    st.caption("Blue line: prediction. Blue shading: 95% band. Green dashed line and green shading: best CO2 time "
               "and its 95% range. Orange dots: data the model was trained on. Red dot: your selected time. "
               "Red diamonds: new lab data you entered.")
    st.markdown(f"**Best CO2 injection time for this recipe: about {o_med:.1f} min** "
                f"(95% range {o_lo:.1f} to {o_hi:.1f} min), with a predicted "
                f"{'lowest' if best == 'min' else 'highest'} {T['label'].lower()} of {y_best:.2f} {unit}.")
    meaning(f"the model drew 2,000 possible curves and recorded where each one is "
            f"{'lowest' if best == 'min' else 'highest'}. In 95% of them the best time was between "
            f"{o_lo:.1f} and {o_hi:.1f} min. The JCP 2025 paper also found about 10 min.")
    st.download_button("Download this curve (CSV)", csv_bytes(curve.round(4).assign(recipe=recipe)),
                       f"curve_{recipe}.csv", "text/csv")

# ---------- 2. compare models ----------
with tabs[1]:
    G = pd.DataFrame({"co2_time_min": bm.GRID, **mix}); G.insert(0, "recipe", recipe)
    curves = pd.concat([G[["co2_time_min"]].assign(model=n, predicted=m.predict(G)) for n, m in models.items()])
    table = pd.DataFrame({"model": list(models),
                          f"prediction at {time:g} min ({unit})": [m.predict(point)[0] for m in models.values()]})
    table = table.join(bm.NOTEBOOK_LOTO[["R2", "optimum found with 10 min hidden (min)"]], on="model")
    table = table.rename(columns={"R2": "honest test score R2 (notebook)"})
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    if table.iloc[0, 1:2].isna().all():
        st.caption("The quadratic fit only works for recipes that were tested, so it is blank here.")
    meaning("all eight models are shown for the same inputs. The honest test score (R2, 1 = perfect) comes from "
            "hiding one CO2 time at a time and predicting it. Models that agree with each other are more "
            "trustworthy; large disagreement means the data cannot settle the answer yet.")
    st.altair_chart(alt.Chart(curves.dropna()).mark_line().encode(
        alt.X("co2_time_min:Q", title=LABELS["co2_time_min"]),
        alt.Y("predicted:Q", title=f"{T['label']} ({unit})", scale=alt.Scale(zero=False)),
        alt.Color("model:N", title="Model"),
        strokeWidth=alt.condition(alt.datum.model == bm.BEST, alt.value(3.5), alt.value(1.5)),
        tooltip=["model", "co2_time_min", alt.Tooltip("predicted:Q", format=".2f")]).properties(height=380),
        width="stretch")
    meaning(f"each line is one model's curve for this recipe. The thick line is the main model ({bm.BEST}). "
            "Tree models (random forest, XGBoost) give step-shaped curves because they cannot interpolate.")
    st.download_button("Download model comparison (CSV)", csv_bytes(
        curves.pivot(index="co2_time_min", columns="model", values="predicted").reset_index().round(4)),
        f"model_comparison_{recipe}.csv", "text/csv")

# ---------- 3. check new lab data ----------
with tabs[2]:
    st.markdown("Enter or upload new lab measurements to see how well the model predicted them. "
                "They are checked against the model trained on the **original data only**, so this is a fair test.")
    cols = bm.INPUTS + [target]
    template = pd.DataFrame({"co2_time_min": bm.NEXT_TIMES, **{k: round(v, 4) for k, v in mix.items()}, target: np.nan})
    st.download_button("Download a blank template (CSV)", csv_bytes(template[cols]), "new_lab_data_template.csv", "text/csv")
    st.caption("Columns: " + ", ".join(cols) + ". cement_fraction = 1 / (1 + ratio), e.g. 1:6 gives 0.1429. "
               "Leave the template's last column for your measured values.")
    how = st.radio("How do you want to add data?", ["Upload a CSV file", "Type values"], horizontal=True)
    raw = None
    if how == "Upload a CSV file":
        f = st.file_uploader("CSV file", type="csv")
        if f is not None:
            try: raw = pd.read_csv(f)
            except Exception as e: st.error(f"Could not read the file: {e}")
    else:
        ss.setdefault("editor_default", template[cols])
        raw = st.data_editor(ss.editor_default, num_rows="dynamic", width="stretch", key="editor")
    if raw is not None:
        new, msgs = bm.clean_new_data(raw, target)
        for m in msgs: st.warning(m)
        ss.new_data = new if new is not None and len(new) else None
    new = ss.get("new_data")
    if new is not None:
        p, l, h = bm.predict_band(base_m[bm.BEST], new)
        base_ranges = bm.input_ranges(base_df)
        res = new.assign(predicted=p, lower_95=l, upper_95=h, error=new[target] - p,
                         inside_95_band=np.where((new[target] >= l) & (new[target] <= h), "yes", "no"),
                         outside_training_range=[", ".join(bm.range_issues(r, base_ranges)[0]) for _, r in new.iterrows()])
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows checked", len(res))
        c2.metric("Average error (MAE)", f"{np.abs(res.error).mean():.2f} {unit}")
        c3.metric("Inside the 95% band", f"{(res.inside_95_band == 'yes').mean():.0%}")
        meaning(f"error = measured minus predicted. If the model were reliable, about 95% of rows would fall "
                f"inside the band; past tests suggest 63-77%. Rows marked in the last column are outside the "
                f"training range, so large errors there are expected.")
        st.dataframe(res.round(3), hide_index=True, width="stretch")
        lim = [min(res[target].min(), res.lower_95.min()) - 0.3, max(res[target].max(), res.upper_95.max()) + 0.3]
        sc = alt.Scale(domain=lim)
        mx, py = f"Measured ({unit})", f"Predicted ({unit})"
        line = alt.Chart(pd.DataFrame({"v": lim})).mark_line(color="gray", strokeDash=[4, 4]).encode(
            alt.X("v:Q", title=mx, scale=sc), alt.Y("v:Q", title=py, scale=sc))
        bars = alt.Chart(res).mark_rule(color=BLUE).encode(
            alt.X(f"{target}:Q", title=mx, scale=sc), alt.Y("lower_95:Q", title=py, scale=sc), alt.Y2("upper_95:Q"))
        pts = alt.Chart(res).mark_circle(size=70).encode(
            alt.X(f"{target}:Q", title=mx, scale=sc), alt.Y("predicted:Q", title=py, scale=sc),
            alt.Color("inside_95_band:N", title="Inside band", scale=alt.Scale(domain=["yes", "no"], range=[GREEN, RED])),
            tooltip=["recipe", "co2_time_min", target, alt.Tooltip("predicted:Q", format=".2f")])
        st.altair_chart((line + bars + pts).properties(height=380), width="stretch")
        meaning("points on the dashed line were predicted exactly. Vertical bars are the 95% bands; a red point "
                "means the measurement fell outside its band.")
        st.download_button("Download check results (CSV)", csv_bytes(res.round(4)), "check_results.csv", "text/csv")

# ---------- 4. retrain ----------
with tabs[3]:
    st.markdown("Add your new rows to the training data and refit all models. The original CSV file is "
                "**never changed**; the retrained model lives only in this browser tab and is lost when you reload.")
    new = ss.get("new_data")
    if new is None:
        st.info("Add new lab data in the **Check new lab data** tab first.")
    elif st.button(f"Retrain with {len(new)} new row(s) (about 1-2 minutes)", type="primary"):
        comb = pd.concat([base_df[["recipe"] + bm.INPUTS + [target]], new], ignore_index=True)
        with st.spinner("Refitting the 8 models..."):
            m2 = bm.fit_all(comb, target)
        before = cv_table(base_df, base_m[bm.BEST].gp.kernel_, target, "original")
        with st.spinner("Scoring the retrained model..."):
            after = cv_table(comb, m2[bm.BEST].gp.kernel_, target, csv_bytes(comb))
        ss.retrained = dict(df=comb, models=m2, before=before, after=after, n=len(new))
        ss.use_retrained = True
        st.rerun()
    if ss.retrained is not None:
        r = ss.retrained
        st.success(f"Retrained with {r['n']} new row(s): {len(r['df'])} rows in total.")
        cmp = pd.concat({"before": r["before"], "after": r["after"]}, axis=1).swaplevel(axis=1)[["R2", "RMSE", "MAE", "95% band coverage"]]
        st.dataframe(cmp.round(3), width="stretch")
        meaning("each test hides part of the data, refits the main model and predicts what was hidden. "
                "R2 closer to 1 and RMSE/MAE closer to 0 are better; band coverage should be near 0.95. "
                "**Leave one recipe out** looks almost perfect because, in this preliminary file, the cement, "
                "stirring and flow effects are fixed percentages. **Leave one CO2 time out** is the harder and "
                "more honest test. For speed, the model's smoothness settings are learned once and kept fixed in "
                "each fold, so scores are slightly optimistic compared with the notebook.")
        st.toggle("Use the retrained model on the other tabs", key="use_retrained")
        c1, c2 = st.columns(2)
        c1.download_button("Download combined data (CSV)", csv_bytes(r["df"]), "combined_training_data.csv", "text/csv")
        c2.button("Discard the retrained model", on_click=lambda: ss.update(retrained=None, use_retrained=False))

# ---------- 5. next tests ----------
with tabs[4]:
    tested_t = ranges["co2_time_min"][2]
    rows = []
    for t in bm.NEXT_TIMES:
        c = curve.iloc[(curve.co2_time_min - t).abs().argmin()]
        rows.append({"CO2 time (min)": t, f"predicted ({unit})": c.predicted, "95% band": f"{c.lower_95:.2f} to {c.upper_95:.2f}",
                     f"band width ({unit})": c.upper_95 - c.lower_95,
                     "chance the best time is within ±0.5 min": f"{np.mean(np.abs(t_opt - t) <= 0.5):.0%}",
                     "already tested": "yes" if np.isclose(tested_t, t).any() else "no"})
    nxt = pd.DataFrame(rows)
    st.markdown(f"**Recommended next CO2 times to test: {', '.join(map(str, bm.NEXT_TIMES))} min** "
                f"(shown for {recipe.replace('_', ', ')}).")
    st.dataframe(nxt.round(2), hide_index=True, width="stretch")
    meaning("the data only have tests at 5, 8, 10 and 15 min around the optimum. When the notebook hid the 8 and "
            "10 min results, most models lost the optimum, so the valley is under-sampled. Testing these times "
            "will confirm where the lowest porosity really is and narrow the 95% range. The 'chance' column shows "
            "how often each time came out best in the 2,000 model curves.")
    st.download_button("Download test plan (CSV)", csv_bytes(nxt.round(4)), f"next_tests_{recipe}.csv", "text/csv")

# ---------- 6. about ----------
with tabs[5]:
    st.markdown(f"""
**Data.** `co2_backfill_train.csv`: {len(base_df)} rows, {base_df.recipe.nunique()} recipes (cement ratio 1:4, 1:6, 1:8;
stirring 500, 700, 800 rpm; CO2 flow 1, 2, 3 L/min; curing 3, 7, 14 days), 10 CO2 times from 0 to 20 min.
Target: porosity (%). Based on Hussain et al., JCP 2025 (CO2 injection into gold-tailings cement paste backfill).

**Main model.** Gaussian process with a quadratic (U-shaped) trend in CO2 time. It gives a prediction and an
uncertainty band, and finds the best CO2 time with a 95% range from 2,000 possible curves.
Seven other models are kept for comparison.

**Limits. Please read.**
- The data look derived from the paper's published results, not raw per-specimen lab data. Cement ratio, stirring and
  flow effects are fixed percentages, so the model cannot learn real interactions between them yet.
- The 95% bands are too narrow: in the honest test they held only 63-77% of hidden points.
- Between 5 and 15 min there are only four tested times, so the optimum is not well pinned down.
- Outside the tested ranges the predictions are unreliable; the app warns you when this happens.

**Honest test scores (leave one CO2 time out, notebook run of 23 Oct 2026):**
""")
    st.dataframe(bm.NOTEBOOK_LOTO, width="stretch")
    meaning("R2 of 1 is perfect and 0 is no better than guessing the average. The best models predict an unseen CO2 "
            "time with an average error of about 0.6-0.7 percentage points of porosity.")
