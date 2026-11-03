# -*- coding: utf-8 -*-
"""
Prototype web application for predicting sand production rate (Section 3.4.4, Figure 3.3).
Run with:  streamlit run app.py
Uses model.py: loads spr_model.joblib, or trains it from synthetic_train_data.csv on first start if it is missing.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sand Production Rate Predictor", layout="wide")
from model import load_or_train


@st.cache_resource(show_spinner="Loading the model…")
def get_model():
    return load_or_train()


def classify(value, caution, alarm):
    """Alarm status of a predicted sand rate against the two thresholds: 'alarm', 'caution' or 'ok'."""
    if value >= alarm:
        return "alarm"
    if value >= caution:
        return "caution"
    return "ok"


bundle = get_model()
model, ranges = bundle["model"], bundle["ranges"]

st.title("Sand Production Rate Predictor")
st.caption("Monotonic-constrained gradient boosting model trained on physics-based synthetic data "
           "(Willson et al., 2002; Udebhulu & Ogbe, 2015). A screening tool, not a substitute for geomechanical analysis.")

mode = st.radio("Input mode", ["Enter the four model inputs directly", "Compute LF and Re from well data"], horizontal=True)
c1, c2 = st.columns([1, 1.2])

with c1:
    st.subheader("Inputs")
    if mode.startswith("Enter"):
        LF = st.number_input("Loading factor, LF (–)", 0.5, 10.0, 1.70, 0.01)
        Re = st.number_input("Reynolds number, Re (–)", 1e-5, 1.0, 0.0200, format="%.5f")
    else:
        st.markdown("**Stress and strength (equations 3.1 to 3.4)**")
        s1 = st.number_input("Maximum horizontal stress, σ1 (psi)", 1000.0, 20000.0, 8500.0, 10.0)
        s2 = st.number_input("Minimum horizontal stress, σ2 (psi)", 1000.0, 20000.0, 5000.0, 10.0)
        pres = st.number_input("Reservoir pressure (psi)", 100.0, 20000.0, 4500.0, 10.0)
        dp = st.number_input("Drawdown (psi)", 0.0, 10000.0, 1500.0, 10.0)
        twc = st.number_input("TWC strength (psi)", 100.0, 20000.0, 2500.0, 10.0)
        nu = st.number_input("Poisson's ratio (–)", 0.05, 0.49, 0.25, 0.01)
        E = st.number_input("Young's modulus (10⁶ psi)", 0.01, 5.0, 0.20, 0.01)
        cr = st.number_input("Grain compressibility (10⁻⁶ psi⁻¹)", 0.1, 10.0, 2.8, 0.1)
        st.markdown("**Flow (equations 3.5 and 3.6)**")
        k = st.number_input("Permeability, k (md)", 1.0, 5000.0, 350.0, 1.0)
        rho = st.number_input("Fluid density, ρ (lb/ft³)", 30.0, 75.0, 55.0, 0.5)
        mu = st.number_input("Fluid viscosity, μ (cp)", 0.1, 20.0, 1.0, 0.1)
        V = st.number_input("Velocity at the perforation surface, V (in/s)", 0.0001, 1.0, 0.04, format="%.4f")
        pwf = max(pres - dp, 50.0)
        Cb = 3 * (1 - 2 * nu) / (E * 1e6); alpha = 1 - cr * 1e-6 / Cb; A = (1 - 2 * nu) * alpha / (1 - nu)
        LF = (3 * s1 - s2 - 2 * pwf - A * (pres - pwf)) / (3.10 * twc)
        beta = 2.65e10 / k ** 1.2; Re = 1.31735e-12 * k * beta * rho * V / mu
        st.info(f"Computed: LF = {LF:.3f}, Re = {Re:.5f} (A = {A:.3f})")
    W = st.number_input("Water cut, W (%)", 0.0, 100.0, 40.0, 0.1)
    G = st.number_input("Gas–liquid ratio, G (scf/bbl)", 0.0, 50000.0, 500.0, 1.0)
    qo = st.number_input("Oil rate (bbl/month) — optional, for pptb", 0.0, 1e7, 15000.0, 100.0)
    with st.expander("Alarm thresholds (set these to your field's sand-handling limit)"):
        basis = st.radio("Threshold unit", ["pptb (lb per 1,000 bbl of oil)", "lb/month"], horizontal=True)
        use_pptb = basis.startswith("pptb")
        caution_lvl = st.number_input("Caution level", 0.0, 1e7, 5.0 if use_pptb else 75.0, 1.0, key=f"caution_{use_pptb}")
        alarm_lvl = st.number_input("Alarm level", 0.0, 1e7, 10.0 if use_pptb else 150.0, 1.0, key=f"alarm_{use_pptb}")
        if alarm_lvl < caution_lvl:
            st.warning("The alarm level should not be below the caution level.")
        st.caption("Default levels are placeholders, not a standard: sand limits depend on the field and its surface "
                   "facilities. 1 pptb is about 1.08 ppmv of sand by volume. Published guidance (SPE 164645-MS) describes "
                   "typical steady-state sand production of 5-10 ppmv, and conventional facilities operating satisfactorily "
                   "below about 5 ppmv between maintenance intervals. In lb/month the defaults equal 5 and 10 pptb at "
                   "15,000 bbl/month of oil. Replace them with your operator's limit.")
    st.markdown("**Calibration to this well (optional)**")
    st.caption("If the well already has measured sand rates, enter at least three months of records "
               "(the four inputs and the measured rate for each). The application computes one correction factor for "
               "the well and applies it to the prediction (Section 4.3.4).")
    cal_default = pd.DataFrame({"LF": [None] * 3, "Re": [None] * 3, "W (%)": [None] * 3, "G (scf/bbl)": [None] * 3,
                                "Measured SPR (lb/month)": [None] * 3}, dtype="float")
    up = st.file_uploader("Upload measured records (CSV with the columns shown below)", type="csv")
    if up is not None:
        cal_default = pd.read_csv(up)[list(cal_default.columns)].astype(float)
    cal = st.data_editor(cal_default, num_rows="dynamic", width="stretch", key=f"cal_{0 if up is None else up.name}")
    run = st.button("Predict", type="primary")

with c2:
    st.subheader("Prediction")
    if run or True:
        x = np.array([[LF, np.log10(max(Re, 1e-12)), W, G]])
        ylog = float(model.predict(x)[0]); spr = max(10 ** ylog - 1.0, 0.0)
        warn = []
        for name, val in [("LF", LF), ("Re", Re), ("W", W), ("G", G)]:
            lo, hi = ranges[name]
            if val < lo or val > hi: warn.append(f"{name} = {val:g} is outside the training range {lo:.4g}–{hi:.4g}")
        if LF < 1.0: warn.append("LF < 1: the formation is below its failure stress; the physics predicts no continuous sanding")
        # per-well correction factor from measured records (mean log-scale residual of the model on those records)
        cal_ok = cal.dropna()
        cal_ok = cal_ok[(cal_ok["Re"] > 0) & (cal_ok["Measured SPR (lb/month)"] >= 0)]
        factor = None
        if len(cal_ok) >= 1:
            Xc = np.column_stack([cal_ok["LF"], np.log10(cal_ok["Re"]), cal_ok["W (%)"], cal_ok["G (scf/bbl)"]])
            delta = float(np.mean(np.log10(1 + cal_ok["Measured SPR (lb/month)"].values) - model.predict(Xc)))
            factor = 10 ** delta
            spr_cal = max(10 ** (ylog + delta) - 1.0, 0.0)
        m1, m2 = st.columns(2)
        if factor is None:
            m1.metric("Sand production rate (uncalibrated)", f"{spr:,.1f} lb/month")
            m2.metric("Sand concentration", f"{spr * 1000 / qo:,.2f} pptb" if qo > 0 else "— (enter oil rate)")
            st.info("Screening estimate for a well without measured sand rates. On the field records of this study, "
                    "about half of such predictions were within a factor of two of the measured rate.")
        else:
            m1.metric("Sand production rate (calibrated)", f"{spr_cal:,.1f} lb/month", delta=f"uncalibrated {spr:,.1f}", delta_color="off")
            m2.metric("Sand concentration", f"{spr_cal * 1000 / qo:,.2f} pptb" if qo > 0 else "— (enter oil rate)")
            msg = f"Correction factor for this well: ×{factor:.2f}, from {len(cal_ok)} measured record(s)."
            (st.success if len(cal_ok) >= 3 else st.warning)(msg + ("" if len(cal_ok) >= 3 else " At least three records are recommended."))
        # ---- alarm: compare the predicted rate (calibrated if available) with the thresholds
        spr_eff = spr_cal if factor is not None else spr
        if use_pptb and qo <= 0:
            st.warning("Alarm thresholds are set in pptb, but no oil rate was entered, so no alarm check was made. "
                       "Enter the oil rate or switch the threshold unit to lb/month.")
        else:
            val = spr_eff * 1000 / qo if use_pptb else spr_eff
            unit = "pptb" if use_pptb else "lb/month"
            status = classify(val, caution_lvl, alarm_lvl)
            if status == "alarm":
                st.error(f"HIGH SAND RATE ALARM: predicted {val:,.1f} {unit} is at or above the alarm level of "
                         f"{alarm_lvl:g} {unit}.", icon="🚨")
            elif status == "caution":
                st.warning(f"CAUTION: predicted {val:,.1f} {unit} is above the caution level of {caution_lvl:g} {unit} "
                           f"(alarm level {alarm_lvl:g} {unit}).", icon="⚠️")
            else:
                st.success(f"Below the caution level: predicted {val:,.1f} {unit} "
                           f"(caution {caution_lvl:g}, alarm {alarm_lvl:g} {unit}).")
            if status != "alarm" and 2 * val >= alarm_lvl > 0:
                st.info(f"Close to the alarm level: if the model under-predicts by a factor of two, the rate would be "
                        f"{2 * val:,.1f} {unit}, at or above the alarm level of {alarm_lvl:g} {unit}. "
                        "Predictions are screening estimates.")
        for w in warn: st.warning(w)
        st.markdown("**Contribution of each input to this prediction (SHAP, log₁₀ units)**")
        # exact TreeSHAP from xgboost itself (same algorithm as shap.TreeExplainer; avoids the shap/xgboost base_score parsing error)
        contrib = model.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[0]
        sv, base_log = contrib[:-1], float(contrib[-1])
        fig, ax = plt.subplots(figsize=(5.5, 2.4))
        labels = ["Loading factor", "Reynolds number", "Water cut", "Gas–liquid ratio"]
        colors = ["#a93226" if v > 0 else "#1f4e79" for v in sv]
        ax.barh(labels, sv, color=colors); ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("SHAP value (log₁₀(1 + SPR))")
        ax.invert_yaxis(); fig.tight_layout(); st.pyplot(fig)
        st.caption(f"Base value (average prediction over the training data): {10 ** base_log - 1:,.0f} lb/month")
