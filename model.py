# -*- coding: utf-8 -*-
"""
model.py — trains and saves the recommended model for the Sand Production Rate Predictor (Section 4.6).

Model: monotonic-constrained gradient boosting (XGBoost), non-decreasing in the loading factor,
log10 Reynolds number and water cut, unconstrained in the gas–liquid ratio (Equation 3.14), with the
hyper-parameters selected by grid search in Table 4.2. Trained on the 1,975-record synthetic dataset
(Section 3.3.2) to predict log10(1 + SPR), SPR in lb/month (Equation 3.10).

Usage
    python model.py                                   # trains on synthetic_train_data.csv, writes spr_model.joblib
    python model.py my_synthetic.csv my_model.joblib  # custom paths

The app (app.py) calls load_or_train(), so if spr_model.joblib is missing on the server it is
trained automatically from synthetic_train_data.csv on first start (about 10–20 seconds).
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import sklearn
import xgboost
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "synthetic_train_data.csv")
MODEL_FILE = os.path.join(HERE, "spr_model.joblib")
INPUTS = ["LF", "log10_Re", "W", "G"]

# Table 4.2: hyper-parameters selected for the monotonic-constrained gradient boosting model
PARAMS = dict(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=1.0, reg_lambda=5,
              monotone_constraints="(1,1,1,0)", random_state=42, n_jobs=1, verbosity=0)


def features(lf, re, w, g):
    """Model inputs in the order used for training: LF, log10(Re), W (%), G (scf/bbl)."""
    return np.column_stack([np.asarray(lf, float), np.log10(np.asarray(re, float)),
                            np.asarray(w, float), np.asarray(g, float)])


def train(data_file=DATA_FILE, model_file=MODEL_FILE):
    syn = pd.read_csv(data_file)
    X = features(syn["loading_factor_LF"], syn["reynolds_number_Re"], syn["watercut_pct"], syn["glr_scf_per_bbl"])
    y = np.log10(1.0 + syn["sand_production_rate_lb_per_month"].values)
    model = XGBRegressor(**PARAMS).fit(X, y)
    ranges = {"LF": [float(syn["loading_factor_LF"].min()), float(syn["loading_factor_LF"].max())],
              "Re": [float(syn["reynolds_number_Re"].min()), float(syn["reynolds_number_Re"].max())],
              "W": [float(syn["watercut_pct"].min()), float(syn["watercut_pct"].max())],
              "G": [float(syn["glr_scf_per_bbl"].min()), float(syn["glr_scf_per_bbl"].max())]}
    bundle = {"model": model, "ranges": ranges, "inputs": INPUTS, "params": PARAMS,
              "versions": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
                           "numpy": np.__version__, "pandas": pd.__version__}}
    try:
        joblib.dump(bundle, model_file)
    except OSError:      # read-only server folder: keep the model in memory only
        pass
    return bundle


def load_or_train(model_file=MODEL_FILE, data_file=DATA_FILE):
    """Load the saved model; retrain if it is missing or was saved with a different XGBoost version."""
    if os.path.exists(model_file):
        bundle = joblib.load(model_file)
        if bundle.get("versions", {}).get("xgboost") == xgboost.__version__:
            return bundle
    return train(data_file, model_file)


def predict_spr(bundle, lf, re, w, g):
    """Predicted sand production rate in lb/month."""
    ylog = bundle["model"].predict(features(lf, re, w, g))
    return np.clip(10.0 ** ylog - 1.0, 0.0, None)


if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else DATA_FILE
    out = sys.argv[2] if len(sys.argv) > 2 else MODEL_FILE
    b = train(data, out)
    print(f"Saved {out}")
    print("Training ranges:", b["ranges"])
    print("Library versions:", b["versions"])
    # check: first A-64/D4 field record (Table 3.2); the thesis reports 309.1 lb/month
    print("Check, first A-64/D4 record:", round(float(predict_spr(b, 1.70, 0.016611, 73.52, 120)[0]), 1), "lb/month")
