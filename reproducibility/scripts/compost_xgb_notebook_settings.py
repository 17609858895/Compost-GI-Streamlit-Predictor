# -*- coding: utf-8 -*-
"""Run compost GI prediction with the XGBoost settings from D:/Jupyter/XGBoost.ipynb.

The notebook's final anti-overfitting setup is adapted to the compost dataset:
KNN imputation, Box-Cox transform, standardization, XGBoost, and Bayesian
hyperparameter search. Preprocessing is fitted inside the training/CV pipeline
to avoid leakage.
"""

from __future__ import annotations

import os
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PowerTransformer, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "1"
np.random.seed(1)
random.seed(1)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Compost dateset.xlsx"
OUTPUT_DIR = ROOT / "generated" / "exploratory_tuning"
TARGET = "GI (%)"
TEST_SIZES = [0.30, 0.25, 0.20]
SEED = 1
NOTEBOOK_BEST_PARAMS = {
    "colsample_bytree": 0.9,
    "learning_rate": 0.05360192649435464,
    "max_depth": 6,
    "min_child_weight": 3,
    "n_estimators": 150,
    "subsample": 0.7431844659980276,
}

P0_FEATURES = [
    "Waste type",
    "Initial pH",
    "Initial carbon to nitrogen",
    "Initial moisture content (%)",
    "Compost time (day)",
    "Temperature_corr",
    "Moisture_corr",
    "pH_corr",
]

FEATURE_SETS = {
    "P0": P0_FEATURES,
    "P0_EC": P0_FEATURES + ["EC（ms/cm-1）"],
    "P1": P0_FEATURES
    + [
        "EC（ms/cm-1）",
        "Ammonia氨mg/kg",
        "Nitrate硝酸盐mg/kg",
        "总氮(%)",
        "总有机碳变化(%)",
        "有机质含量变化(%)",
    ],
}


def make_positive(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0, 1e-5, values)


def load_compost_data(path: Path = DATA_PATH) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    columns = [str(col).strip() for col in raw.iloc[1].tolist()]
    df = raw.iloc[3:].copy()
    df.columns = columns
    df = df.dropna(how="all").reset_index(drop=True)

    for col in df.columns:
        if col != "Waste type":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Waste type"] = df["Waste type"].astype(str).str.strip().replace({"nan": np.nan})

    shifted_columns = [
        "EC（ms/cm-1）",
        "Ammonia氨mg/kg",
        "Nitrate硝酸盐mg/kg",
        "总氮(%)",
        "总有机碳变化(%)",
        "有机质含量变化(%)",
    ]
    shifted = df[shifted_columns].notna().any(axis=1)
    df["Temperature_corr"] = np.where(shifted, df["pH"], df["Temperature (℃)"])
    df["Moisture_corr"] = np.where(shifted, df["Temperature (℃)"], df["Moisture content (%)"])
    df["pH_corr"] = np.where(shifted, df["Moisture content (%)"], df["pH"])
    return df


def build_pipeline(features: list[str]) -> Pipeline:
    numeric_features = [feature for feature in features if feature != "Waste type"]
    numeric_pipe = Pipeline(
        [
            ("imputer", KNNImputer(n_neighbors=5)),
            ("positive", FunctionTransformer(make_positive, validate=False)),
            ("boxcox", PowerTransformer(method="box-cox")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, ["Waste type"]),
        ],
        sparse_threshold=0,
    )
    model = XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_fitted(model: Pipeline, X_train, X_test, y_train, y_test, extra: dict) -> dict:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    return {
        **extra,
        "train_n": len(X_train),
        "test_n": len(X_test),
        "train_r2": train_r2,
        "test_r2": test_r2,
        "gap": train_r2 - test_r2,
        "train_rmse": mean_squared_error(y_train, train_pred, squared=False),
        "test_rmse": mean_squared_error(y_test, test_pred, squared=False),
        "train_mae": mean_absolute_error(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
    }


def run_one(
    feature_set: str,
    features: list[str],
    test_size: float,
    X: pd.DataFrame,
    y: pd.Series,
    mode: str,
    n_iter: int,
) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X[features],
        y,
        test_size=test_size,
        random_state=SEED,
    )
    pipeline = build_pipeline(features)
    if mode == "fixed":
        pipeline.set_params(**{f"model__{key}": value for key, value in NOTEBOOK_BEST_PARAMS.items()})
        pipeline.fit(X_train, y_train)
        return evaluate_fitted(
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
            {
                "mode": mode,
                "feature_set": feature_set,
                "test_size": test_size,
                "best_params": NOTEBOOK_BEST_PARAMS,
            },
        )

    search_space = {
        "model__learning_rate": Real(0.01, 0.1),
        "model__n_estimators": Integer(50, 150),
        "model__max_depth": Integer(1, 10),
        "model__min_child_weight": Integer(1, 10),
        "model__subsample": Real(0.6, 0.9),
        "model__colsample_bytree": Real(0.6, 0.9),
    }
    optimizer = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=5,
        random_state=SEED,
        n_jobs=1,
    )
    optimizer.fit(X_train, y_train)

    best_params = {key.replace("model__", ""): value for key, value in optimizer.best_params_.items()}
    return evaluate_fitted(
        optimizer,
        X_train,
        X_test,
        y_train,
        y_test,
        {
        "mode": mode,
        "feature_set": feature_set,
        "test_size": test_size,
        "best_params": best_params,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixed", "search"], default="fixed")
    parser.add_argument("--n-iter", type=int, default=32)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_compost_data()
    data = df[df[TARGET].notna()].copy()
    y = data[TARGET].astype(float)

    rows = []
    for feature_set, features in FEATURE_SETS.items():
        for test_size in TEST_SIZES:
            print(f"Running {feature_set}, test_size={test_size} ...", flush=True)
            rows.append(run_one(feature_set, features, test_size, data, y, args.mode, args.n_iter))

    results = pd.DataFrame(rows).sort_values(["test_r2", "gap"], ascending=[False, True])
    output_csv = OUTPUT_DIR / f"compost_xgb_notebook_settings_{args.mode}.csv"
    results.to_csv(output_csv, index=False, encoding="utf-8-sig")

    report_lines = [
        "# Compost XGBoost notebook-settings results",
        "",
        "Adapted from D:/Jupyter/XGBoost.ipynb final XGBoost setup.",
        f"Mode = {args.mode}; seed = 1; BayesSearchCV n_iter = {args.n_iter} when mode=search.",
        "",
        results.to_markdown(index=False, floatfmt=".3f"),
    ]
    (OUTPUT_DIR / f"compost_xgb_notebook_settings_{args.mode}.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
