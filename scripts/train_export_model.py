# -*- coding: utf-8 -*-
"""Train and export the compost GI prediction model for the Streamlit app."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


APP_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = APP_ROOT.parent
DEFAULT_DATA_PATH = SOURCE_ROOT / "Compost dateset.xlsx"
TARGET = "GI (%)"
FEATURE_SET = "P1"
MODEL_NAME = "XGBoost"
SEEDS = [0, 7, 21, 42, 84]
TEST_SIZE = 0.25

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

FEATURES = P0_FEATURES + [
    "EC（ms/cm-1）",
    "Ammonia氨mg/kg",
    "Nitrate硝酸盐mg/kg",
    "总氮(%)",
    "总有机碳变化(%)",
    "有机质含量变化(%)",
]

DISPLAY_NAMES = {
    "Waste type": "Waste type",
    "Initial pH": "Initial pH",
    "Initial carbon to nitrogen": "Initial C/N",
    "Initial moisture content (%)": "Initial moisture (%)",
    "Compost time (day)": "Compost time (day)",
    "Temperature_corr": "Temperature (deg C)",
    "Moisture_corr": "Moisture (%)",
    "pH_corr": "Process pH",
    "EC（ms/cm-1）": "EC (mS cm-1)",
    "Ammonia氨mg/kg": "Ammonia (mg kg-1)",
    "Nitrate硝酸盐mg/kg": "Nitrate (mg kg-1)",
    "总氮(%)": "Total nitrogen (%)",
    "总有机碳变化(%)": "TOC change (%)",
    "有机质含量变化(%)": "Organic matter change (%)",
}

FEATURE_GROUPS = {
    "Feedstock and initial conditions": [
        "Waste type",
        "Initial pH",
        "Initial carbon to nitrogen",
        "Initial moisture content (%)",
    ],
    "Composting process": [
        "Compost time (day)",
        "Temperature_corr",
        "Moisture_corr",
        "pH_corr",
    ],
    "Laboratory chemistry": [
        "EC（ms/cm-1）",
        "Ammonia氨mg/kg",
        "Nitrate硝酸盐mg/kg",
        "总氮(%)",
        "总有机碳变化(%)",
        "有机质含量变化(%)",
    ],
}


def load_compost_data(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    columns = [str(col).strip() for col in raw.iloc[1].tolist()]
    df = raw.iloc[3:].copy()
    df.columns = columns
    df = df.dropna(how="all").reset_index(drop=True)

    for col in df.columns:
        if col != "Waste type":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Waste type"] = df["Waste type"].astype(str).str.strip().replace({"nan": np.nan})

    shifted_cols = [
        "EC（ms/cm-1）",
        "Ammonia氨mg/kg",
        "Nitrate硝酸盐mg/kg",
        "总氮(%)",
        "总有机碳变化(%)",
        "有机质含量变化(%)",
    ]
    shifted = df[shifted_cols].notna().any(axis=1)
    df["Temperature_corr"] = np.where(shifted, df["pH"], df["Temperature (℃)"])
    df["Moisture_corr"] = np.where(shifted, df["Temperature (℃)"], df["Moisture content (%)"])
    df["pH_corr"] = np.where(shifted, df["Moisture content (%)"], df["pH"])
    df["batch_id"] = (
        df["Waste type"].fillna("NA")
        + "|"
        + df["Initial pH"].round(3).astype(str)
        + "|"
        + df["Initial carbon to nitrogen"].round(3).astype(str)
        + "|"
        + df["Initial moisture content (%)"].round(3).astype(str)
    )
    return df


def q_strata(y: pd.Series) -> pd.Series | None:
    try:
        strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
        if pd.Series(strata).value_counts().min() >= 2:
            return strata
    except Exception:
        return None
    return None


def build_pipeline() -> Pipeline:
    numeric = [feature for feature in FEATURES if feature != "Waste type"]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, ["Waste type"])],
        sparse_threshold=0,
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        colsample_bytree=0.9,
        learning_rate=0.05360192649435464,
        max_depth=6,
        min_child_weight=3,
        n_estimators=150,
        subsample=0.7431844659980276,
        reg_lambda=3.0,
        random_state=1,
        n_jobs=-1,
        verbosity=0,
    )
    return Pipeline([("pre", preprocessor), ("model", model)])


def split_indices(data: pd.DataFrame, y: pd.Series, validation: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(data))
    if validation == "random":
        return train_test_split(idx, test_size=TEST_SIZE, random_state=seed, stratify=q_strata(y))
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    return next(splitter.split(data, y, groups=data["batch_id"]))


def evaluate(data: pd.DataFrame, y: pd.Series, validation: str, seed: int) -> dict:
    tr, te = split_indices(data, y, validation, seed)
    pipe = build_pipeline()
    pipe.fit(data.iloc[tr][FEATURES], y.iloc[tr])
    train_pred = pipe.predict(data.iloc[tr][FEATURES])
    test_pred = pipe.predict(data.iloc[te][FEATURES])
    return {
        "validation": validation,
        "seed": seed,
        "train_n": int(len(tr)),
        "test_n": int(len(te)),
        "train_r2": float(r2_score(y.iloc[tr], train_pred)),
        "test_r2": float(r2_score(y.iloc[te], test_pred)),
        "gap": float(r2_score(y.iloc[tr], train_pred) - r2_score(y.iloc[te], test_pred)),
        "rmse": float(mean_squared_error(y.iloc[te], test_pred, squared=False)),
        "mae": float(mean_absolute_error(y.iloc[te], test_pred)),
    }


def numeric_profile(data: pd.DataFrame) -> dict:
    profile = {}
    for feature in FEATURES:
        if feature == "Waste type":
            continue
        s = pd.to_numeric(data[feature], errors="coerce")
        profile[feature] = {
            "median": float(s.median()),
            "mean": float(s.mean()),
            "min": float(s.min()),
            "max": float(s.max()),
            "q01": float(s.quantile(0.01)),
            "q05": float(s.quantile(0.05)),
            "q95": float(s.quantile(0.95)),
            "q99": float(s.quantile(0.99)),
        }
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", type=Path, default=APP_ROOT)
    args = parser.parse_args()

    df = load_compost_data(args.data)
    data = df[df[TARGET].notna()].copy().reset_index(drop=True)
    y = data[TARGET].astype(float)

    rows = []
    for validation in ["random", "group"]:
        for seed in SEEDS:
            rows.append(evaluate(data, y, validation, seed))
    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby("validation")
        .agg(
            train_r2=("train_r2", "mean"),
            test_r2=("test_r2", "mean"),
            gap=("gap", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            test_r2_sd=("test_r2", "std"),
        )
        .round(4)
        .to_dict(orient="index")
    )

    final_model = build_pipeline()
    final_model.fit(data[FEATURES], y)

    waste_types = sorted([str(v) for v in data["Waste type"].dropna().unique()])
    profile = numeric_profile(data)
    default_row = {feature: profile[feature]["median"] for feature in FEATURES if feature != "Waste type"}
    default_row["Waste type"] = data["Waste type"].mode(dropna=True).iloc[0]

    example = data[FEATURES + [TARGET]].dropna(subset=[TARGET]).sample(min(8, len(data)), random_state=8)
    template = pd.DataFrame([default_row])[FEATURES]

    metadata = {
        "model_name": MODEL_NAME,
        "target": TARGET,
        "target_unit": "%",
        "feature_set": FEATURE_SET,
        "features": FEATURES,
        "display_names": DISPLAY_NAMES,
        "feature_groups": FEATURE_GROUPS,
        "waste_types": waste_types,
        "numeric_profile": profile,
        "training_rows": int(len(data)),
        "training_target_min": float(y.min()),
        "training_target_max": float(y.max()),
        "metrics_mean": summary,
        "metrics_raw": rows,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "caveats": [
            "Predictions are for GI (%) in compost maturity data represented by the training set.",
            "Random-split accuracy reflects within-distribution interpolation; group-holdout accuracy is the conservative robustness estimate.",
            "Inputs outside the training range are flagged and should be interpreted cautiously.",
            "Missing numeric inputs are imputed by the fitted median imputer; unseen waste types are handled by one-hot encoding as unknown categories.",
        ],
    }

    model_dir = args.out / "model"
    data_dir = args.out / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": final_model,
        "metadata": metadata,
        "feature_columns": FEATURES,
        "target": TARGET,
    }
    joblib.dump(bundle, model_dir / "model_bundle.joblib")
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics.to_csv(model_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    example.to_csv(data_dir / "example_input.csv", index=False, encoding="utf-8-sig")
    template.to_csv(data_dir / "single_prediction_template.csv", index=False, encoding="utf-8-sig")
    print(f"Exported model to {model_dir / 'model_bundle.joblib'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
