# -*- coding: utf-8 -*-
"""Tune GI prediction models on the compost maturity dataset.

The script keeps the original Excel file untouched, applies the segment-wise
field calibration described in the project note, and compares several
train/test ratios for random stratified holdout and batch-proxy group holdout.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Compost dateset.xlsx"
OUTPUT_DIR = ROOT / "generated" / "exploratory_tuning"
TARGET = "GI (%)"
SEEDS = [0, 7, 21, 42, 84]
TEST_SIZES = [0.30, 0.25, 0.20]

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

    batch_parts = [
        df["Waste type"].fillna("NA"),
        df["Initial pH"].round(3).astype(str),
        df["Initial carbon to nitrogen"].round(3).astype(str),
        df["Initial moisture content (%)"].round(3).astype(str),
    ]
    df["batch_id"] = batch_parts[0] + "|" + batch_parts[1] + "|" + batch_parts[2] + "|" + batch_parts[3]
    return df


def make_strata(y: pd.Series, bins: int = 10) -> pd.Series | None:
    try:
        strata = pd.qcut(y, q=bins, labels=False, duplicates="drop")
        if pd.Series(strata).value_counts().min() >= 2:
            return strata
    except ValueError:
        pass
    return None


def make_preprocessor(features: list[str], sparse: bool) -> ColumnTransformer:
    numeric_features = [col for col in features if col != "Waste type"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            ("categorical", categorical_pipe, ["Waste type"]),
        ]
    )


def model_factory(name: str, seed: int):
    if name == "RF_leaf10_depth12":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=10,
            max_features=0.75,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "RF_leaf15_depth10":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=15,
            max_features=0.75,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "ET_leaf10_depth12":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=10,
            max_features=0.75,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "ET_leaf15_depth10":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=15,
            max_features=0.75,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "HGB_lite":
        return HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.055,
            max_leaf_nodes=19,
            min_samples_leaf=35,
            l2_regularization=2.0,
            random_state=seed,
        )
    if name == "HGB_mid":
        return HistGradientBoostingRegressor(
            max_iter=320,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=seed,
        )
    if name == "LGBM_cons":
        return LGBMRegressor(
            n_estimators=420,
            learning_rate=0.035,
            num_leaves=19,
            max_depth=5,
            min_child_samples=28,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5,
            reg_alpha=0.1,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "XGB_cons":
        return XGBRegressor(
            n_estimators=420,
            learning_rate=0.035,
            max_depth=4,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5,
            reg_alpha=0.1,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(f"Unknown model: {name}")


def model_names(preset: str) -> list[str]:
    compact = ["HGB_mid", "HGB_lite", "LGBM_cons", "XGB_cons"]
    full = [
        "RF_leaf10_depth12",
        "RF_leaf15_depth10",
        "ET_leaf10_depth12",
        "ET_leaf15_depth10",
        *compact,
    ]
    return compact if preset == "compact" else full


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    seed: int,
    validation: str,
):
    if validation == "random_stratified":
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=make_strata(y))

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def run_experiments(validation: str, preset: str) -> pd.DataFrame:
    df = load_compost_data()
    data = df[df[TARGET].notna()].copy()
    y = data[TARGET].astype(float)
    groups = data["batch_id"]

    rows = []
    for feature_set, features in FEATURE_SETS.items():
        X = data[features].copy()
        for test_size in TEST_SIZES:
            for model_name in model_names(preset):
                for seed in SEEDS:
                    X_train, X_test, y_train, y_test = split_data(X, y, groups, test_size, seed, validation)
                    sparse = model_name.startswith(("RF", "ET"))
                    pipe = Pipeline(
                        [
                            ("preprocessor", make_preprocessor(features, sparse=sparse)),
                            ("model", model_factory(model_name, seed)),
                        ]
                    )
                    pipe.fit(X_train, y_train)
                    train_pred = pipe.predict(X_train)
                    test_pred = pipe.predict(X_test)
                    train_r2 = r2_score(y_train, train_pred)
                    test_r2 = r2_score(y_test, test_pred)
                    rows.append(
                        {
                            "validation": validation,
                            "feature_set": feature_set,
                            "test_size": test_size,
                            "model": model_name,
                            "seed": seed,
                            "train_n": len(X_train),
                            "test_n": len(X_test),
                            "train_r2": train_r2,
                            "test_r2": test_r2,
                            "gap": train_r2 - test_r2,
                            "test_rmse": mean_squared_error(y_test, test_pred, squared=False),
                            "test_mae": mean_absolute_error(y_test, test_pred),
                        }
                    )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["validation", "feature_set", "test_size", "model"], as_index=False)
    summary = grouped.agg(
        train_r2_mean=("train_r2", "mean"),
        train_r2_sd=("train_r2", "std"),
        test_r2_mean=("test_r2", "mean"),
        test_r2_sd=("test_r2", "std"),
        gap_mean=("gap", "mean"),
        gap_sd=("gap", "std"),
        gap_max=("gap", "max"),
        test_r2_min=("test_r2", "min"),
        test_rmse_mean=("test_rmse", "mean"),
        test_mae_mean=("test_mae", "mean"),
        train_n_mean=("train_n", "mean"),
        test_n_mean=("test_n", "mean"),
    )
    return summary.sort_values(["validation", "test_r2_mean"], ascending=[True, False])


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    random_summary = summary[summary["validation"] == "random_stratified"].copy()
    feasible = random_summary[
        (random_summary["gap_mean"].between(0.0, 0.15))
        & (random_summary["gap_max"] <= 0.15)
        & (random_summary["test_r2_mean"] >= 0.80)
    ].sort_values(["test_r2_mean", "gap_mean"], ascending=[False, True])

    group_summary = summary[summary["validation"] == "group_holdout"].copy()
    lines = [
        "# Compost GI model tuning summary",
        "",
        "Data: Compost dateset.xlsx; target: GI (%); seeds: 0, 7, 21, 42, 84.",
        "Preprocessing: segment-wise calibration for Temperature, Moisture, and pH; median numeric imputation; one-hot Waste type.",
        "",
        "## Best random-stratified candidates with gap <= 0.15",
        "",
    ]
    if feasible.empty:
        lines.append("No candidate met the configured gap/test-R2 filters.")
    else:
        lines.append(
            feasible.head(12).to_markdown(
                index=False,
                floatfmt=".3f",
            )
        )

    if not group_summary.empty:
        lines += [
            "",
            "## Best group-holdout candidates",
            "",
            group_summary.head(12).to_markdown(index=False, floatfmt=".3f"),
        ]

    (output_dir / "compost_tuning_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        choices=["random_stratified", "group_holdout", "both"],
        default="both",
    )
    parser.add_argument("--preset", choices=["compact", "full"], default="compact")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    validations = ["random_stratified", "group_holdout"] if args.validation == "both" else [args.validation]
    result_frames = []
    for validation in validations:
        results = run_experiments(validation=validation, preset=args.preset)
        results.to_csv(OUTPUT_DIR / f"compost_tuning_{validation}_raw.csv", index=False, encoding="utf-8-sig")
        result_frames.append(results)

    all_results = pd.concat(result_frames, ignore_index=True)
    summary = summarize(all_results)
    summary.to_csv(OUTPUT_DIR / "compost_tuning_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary, OUTPUT_DIR)
    print(summary.head(20).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
