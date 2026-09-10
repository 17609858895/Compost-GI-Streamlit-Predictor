from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generate_compost_figures as gcf  # noqa: E402


OUT_DIR = ROOT / "generated" / "revision"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def metric_row(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def evaluate_feature_sets(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    validation_modes=("random", "group"),
    seeds=gcf.SEEDS,
    label_prefix: str = "",
) -> pd.DataFrame:
    data = df[df[gcf.TARGET].notna()].copy().reset_index(drop=True)
    y = data[gcf.TARGET].astype(float)
    rows = []
    for feature_set, features in feature_sets.items():
        for validation in validation_modes:
            for seed in seeds:
                tr, te = gcf.split_indices(data, y, validation, seed, test_size=0.25)
                pipe = gcf.build_pipe(gcf.PRIMARY_MODEL, features)
                pipe.fit(data.iloc[tr][features], y.iloc[tr])
                pred_train = pipe.predict(data.iloc[tr][features])
                pred_test = pipe.predict(data.iloc[te][features])
                rows.append(
                    {
                        "analysis": label_prefix,
                        "feature_set": feature_set,
                        "validation": validation,
                        "seed": seed,
                        "train_n": len(tr),
                        "test_n": len(te),
                        "train_groups": data.iloc[tr]["batch_id"].nunique(),
                        "test_groups": data.iloc[te]["batch_id"].nunique(),
                        "train_r2": r2_score(y.iloc[tr], pred_train),
                        "test_r2": r2_score(y.iloc[te], pred_test),
                        "gap": r2_score(y.iloc[tr], pred_train) - r2_score(y.iloc[te], pred_test),
                        "rmse": mean_squared_error(y.iloc[te], pred_test, squared=False),
                        "mae": mean_absolute_error(y.iloc[te], pred_test),
                    }
                )
    return pd.DataFrame(rows)


def finite_sample_quantile(errors: np.ndarray, alpha: float) -> float:
    n = len(errors)
    if n == 0:
        return float("nan")
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(errors, q_level, method="higher"))


def split_conformal(df: pd.DataFrame, alpha: float, seeds=gcf.SEEDS) -> pd.DataFrame:
    data = df[df[gcf.TARGET].notna()].copy().reset_index(drop=True)
    y = data[gcf.TARGET].astype(float).reset_index(drop=True)
    features = gcf.FEATURE_SETS["P1"]
    rows = []
    for seed in seeds:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_cal_idx, test_idx = next(outer.split(data, y, groups=data["batch_id"]))
        train_cal = data.iloc[train_cal_idx].reset_index(drop=True)
        y_train_cal = train_cal[gcf.TARGET].astype(float).reset_index(drop=True)
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1000)
        train_idx, cal_idx = next(inner.split(train_cal, y_train_cal, groups=train_cal["batch_id"]))

        pipe = gcf.build_pipe(gcf.PRIMARY_MODEL, features)
        pipe.fit(train_cal.iloc[train_idx][features], y_train_cal.iloc[train_idx])

        cal_pred = pipe.predict(train_cal.iloc[cal_idx][features])
        cal_err = np.abs(cal_pred - y_train_cal.iloc[cal_idx].to_numpy())
        qhat = finite_sample_quantile(cal_err, alpha)

        test = data.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].to_numpy()
        test_pred = pipe.predict(test[features])
        lower = test_pred - qhat
        upper = test_pred + qhat
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        met = metric_row(y_test, test_pred)
        rows.append(
            {
                "alpha": alpha,
                "nominal_coverage": 1 - alpha,
                "seed": seed,
                "proper_train_n": len(train_idx),
                "calibration_n": len(cal_idx),
                "test_n": len(test_idx),
                "proper_train_groups": train_cal.iloc[train_idx]["batch_id"].nunique(),
                "calibration_groups": train_cal.iloc[cal_idx]["batch_id"].nunique(),
                "test_groups": test["batch_id"].nunique(),
                "qhat_gi_units": qhat,
                "interval_width_gi_units": 2 * qhat,
                "coverage": coverage,
                **met,
            }
        )
    return pd.DataFrame(rows)


def shap_group_normalisation(df: pd.DataFrame) -> pd.DataFrame:
    _, _, _, _, _, shap_values, names = gcf.shap_for_primary(df)
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=names)
    group_vals = {"Initial": 0.0, "Process": 0.0, "Chemistry": 0.0, "Waste type": 0.0}
    group_counts = {"Initial": 0, "Process": 0, "Chemistry": 0, "Waste type": 0}
    initial_names = {gcf.label(c) for c in gcf.feature_groups()["Initial"]}
    process_names = {gcf.label(c) for c in gcf.feature_groups()["Process"]}
    chemistry_names = {gcf.label(c) for c in gcf.feature_groups()["Chemistry"]}
    for name, value in mean_abs.items():
        if name.startswith("Waste:"):
            group = "Waste type"
        elif name in initial_names:
            group = "Initial"
        elif name in process_names:
            group = "Process"
        elif name in chemistry_names:
            group = "Chemistry"
        else:
            group = "Chemistry"
        group_vals[group] += float(value)
        group_counts[group] += 1
    total = sum(group_vals.values())
    return pd.DataFrame(
        [
            {
                "group": group,
                "encoded_feature_count": group_counts[group],
                "cumulative_mean_abs_shap": group_vals[group],
                "cumulative_share_percent": group_vals[group] / total * 100 if total else np.nan,
                "per_encoded_feature_mean_abs_shap": group_vals[group] / group_counts[group]
                if group_counts[group]
                else np.nan,
            }
            for group in group_vals
        ]
    )


def main() -> None:
    df = gcf.load_compost_data()
    raw_cols = ["pH", "Temperature (℃)", "Moisture content (%)"]
    corr_cols = ["pH_corr", "Temperature_corr", "Moisture_corr"]

    segment_ranges = []
    for seg_name, sub in df.groupby("segment"):
        for raw_col, corr_col, variable in zip(raw_cols, corr_cols, ["pH", "Temperature", "Moisture"]):
            raw = pd.to_numeric(sub[raw_col], errors="coerce")
            corr = pd.to_numeric(sub[corr_col], errors="coerce")
            segment_ranges.append(
                {
                    "segment": seg_name,
                    "variable": variable,
                    "original_column": raw_col,
                    "calibrated_column": corr_col,
                    "raw_n": raw.notna().sum(),
                    "raw_min": raw.min(),
                    "raw_median": raw.median(),
                    "raw_max": raw.max(),
                    "calibrated_min": corr.min(),
                    "calibrated_median": corr.median(),
                    "calibrated_max": corr.max(),
                }
            )
    segment_ranges = pd.DataFrame(segment_ranges)

    group_counts = df.loc[df[gcf.TARGET].notna(), "batch_id"].value_counts()
    group_summary = pd.DataFrame(
        [
            {
                "gi_records": int(df[gcf.TARGET].notna().sum()),
                "batch_proxy_groups": int(group_counts.size),
                "median_group_size": float(group_counts.median()),
                "mean_group_size": float(group_counts.mean()),
                "max_group_size": int(group_counts.max()),
                "singleton_groups": int((group_counts == 1).sum()),
                "singleton_group_percent": float((group_counts == 1).mean() * 100),
                "two_record_groups": int((group_counts == 2).sum()),
                "two_record_group_percent": float((group_counts == 2).mean() * 100),
                "one_or_two_record_groups": int((group_counts <= 2).sum()),
                "one_or_two_record_group_percent": float((group_counts <= 2).mean() * 100),
                "records_in_one_or_two_record_groups": int(group_counts[group_counts <= 2].sum()),
                "records_in_one_or_two_record_groups_percent": float(group_counts[group_counts <= 2].sum() / group_counts.sum() * 100),
            }
        ]
    )

    source_like_cols = [
        c
        for c in df.columns
        if any(key in str(c).lower() for key in ["doi", "source", "paper", "study", "reactor", "pile", "batch", "reference"])
    ]
    source_columns = pd.DataFrame({"available_source_identifier_columns": source_like_cols or ["None found"]})

    chem_cols = [
        "EC（ms/cm-1）",
        "Ammonia氨mg/kg",
        "Nitrate硝酸盐mg/kg",
        "总氮(%)",
        "总有机碳变化(%)",
        "有机质含量变化(%)",
    ]
    missing_rows = []
    for col in chem_cols:
        row = {"feature": gcf.label(col)}
        row["overall_missing_percent"] = df[col].isna().mean() * 100
        for seg in ["Segment A: calibrated", "Segment B: native"]:
            sub = df[df["segment"] == seg]
            row[f"{seg}_missing_percent"] = sub[col].isna().mean() * 100
            row[f"{seg}_observed_n"] = int(sub[col].notna().sum())
        missing_rows.append(row)
    p1_missingness = pd.DataFrame(missing_rows)

    segment_a = df[df["segment"] == "Segment A: calibrated"].copy()
    segment_a_feature_sets = {"P0": gcf.FEATURE_SETS["P0"], "P1": gcf.FEATURE_SETS["P1"]}
    segment_a_perf = evaluate_feature_sets(segment_a, segment_a_feature_sets, label_prefix="segment_a_only")

    no_day_sets = {
        "P0_no_day": [f for f in gcf.FEATURE_SETS["P0"] if f != "Compost time (day)"],
        "P1_no_day": [f for f in gcf.FEATURE_SETS["P1"] if f != "Compost time (day)"],
        "Day_plus_waste": ["Waste type", "Compost time (day)"],
    }
    no_day_perf = evaluate_feature_sets(df, no_day_sets, label_prefix="day_sensitivity")

    conformal90 = split_conformal(df, alpha=0.10)
    conformal95 = split_conformal(df, alpha=0.05)
    conformal = pd.concat([conformal90, conformal95], ignore_index=True)

    shap_norm = shap_group_normalisation(df)

    outputs = {
        "segment_ranges.csv": segment_ranges,
        "batch_proxy_group_summary.csv": group_summary,
        "source_identifier_columns.csv": source_columns,
        "p1_missingness.csv": p1_missingness,
        "segment_a_p0_p1_control_raw.csv": segment_a_perf,
        "day_sensitivity_raw.csv": no_day_perf,
        "split_conformal_raw.csv": conformal,
        "shap_group_normalisation.csv": shap_norm,
    }
    for filename, table in outputs.items():
        table.to_csv(OUT_DIR / filename, index=False, encoding="utf-8-sig")

    conformal_ranges = conformal.groupby("nominal_coverage")[["coverage", "qhat_gi_units", "interval_width_gi_units"]].agg(["min", "max"]).reset_index()
    conformal_ranges.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in conformal_ranges.columns
    ]

    summary = {
        "segment_ranges": segment_ranges.to_dict(orient="records"),
        "group_summary": group_summary.iloc[0].to_dict(),
        "source_identifier_columns": source_like_cols,
        "p1_missingness": p1_missingness.to_dict(orient="records"),
        "segment_a_perf_mean": segment_a_perf.groupby(["feature_set", "validation"])[["train_r2", "test_r2", "gap", "rmse", "mae"]].mean().reset_index().to_dict(orient="records"),
        "day_sensitivity_mean": no_day_perf.groupby(["feature_set", "validation"])[["train_r2", "test_r2", "gap", "rmse", "mae"]].mean().reset_index().to_dict(orient="records"),
        "split_conformal_mean": conformal.groupby("nominal_coverage")[["coverage", "qhat_gi_units", "interval_width_gi_units", "r2", "rmse", "mae"]].mean().reset_index().to_dict(orient="records"),
        "split_conformal_seed_ranges": conformal_ranges.to_dict(orient="records"),
        "shap_group_normalisation": shap_norm.to_dict(orient="records"),
    }
    (OUT_DIR / "major_revision_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
