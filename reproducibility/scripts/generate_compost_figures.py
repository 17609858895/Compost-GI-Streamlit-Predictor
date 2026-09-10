# -*- coding: utf-8 -*-
"""Generate Nature-style figures for the compost maturity ML paper.

Outputs are written to figures_compost/FigXX_*/ as PNG and PDF files. The code
uses conservative model settings and group-aware validation for the main claims.
"""

from __future__ import annotations

import json
import math
import random
import warnings
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
random.seed(1)
np.random.seed(1)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Compost dateset.xlsx"
OUT = ROOT / "generated" / "figures_compost"
DATA_OUT = OUT / "_data"
TARGET = "GI (%)"
SEEDS = [0, 7, 21, 42, 84]
PRIMARY_MODEL = "XGBoost"
CONSERVATIVE_MODEL = "CatBoost"
ACCURACY_UPPER_MODEL = "XGBoost"
WITHIN_DISTRIBUTION_MODEL = "HGB"
PRIMARY_FEATURE_SET = "P1"
PRIMARY_GAP_LIMIT = 0.15

NATURE = {
    "blue": "#6EC6DC",
    "green": "#4DB6AC",
    "red": "#E76F61",
    "navy": "#5874A6",
    "coral": "#F4A582",
    "teal": "#9DD7D1",
    "lavender": "#A7B3D3",
    "brown": "#9C7A62",
    "sand": "#F1D36B",
    "grey": "#8A8F93",
    "light": "#F7FAFA",
    "ink": "#2B2B2B",
}

WASTE_COLORS = {
    "poultry manure": NATURE["green"],
    "pig manure": NATURE["blue"],
    "livestock manure compost": NATURE["navy"],
    "sludge waste composting": NATURE["coral"],
    "Composting kitchen waste": NATURE["sand"],
    "sewage sludge": NATURE["lavender"],
    "food waste": NATURE["teal"],
    "green waste": "#8BC34A",
    "sheep manure": NATURE["brown"],
    "cattle manure": NATURE["red"],
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
    "P0+EC": P0_FEATURES + ["EC（ms/cm-1）"],
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

TARGETS = {
    "GI": "GI (%)",
    "C/N": "C/N",
    "NH4-N": "NH4+-N (g/Kg)",
    "NO3-N": "NO3--N (g/Kg)",
    "CO2": "CO2 (g/Kg)",
    "CH4": "CH4 (g/Kg)",
    "N2O": "N2O (g/Kg)",
    "NH3": "NH3 (g/Kg)",
}

TARGET_LABELS = {
    "GI (%)": "GI (%)",
    "C/N": "C/N",
    "NH4+-N (g/Kg)": r"$\mathrm{NH_4^+-N}$ (g kg$^{-1}$)",
    "NO3--N (g/Kg)": r"$\mathrm{NO_3^--N}$ (g kg$^{-1}$)",
    "CO2 (g/Kg)": r"$\mathrm{CO_2}$ (g kg$^{-1}$)",
    "CH4 (g/Kg)": r"$\mathrm{CH_4}$ (g kg$^{-1}$)",
    "N2O (g/Kg)": r"$\mathrm{N_2O}$ (g kg$^{-1}$)",
    "NH3 (g/Kg)": r"$\mathrm{NH_3}$ (g kg$^{-1}$)",
}

SHORT_LABELS = {
    "GI": "GI",
    "C/N": "C/N",
    "NH4-N": r"$\mathrm{NH_4^+-N}$",
    "NO3-N": r"$\mathrm{NO_3^--N}$",
    "CO2": r"$\mathrm{CO_2}$",
    "CH4": r"$\mathrm{CH_4}$",
    "N2O": r"$\mathrm{N_2O}$",
    "NH3": r"$\mathrm{NH_3}$",
}

DISPLAY = {
    "Initial pH": "Initial pH",
    "Initial carbon to nitrogen": "Initial C/N",
    "Initial moisture content (%)": "Initial MC",
    "Compost time (day)": "Day",
    "Temperature_corr": "Temperature",
    "Moisture_corr": "Moisture",
    "pH_corr": "pH",
    "EC（ms/cm-1）": "EC",
    "Ammonia氨mg/kg": "Ammonia",
    "Nitrate硝酸盐mg/kg": "Nitrate",
    "总氮(%)": "TN",
    "总有机碳变化(%)": "TOC change",
    "有机质含量变化(%)": "OM change",
    "Waste type": "Waste type",
}


def configure_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.4,
            "axes.labelweight": "bold",
            "axes.labelsize": 19.2,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "legend.fontsize": 14.5,
            "axes.grid": False,
        }
    )


def style_axis(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(NATURE["ink"])
        ax.spines[spine].set_linewidth(1.4)
    ax.tick_params(axis="both", width=1.35, length=6.4, colors=NATURE["ink"], direction="out", bottom=True, left=True)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold")


def panel_label(ax, label: str) -> None:
    ax._panel_label = label


def export_subfigures(fig: plt.Figure, path: Path, stem: str) -> None:
    subdir = path / "subfigures"
    subdir.mkdir(parents=True, exist_ok=True)
    panels = [(ax, getattr(ax, "_panel_label")) for ax in fig.axes if hasattr(ax, "_panel_label")]
    if not panels:
        (subdir / "README.md").write_text("No labelled panels were detected for subfigure export.\n", encoding="utf-8")
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lines = []
    for ax, letter in panels:
        bbox = ax.get_tightbbox(renderer)
        # Include adjacent colorbar axes when they share the same vertical band.
        for other in fig.axes:
            if other is ax or hasattr(other, "_panel_label"):
                continue
            obox = other.get_tightbbox(renderer)
            vertical_overlap = max(0, min(bbox.y1, obox.y1) - max(bbox.y0, obox.y0))
            overlap_frac = vertical_overlap / max(1, min(bbox.height, obox.height))
            close_right = 0 <= obox.x0 - bbox.x1 <= 140
            horizontal_overlap = max(0, min(bbox.x1, obox.x1) - max(bbox.x0, obox.x0))
            h_overlap_frac = horizontal_overlap / max(1, min(bbox.width, obox.width))
            close_below = 0 <= bbox.y0 - obox.y1 <= 90
            if (close_right and overlap_frac > 0.45) or (close_below and h_overlap_frac > 0.45):
                bbox = Bbox.union([bbox, obox])
        bbox = bbox.expanded(1.08, 1.12).transformed(fig.dpi_scale_trans.inverted())
        png = subdir / f"{stem}_{letter}.png"
        pdf = subdir / f"{stem}_{letter}.pdf"
        fig.savefig(png, dpi=600, bbox_inches=bbox, facecolor="white")
        fig.savefig(pdf, bbox_inches=bbox, facecolor="white")
        lines.append(f"- {letter}: `{png.name}`, `{pdf.name}`")
    (subdir / "README.md").write_text("# Subfigures\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def numeric_ticks(ax, x: bool = True, y: bool = True, n: int = 5) -> None:
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n))


def save_figure(fig: plt.Figure, folder: str, stem: str) -> None:
    path = OUT / folder
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(path / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    export_subfigures(fig, path, stem)
    plt.close(fig)


def load_compost_data() -> pd.DataFrame:
    raw = pd.read_excel(DATA_PATH, header=None)
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
    df["segment"] = np.where(shifted, "Segment A: calibrated", "Segment B: native")
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


def feature_groups() -> dict[str, list[str]]:
    return {
        "Initial": ["Initial pH", "Initial carbon to nitrogen", "Initial moisture content (%)"],
        "Process": ["Compost time (day)", "Temperature_corr", "Moisture_corr", "pH_corr"],
        "Chemistry": ["EC（ms/cm-1）", "Ammonia氨mg/kg", "Nitrate硝酸盐mg/kg", "总氮(%)", "总有机碳变化(%)", "有机质含量变化(%)"],
        "Gas": ["CO2 (g/Kg)", "CH4 (g/Kg)", "N2O (g/Kg)", "NH3 (g/Kg)"],
        "Maturity": ["GI (%)", "C/N", "NH4+-N (g/Kg)", "NO3--N (g/Kg)"],
    }


def label(col: str) -> str:
    return DISPLAY.get(col, TARGET_LABELS.get(col, col))


def safe_token(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def model_library() -> dict[str, object]:
    return {
        "Ridge": Ridge(alpha=3.0),
        "KNN": KNeighborsRegressor(n_neighbors=18, weights="distance"),
        "SVR": SVR(C=12.0, epsilon=2.0, gamma="scale"),
        "RF": RandomForestRegressor(
            n_estimators=260, max_depth=10, min_samples_leaf=12, max_features=0.75, random_state=1, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=320, max_depth=11, min_samples_leaf=10, max_features=0.75, random_state=1, n_jobs=-1
        ),
        "HGB": HistGradientBoostingRegressor(
            max_iter=260, learning_rate=0.055, max_leaf_nodes=19, min_samples_leaf=35, l2_regularization=2.0, random_state=1
        ),
        "XGBoost": xgb.XGBRegressor(
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
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=260,
            learning_rate=0.04,
            num_leaves=19,
            max_depth=5,
            min_child_samples=28,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            reg_alpha=0.1,
            random_state=1,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": cb.CatBoostRegressor(
            iterations=260,
            learning_rate=0.045,
            depth=5,
            l2_leaf_reg=8.0,
            loss_function="RMSE",
            random_seed=1,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def make_preprocessor(features: list[str], scale: bool = False) -> ColumnTransformer:
    numeric = [f for f in features if f != "Waste type"]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, ["Waste type"])],
        sparse_threshold=0,
    )


def build_pipe(model_name: str, features: list[str]) -> Pipeline:
    scale = model_name in {"Ridge", "KNN", "SVR"}
    return Pipeline([("pre", make_preprocessor(features, scale=scale)), ("model", clone(model_library()[model_name]))])


def q_strata(y: pd.Series) -> pd.Series | None:
    try:
        strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
        if pd.Series(strata).value_counts().min() >= 2:
            return strata
    except Exception:
        return None
    return None


def split_indices(data: pd.DataFrame, y: pd.Series, validation: str, seed: int, test_size: float = 0.25):
    idx = np.arange(len(data))
    if validation == "random":
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=q_strata(y))
        return tr, te
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(splitter.split(data, y, groups=data["batch_id"]))
    return tr, te


def evaluate_models(df: pd.DataFrame, target: str = TARGET, feature_set: str = "P1", force: bool = False) -> pd.DataFrame:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    cache = DATA_OUT / f"model_comparison_{target.replace('/', '_')}_{feature_set}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    data = df[df[target].notna()].copy().reset_index(drop=True)
    y = data[target].astype(float)
    features = FEATURE_SETS[feature_set]
    rows = []
    for validation in ["random", "group"]:
        for seed in SEEDS:
            tr, te = split_indices(data, y, validation, seed, test_size=0.25)
            for model_name in model_library():
                pipe = build_pipe(model_name, features)
                pipe.fit(data.iloc[tr][features], y.iloc[tr])
                pred_train = pipe.predict(data.iloc[tr][features])
                pred_test = pipe.predict(data.iloc[te][features])
                train_r2 = r2_score(y.iloc[tr], pred_train)
                test_r2 = r2_score(y.iloc[te], pred_test)
                rows.append(
                    {
                        "validation": validation,
                        "seed": seed,
                        "model": model_name,
                        "feature_set": feature_set,
                        "target": target,
                        "train_n": len(tr),
                        "test_n": len(te),
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "gap": train_r2 - test_r2,
                        "test_rmse": mean_squared_error(y.iloc[te], pred_test, squared=False),
                        "test_mae": mean_absolute_error(y.iloc[te], pred_test),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def evaluate_primary_feature_sets(df: pd.DataFrame, target: str = TARGET, force: bool = False) -> pd.DataFrame:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    cache = DATA_OUT / f"primary_feature_sets_{safe_token(PRIMARY_MODEL)}_{safe_token(target)}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    data = df[df[target].notna()].copy().reset_index(drop=True)
    y = data[target].astype(float)
    rows = []
    for feature_set, features in FEATURE_SETS.items():
        for validation in ["random", "group"]:
            for seed in SEEDS:
                tr, te = split_indices(data, y, validation, seed, test_size=0.25)
                pipe = build_pipe(PRIMARY_MODEL, features)
                pipe.fit(data.iloc[tr][features], y.iloc[tr])
                pred_train = pipe.predict(data.iloc[tr][features])
                pred_test = pipe.predict(data.iloc[te][features])
                train_r2 = r2_score(y.iloc[tr], pred_train)
                test_r2 = r2_score(y.iloc[te], pred_test)
                rows.append(
                    {
                        "feature_set": feature_set,
                        "validation": validation,
                        "seed": seed,
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "gap": train_r2 - test_r2,
                        "test_rmse": mean_squared_error(y.iloc[te], pred_test, squared=False),
                        "test_mae": mean_absolute_error(y.iloc[te], pred_test),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def primary_fit(df: pd.DataFrame, target: str = TARGET, feature_set: str = PRIMARY_FEATURE_SET, seed: int = 1, model_name: str = PRIMARY_MODEL):
    data = df[df[target].notna()].copy().reset_index(drop=True)
    y = data[target].astype(float)
    features = FEATURE_SETS[feature_set]
    tr, te = split_indices(data, y, "group", seed, test_size=0.25)
    pipe = build_pipe(model_name, features)
    pipe.fit(data.iloc[tr][features], y.iloc[tr])
    train = data.iloc[tr].copy()
    test = data.iloc[te].copy()
    train["y_true"] = y.iloc[tr].values
    test["y_true"] = y.iloc[te].values
    train["y_pred"] = pipe.predict(train[features])
    test["y_pred"] = pipe.predict(test[features])
    return pipe, features, train, test, data


def fig1_data_audit(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16.6, 9.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    stages = ["Excel rows", "Valid records", "Segment A", "Segment B", "GI target", "GI groups"]
    values = [4251, len(df), int((df["segment"] == "Segment A: calibrated").sum()), int((df["segment"] == "Segment B: native").sum()), int(df[TARGET].notna().sum()), int(df.loc[df[TARGET].notna(), "batch_id"].nunique())]
    x = np.arange(len(stages))
    ax.plot(x, values, color=NATURE["navy"], lw=2.6, marker="o", ms=9, mfc="white", mec=NATURE["navy"], mew=2)
    ax.fill_between(x, values, 0, color=NATURE["blue"], alpha=0.10)
    for xi, value in zip(x, values):
        ax.text(xi, value + max(values) * 0.025, f"{value:,}", ha="center", va="bottom", fontsize=13.1, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=24, ha="right")
    ax.set_ylim(0, max(values) * 1.16)
    style_axis(ax, ylabel="Records")
    numeric_ticks(ax, x=False, y=True)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    raw_long = df[["pH", "Temperature (℃)", "Moisture content (%)"]].rename(
        columns={"Temperature (℃)": "Temperature", "Moisture content (%)": "Moisture"}
    )
    corrected = df[["pH_corr", "Temperature_corr", "Moisture_corr"]].rename(
        columns={"pH_corr": "pH", "Temperature_corr": "Temperature", "Moisture_corr": "Moisture"}
    )
    range_rows = []
    for state, table in [("Raw", raw_long), ("Calibrated", corrected)]:
        for col in ["pH", "Temperature", "Moisture"]:
            s = pd.to_numeric(table[col], errors="coerce")
            range_rows.append({"State": state, "Variable": col, "Min": s.min(), "Max": s.max(), "Median": s.median()})
    ranges = pd.DataFrame(range_rows)
    ypos = np.arange(3)
    for offset, state, color in [(-0.16, "Raw", NATURE["coral"]), (0.16, "Calibrated", NATURE["green"])]:
        sub = ranges[ranges["State"] == state].set_index("Variable").loc[["pH", "Temperature", "Moisture"]]
        ax.hlines(ypos + offset, sub["Min"], sub["Max"], color=color, lw=4, alpha=0.86, label=state)
        ax.scatter(sub["Median"], ypos + offset, s=52, facecolor="white", edgecolor=color, lw=1.6, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(["pH", "Temperature", "Moisture"])
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax, xlabel="Observed range")
    numeric_ticks(ax, x=True, y=False)
    ax.xaxis.label.set_size(18.4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(15.8)
        tick.set_fontweight("bold")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    missing_cols = ["Initial pH", "Initial carbon to nitrogen", "Initial moisture content (%)", "Compost time (day)", "pH_corr", "Temperature_corr", "Moisture_corr", "EC（ms/cm-1）", "Ammonia氨mg/kg", "Nitrate硝酸盐mg/kg", "GI (%)", "C/N", "CO2 (g/Kg)", "NH3 (g/Kg)"]
    miss_labels = {
        "Initial pH": "Initial pH",
        "Initial carbon to nitrogen": "Initial C/N",
        "Initial moisture content (%)": "Initial MC",
        "Compost time (day)": "Day",
        "pH_corr": "pH",
        "Temperature_corr": "Temp.",
        "Moisture_corr": "Moisture",
        "EC（ms/cm-1）": "EC",
        "Ammonia氨mg/kg": "Ammonia",
        "Nitrate硝酸盐mg/kg": "Nitrate",
        "GI (%)": "GI",
        "C/N": "C/N",
        "CO2 (g/Kg)": r"$\mathrm{CO_2}$",
        "NH3 (g/Kg)": r"$\mathrm{NH_3}$",
    }
    miss = df.groupby("segment")[missing_cols].apply(lambda x: x.isna().mean() * 100)
    miss.columns = [miss_labels.get(c, label(c)) for c in miss.columns]
    hm = sns.heatmap(
        miss,
        cmap=sns.light_palette(NATURE["red"], as_cmap=True),
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 10.4, "fontweight": "bold"},
        cbar_kws={"label": "Missing (%)", "shrink": 0.82},
        linewidths=0.8,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=48, ha="right", rotation_mode="anchor", fontsize=11.3)
    ax.tick_params(axis="y", rotation=0)
    hm.collections[0].colorbar.ax.tick_params(labelsize=10.8, width=1.0, length=4, direction="out")
    hm.collections[0].colorbar.set_label("Missing (%)", fontweight="bold", fontsize=12.2)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    counts = df.loc[df[TARGET].notna(), "batch_id"].value_counts()
    bins = np.arange(1, min(counts.max(), 32) + 2) - 0.5
    ax.hist(np.clip(counts.values, 1, 32), bins=bins, color=NATURE["blue"], alpha=0.78, edgecolor="white")
    ax.axvline(counts.median(), color=NATURE["red"], lw=1.8, ls="--", label=f"Median = {counts.median():.0f}")
    ax.text(0.98, 0.92, f"Groups = {counts.size}\nMax size = {counts.max()}", transform=ax.transAxes, ha="right", va="top", fontsize=13.0, fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.16), ncol=3, fontsize=11.6, handletextpad=0.45, columnspacing=1.0)
    style_axis(ax, xlabel="Records per batch-proxy group", ylabel="Group count")
    numeric_ticks(ax, x=True, y=True)
    ax.xaxis.label.set_size(18.4)
    ax.yaxis.label.set_size(18.4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(15.8)
        tick.set_fontweight("bold")
    panel_label(ax, "d")
    save_figure(fig, "Fig01_data_audit", "Fig01_data_audit")


def fig2_target_distribution(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(18.0, 10.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.40)
    target_cols = list(TARGETS.values())

    ax = fig.add_subplot(gs[0, 0])
    violin_rows = []
    for col in ["GI (%)", "C/N", "NH4+-N (g/Kg)", "NO3--N (g/Kg)"]:
        s = df[col].dropna()
        if len(s) > 20:
            iqr = s.quantile(0.75) - s.quantile(0.25)
            scaled = (s - s.median()) / (iqr if iqr else s.std())
            scaled = scaled.clip(-4, 4)
            violin_rows.append(pd.DataFrame({"Target": TARGET_LABELS[col], "Robust scaled value": scaled}))
    violin = pd.concat(violin_rows, ignore_index=True)
    sns.violinplot(
        data=violin,
        y="Target",
        x="Robust scaled value",
        inner="quartile",
        linewidth=1.0,
        palette=[NATURE["green"], NATURE["blue"], NATURE["coral"], NATURE["lavender"]],
        ax=ax,
    )
    ax.axvline(0, color=NATURE["ink"], lw=1.1, ls="--")
    style_axis(ax, xlabel="Robust scaled value", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    availability = pd.Series({name: df[col].notna().sum() for name, col in TARGETS.items()}).sort_values()
    colors = [NATURE["green"] if v >= 2000 else NATURE["blue"] if v >= 900 else NATURE["coral"] for v in availability.values]
    ax.barh([SHORT_LABELS.get(i, i) for i in availability.index], availability.values, color=colors, edgecolor="white", lw=1.0)
    for yi, value in enumerate(availability.values):
        ax.text(value + 35, yi, f"{int(value)}", va="center", fontsize=12.5, fontweight="bold")
    style_axis(ax, xlabel="Available records", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    panel_label(ax, "b")

    gi = df[df["GI (%)"].notna()].copy()
    ax = fig.add_subplot(gs[1, 0])
    top_waste = gi["Waste type"].value_counts().head(7).index.tolist()
    gi_plot = gi[gi["Waste type"].isin(top_waste)].copy()
    order = gi_plot.groupby("Waste type")["GI (%)"].median().sort_values().index.tolist()
    sns.boxplot(data=gi_plot, x="GI (%)", y="Waste type", order=order, color="white", fliersize=0, linewidth=1.2, ax=ax)
    sns.stripplot(
        data=gi_plot.sample(min(900, len(gi_plot)), random_state=1),
        x="GI (%)",
        y="Waste type",
        order=order,
        palette=WASTE_COLORS,
        size=2.3,
        alpha=0.38,
        ax=ax,
    )
    style_axis(ax, xlabel="GI (%)", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[0, 2])
    sample = gi.sample(min(1800, len(gi)), random_state=1)
    colors = sample["Waste type"].map(WASTE_COLORS).fillna(NATURE["grey"])
    ax.scatter(sample["Compost time (day)"], sample["GI (%)"], c=colors, s=18, alpha=0.50, edgecolors="none")
    x = sample["Compost time (day)"].to_numpy()
    y = sample["GI (%)"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 20:
        xs = np.linspace(np.nanquantile(x[ok], 0.02), np.nanquantile(x[ok], 0.98), 150)
        bins = pd.qcut(x[ok], q=20, duplicates="drop")
        smooth = pd.DataFrame({"x": x[ok], "y": y[ok], "bin": bins}).groupby("bin", observed=True).agg(x=("x", "median"), y=("y", "median"))
        ax.plot(smooth["x"], smooth["y"], color=NATURE["ink"], lw=2.0)
    style_axis(ax, xlabel="Compost time (day)", ylabel="GI (%)")
    numeric_ticks(ax, x=True, y=True)
    panel_label(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    co_targets = ["GI", "C/N", "NH4-N", "NO3-N"]
    co_cols = [TARGETS[t] for t in co_targets]
    co = pd.DataFrame(index=co_targets, columns=co_targets, dtype=float)
    for a_name, a_col in zip(co_targets, co_cols):
        for b_name, b_col in zip(co_targets, co_cols):
            co.loc[a_name, b_name] = df[[a_col, b_col]].dropna().shape[0]
    hm = sns.heatmap(
        co,
        cmap=sns.light_palette(NATURE["navy"], as_cmap=True),
        annot=True,
        fmt=".0f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Co-observed n", "shrink": 0.76},
        ax=ax,
    )
    ax.set_xticklabels([SHORT_LABELS.get(t, t) for t in co_targets], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticklabels([SHORT_LABELS.get(t, t) for t in co_targets], rotation=0)
    ax.tick_params(axis="x", pad=7)
    hm.collections[0].colorbar.ax.tick_params(labelsize=10.8, width=1.0, length=4, direction="out")
    hm.collections[0].colorbar.set_label("Co-observed n", fontweight="bold", fontsize=12.0)
    style_axis(ax, xlabel="", ylabel="")
    panel_label(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    seg_targets = ["GI", "C/N", "NH4-N", "NO3-N", "CO2", "NH3", "CH4", "N2O"]
    seg_rows = []
    for target_name in seg_targets:
        col = TARGETS[target_name]
        counts = df.groupby("segment")[col].apply(lambda s: s.notna().sum())
        seg_rows.append(
            {
                "Target": target_name,
                "Segment A": counts.get("Segment A: calibrated", 0),
                "Segment B": counts.get("Segment B: native", 0),
            }
        )
    seg_avail = pd.DataFrame(seg_rows).set_index("Target")
    y_pos = np.arange(len(seg_avail))
    left = np.zeros(len(seg_avail))
    for seg, color in [("Segment A", NATURE["teal"]), ("Segment B", NATURE["lavender"])]:
        ax.barh(y_pos, seg_avail[seg].values, left=left, color=color, edgecolor="white", label=seg)
        left += seg_avail[seg].values
    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_LABELS.get(t, t) for t in seg_avail.index])
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.62, -0.16), ncol=2, fontsize=11.8)
    style_axis(ax, xlabel="Available records by segment", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    panel_label(ax, "f")
    save_figure(fig, "Fig02_target_distribution", "Fig02_target_distribution")


def corr_pvalues(data: pd.DataFrame, method: str) -> pd.DataFrame:
    cols = data.columns
    p = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                p.loc[a, b] = 0.0
                continue
            sub = data[[a, b]].dropna()
            if len(sub) < 8:
                continue
            try:
                if method == "pearson":
                    _, pv = stats.pearsonr(sub[a], sub[b])
                else:
                    _, pv = stats.spearmanr(sub[a], sub[b])
                p.loc[a, b] = pv
            except Exception:
                p.loc[a, b] = 1.0
    return p


def star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def draw_grouped_corr(
    ax,
    data: pd.DataFrame,
    method: str,
    group_spans: list[tuple[str, int, int]],
    cbar_label: str,
    cbar_ax=None,
    compact: bool = False,
) -> None:
    corr = data.corr(method=method)
    pvals = corr_pvalues(data, method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    labels = [label(c) for c in corr.columns]
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("white")
    values = np.ma.array(corr.values, mask=mask)
    im = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1, origin="upper")
    n = len(corr)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, ha="center")
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.78)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i >= j:
                val = corr.iloc[i, j]
                if not np.isfinite(val):
                    continue
                text = f"{val:.2f}\n{star(pvals.iloc[i, j])}" if i != j else "1.00"
                ax.text(j, i, text, ha="center", va="center", fontsize=7.0 if compact else 8.4, fontweight="bold", color="black", linespacing=0.68)
    ax.tick_params(axis="both", which="major", direction="out", length=4.0, width=1.0, bottom=True, left=True)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_fontsize(10.0 if compact else 12.4)
    for _, _, end in group_spans[:-1]:
        boundary = end + 0.5
        ax.plot([-0.5, boundary], [boundary, boundary], color=NATURE["ink"], lw=1.15)
        ax.plot([boundary, boundary], [boundary, n - 0.5], color=NATURE["ink"], lw=1.15)
    for group_name, start, end in group_spans:
        mid = (start + end + 1) / 2
        x0, x1 = start - 0.5, end + 0.5
        y0 = -1.08
        tick = 0.22
        label_text = group_name.replace(" (", "\n(")
        ax.plot([x0, x1], [y0, y0], color=NATURE["ink"], lw=1.25, clip_on=False)
        ax.plot([x0, x0], [y0 - tick, y0 + tick], color=NATURE["ink"], lw=1.25, clip_on=False)
        ax.plot([x1, x1], [y0 - tick, y0 + tick], color=NATURE["ink"], lw=1.25, clip_on=False)
        ax.text(
            mid,
            y0 - 0.42,
            label_text,
            rotation=0,
            ha="center",
            va="top",
            fontsize=10.6 if compact else 12.4,
            fontweight="bold",
            color=NATURE["ink"],
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.82),
            clip_on=False,
        )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(
        im,
        cax=cbar_ax,
        ax=None if cbar_ax is not None else ax,
        orientation="horizontal" if cbar_ax is not None else "vertical",
        fraction=0.046,
        pad=0.075,
    )
    cbar.ax.tick_params(labelsize=10.5, width=1.0, length=4, direction="out")
    for tick in cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    cbar.set_label(cbar_label, fontweight="bold", fontsize=12.4 if not compact else 10.9)


def fig3_correlation(df: pd.DataFrame) -> None:
    groups = [
        ("Maturity", ["GI (%)", "C/N"]),
        ("Initial", ["Initial pH", "Initial carbon to nitrogen", "Initial moisture content (%)"]),
        ("Process", ["Compost time (day)", "Temperature_corr", "Moisture_corr", "pH_corr"]),
        ("Chemistry", ["EC（ms/cm-1）", "Ammonia氨mg/kg", "Nitrate硝酸盐mg/kg", "总氮(%)"]),
    ]
    cols = [col for _, gcols in groups for col in gcols]
    spans = []
    start = 0
    for group_name, gcols in groups:
        end = start + len(gcols) - 1
        spans.append((group_name, start, end))
        start = end + 1
    data = df[cols].copy()
    fig = plt.figure(figsize=(16.4, 12.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    draw_grouped_corr(ax, data, "pearson", spans, "Pearson correlation coefficient (r)")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    draw_grouped_corr(ax, data, "spearman", spans, "Spearman rho", compact=True)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    mi_data = df[df["GI (%)"].notna()][FEATURE_SETS["P1"]].copy()
    y = df.loc[df["GI (%)"].notna(), "GI (%)"].astype(float)
    num_cols = [c for c in FEATURE_SETS["P1"] if c != "Waste type"]
    mi_num = mi_data[num_cols].copy()
    mi_num = mi_num.fillna(mi_num.median())
    mi = mutual_info_regression(mi_num, y, random_state=1)
    mi_s = pd.Series(mi, index=num_cols).sort_values(ascending=True)
    ax.barh([label(c) for c in mi_s.index], mi_s.values, color=NATURE["green"], edgecolor="white")
    style_axis(ax, xlabel="Mutual information with GI", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    rows = []
    for gname, gcols in feature_groups().items():
        if gname == "Gas":
            continue
        valid = [c for c in gcols if c in data.columns and c != "GI (%)"]
        if not valid:
            continue
        vals = data[valid].corrwith(data["GI (%)"], method="spearman").abs().dropna()
        if vals.empty:
            continue
        rows.append({"Group": gname, "Mean |rho|": vals.mean(), "Features": len(vals)})
    assoc = pd.DataFrame(rows).sort_values("Mean |rho|")
    ax.barh(assoc["Group"], assoc["Mean |rho|"], color=[NATURE["blue"], NATURE["coral"], NATURE["green"], NATURE["navy"]][: len(assoc)], edgecolor="white")
    for yi, row in assoc.reset_index(drop=True).iterrows():
        ax.text(row["Mean |rho|"] + 0.01, yi, f"n={int(row['Features'])}", va="center", fontsize=12.3, fontweight="bold")
    ax.set_xlim(0, max(0.55, assoc["Mean |rho|"].max() * 1.25))
    style_axis(ax, xlabel="Mean absolute Spearman correlation with GI", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    panel_label(ax, "d")
    save_figure(fig, "Fig03_correlation", "Fig03_correlation")


def fig4_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    axis_label_fs = 21.2
    tick_fs = 16.4
    heat_tick_fs = 15.0
    heat_annot_fs = 13.4
    compact_heat_tick_fs = 13.6
    compact_annot_fs = 12.4
    cbar_tick_fs = 12.6
    cbar_label_fs = 14.8
    candidate_tick_fs = 14.0
    candidate_annot_fs = 11.4
    legend_fs = 14.0

    def set_fig4_axis_text(ax, x_size: float = tick_fs, y_size: float = tick_fs, label_size: float = axis_label_fs) -> None:
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_weight("bold")
        for tick in ax.get_xticklabels():
            tick.set_fontsize(x_size)
            tick.set_fontweight("bold")
        for tick in ax.get_yticklabels():
            tick.set_fontsize(y_size)
            tick.set_fontweight("bold")

    def set_fig4_cbar(cbar, label: str, tick_size: float = cbar_tick_fs, label_size: float = cbar_label_fs) -> None:
        cbar.ax.tick_params(labelsize=tick_size, width=1.0, length=4, direction="out")
        for tick in cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")
        cbar.set_label(label, fontweight="bold", fontsize=label_size)

    perf = evaluate_models(df, TARGET, "P1")
    summary = (
        perf.groupby(["validation", "model"])
        .agg(train_r2=("train_r2", "mean"), test_r2=("test_r2", "mean"), test_r2_sd=("test_r2", "std"), gap=("gap", "mean"), rmse=("test_rmse", "mean"), mae=("test_mae", "mean"))
        .reset_index()
    )
    group = summary[summary["validation"] == "group"].sort_values("test_r2", ascending=False)
    order = group["model"].tolist()
    colors = dict(zip(order, [NATURE["navy"], NATURE["blue"], NATURE["green"], NATURE["coral"], NATURE["lavender"], NATURE["sand"], NATURE["red"], NATURE["teal"], NATURE["brown"]]))

    fig = plt.figure(figsize=(19.2, 21.6))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.40, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    sub = group.set_index("model").loc[order]
    ax.bar(np.arange(len(order)), sub["test_r2"], yerr=sub["test_r2_sd"], color=[colors[o] for o in order], edgecolor="white", capsize=3)
    ax.scatter(np.arange(len(order)), sub["train_r2"], s=48, facecolor="white", edgecolor=NATURE["ink"], lw=1.5, zorder=3, label="Train R$^2$")
    if PRIMARY_MODEL in order:
        primary_idx = order.index(PRIMARY_MODEL)
        ax.scatter(primary_idx, sub.loc[PRIMARY_MODEL, "test_r2"] + 0.035, marker="*", s=170, color=NATURE["red"], edgecolor="white", lw=0.6, zorder=4, label="Best predictive")
    if CONSERVATIVE_MODEL in order:
        conservative_idx = order.index(CONSERVATIVE_MODEL)
        ax.scatter(conservative_idx, sub.loc[CONSERVATIVE_MODEL, "test_r2"] + 0.065, marker="D", s=110, facecolor="white", edgecolor=NATURE["navy"], lw=1.5, zorder=4, label="Gap-controlled")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylim(0, min(1.02, max(sub["train_r2"].max(), sub["test_r2"].max()) * 1.14))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.14), ncol=3, fontsize=legend_fs, handletextpad=0.45, columnspacing=0.90)
    style_axis(ax, ylabel="R$^2$")
    numeric_ticks(ax, x=False, y=True)
    set_fig4_axis_text(ax)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(order))
    width = 0.36
    ax.bar(x - width / 2, sub["rmse"], width=width, color=NATURE["blue"], edgecolor="white", label="RMSE")
    ax.bar(x + width / 2, sub["mae"], width=width, color=NATURE["coral"], edgecolor="white", label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.legend(frameon=False, loc="upper left", fontsize=legend_fs, handlelength=1.15, handletextpad=0.45)
    style_axis(ax, ylabel="Error")
    numeric_ticks(ax, x=False, y=True)
    set_fig4_axis_text(ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    selection_heat = sub[["test_r2", "gap"]].rename(columns={"test_r2": "Group test R$^2$", "gap": "Train-test gap"})
    sel_cmap = LinearSegmentedColormap.from_list("fig4_selection", [NATURE["light"], NATURE["teal"], NATURE["navy"]])
    sns.heatmap(
        selection_heat,
        cmap=sel_cmap,
        vmin=0,
        vmax=max(float(selection_heat.max().max()) + 0.03, 0.80),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": heat_annot_fs, "fontweight": "bold"},
        linewidths=0.9,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Metric value", "shrink": 0.80, "pad": 0.025},
        ax=ax,
    )
    for yi, model in enumerate(selection_heat.index):
        if sub.loc[model, "gap"] > PRIMARY_GAP_LIMIT:
            ax.add_patch(patches.Rectangle((1, yi), 1, 1, fill=False, edgecolor=NATURE["red"], lw=1.9))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(selection_heat.columns, rotation=0, fontsize=heat_tick_fs, fontweight="bold")
    ax.set_yticklabels(order, rotation=0, fontsize=heat_tick_fs, fontweight="bold")
    ax.tick_params(axis="both", which="major", length=0, pad=5)
    set_fig4_cbar(ax.collections[0].colorbar, "Metric value")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    merged = summary.pivot(index="model", columns="validation", values="test_r2").loc[order]
    validation_heat = merged[["random", "group"]].rename(columns={"random": "Random split", "group": "Group split"})
    val_cmap = LinearSegmentedColormap.from_list("fig4_validation", [NATURE["light"], NATURE["teal"], NATURE["navy"]])
    sns.heatmap(
        validation_heat,
        cmap=val_cmap,
        vmin=max(0.35, float(validation_heat.min().min()) - 0.03),
        vmax=min(0.90, float(validation_heat.max().max()) + 0.03),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": heat_annot_fs, "fontweight": "bold"},
        linewidths=0.9,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Test R$^2$", "shrink": 0.80, "pad": 0.025},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(validation_heat.columns, rotation=0, fontsize=heat_tick_fs, fontweight="bold")
    ax.set_yticklabels(order, rotation=0, fontsize=heat_tick_fs, fontweight="bold")
    ax.tick_params(axis="both", which="major", length=0, pad=5)
    set_fig4_cbar(ax.collections[0].colorbar, "Test R$^2$")
    panel_label(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    pmat = pd.DataFrame(np.ones((len(order), len(order))), index=order, columns=order)
    gperf = perf[perf["validation"] == "group"]
    for a in order:
        for b in order:
            xa = gperf[gperf["model"] == a].sort_values("seed")["test_r2"].values
            xb = gperf[gperf["model"] == b].sort_values("seed")["test_r2"].values
            if len(xa) == len(xb) and a != b:
                try:
                    pmat.loc[a, b] = stats.wilcoxon(xa, xb).pvalue
                except Exception:
                    pmat.loc[a, b] = 1.0
    sns.heatmap(
        -np.log10(pmat.clip(lower=1e-4)),
        cmap=sns.light_palette(NATURE["navy"], as_cmap=True),
        cbar_kws={"label": "-log10(P)", "shrink": 0.82},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=heat_tick_fs, fontweight="bold")
    ax.set_yticklabels(order, rotation=0, fontsize=heat_tick_fs, fontweight="bold")
    ax.tick_params(axis="both", which="major", length=0, pad=5)
    set_fig4_cbar(ax.collections[0].colorbar, "-log10(P)")
    panel_label(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    decision = sub.copy()
    decision["Eligible"] = decision["gap"] <= PRIMARY_GAP_LIMIT
    decision = decision.sort_values(["Eligible", "test_r2"], ascending=[False, False]).head(7).sort_values("test_r2")
    bar_colors = np.where(decision["Eligible"], NATURE["green"], NATURE["coral"])
    ax.barh(decision.index, decision["test_r2"], color=bar_colors, edgecolor="white", alpha=0.88)
    if PRIMARY_MODEL in decision.index:
        ax.scatter(decision.loc[PRIMARY_MODEL, "test_r2"] + 0.012, list(decision.index).index(PRIMARY_MODEL), marker="*", s=180, color=NATURE["red"], edgecolor="white", lw=0.7, zorder=4)
    if CONSERVATIVE_MODEL in decision.index:
        ax.scatter(decision.loc[CONSERVATIVE_MODEL, "test_r2"] + 0.030, list(decision.index).index(CONSERVATIVE_MODEL), marker="D", s=120, facecolor="white", edgecolor=NATURE["navy"], lw=1.5, zorder=4)
    ax.set_xlim(0, min(1.0, max(0.78, decision["test_r2"].max() + 0.08)))
    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=8, markerfacecolor=NATURE["green"], markeredgecolor="white", label=f"gap <= {PRIMARY_GAP_LIMIT:.2f}"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=8, markerfacecolor=NATURE["coral"], markeredgecolor="white", label=f"gap > {PRIMARY_GAP_LIMIT:.2f}"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=legend_fs)
    style_axis(ax, xlabel="Group test R$^2$", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    set_fig4_axis_text(ax)
    panel_label(ax, "f")

    candidate_path = DATA_OUT / "overfit_reduction_candidates.csv"
    if candidate_path.exists():
        raw_hp = pd.read_csv(candidate_path)
        if raw_hp.duplicated(["candidate", "validation"]).any():
            hp = (
                raw_hp.groupby(["candidate", "validation"])
                .agg(train_r2=("train_r2", "mean"), test_r2=("test_r2", "mean"), gap=("gap", "mean"), rmse=("rmse", "mean"), test_sd=("test_r2", "std"), gap_max=("gap", "max"))
                .reset_index()
            )
        else:
            hp = raw_hp.copy()
            if "test_sd" not in hp.columns:
                hp["test_sd"] = np.nan
            if "gap_max" not in hp.columns:
                hp["gap_max"] = hp["gap"]
    else:
        hp = (
            perf.groupby(["model", "validation"])
            .agg(train_r2=("train_r2", "mean"), test_r2=("test_r2", "mean"), gap=("gap", "mean"), rmse=("test_rmse", "mean"), test_sd=("test_r2", "std"))
            .reset_index()
            .rename(columns={"model": "candidate"})
        )
        hp["gap_max"] = hp["gap"]
    group_hp = hp[hp["validation"] == "group"].copy()
    random_hp = hp[hp["validation"] == "random"].copy()
    stable_hp = group_hp[group_hp["gap"] <= PRIMARY_GAP_LIMIT]
    selected_hp_idx = stable_hp["test_r2"].idxmax() if not stable_hp.empty else group_hp["test_r2"].idxmax()
    selected_hp = group_hp.loc[selected_hp_idx, "candidate"]

    ax = fig.add_subplot(gs[2, 0])
    candidate_rank = group_hp.copy()
    candidate_rank["Gap-adjusted score"] = candidate_rank["test_r2"] - candidate_rank["gap"]
    candidate_rank = candidate_rank.sort_values("test_r2", ascending=False).head(10)
    candidate_heat = candidate_rank.set_index("candidate")[["test_r2", "gap", "Gap-adjusted score"]]
    candidate_heat.columns = ["Group R$^2$", "Gap", "R$^2$ - gap"]
    candidate_cmap = LinearSegmentedColormap.from_list("fig4_candidate_selection", [NATURE["light"], NATURE["teal"], NATURE["navy"]])
    sns.heatmap(
        candidate_heat,
        ax=ax,
        cmap=candidate_cmap,
        vmin=0,
        vmax=max(0.76, float(candidate_heat.max().max()) + 0.025),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": compact_annot_fs, "fontweight": "bold"},
        linewidths=0.9,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Metric value", "shrink": 0.76, "pad": 0.025},
    )
    for yi, candidate in enumerate(candidate_heat.index):
        if candidate == selected_hp:
            ax.add_patch(patches.Rectangle((0, yi), candidate_heat.shape[1], 1, fill=False, edgecolor=NATURE["red"], lw=2.2))
        if candidate_rank.set_index("candidate").loc[candidate, "gap"] > PRIMARY_GAP_LIMIT:
            ax.add_patch(patches.Rectangle((1, yi), 1, 1, fill=False, edgecolor=NATURE["coral"], lw=1.8))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0, labelsize=compact_heat_tick_fs, bottom=True, top=False, length=0)
    ax.tick_params(axis="y", labelrotation=0, labelsize=candidate_tick_fs, left=True, right=False, length=0)
    set_fig4_axis_text(ax, x_size=compact_heat_tick_fs, y_size=candidate_tick_fs)
    set_fig4_cbar(ax.collections[0].colorbar, "Metric value", tick_size=10.2, label_size=11.6)
    panel_label(ax, "g")

    ax = fig.add_subplot(gs[2, 1])
    dumb = group_hp.sort_values("test_r2", ascending=True).reset_index(drop=True)
    y = np.arange(len(dumb))
    for yi, row in dumb.iterrows():
        lw = 2.0 if row["candidate"] == selected_hp else 1.15
        alpha = 0.92 if row["candidate"] == selected_hp else 0.62
        ax.plot([row["test_r2"], row["train_r2"]], [yi, yi], color=NATURE["grey"], lw=lw, alpha=alpha, zorder=1)
    ax.scatter(dumb["train_r2"], y, s=58, marker="^", color=NATURE["navy"], edgecolor="white", lw=0.7, label="Train R$^2$", zorder=3)
    ax.scatter(dumb["test_r2"], y, s=64, facecolors="white", edgecolors=NATURE["green"], lw=1.6, label="Group test R$^2$", zorder=4)
    sel_y = int(dumb.index[dumb["candidate"].eq(selected_hp)][0])
    ax.scatter(dumb.loc[sel_y, "test_r2"], sel_y, marker="*", s=160, color=NATURE["red"], edgecolor="white", lw=0.6, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(dumb["candidate"])
    ax.set_xlim(max(0, dumb[["test_r2", "train_r2"]].min().min() - 0.05), min(1.0, dumb[["test_r2", "train_r2"]].max().max() + 0.05))
    style_axis(ax, xlabel="R$^2$", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=4)
    set_fig4_axis_text(ax, y_size=candidate_tick_fs)
    ax.legend(frameon=False, loc="lower right", fontsize=legend_fs, handlelength=1.0, handletextpad=0.35)
    panel_label(ax, "h")

    ax = fig.add_subplot(gs[2, 2])
    pivot_hp = pd.merge(
        group_hp[["candidate", "test_r2", "gap"]],
        random_hp[["candidate", "test_r2"]].rename(columns={"test_r2": "random_test_r2"}),
        on="candidate",
        how="inner",
    ).sort_values("test_r2", ascending=False)
    hp_heat = pivot_hp.set_index("candidate")[["random_test_r2", "test_r2"]]
    hp_heat.columns = ["Random split", "Group split"]
    hp_cmap = LinearSegmentedColormap.from_list("fig4_hp_validation", [NATURE["light"], NATURE["teal"], NATURE["navy"]])
    sns.heatmap(
        hp_heat,
        ax=ax,
        cmap=hp_cmap,
        vmin=max(0.5, float(hp_heat.min().min()) - 0.03),
        vmax=min(0.9, float(hp_heat.max().max()) + 0.03),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": compact_annot_fs, "fontweight": "bold"},
        linewidths=0.9,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Test R$^2$", "shrink": 0.76, "pad": 0.025},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0, labelsize=compact_heat_tick_fs, bottom=True, top=False, length=0)
    ax.tick_params(axis="y", labelrotation=0, labelsize=candidate_tick_fs, left=True, right=False, length=0)
    set_fig4_axis_text(ax, x_size=compact_heat_tick_fs, y_size=candidate_tick_fs)
    set_fig4_cbar(ax.collections[0].colorbar, "Test R$^2$", tick_size=10.2, label_size=11.6)
    panel_label(ax, "i")

    ax = fig.add_subplot(gs[3, 0])
    top_hp = group_hp.sort_values(["gap", "test_r2"], ascending=[True, False]).head(10).sort_values("gap", ascending=True)
    ax.barh(top_hp["candidate"], top_hp["gap"], color=np.where(top_hp["candidate"].eq(selected_hp), NATURE["red"], NATURE["blue"]), edgecolor="white", alpha=0.86)
    ax.axvline(PRIMARY_GAP_LIMIT, color=NATURE["ink"], lw=1.1, ls="--")
    style_axis(ax, xlabel="Train-test gap", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=4)
    set_fig4_axis_text(ax, y_size=candidate_tick_fs)
    panel_label(ax, "j")

    ax = fig.add_subplot(gs[3, 1])
    rmse_pivot = pd.merge(
        group_hp[["candidate", "rmse"]],
        random_hp[["candidate", "rmse"]].rename(columns={"rmse": "random_rmse"}),
        on="candidate",
        how="inner",
    ).set_index("candidate").loc[pivot_hp["candidate"]]
    rmse_heat = rmse_pivot[["random_rmse", "rmse"]].rename(columns={"random_rmse": "Random RMSE", "rmse": "Group RMSE"})
    rmse_cmap = LinearSegmentedColormap.from_list("fig4_rmse", [NATURE["light"], NATURE["sand"], NATURE["red"]])
    sns.heatmap(
        rmse_heat,
        ax=ax,
        cmap=rmse_cmap,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": compact_annot_fs, "fontweight": "bold"},
        linewidths=0.9,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "RMSE", "shrink": 0.76, "pad": 0.025},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0, labelsize=compact_heat_tick_fs, bottom=True, top=False, length=0)
    ax.tick_params(axis="y", labelrotation=0, labelsize=candidate_tick_fs, left=True, right=False, length=0)
    set_fig4_axis_text(ax, x_size=compact_heat_tick_fs, y_size=candidate_tick_fs)
    set_fig4_cbar(ax.collections[0].colorbar, "RMSE", tick_size=10.2, label_size=11.6)
    panel_label(ax, "k")

    ax = fig.add_subplot(gs[3, 2])
    score = group_hp.copy()
    score["Gap-adjusted score"] = score["test_r2"] - score["gap"]
    score = score.sort_values("Gap-adjusted score", ascending=True).tail(10)
    ax.barh(score["candidate"], score["Gap-adjusted score"], color=np.where(score["candidate"].eq(selected_hp), NATURE["red"], NATURE["green"]), edgecolor="white", alpha=0.88)
    if selected_hp in score["candidate"].values:
        ax.scatter(score.loc[score["candidate"].eq(selected_hp), "Gap-adjusted score"].iloc[0] + 0.006, list(score["candidate"]).index(selected_hp), marker="*", s=145, color=NATURE["red"], edgecolor="white", lw=0.6, zorder=4)
    style_axis(ax, xlabel="Group R$^2$ - gap", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=4)
    set_fig4_axis_text(ax, y_size=candidate_tick_fs)
    panel_label(ax, "l")
    save_figure(fig, "Fig04_model_comparison", "Fig04_model_comparison")
    perf.to_csv(DATA_OUT / "Fig04_model_comparison_raw.csv", index=False, encoding="utf-8-sig")
    high_r2 = perf[(perf["validation"] == "random") & (perf["test_r2"] >= 0.80) & (perf["gap"] <= 0.10)].copy()
    high_r2.sort_values(["test_r2", "gap"], ascending=[False, True]).to_csv(
        DATA_OUT / "within_distribution_high_r2_models.csv", index=False, encoding="utf-8-sig"
    )
    return perf


def fit_target_predictions(
    df: pd.DataFrame,
    target: str,
    seed: int = 0,
    validation: str = "group",
    model_name: str = PRIMARY_MODEL,
    feature_set: str = PRIMARY_FEATURE_SET,
):
    data = df[df[target].notna()].copy().reset_index(drop=True)
    y = data[target].astype(float)
    features = FEATURE_SETS[feature_set]
    tr, te = split_indices(data, y, validation, seed, test_size=0.25)
    pipe = build_pipe(model_name, features)
    pipe.fit(data.iloc[tr][features], y.iloc[tr])
    train = pd.DataFrame(
        {
            "set": "Train",
            "model": model_name,
            "validation": validation,
            "target": target,
            "feature_set": feature_set,
            "seed": seed,
            "y_true": y.iloc[tr],
            "y_pred": pipe.predict(data.iloc[tr][features]),
        }
    )
    test = pd.DataFrame(
        {
            "set": "Test",
            "model": model_name,
            "validation": validation,
            "target": target,
            "feature_set": feature_set,
            "seed": seed,
            "y_true": y.iloc[te],
            "y_pred": pipe.predict(data.iloc[te][features]),
        }
    )
    return pd.concat([train, test], ignore_index=True)


def pred_panel(ax, pred: pd.DataFrame, target: str) -> None:
    train = pred[pred["set"] == "Train"]
    test = pred[pred["set"] == "Test"]
    model_name = str(pred["model"].iloc[0])
    validation = str(pred["validation"].iloc[0]).capitalize()
    test_color = NATURE["green"] if validation.lower() == "random" else NATURE["red"]
    ax.scatter(train["y_true"], train["y_pred"], s=22, facecolor=NATURE["blue"], edgecolor="white", lw=0.4, alpha=0.45, label="Train")
    ax.scatter(test["y_true"], test["y_pred"], s=34, facecolors="none", edgecolor=test_color, lw=1.0, alpha=0.85, label=f"{validation} test")
    vals = np.r_[pred["y_true"].values, pred["y_pred"].values]
    lo, hi = np.nanquantile(vals, [0.01, 0.99])
    pad = (hi - lo) * 0.08 if hi > lo else 1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=NATURE["ink"], ls="--", lw=1.2)
    r2_tr = r2_score(train["y_true"], train["y_pred"])
    r2_te = r2_score(test["y_true"], test["y_pred"])
    rmse_te = mean_squared_error(test["y_true"], test["y_pred"], squared=False)
    ax.text(
        0.05,
        0.94,
        f"{model_name} ({validation})\nTrain R$^2$={r2_tr:.3f}\nTest R$^2$={r2_te:.3f}\nGap={r2_tr - r2_te:.3f}\nRMSE={rmse_te:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="bold",
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    style_axis(ax, xlabel=f"Observed {TARGET_LABELS[target]}", ylabel=f"Predicted {TARGET_LABELS[target]}")
    numeric_ticks(ax, x=True, y=True)


def target_perf(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    cache = DATA_OUT / f"target_random_group_{safe_token(PRIMARY_MODEL)}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    rows = []
    for short, target in TARGETS.items():
        data = df[df[target].notna()].copy().reset_index(drop=True)
        if len(data) < 80 or data["batch_id"].nunique() < 10:
            continue
        y = data[target].astype(float)
        for validation in ["random", "group"]:
            for seed in SEEDS:
                tr, te = split_indices(data, y, validation, seed, test_size=0.25)
                features = FEATURE_SETS[PRIMARY_FEATURE_SET]
                pipe = build_pipe(PRIMARY_MODEL, features)
                pipe.fit(data.iloc[tr][features], y.iloc[tr])
                pred_train = pipe.predict(data.iloc[tr][features])
                pred = pipe.predict(data.iloc[te][features])
                train_r2 = r2_score(y.iloc[tr], pred_train)
                test_r2 = r2_score(y.iloc[te], pred)
                rows.append(
                    {
                        "target_short": short,
                        "target": target,
                        "model": PRIMARY_MODEL,
                        "validation": validation,
                        "seed": seed,
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "gap": train_r2 - test_r2,
                        "test_rmse": mean_squared_error(y.iloc[te], pred, squared=False),
                        "n": len(data),
                        "groups": data["batch_id"].nunique(),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def fig6_optimism_gap(df: pd.DataFrame) -> None:
    perf = target_perf(df)
    mean = perf.groupby(["target_short", "validation"]).agg(r2=("test_r2", "mean"), sd=("test_r2", "std"), n=("n", "first"), groups=("groups", "first")).reset_index()
    pivot = mean.pivot(index="target_short", columns="validation", values="r2")
    pivot["gap"] = pivot["random"] - pivot["group"]
    order = pivot.sort_values("gap", ascending=False).index.tolist()

    fig = plt.figure(figsize=(15.6, 9.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order))
    width = 0.36
    ax.bar(x - width / 2, pivot.loc[order, "random"], width=width, color=NATURE["blue"], edgecolor="white", label="Random")
    ax.bar(x + width / 2, pivot.loc[order, "group"], width=width, color=NATURE["coral"], edgecolor="white", label="Group")
    ax.set_xticks(x)
    # Rotate around each label's centre, then add enough downward padding that
    # the long chemical formulae remain centred on their ticks without rising
    # into the plotting area.
    ax.set_xticklabels(
        [SHORT_LABELS.get(i, i) for i in order],
        rotation=30,
        ha="center",
        va="top",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", labelsize=14.2, pad=16)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.68, 1.14), ncol=2)
    style_axis(ax, ylabel="Test R$^2$")
    numeric_ticks(ax, x=False, y=True)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.barh([SHORT_LABELS.get(i, i) for i in order], pivot.loc[order, "gap"], color=NATURE["red"], edgecolor="white", alpha=0.82)
    ax.axvline(0, color=NATURE["ink"], lw=1.0)
    style_axis(ax, xlabel="Optimism gap (Random - Group)", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    ax.text(0.05, 0.90, "Split logic", fontsize=18.0, fontweight="bold", color=NATURE["ink"], transform=ax.transAxes)

    def draw_token(x0, y0, text, fc, ec="white"):
        box = patches.FancyBboxPatch(
            (x0, y0),
            0.105,
            0.105,
            boxstyle="round,pad=0.010",
            ec=ec,
            fc=fc,
            lw=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x0 + 0.0525, y0 + 0.0525, text, ha="center", va="center", fontsize=13.0, fontweight="bold", color=NATURE["ink"], transform=ax.transAxes)

    # Reserve a dedicated left text column.  The previous first token started
    # at x=0.30, so the long labels extended into the t1 boxes.
    x_positions = [0.36, 0.49, 0.62, 0.75]
    ax.text(0.05, 0.70, "Random", fontsize=16.2, fontweight="bold", color=NATURE["red"], transform=ax.transAxes)
    ax.text(0.05, 0.625, "mixed within batch", fontsize=11.6, fontweight="bold", color=NATURE["grey"], transform=ax.transAxes)
    for k, x0 in enumerate(x_positions):
        color = NATURE["blue"] if k in [0, 2] else NATURE["coral"]
        draw_token(x0, 0.64, f"t{k+1}", color)
    token_centres = [x0 + 0.0525 for x0 in x_positions]
    ax.plot([token_centres[0], token_centres[-1]], [0.605, 0.605], color=NATURE["red"], lw=1.5, ls="--", transform=ax.transAxes, clip_on=False)
    ax.text(np.mean(token_centres), 0.555, "same batch in both sets", ha="center", va="top", fontsize=11.8, fontweight="bold", color=NATURE["red"], transform=ax.transAxes)

    ax.text(0.05, 0.36, "Group-aware", fontsize=16.2, fontweight="bold", color=NATURE["navy"], transform=ax.transAxes)
    ax.text(0.05, 0.285, "batch kept intact", fontsize=11.6, fontweight="bold", color=NATURE["grey"], transform=ax.transAxes)
    for k, x0 in enumerate(x_positions):
        draw_token(x0, 0.30, f"t{k+1}", NATURE["blue"])
    for k, x0 in enumerate(x_positions):
        draw_token(x0, 0.14, f"t{k+1}", NATURE["coral"])
    ax.plot([x_positions[0] - 0.01, x_positions[-1] + 0.115], [0.455, 0.455], color=NATURE["ink"], lw=1.0, transform=ax.transAxes, clip_on=False)

    handles = [
        patches.Patch(facecolor=NATURE["blue"], edgecolor="white", label="Train"),
        patches.Patch(facecolor=NATURE["coral"], edgecolor="white", label="Test"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0.03, -0.02), ncol=2, fontsize=12.4, handlelength=1.2, columnspacing=1.0)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    gi_group = perf[(perf["target_short"] == "GI") & (perf["validation"] == "group")]
    gi_random = perf[(perf["target_short"] == "GI") & (perf["validation"] == "random")]
    ax.violinplot([gi_random["test_r2"], gi_group["test_r2"]], positions=[0, 1], widths=0.6, showmeans=True)
    for pos, vals, color in [(0, gi_random["test_r2"], NATURE["blue"]), (1, gi_group["test_r2"], NATURE["coral"])]:
        ax.scatter(np.full(len(vals), pos) + np.linspace(-0.08, 0.08, len(vals)), vals, s=38, color=color, edgecolor="white", zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random", "Group"])
    style_axis(ax, ylabel="GI test R$^2$ across seeds")
    numeric_ticks(ax, x=False, y=True)
    panel_label(ax, "d")
    save_figure(fig, "Fig06_temporal_validation", "Fig06_temporal_validation")


def fig7_residual_reliability(df: pd.DataFrame) -> None:
    pipe, features, train, test, data = primary_fit(df)
    residual = test["y_pred"] - test["y_true"]
    fig = plt.figure(figsize=(17.4, 10.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 0.86], hspace=0.46, wspace=0.52)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])

    ax = axes[0, 0]
    ax.hist(residual, bins=34, color=NATURE["blue"], alpha=0.72, edgecolor="white", density=True)
    xs = np.linspace(residual.min(), residual.max(), 200)
    if residual.std() > 0:
        ax.plot(xs, stats.norm.pdf(xs, residual.mean(), residual.std()), color=NATURE["ink"], lw=2)
    ax.axvline(0, color=NATURE["red"], lw=1.6, ls="--")
    style_axis(ax, xlabel="Residual (predicted - observed)", ylabel="Density")
    numeric_ticks(ax, x=True, y=True)
    panel_label(ax, "a")

    ax = axes[0, 1]
    colors = test["Waste type"].map(WASTE_COLORS).fillna(NATURE["grey"])
    ax.scatter(test["y_pred"], residual, s=30, c=colors, alpha=0.58, edgecolors="none")
    ax.axhline(0, color=NATURE["ink"], lw=1.1, ls="--")
    ax.axhline(residual.mean() + 1.96 * residual.std(), color=NATURE["red"], lw=1.1, ls=":")
    ax.axhline(residual.mean() - 1.96 * residual.std(), color=NATURE["red"], lw=1.1, ls=":")
    style_axis(ax, xlabel="Predicted GI (%)", ylabel="Residual")
    numeric_ticks(ax, x=True, y=True)
    panel_label(ax, "b")

    ax = axes[1, 0]
    data_y = data["GI (%)"].astype(float)
    tr, te = split_indices(data, data_y, "group", 1, test_size=0.25)
    rand_r2 = []
    for seed in range(30):
        y_perm = data_y.copy().sample(frac=1.0, random_state=seed).reset_index(drop=True)
        model = build_pipe(PRIMARY_MODEL, features)
        model.fit(data.iloc[tr][features], y_perm.iloc[tr])
        rand_r2.append(r2_score(data_y.iloc[te], model.predict(data.iloc[te][features])))
    true_r2 = r2_score(test["y_true"], test["y_pred"])
    ax.hist(rand_r2, bins=16, color=NATURE["lavender"], edgecolor="white", alpha=0.8)
    ax.axvline(true_r2, color=NATURE["red"], lw=2.0, label=f"True R$^2$={true_r2:.3f}")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax, xlabel="R$^2$ after Y-randomisation", ylabel="Frequency")
    numeric_ticks(ax, x=True, y=True)
    panel_label(ax, "c")

    ax = axes[1, 1]
    Xtr = pipe.named_steps["pre"].transform(train[features])
    Xte = pipe.named_steps["pre"].transform(test[features])
    xtx_inv = np.linalg.pinv(Xtr.T @ Xtr)
    leverage = np.einsum("ij,jk,ik->i", Xte, xtx_inv, Xte)
    std_res = (residual - residual.mean()) / (residual.std() if residual.std() else 1)
    ax.scatter(leverage, std_res, s=32, facecolors="none", edgecolors=NATURE["navy"], lw=1.0, alpha=0.82)
    h_star = 3 * Xtr.shape[1] / max(Xtr.shape[0], 1)
    ax.axhline(3, color=NATURE["red"], lw=1.1, ls="--")
    ax.axhline(-3, color=NATURE["red"], lw=1.1, ls="--")
    ax.axvline(h_star, color=NATURE["coral"], lw=1.2, ls=":")
    style_axis(ax, xlabel="Leverage h", ylabel="Standardised residual")
    numeric_ticks(ax, x=True, y=True)
    panel_label(ax, "d")

    ax = axes[0, 2]
    train_resid = train["y_pred"] - train["y_true"]
    q90 = np.nanquantile(np.abs(train_resid), 0.90)
    q95 = np.nanquantile(np.abs(train_resid), 0.95)
    interval_rows = []
    pred_bins = pd.qcut(test["y_pred"], q=5, duplicates="drop")
    for interval, sub in test.assign(abs_error=np.abs(residual), bin=pred_bins).groupby("bin", observed=True):
        interval_rows.append(
            {
                "bin_mid": sub["y_pred"].median(),
                "coverage90": (sub["abs_error"] <= q90).mean() * 100,
                "coverage95": (sub["abs_error"] <= q95).mean() * 100,
            }
        )
    coverage = pd.DataFrame(interval_rows).sort_values("bin_mid")
    ax.plot(coverage["bin_mid"], coverage["coverage90"], marker="o", color=NATURE["navy"], lw=2.0, label="90% interval")
    ax.plot(coverage["bin_mid"], coverage["coverage95"], marker="s", color=NATURE["teal"], lw=2.0, label="95% interval")
    ax.axhline(90, color=NATURE["navy"], lw=1.0, ls=":")
    ax.axhline(95, color=NATURE["teal"], lw=1.0, ls=":")
    ax.set_ylim(45, 103)
    ax.legend(frameon=False, fontsize=11.2, loc="lower right")
    style_axis(ax, xlabel="Predicted GI bin median", ylabel="Empirical coverage (%)")
    numeric_ticks(ax, x=True, y=True)
    panel_label(ax, "e")

    ax = axes[1, 2]
    error_df = test.assign(abs_error=np.abs(residual))
    top_waste = error_df["Waste type"].value_counts().head(7).index.tolist()
    err_plot = error_df[error_df["Waste type"].isin(top_waste)].copy()
    short_waste = {
        "livestock manure compost": "livestock manure",
        "Composting kitchen waste": "kitchen waste",
        "sludge waste composting": "sludge compost",
        "poultry manure": "poultry manure",
        "pig manure": "pig manure",
        "green waste": "green waste",
        "food waste": "food waste",
        "sewage sludge": "sewage sludge",
    }
    err_plot["Waste group"] = err_plot["Waste type"].map(short_waste).fillna(err_plot["Waste type"])
    order = err_plot.groupby("Waste group")["abs_error"].median().sort_values().index.tolist()
    palette_short = {}
    for waste_name in top_waste:
        palette_short[short_waste.get(waste_name, waste_name)] = WASTE_COLORS.get(waste_name, NATURE["grey"])
    sns.boxplot(data=err_plot, x="abs_error", y="Waste group", order=order, color="white", fliersize=0, linewidth=1.1, ax=ax)
    sns.stripplot(
        data=err_plot.sample(min(700, len(err_plot)), random_state=3),
        x="abs_error",
        y="Waste group",
        order=order,
        palette=palette_short,
        size=2.4,
        alpha=0.45,
        ax=ax,
    )
    style_axis(ax, xlabel="Absolute error in GI (%)", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=4)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12.8)
        tick.set_fontweight("bold")
    panel_label(ax, "f")
    save_figure(fig, "Fig07_residual_reliability", "Fig07_residual_reliability")


def shap_for_primary(df: pd.DataFrame):
    cache_stem = f"primary_shap_{safe_token(PRIMARY_MODEL)}_{safe_token(PRIMARY_FEATURE_SET)}"
    cache_np = DATA_OUT / f"{cache_stem}_values.npy"
    cache_json = DATA_OUT / f"{cache_stem}_meta.json"
    pipe, features, train, test, data = primary_fit(df)
    X_test_trans = pipe.named_steps["pre"].transform(test[features])
    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    clean_names = []
    for name in feature_names:
        if name.startswith("num__"):
            clean_names.append(label(name.replace("num__", "")))
        elif name.startswith("cat__Waste type_"):
            clean_names.append(name.replace("cat__Waste type_", "Waste: "))
        else:
            clean_names.append(name)
    if cache_np.exists() and cache_json.exists():
        shap_values = np.load(cache_np)
    else:
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        shap_values = explainer.shap_values(X_test_trans)
        np.save(cache_np, shap_values)
        cache_json.write_text(json.dumps({"model": PRIMARY_MODEL, "feature_set": PRIMARY_FEATURE_SET, "features": clean_names}, ensure_ascii=False, indent=2), encoding="utf-8")
    x_df = pd.DataFrame(X_test_trans, columns=clean_names)
    return pipe, features, train, test, x_df, shap_values, clean_names


def fig8_shap_importance(df: pd.DataFrame) -> None:
    pipe, features, train, test, x_df, shap_values, names = shap_for_primary(df)
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=names).sort_values(ascending=False)
    top = mean_abs.head(12).index.tolist()
    fig = plt.figure(figsize=(17.8, 11.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.44)

    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(1)
    for yi, feat in enumerate(reversed(top)):
        idx = names.index(feat)
        vals = shap_values[:, idx]
        fvals = x_df[feat].values
        if np.nanmax(fvals) > np.nanmin(fvals):
            c = (fvals - np.nanmin(fvals)) / (np.nanmax(fvals) - np.nanmin(fvals))
        else:
            c = np.zeros_like(fvals)
        jitter = rng.normal(0, 0.075, len(vals))
        ax.scatter(vals, np.full(len(vals), yi) + jitter, c=c, cmap="coolwarm", s=13, alpha=0.62, edgecolors="none")
    ax.axvline(0, color=NATURE["ink"], lw=1.0, ls="--")
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="coolwarm")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.18, fraction=0.08, aspect=24)
    cbar.set_label("Feature value", fontweight="bold", fontsize=12.3)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(length=4.0, width=1.1, labelsize=11.0, direction="out")
    for t in cbar.ax.get_xticklabels():
        t.set_fontweight("bold")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(list(reversed(top)), fontsize=11.0)
    style_axis(ax, xlabel="SHAP value for GI", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.barh(list(reversed(top)), mean_abs.loc[top].iloc[::-1], color=NATURE["green"], edgecolor="white")
    style_axis(ax, xlabel="Mean |SHAP|", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    result = permutation_importance(pipe, test[features], test["y_true"], n_repeats=12, random_state=1, scoring="neg_root_mean_squared_error")
    imp = pd.Series(result.importances_mean, index=[label(f) for f in features]).sort_values(ascending=True)
    err = pd.Series(result.importances_std, index=[label(f) for f in features]).loc[imp.index]
    ax.barh(imp.index, imp.values, xerr=err.values, color=NATURE["blue"], edgecolor="white", capsize=2)
    style_axis(ax, xlabel="Permutation RMSE increase", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    group_vals = {"Initial": 0.0, "Process": 0.0, "Chemistry": 0.0, "Waste type": 0.0}
    for name, value in mean_abs.items():
        if name.startswith("Waste:"):
            group_vals["Waste type"] += value
        elif name in [label(c) for c in feature_groups()["Initial"]]:
            group_vals["Initial"] += value
        elif name in [label(c) for c in feature_groups()["Process"]]:
            group_vals["Process"] += value
        else:
            group_vals["Chemistry"] += value
    gp = pd.Series(group_vals).sort_values()
    gp_pct = gp / gp.sum() * 100
    gp_colors = [NATURE["blue"], NATURE["green"], NATURE["coral"], NATURE["lavender"]][: len(gp_pct)]
    ax.barh(gp_pct.index, gp_pct.values, color=gp_colors, edgecolor="white")
    for yi, value in enumerate(gp_pct.values):
        ax.text(value + 1.0, yi, f"{value:.1f}%", va="center", fontsize=11.4, fontweight="bold")
    ax.set_xlim(0, max(55, gp_pct.max() * 1.22))
    style_axis(ax, xlabel="Grouped SHAP contribution (%)", ylabel="")
    numeric_ticks(ax, x=True, y=False)
    panel_label(ax, "d")

    numeric_top = [feat for feat in top if feat in x_df.columns and not feat.startswith("Waste:") and np.nanstd(x_df[feat].values) > 0]
    dep_feats = []
    for feat in numeric_top + [label("Compost time (day)"), label("Moisture_corr"), label("Temperature_corr")]:
        if feat in x_df.columns and feat not in dep_feats:
            dep_feats.append(feat)
        if len(dep_feats) == 2:
            break
    dep_axes = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    for ax, feat, letter in zip(dep_axes, dep_feats, ["e", "f"]):
        idx = names.index(feat)
        color_feat = dep_feats[1] if feat == dep_feats[0] and len(dep_feats) > 1 else dep_feats[0]
        cvals = x_df[color_feat].values if color_feat in x_df.columns else np.zeros(len(x_df))
        sc = ax.scatter(x_df[feat], shap_values[:, idx], c=cvals, cmap="YlGnBu", s=24, alpha=0.72, edgecolors="white", linewidths=0.25)
        ax.axhline(0, color=NATURE["ink"], lw=1.0, ls="--")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.76, pad=0.025)
        cbar.set_label(color_feat, fontweight="bold", fontsize=10.3)
        cbar.ax.tick_params(labelsize=9.6, width=1.0, length=3.5, direction="out")
        style_axis(ax, xlabel=feat, ylabel="SHAP value")
        numeric_ticks(ax, x=True, y=True, n=4)
        panel_label(ax, letter)
    save_figure(fig, "Fig08_SHAP_importance", "Fig08_SHAP_importance")


def pdp_curve(pipe: Pipeline, base: pd.DataFrame, features: list[str], feat: str, grid: np.ndarray) -> np.ndarray:
    preds = []
    sample = base.sample(min(600, len(base)), random_state=1).copy()
    for value in grid:
        mod = sample.copy()
        mod[feat] = value
        preds.append(pipe.predict(mod[features]).mean())
    return np.array(preds)


def ale_curve(pipe: Pipeline, base: pd.DataFrame, features: list[str], feat: str, bins: int = 18):
    sample = base.copy()
    x = sample[feat].to_numpy()
    finite = np.isfinite(x)
    cuts = np.unique(np.nanquantile(x[finite], np.linspace(0.02, 0.98, bins + 1)))
    centers, effects = [], []
    running = 0.0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mask = (sample[feat] >= lo) & (sample[feat] <= hi)
        if mask.sum() < 5:
            continue
        low = sample.loc[mask].copy()
        high = sample.loc[mask].copy()
        low[feat] = lo
        high[feat] = hi
        diff = pipe.predict(high[features]) - pipe.predict(low[features])
        running += diff.mean()
        centers.append((lo + hi) / 2)
        effects.append(running)
    effects = np.array(effects)
    if len(effects):
        effects = effects - effects.mean()
    return np.array(centers), effects


def fig9_pdp_ale(df: pd.DataFrame) -> None:
    pipe, features, train, test, data = primary_fit(df)
    feats = ["Compost time (day)", "Temperature_corr", "pH_corr", "Moisture_corr"]
    fig = plt.figure(figsize=(16.8, 14.6))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.58, wspace=0.40)
    labs = list("abcdefghijkl")
    k = 0
    for i, feat in enumerate(feats):
        grid = np.linspace(data[feat].quantile(0.03), data[feat].quantile(0.97), 45)
        ax = fig.add_subplot(gs[i // 2, (i % 2) * 2])
        pdp = pdp_curve(pipe, data, features, feat, grid)
        ax.plot(grid, pdp, color=NATURE["blue"], lw=2.0)
        ax.fill_between(grid, pdp - pdp.std() * 0.10, pdp + pdp.std() * 0.10, color=NATURE["blue"], alpha=0.14)
        style_axis(ax, xlabel=label(feat), ylabel="PDP")
        numeric_ticks(ax, x=True, y=True)
        panel_label(ax, labs[k])
        k += 1

        ax = fig.add_subplot(gs[i // 2, (i % 2) * 2 + 1])
        cx, eff = ale_curve(pipe, data, features, feat)
        ax.plot(cx, eff, color=NATURE["coral"], lw=2.0)
        ax.axhline(0, color=NATURE["ink"], lw=1.0, ls="--")
        if len(eff):
            ax.fill_between(cx, eff - np.nanstd(eff) * 0.12, eff + np.nanstd(eff) * 0.12, color=NATURE["coral"], alpha=0.16)
        style_axis(ax, xlabel=label(feat), ylabel="ALE")
        numeric_ticks(ax, x=True, y=True)
        panel_label(ax, labs[k])
        k += 1

    base = data.sample(min(500, len(data)), random_state=2).copy()

    def draw_interaction(ax, xfeat: str, yfeat: str, cmap, letter: str) -> None:
        gx = np.linspace(data[xfeat].quantile(0.05), data[xfeat].quantile(0.95), 32)
        gy = np.linspace(data[yfeat].quantile(0.05), data[yfeat].quantile(0.95), 32)
        z = np.zeros((len(gy), len(gx)))
        for yi, yv in enumerate(gy):
            for xi, xv in enumerate(gx):
                mod = base.copy()
                mod[xfeat] = xv
                mod[yfeat] = yv
                z[yi, xi] = pipe.predict(mod[features]).mean()
        im = ax.imshow(z, origin="lower", aspect="auto", extent=[gx.min(), gx.max(), gy.min(), gy.max()], cmap=cmap)
        cbar = fig.colorbar(im, ax=ax, shrink=0.74, pad=0.022)
        cbar.set_label("Predicted GI", fontweight="bold", fontsize=11.6)
        cbar.ax.tick_params(labelsize=10.3, width=1.0, length=4, direction="out")
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")
        style_axis(ax, xlabel=label(xfeat), ylabel=label(yfeat))
        numeric_ticks(ax, x=True, y=True, n=4)
        panel_label(ax, letter)

    interaction_specs = [
        (gs[2, 0:2], "Compost time (day)", "Temperature_corr", sns.light_palette(NATURE["green"], as_cmap=True)),
        (gs[2, 2:4], "pH_corr", "Moisture_corr", sns.light_palette(NATURE["blue"], as_cmap=True)),
        (gs[3, 0:2], "Compost time (day)", "pH_corr", sns.light_palette(NATURE["coral"], as_cmap=True)),
        (gs[3, 2:4], "Temperature_corr", "Moisture_corr", sns.light_palette(NATURE["navy"], as_cmap=True)),
    ]
    for spec, xfeat, yfeat, cmap in interaction_specs:
        ax = fig.add_subplot(spec)
        draw_interaction(ax, xfeat, yfeat, cmap, labs[k])
        k += 1
    save_figure(fig, "Fig09_PDP_ALE_interaction", "Fig09_PDP_ALE_interaction")


def fig10_feature_ablation(df: pd.DataFrame) -> None:
    perf = evaluate_primary_feature_sets(df)
    summary = perf.groupby(["feature_set", "validation"]).agg(train_r2=("train_r2", "mean"), test_r2=("test_r2", "mean"), gap=("gap", "mean"), rmse=("test_rmse", "mean"), mae=("test_mae", "mean")).reset_index()
    order = ["P0", "P0+EC", "P1"]
    fig = plt.figure(figsize=(15.2, 9.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order))
    width = 0.36
    for off, validation, color in [(-width / 2, "random", NATURE["blue"]), (width / 2, "group", NATURE["coral"])]:
        sub = summary[summary["validation"] == validation].set_index("feature_set").loc[order]
        ax.bar(x + off, sub["test_r2"], width=width, color=color, edgecolor="white", label=validation.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.68, 1.10), ncol=2)
    style_axis(ax, ylabel="Test R$^2$")
    numeric_ticks(ax, x=False, y=True)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    sub = summary[summary["validation"] == "group"].set_index("feature_set").loc[order]
    ax.plot(order, sub["rmse"], marker="o", color=NATURE["navy"], lw=2.0, label="RMSE")
    ax.plot(order, sub["mae"], marker="s", color=NATURE["green"], lw=2.0, label="MAE")
    ax.legend(frameon=False)
    style_axis(ax, xlabel="Feature protocol", ylabel="Group test error")
    numeric_ticks(ax, x=False, y=True)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    _, _, _, _, _, shap_values, names = shap_for_primary(df)
    group_vals = {"Initial": 0.0, "Process": 0.0, "Chemistry": 0.0, "Waste type": 0.0}
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=names)
    for name, value in mean_abs.items():
        if name.startswith("Waste:"):
            group_vals["Waste type"] += value
        elif name in [label(c) for c in feature_groups()["Initial"]]:
            group_vals["Initial"] += value
        elif name in [label(c) for c in feature_groups()["Process"]]:
            group_vals["Process"] += value
        else:
            group_vals["Chemistry"] += value
    gp = pd.Series(group_vals)
    gp = gp / gp.sum() * 100
    ax.bar(gp.index, gp.values, color=[NATURE["blue"], NATURE["green"], NATURE["coral"], NATURE["lavender"]], edgecolor="white")
    style_axis(ax, xlabel="", ylabel="SHAP contribution (%)")
    numeric_ticks(ax, x=False, y=True)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    group = summary[summary["validation"] == "group"].set_index("feature_set").loc[order]
    sizes = [len(FEATURE_SETS[o]) * 80 for o in order]
    ax.scatter([len(FEATURE_SETS[o]) for o in order], group["test_r2"], s=sizes, c=[NATURE["blue"], NATURE["green"], NATURE["coral"]], alpha=0.78, edgecolor="white")
    for o in order:
        ax.text(len(FEATURE_SETS[o]) + 0.2, group.loc[o, "test_r2"], o, fontsize=12.7, fontweight="bold", va="center")
    ax.margins(x=0.15, y=0.20)  # Keep protocol markers and labels inside the axes.
    style_axis(ax, xlabel="Number of input features", ylabel="Group test R$^2$")
    numeric_ticks(ax, x=True, y=True)
    panel_label(ax, "d")
    save_figure(fig, "Fig10_feature_ablation", "Fig10_feature_ablation")


def fig_s1_learning_curves(df: pd.DataFrame) -> None:
    data = df[df[TARGET].notna()].copy().reset_index(drop=True)
    y = data[TARGET].astype(float)
    features = FEATURE_SETS[PRIMARY_FEATURE_SET]
    tr, te = split_indices(data, y, "group", 1, test_size=0.25)
    train_pool = data.iloc[tr].copy().reset_index(drop=True)
    y_pool = train_pool[TARGET].astype(float).reset_index(drop=True)
    test = data.iloc[te].copy()
    y_test = y.iloc[te]
    fracs = [0.18, 0.32, 0.46, 0.60, 0.75, 1.0]
    rows = []
    for frac in fracs:
        n = max(80, int(len(train_pool) * frac))
        sub = train_pool.sample(n=n, random_state=int(frac * 1000))
        pipe = build_pipe(PRIMARY_MODEL, features)
        pipe.fit(sub[features], sub[TARGET])
        rows.append({"frac": frac, "n": n, "train_r2": r2_score(sub[TARGET], pipe.predict(sub[features])), "test_r2": r2_score(y_test, pipe.predict(test[features])), "test_rmse": mean_squared_error(y_test, pipe.predict(test[features]), squared=False)})
    lc = pd.DataFrame(rows)
    lc["model"] = PRIMARY_MODEL
    lc.to_csv(DATA_OUT / "FigS1_learning_curve.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.1))
    axes[0].plot(lc["n"], lc["train_r2"], marker="o", color=NATURE["blue"], lw=2, label="Train")
    axes[0].plot(lc["n"], lc["test_r2"], marker="s", color=NATURE["coral"], lw=2, label="Group test")
    axes[0].legend(frameon=False)
    style_axis(axes[0], xlabel="Training records", ylabel="R$^2$")
    panel_label(axes[0], "a")
    axes[1].plot(lc["n"], lc["train_r2"] - lc["test_r2"], marker="o", color=NATURE["red"], lw=2)
    style_axis(axes[1], xlabel="Training records", ylabel="Train-test gap")
    panel_label(axes[1], "b")
    axes[2].plot(lc["n"], lc["test_rmse"], marker="o", color=NATURE["green"], lw=2)
    style_axis(axes[2], xlabel="Training records", ylabel="Group test RMSE")
    panel_label(axes[2], "c")
    for ax in axes:
        numeric_ticks(ax, x=True, y=True)
    save_figure(fig, "FigS1_learning_curves", "FigS1_learning_curves")


def fig_s2_hyperparameter_sensitivity(df: pd.DataFrame) -> None:
    candidate_path = DATA_OUT / "overfit_reduction_candidates.csv"
    if candidate_path.exists():
        raw_hp = pd.read_csv(candidate_path)
        if raw_hp.duplicated(["candidate", "validation"]).any():
            hp = (
                raw_hp.groupby(["candidate", "validation"])
                .agg(train_r2=("train_r2", "mean"), test_r2=("test_r2", "mean"), gap=("gap", "mean"), rmse=("rmse", "mean"), test_sd=("test_r2", "std"), gap_max=("gap", "max"))
                .reset_index()
            )
        else:
            hp = raw_hp
    else:
        base = evaluate_models(df, TARGET, PRIMARY_FEATURE_SET).copy()
        hp = (
            base.groupby(["model", "validation"])
            .agg(train_r2=("train_r2", "mean"), test_r2=("test_r2", "mean"), gap=("gap", "mean"), rmse=("test_rmse", "mean"), test_sd=("test_r2", "std"))
            .reset_index()
            .rename(columns={"model": "candidate"})
        )
    hp.to_csv(DATA_OUT / "FigS2_regularisation_sensitivity.csv", index=False, encoding="utf-8-sig")
    group = hp[hp["validation"] == "group"].copy()
    random_rows = hp[hp["validation"] == "random"].copy()
    stable = group[group["gap"] <= PRIMARY_GAP_LIMIT]
    selected_idx = stable["test_r2"].idxmax() if not stable.empty else group["test_r2"].idxmax()
    selected_name = group.loc[selected_idx, "candidate"]

    fig = plt.figure(figsize=(15.0, 9.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.08, 0.86], hspace=0.34, wspace=0.46)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)])

    def annotate_s2_candidates(ax, data, x_col, y_col, offsets, fontsize=9.2):
        for _, row in data.iterrows():
            name = str(row["candidate"])
            dx, dy = offsets.get(name, (7, 4))
            ax.annotate(
                name,
                (row[x_col], row[y_col]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                fontweight="bold",
                va="center",
                ha="left" if dx >= 0 else "right",
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.76),
                arrowprops=dict(arrowstyle="-", color=NATURE["grey"], lw=0.75, shrinkA=1, shrinkB=4),
                zorder=5,
            )

    colors = np.where(group["gap"] <= PRIMARY_GAP_LIMIT, NATURE["green"], NATURE["coral"])
    axes[0, 0].scatter(group["gap"], group["test_r2"], s=78, c=colors, edgecolor="white", lw=0.8, alpha=0.9)
    axes[0, 0].axvline(PRIMARY_GAP_LIMIT, color=NATURE["ink"], ls="--", lw=1.2)
    axes[0, 0].scatter(group.loc[selected_idx, "gap"], group.loc[selected_idx, "test_r2"], marker="*", s=210, color=NATURE["red"], edgecolor="white", lw=0.7, zorder=4)
    annotate_s2_candidates(
        axes[0, 0],
        group.sort_values("test_r2", ascending=False),
        "gap",
        "test_r2",
        {
            "XGB_current": (-18, -12),
            "HGB_reg1": (9, -2),
            "Cat_reg_current": (-68, 10),
            "XGB_reg1": (-66, -7),
            "HGB_reg2": (9, -14),
            "LGBM_reg1": (9, 2),
            "RF_reg": (9, -13),
            "Cat_reg2": (9, 2),
            "XGB_reg2": (9, -10),
            "XGB_reg3": (9, 8),
            "ET_reg": (9, -2),
        },
    )
    style_axis(axes[0, 0], xlabel="Train-test gap", ylabel="Group test R$^2$")
    panel_label(axes[0, 0], "a")

    bdata = group.sort_values("test_r2", ascending=True).reset_index(drop=True)
    y = np.arange(len(bdata))
    for yi, row in bdata.iterrows():
        lw = 2.0 if row["candidate"] == selected_name else 1.25
        alpha = 0.92 if row["candidate"] == selected_name else 0.62
        axes[0, 1].plot([row["test_r2"], row["train_r2"]], [yi, yi], color=NATURE["grey"], lw=lw, alpha=alpha, zorder=1)
    axes[0, 1].scatter(bdata["train_r2"], y, s=72, marker="^", color=NATURE["navy"], edgecolor="white", lw=0.8, label="Train R$^2$", zorder=3)
    axes[0, 1].scatter(bdata["test_r2"], y, s=76, facecolors="white", edgecolors=NATURE["green"], lw=1.8, label="Group test R$^2$", zorder=4)
    sel_y = int(bdata.index[bdata["candidate"].eq(selected_name)][0])
    axes[0, 1].scatter(bdata.loc[sel_y, "test_r2"], sel_y, marker="*", s=190, color=NATURE["red"], edgecolor="white", lw=0.7, zorder=5)
    axes[0, 1].set_yticks(y)
    axes[0, 1].set_yticklabels(bdata["candidate"])
    axes[0, 1].set_xlim(max(0, bdata[["test_r2", "train_r2"]].min().min() - 0.05), min(1.0, bdata[["test_r2", "train_r2"]].max().max() + 0.05))
    for tick in axes[0, 1].get_yticklabels():
        tick.set_fontsize(9.6)
        tick.set_fontweight("bold")
    style_axis(axes[0, 1], xlabel="R$^2$", ylabel="")
    axes[0, 1].legend(loc="lower right", fontsize=9.8, handlelength=1.1, handletextpad=0.4)
    panel_label(axes[0, 1], "b")

    pivot = pd.merge(
        group[["candidate", "test_r2", "gap"]],
        random_rows[["candidate", "test_r2"]].rename(columns={"test_r2": "random_test_r2"}),
        on="candidate",
        how="inner",
    )
    pivot = pivot.sort_values("test_r2", ascending=False)
    heat = pivot.set_index("candidate")[["random_test_r2", "test_r2"]]
    heat.columns = ["Random split", "Group split"]
    hm_cmap = LinearSegmentedColormap.from_list("s2_validation", [NATURE["light"], NATURE["teal"], NATURE["navy"]])
    sns.heatmap(
        heat,
        ax=axes[1, 0],
        cmap=hm_cmap,
        vmin=max(0.5, float(heat.min().min()) - 0.03),
        vmax=min(0.9, float(heat.max().max()) + 0.03),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 10.0, "fontweight": "bold"},
        linewidths=1.0,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Test R$^2$", "shrink": 0.72, "pad": 0.025},
    )
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("")
    axes[1, 0].tick_params(axis="x", labelrotation=0, labelsize=11.5, bottom=True, top=False, length=0)
    axes[1, 0].tick_params(axis="y", labelrotation=0, labelsize=9.7, left=True, right=False, length=0)
    for label in axes[1, 0].get_xticklabels() + axes[1, 0].get_yticklabels():
        label.set_fontweight("bold")
    cbar = axes[1, 0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=9.2, width=1.0, length=4)
    cbar.ax.yaxis.label.set_size(10.2)
    cbar.ax.yaxis.label.set_weight("bold")
    panel_label(axes[1, 0], "c")

    top = group.sort_values(["gap", "test_r2"], ascending=[True, False]).head(10).copy()
    top = top.sort_values("gap", ascending=True)
    axes[1, 1].barh(top["candidate"], top["gap"], color=np.where(top["candidate"].eq(selected_name), NATURE["red"], NATURE["blue"]), edgecolor="white", alpha=0.86)
    axes[1, 1].axvline(PRIMARY_GAP_LIMIT, color=NATURE["ink"], lw=1.1, ls="--")
    style_axis(axes[1, 1], xlabel="Train-test gap", ylabel="")
    panel_label(axes[1, 1], "d")
    numeric_ticks(axes[0, 0], x=True, y=True)
    numeric_ticks(axes[0, 1], x=True, y=False, n=4)
    numeric_ticks(axes[1, 1], x=True, y=False, n=4)
    for tick in axes[1, 1].get_yticklabels():
        tick.set_fontsize(10.6)
        tick.set_fontweight("bold")
    save_figure(fig, "_archive_merged_into_Fig04/regularisation_sensitivity_backup", "regularisation_sensitivity_backup")


def fig_s3_gas_targets(df: pd.DataFrame) -> None:
    perf = target_perf(df)
    gas = perf[perf["target_short"].isin(["CO2", "CH4", "N2O", "NH3"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.1))
    avail = pd.Series({k: df[v].notna().sum() for k, v in TARGETS.items() if k in ["CO2", "CH4", "N2O", "NH3"]}).sort_values()
    axes[0].barh([SHORT_LABELS.get(i, i) for i in avail.index], avail.values, color=NATURE["teal"], edgecolor="white")
    style_axis(axes[0], xlabel="Available records", ylabel="")
    panel_label(axes[0], "a")
    mean = gas.groupby(["target_short", "validation"])["test_r2"].mean().reset_index()
    pivot = mean.pivot(index="target_short", columns="validation", values="test_r2").loc[avail.index]
    x = np.arange(len(pivot))
    axes[1].bar(x - 0.18, pivot["random"], width=0.36, color=NATURE["blue"], edgecolor="white", label="Random")
    axes[1].bar(x + 0.18, pivot["group"], width=0.36, color=NATURE["coral"], edgecolor="white", label="Group")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SHORT_LABELS.get(i, i) for i in pivot.index])
    axes[1].legend(frameon=False)
    style_axis(axes[1], ylabel="Test R$^2$")
    panel_label(axes[1], "b")
    gap = (pivot["random"] - pivot["group"]).sort_values()
    axes[2].barh([SHORT_LABELS.get(i, i) for i in gap.index], gap.values, color=NATURE["red"], edgecolor="white", alpha=0.82)
    axes[2].axvline(0, color=NATURE["ink"], lw=1)
    style_axis(axes[2], xlabel="Optimism gap", ylabel="")
    panel_label(axes[2], "c")
    for ax in axes:
        numeric_ticks(ax, x=True, y=True)
    save_figure(fig, "FigS2_gas_targets", "FigS2_gas_targets")


def fig5_pred_vs_obs(df: pd.DataFrame) -> None:
    data = df[df[TARGET].notna()].copy().reset_index(drop=True)
    y = data[TARGET].astype(float)
    features = FEATURE_SETS[PRIMARY_FEATURE_SET]
    model_order = ["XGBoost", "HGB", "LightGBM", "CatBoost", "KNN", "RF", "SVR", "ExtraTrees", "Ridge"]
    tr, te = split_indices(data, y, "random", 0, test_size=0.25)
    rows = []
    fitted = {}
    all_values = [y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()]
    for model_name in model_order:
        pipe = build_pipe(model_name, features)
        pipe.fit(data.iloc[tr][features], y.iloc[tr])
        pred_train = pipe.predict(data.iloc[tr][features])
        pred_test = pipe.predict(data.iloc[te][features])
        fitted[model_name] = (pred_train, pred_test)
        all_values.extend([pred_train, pred_test])
        rows.append(
            {
                "model": model_name,
                "validation": "random",
                "seed": 0,
                "train_r2": r2_score(y.iloc[tr], pred_train),
                "test_r2": r2_score(y.iloc[te], pred_test),
                "gap": r2_score(y.iloc[tr], pred_train) - r2_score(y.iloc[te], pred_test),
                "test_rmse": mean_squared_error(y.iloc[te], pred_test, squared=False),
                "test_mae": mean_absolute_error(y.iloc[te], pred_test),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(DATA_OUT / "Fig05_nine_model_prediction_error_metrics.csv", index=False, encoding="utf-8-sig")

    vals = np.concatenate([np.asarray(v, dtype=float) for v in all_values])
    lo, hi = np.nanquantile(vals, [0.005, 0.995])
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    model_colors = {
        "XGBoost": NATURE["navy"],
        "HGB": NATURE["blue"],
        "LightGBM": NATURE["green"],
        "CatBoost": NATURE["coral"],
        "KNN": NATURE["lavender"],
        "RF": NATURE["sand"],
        "SVR": NATURE["red"],
        "ExtraTrees": NATURE["teal"],
        "Ridge": NATURE["brown"],
    }
    fig = plt.figure(figsize=(18.8, 14.7))
    outer = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.22)
    labs = list("abcdefghi")
    rng = np.random.default_rng(2)
    for i, model_name in enumerate(model_order):
        sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i // 3, i % 3], width_ratios=[4.7, 1.0], wspace=0.050)
        ax = fig.add_subplot(sub[0, 0])
        err_ax = fig.add_subplot(sub[0, 1], sharey=ax)
        pred_train, pred_test = fitted[model_name]
        metric = metrics.set_index("model").loc[model_name]
        color = model_colors[model_name]
        train_idx = np.arange(len(tr))
        if len(train_idx) > 900:
            train_idx = rng.choice(train_idx, size=900, replace=False)
        ax.scatter(
            y.iloc[tr].to_numpy()[train_idx],
            pred_train[train_idx],
            s=34,
            marker="^",
            facecolors=color,
            edgecolors=color,
            lw=0.25,
            alpha=0.58,
            zorder=2,
            label="Train fit",
        )
        ax.scatter(
            y.iloc[te],
            pred_test,
            s=40,
            marker="o",
            facecolors="none",
            edgecolor=color,
            lw=1.45,
            alpha=0.95,
            zorder=3,
            label="Random test",
        )
        ax.plot([lo, hi], [lo, hi], color=NATURE["ink"], ls="--", lw=1.1)
        band = (hi - lo) * 0.10
        ax.fill_between([lo, hi], [lo - band, hi - band], [lo + band, hi + band], color=color, alpha=0.07, lw=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.text(
            0.05,
            0.95,
            f"{model_name}\nTrain R$^2$={metric['train_r2']:.3f}\nTest R$^2$={metric['test_r2']:.3f}\nGap={metric['gap']:.3f}\nRMSE={metric['test_rmse']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12.7,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#B8C3CC", lw=0.9, alpha=0.90),
        )
        xlabel = "Observed GI (%)"
        ylabel = "Predicted GI (%)" if i % 3 == 0 else None
        style_axis(ax, xlabel=xlabel, ylabel=ylabel)
        numeric_ticks(ax, x=True, y=True, n=4)
        ax.xaxis.label.set_size(18.8)
        ax.xaxis.labelpad = 6
        if ylabel:
            ax.yaxis.label.set_size(21.0)
            ax.yaxis.labelpad = 8
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(16.0)
            tick.set_fontweight("bold")
        panel_label(ax, labs[i])

        residual = pred_test - y.iloc[te].to_numpy()
        jitter_y = y.iloc[te].to_numpy() + rng.normal(0, (hi - lo) * 0.004, len(te))
        err_ax.scatter(residual, jitter_y, s=22, marker="o", facecolors="none", edgecolor=color, lw=0.95, alpha=0.46)
        err_ax.axvline(0, color=NATURE["ink"], ls="--", lw=1.0)
        emax = np.nanquantile(np.abs(residual), 0.98)
        err_ax.set_xlim(-emax * 1.15, emax * 1.15)
        err_ax.set_ylim(lo, hi)
        style_axis(err_ax, xlabel="Error", ylabel=None)
        err_ax.tick_params(labelleft=False)
        numeric_ticks(err_ax, x=True, y=False, n=3)
        err_ax.xaxis.label.set_size(17.8)
        err_ax.xaxis.labelpad = 6
        for tick in err_ax.get_xticklabels():
            tick.set_fontsize(14.6)
            tick.set_fontweight("bold")
    save_figure(fig, "Fig05_pred_vs_obs", "Fig05_pred_vs_obs")


def write_captions() -> None:
    text = """# Compost figure notes

All figures were generated with a local Nature-style visual system adapted from the templates in `参考/figures` and the HTML examples under `参考/参考文献`. The main claims separate within-distribution prediction from unseen-batch generalisation. Random/stratified splits are reported as within-distribution performance controls, while group-aware validation is used for the conservative generalisation claim.

## Model-selection note

TOPSIS is not used as the main model-selection rule. XGBoost is used as the best predictive model because it gives the highest group-holdout GI accuracy among the nine compared algorithms and also achieves high within-distribution test R^2. Because its group-holdout train-test gap is larger than the conservative 0.15 threshold, CatBoost is retained as a gap-controlled robustness comparator rather than replacing XGBoost as the accuracy-leading model.

## Main figures

- Fig. 1. Data audit and segment-wise field calibration.
- Fig. 2. Target availability, co-observation structure, GI distribution and material-type heterogeneity.
- Fig. 3. Grouped lower-triangle Pearson/Spearman correlation and nonlinear association screening.
- Fig. 4. Merged 4 x 3 model-comparison and regularisation-sensitivity figure. The upper rows report nine-model accuracy, error, selection, split-sensitivity and pairwise-comparison diagnostics; the lower rows integrate the former Fig. S2 overfitting-control and candidate-selection diagnostics.
- Fig. 5. Nine-model within-distribution GI prediction and residual-error panels. Each model uses a different Nature-style colour; train points are filled triangles, random-test points are hollow circles, and side-panel errors are hollow circles on the paired error axis. This is a benchmark diagnostic, not an unseen-batch generalisation claim.
- Fig. 6. Random-split optimism gap and batch-proxy leakage logic.
- Fig. 7. Residual diagnostics, Y-randomisation, interval coverage and Williams-style applicability domain.
- Fig. 8. Global SHAP, permutation importance, grouped SHAP contribution and SHAP dependence diagnostics for the XGBoost primary model.
- Fig. 9. PDP/ALE nonlinear effects and four two-dimensional interaction surfaces.
- Fig. 10. P0/P0+EC/P1 feature-protocol ablation.

## Supplementary figures

- Fig. S1. Learning curves and train-test gap.
- Fig. S2. Gas-emission target availability and robustness.
"""
    (OUT / "图片说明.md").write_text(text, encoding="utf-8")


def make_contact_sheet() -> None:
    pngs = [
        OUT / "Fig01_data_audit" / "Fig01_data_audit.png",
        OUT / "Fig02_target_distribution" / "Fig02_target_distribution.png",
        OUT / "Fig03_correlation" / "Fig03_correlation.png",
        OUT / "Fig04_model_comparison" / "Fig04_model_comparison.png",
        OUT / "Fig05_pred_vs_obs" / "Fig05_pred_vs_obs.png",
        OUT / "Fig06_temporal_validation" / "Fig06_temporal_validation.png",
        OUT / "Fig07_residual_reliability" / "Fig07_residual_reliability.png",
        OUT / "Fig08_SHAP_importance" / "Fig08_SHAP_importance.png",
        OUT / "Fig09_PDP_ALE_interaction" / "Fig09_PDP_ALE_interaction.png",
        OUT / "Fig10_feature_ablation" / "Fig10_feature_ablation.png",
        OUT / "FigS1_learning_curves" / "FigS1_learning_curves.png",
        OUT / "FigS2_gas_targets" / "FigS2_gas_targets.png",
    ]
    thumbs = []
    for p in pngs:
        if p.exists():
            img = Image.open(p).convert("RGB")
            img.thumbnail((520, 360))
            canvas = Image.new("RGB", (540, 400), "white")
            canvas.paste(img, ((540 - img.width) // 2, 10))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 370), p.parent.name, fill=(35, 35, 35))
            thumbs.append(canvas)
    if not thumbs:
        return
    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 540, rows * 400), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 540, (i // cols) * 400))
    sheet.save(OUT / "_contact_sheet.png")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    df = load_compost_data()
    print("Generating Fig01...")
    fig1_data_audit(df)
    print("Generating Fig02...")
    fig2_target_distribution(df)
    print("Generating Fig03...")
    fig3_correlation(df)
    print("Generating Fig04...")
    fig4_model_comparison(df)
    print("Generating Fig05...")
    fig5_pred_vs_obs(df)
    print("Generating Fig06...")
    fig6_optimism_gap(df)
    print("Generating Fig07...")
    fig7_residual_reliability(df)
    print("Generating Fig08...")
    fig8_shap_importance(df)
    print("Generating Fig09...")
    fig9_pdp_ale(df)
    print("Generating Fig10...")
    fig10_feature_ablation(df)
    print("Generating FigS1...")
    fig_s1_learning_curves(df)
    print("Generating FigS2...")
    fig_s3_gas_targets(df)
    write_captions()
    make_contact_sheet()
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()
