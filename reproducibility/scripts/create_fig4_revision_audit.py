from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "generated"


ANALYSIS = ROOT / "generated" / "revision"
OUT = PROJECT_ROOT / "figures_compost" / "FigS8_revision_audit"
SUB = OUT / "subfigures"
OUT.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)

NATURE = {
    "blue": "#6EC6DC",
    "green": "#4DB6AC",
    "red": "#E76F61",
    "navy": "#5874A6",
    "coral": "#F4A582",
    "teal": "#9DD7D1",
    "lavender": "#A7B3D3",
    "sand": "#F1D36B",
    "grey": "#8A8F93",
    "light": "#F7FAFA",
    "ink": "#2B2B2B",
}


plt.rcParams.update(
    {
        "font.family": "Arial",
        "savefig.dpi": 600,
        "figure.dpi": 160,
        "axes.linewidth": 1.55,
        "axes.labelweight": "bold",
        "axes.labelsize": 14.6,
        "xtick.labelsize": 12.4,
        "ytick.labelsize": 12.4,
        "legend.fontsize": 11.0,
        "xtick.major.size": 6.2,
        "ytick.major.size": 6.2,
        "xtick.major.width": 1.45,
        "ytick.major.width": 1.45,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axis(ax, xlabel=None, ylabel=None):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="out", bottom=True, left=True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def panel_label(ax, label: str, x=-0.18, y=1.13):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=17.0,
        fontweight="bold",
        va="top",
        ha="left",
    )


def export_panel(fig, ax, stem: str, extra_axes=None):
    renderer = fig.canvas.get_renderer()
    axes = [ax] + list(extra_axes or [])
    bbox = Bbox.union([axis.get_tightbbox(renderer) for axis in axes]).expanded(1.08, 1.12)
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(SUB / f"{stem}.png", dpi=600, bbox_inches=bbox, facecolor="white")
    fig.savefig(SUB / f"{stem}.pdf", bbox_inches=bbox, facecolor="white")


def main() -> None:
    seg_perf = pd.read_csv(ANALYSIS / "segment_a_p0_p1_control_raw.csv")
    day_perf = pd.read_csv(ANALYSIS / "day_sensitivity_raw.csv")
    conf = pd.read_csv(ANALYSIS / "split_conformal_raw.csv")
    shap_norm = pd.read_csv(ANALYSIS / "shap_group_normalisation.csv")

    fig = plt.figure(figsize=(11.2, 6.85), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.52)
    panel_axes = {}
    extra_axes = {}

    # a. Composting-day sensitivity.
    ax = fig.add_subplot(gs[0, 0])
    panel_axes["a"] = ax
    day_group = day_perf[day_perf["validation"].eq("group")].groupby("feature_set").agg(test_r2=("test_r2", "mean"), rmse=("rmse", "mean"))
    order = ["Day_plus_waste", "P0_no_day", "P1_no_day"]
    labels = ["Day + waste", "P0\nno day", "P1\nno day"]
    vals = day_group.loc[order, "test_r2"]
    colors = [NATURE["lavender"], NATURE["blue"], NATURE["green"]]
    ax.bar(np.arange(3), vals, color=colors, edgecolor="white", width=0.64)
    ax.axhline(0.746, color=NATURE["red"], ls="--", lw=1.35)
    ax.text(2.42, 0.752, "Full P1", color=NATURE["red"], fontsize=11.2, fontweight="bold", va="bottom")
    for xi, val in enumerate(vals):
        ax.text(xi, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=11.4, fontweight="bold")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.38, 0.82)
    style_axis(ax, xlabel="", ylabel="Group test R$^2$")
    panel_label(ax, "a")

    # b. Split-conformal coverage and interval width.
    ax = fig.add_subplot(gs[0, 1])
    panel_axes["b"] = ax
    conf_sum = conf.groupby("nominal_coverage").agg(coverage=("coverage", "mean"), width=("interval_width_gi_units", "mean")).reset_index()
    x = np.arange(len(conf_sum))
    ax.bar(x - 0.18, conf_sum["coverage"] * 100, width=0.36, color=NATURE["navy"], edgecolor="white", label="Empirical coverage")
    ax.scatter(x - 0.18, conf_sum["nominal_coverage"] * 100, s=72, marker="D", color=NATURE["sand"], edgecolor=NATURE["ink"], linewidth=0.6, zorder=4, label="Nominal")
    ax2 = ax.twinx()
    extra_axes["b"] = [ax2]
    ax2.bar(x + 0.18, conf_sum["width"], width=0.36, color=NATURE["coral"], edgecolor="white", alpha=0.82, label="Mean interval width")
    for xi, row in conf_sum.iterrows():
        ax.text(xi - 0.18, row["coverage"] * 100 + 1.3, f"{row['coverage'] * 100:.1f}%", ha="center", va="bottom", fontsize=11.2, fontweight="bold")
        ax2.text(xi + 0.18, row["width"] + 2.5, f"{row['width']:.1f}", ha="center", va="bottom", fontsize=11.2, fontweight="bold", color=NATURE["red"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v * 100)}%" for v in conf_sum["nominal_coverage"]])
    ax.set_ylim(78, 100)
    ax2.set_ylim(0, 95)
    style_axis(ax, xlabel="Conformal interval", ylabel="Coverage (%)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", direction="out")
    for lab in ax2.get_yticklabels():
        lab.set_fontweight("bold")
    ax2.set_ylabel("Interval width (GI)", fontweight="bold")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower center", bbox_to_anchor=(0.50, 1.005), ncol=3)
    panel_label(ax, "b", x=-0.31, y=1.10)

    # c. Grouped SHAP normalised by encoded feature count.
    ax = fig.add_subplot(gs[1, 0])
    panel_axes["c"] = ax
    shap_order = ["Process", "Initial", "Chemistry", "Waste type"]
    shap_plot = shap_norm.set_index("group").loc[shap_order]
    colors = [NATURE["green"], NATURE["blue"], NATURE["coral"], NATURE["lavender"]]
    ax.bar(np.arange(len(shap_order)), shap_plot["per_encoded_feature_mean_abs_shap"], color=colors, edgecolor="white", width=0.64)
    for xi, (group, row) in enumerate(shap_plot.iterrows()):
        ax.text(
            xi,
            row["per_encoded_feature_mean_abs_shap"] + 0.20,
            f"{row['cumulative_share_percent']:.1f}%\nn={int(row['encoded_feature_count'])}",
            ha="center",
            va="bottom",
            fontsize=10.8,
            fontweight="bold",
        )
    ax.set_xticks(np.arange(len(shap_order)))
    ax.set_xticklabels(["Process", "Initial", "Chem.", "Waste"], rotation=0)
    ax.set_ylim(0, max(shap_plot["per_encoded_feature_mean_abs_shap"]) * 1.32)
    style_axis(ax, xlabel="", ylabel="Mean |SHAP| / feature")
    panel_label(ax, "c")

    # d. Segment A-only P0/P1 control.
    ax = fig.add_subplot(gs[1, 1])
    panel_axes["d"] = ax
    seg_group = seg_perf[seg_perf["validation"].eq("group")].groupby("feature_set").agg(test_r2=("test_r2", "mean"), rmse=("rmse", "mean")).loc[["P0", "P1"]]
    x = np.arange(2)
    ax.bar(x, seg_group["test_r2"], color=[NATURE["blue"], NATURE["green"]], edgecolor="white", width=0.62)
    for xi, (feature_set, row) in enumerate(seg_group.iterrows()):
        ax.text(xi, row["test_r2"] + 0.018, f"R$^2$={row['test_r2']:.3f}\nRMSE={row['rmse']:.1f}", ha="center", va="bottom", fontsize=11.2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["P0", "P1"])
    ax.set_ylim(0.70, 0.92)
    style_axis(ax, xlabel="Segment A only", ylabel="Group test R$^2$")
    panel_label(ax, "d")

    fig.savefig(OUT / "FigS8_revision_audit.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "FigS8_revision_audit.pdf", bbox_inches="tight", facecolor="white")
    fig.canvas.draw()
    for label, axis in panel_axes.items():
        export_panel(fig, axis, f"FigS8_revision_audit_{label}", extra_axes.get(label))
    plt.close(fig)

    (SUB / "README.md").write_text(
        "Subfigures are exported from the current FigS8_revision_audit combined figure.\n",
        encoding="utf-8",
    )
    print(OUT / "FigS8_revision_audit.png")
    print(OUT / "FigS8_revision_audit.pdf")


if __name__ == "__main__":
    main()
