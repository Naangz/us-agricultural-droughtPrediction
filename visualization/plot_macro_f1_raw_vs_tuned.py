from pathlib import Path
import re

HAS_PLOTTING_LIBRARIES = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    HAS_PLOTTING_LIBRARIES = False
    plt = None

try:
    import seaborn as sns

    if HAS_PLOTTING_LIBRARIES:
        sns.set_theme(style="whitegrid")
except ImportError:
    sns = None


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
SCENARIOS = [
    "Scenario 1",
    "Scenario 2",
    "Scenario 2A",
    "Scenario 2B",
    "Scenario 2C",
    "Scenario 3",
    "Scenario 4",
    "Scenario 5",
    "Scenario 6",
    "Scenario 7",
]
REGIONS = {
    "kansas": {
        "label": "Kansas",
        "color_raw": "#90CAF9",
        "color_tuned": "#1565C0",
    },
    "nebraska": {
        "label": "Nebraska",
        "color_raw": "#FFCC80",
        "color_tuned": "#E65100",
    },
}
SUMMARY_PATHS = {
    "Scenario 1": {
        "kansas": Path("kansas/output_weekly_kansas_20counties/results_summary.txt"),
        "nebraska": Path("nebraska/output_weekly_nebraska_20counties/results_summary.txt"),
    },
    "Scenario 2": {
        "kansas": Path("kansas/output_weekly_kansas_scenario2/results_summary.txt"),
        "nebraska": Path("nebraska/output_weekly_nebraska_scenario2/results_summary.txt"),
    },
    "Scenario 2A": {
        "kansas": Path("kansas/output_weekly_kansas_scenario2A/results_summary.txt"),
        "nebraska": Path("nebraska/output_weekly_nebraska_scenario2A/results_summary.txt"),
    },
    "Scenario 2B": {
        "kansas": Path("kansas/output_weekly_kansas_scenario2B/results_summary.txt"),
        "nebraska": Path("nebraska/output_weekly_nebraska_scenario2B/results_summary.txt"),
    },
    "Scenario 2C": {
        "kansas": Path("kansas/output_weekly_kansas_scenario2C/results_summary.txt"),
        "nebraska": Path("nebraska/output_weekly_nebraska_scenario2C/results_summary.txt"),
    },
    "Scenario 3": {
        "kansas": Path("kansas/output_weekly_kansas_scenario3/results_summary.txt"),
        "nebraska": Path("nebraska/output_weekly_nebraska_scenario3/results_summary.txt"),
    },
    "Scenario 4": {
        "kansas": Path("kansas/output_weekly_kansas_scenario4/results_summary.txt"),
        "nebraska": Path("nebraska/output_weekly_nebraska_scenario4/results_summary.txt"),
    },
    "Scenario 5": {
        "kansas": Path("kansas/output_weekly_kansas_scenario5/results_summary.txt"),
        "nebraska": Path("nebraska/output_weekly_nebraska_scenario5/results_summary.txt"),
    },
    "Scenario 6": {
        "kansas": Path("kansas/output_weekly_kansas_scenario6/results_summary.txt"),
        "nebraska": Path("nebraska/output_weekly_nebraska_scenario6/results_summary.txt"),
    },
    "Scenario 7": {
        "kansas": Path("kansas/output_weekly_kansas_scenario7/results_summary.txt"),
        "nebraska": Path("nebraska/output_weekly_nebraska_scenario7/results_summary.txt"),
    },
}


if HAS_PLOTTING_LIBRARIES:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI", "Helvetica"],
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
        }
    )


def summary_path(region, scenario):
    return SUMMARY_PATHS[scenario][region]


def require_plotting_libraries():
    if not HAS_PLOTTING_LIBRARIES:
        raise ImportError("matplotlib belum terinstall sehingga plot tidak bisa dibuat.")


def parse_macro_f1_summary(summary_file):
    content = Path(summary_file).read_text(encoding="utf-8")

    raw_match = re.findall(r"^\s*Macro F1 \(raw\):\s*([\d.]+)\s*$", content, flags=re.MULTILINE)
    if not raw_match:
        raise ValueError(f"Gagal parse Macro F1 (raw) dari {summary_file}")

    tuned_matches = re.findall(r"^\s*Macro F1:\s*([\d.]+)\s*$", content, flags=re.MULTILINE)
    if not tuned_matches:
        raise ValueError(f"Gagal parse Macro F1 tuned dari {summary_file}")

    best_trial_match = re.search(r"^\s*Best trial:\s*(.+?)\s*$", content, flags=re.MULTILINE)

    return {
        "title": content.splitlines()[0].strip(),
        "best_trial": best_trial_match.group(1).strip() if best_trial_match else None,
        "raw_macro_f1": float(raw_match[-1]),
        "tuned_macro_f1": float(tuned_matches[-1]),
    }


def load_region_scores(region):
    scores = []
    for scenario in SCENARIOS:
        parsed = parse_macro_f1_summary(REPO_ROOT / summary_path(region, scenario))
        scores.append(
            {
                "scenario": scenario,
                "raw_macro_f1": parsed["raw_macro_f1"],
                "tuned_macro_f1": parsed["tuned_macro_f1"],
                "best_trial": parsed["best_trial"],
            }
        )
    return scores


def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            rotation=90,
        )


def plot_region_comparison(ax, region, region_scores):
    config = REGIONS[region]
    positions = list(range(len(region_scores)))
    raw_scores = [item["raw_macro_f1"] for item in region_scores]
    tuned_scores = [item["tuned_macro_f1"] for item in region_scores]
    labels = [item["scenario"].replace("Scenario ", "S") for item in region_scores]
    width = 0.36

    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, color="#BDBDBD", zorder=0)

    raw_bars = ax.bar(
        [x - width / 2 for x in positions],
        raw_scores,
        width=width,
        color=config["color_raw"],
        edgecolor="white",
        linewidth=1.2,
        label="Raw",
        zorder=3,
    )
    tuned_bars = ax.bar(
        [x + width / 2 for x in positions],
        tuned_scores,
        width=width,
        color=config["color_tuned"],
        edgecolor="white",
        linewidth=1.2,
        label="Tuned",
        zorder=3,
    )

    add_value_labels(ax, raw_bars)
    add_value_labels(ax, tuned_bars)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 0.92)
    ax.set_ylabel("Macro F1-Score")
    ax.set_xlabel("Skenario")
    ax.set_title(f"{config['label']} - Raw vs Tuned", fontweight="bold", pad=12)


def create_plot(output_path=None):
    require_plotting_libraries()
    kansas_scores = load_region_scores("kansas")
    nebraska_scores = load_region_scores("nebraska")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.patch.set_facecolor("#F8F9FA")

    plot_region_comparison(axes[0], "kansas", kansas_scores)
    plot_region_comparison(axes[1], "nebraska", nebraska_scores)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle(
        "Perbandingan Macro F1 Antar Skenario\nRaw vs Tuned per Wilayah (Kansas dan Nebraska)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
        color="#0F172A",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    save_path = output_path or OUTPUT_DIR / "macro_f1_raw_vs_tuned_by_region.png"
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return Path(save_path)


def main():
    output_path = create_plot()
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
