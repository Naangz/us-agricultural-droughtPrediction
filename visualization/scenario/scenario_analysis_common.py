import os
import re
import ast

HAS_CORE_DATA_LIBRARIES = True
try:
    import numpy as np
    import pandas as pd
except ImportError:
    HAS_CORE_DATA_LIBRARIES = False
    np = None
    pd = None

HAS_PLOTTING_LIBRARIES = True
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    HAS_PLOTTING_LIBRARIES = False

try:
    import seaborn as sns

    if HAS_PLOTTING_LIBRARIES:
        sns.set_theme(style="whitegrid")
except ImportError:
    sns = None

HAS_ML_LIBRARIES = True
try:
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
except ImportError:
    HAS_ML_LIBRARIES = False


CLASSES = ["None", "D0", "D1", "D2", "D3", "D4"]
PMF_COLS = ["PMF_None", "PMF_D0", "PMF_D1", "PMF_D2", "PMF_D3", "PMF_D4"]
TRAIN_END_DATE = "2019-12-31"
TEST_START_DATE = "2022-01-01"
DATASET_PATHS = {
    "kansas": "Integrated_weekly_KAN_20counties.csv",
    "nebraska": "Integrated_weekly_NEB_20counties.csv",
}
BASE_WEATHER_FEATURES = [
    "ALLSKY_SFC_SW_DWN",
    "PRECTOTCORR",
    "PS",
    "RH2M",
    "T2M",
    "WS2M",
]
WEATHER_LAG_FEATURES = [
    "PREC_lag1",
    "PREC_lag2",
    "PREC_lag4",
    "PREC_lag8",
    "T2M_lag1",
    "T2M_lag2",
    "T2M_lag4",
    "T2M_lag8",
    "RH2M_lag1",
    "RH2M_lag2",
    "RH2M_lag4",
    "RH2M_lag8",
]
ROLLING_FEATURES = [
    "PREC_roll4_mean",
    "PREC_roll4_std",
    "PREC_roll12_mean",
    "PREC_roll12_std",
    "T2M_roll4_mean",
    "T2M_roll12_mean",
]
SEASONAL_FEATURES = ["week_sin", "week_cos"]
DROUGHT_LAG_FEATURES = [
    "None_lag1",
    "D0_lag1",
    "D1_lag1",
    "D2_lag1",
    "D3_lag1",
    "D4_lag1",
    "None_lag2",
    "D0_lag2",
    "D1_lag2",
    "D2_lag2",
    "D3_lag2",
    "D4_lag2",
]
ALL_ENGINEERED_FEATURES = (
    BASE_WEATHER_FEATURES
    + WEATHER_LAG_FEATURES
    + ROLLING_FEATURES
    + SEASONAL_FEATURES
    + DROUGHT_LAG_FEATURES
)
REGIONS = {
    "kansas": {
        "label": "Kansas",
        "summary_template": os.path.join("kansas", "output_weekly_kansas_scenario{scenario}", "results_summary.txt"),
        "model_template": os.path.join("kansas", "output_weekly_kansas_scenario{scenario}", "best_model.keras"),
        "actual_color": "#1565C0",
        "pred_color": "#FF7043",
        "cm_cmap": "Blues",
    },
    "nebraska": {
        "label": "Nebraska",
        "summary_template": os.path.join("nebraska", "output_weekly_nebraska_scenario{scenario}", "results_summary.txt"),
        "model_template": os.path.join("nebraska", "output_weekly_nebraska_scenario{scenario}", "best_model.keras"),
        "actual_color": "#E65100",
        "pred_color": "#6A1B9A",
        "cm_cmap": "Oranges",
    },
}
SCENARIO_STYLES = {
    4: {"label": "Weather Only", "color": "#D32F2F"},
    5: {"label": "Weather + Lag", "color": "#F57C00"},
    6: {"label": "No Drought History", "color": "#1976D2"},
    7: {"label": "Drought History Only", "color": "#5E35B1"},
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


def summary_path(region_key, scenario):
    return REGIONS[region_key]["summary_template"].format(scenario=scenario)


def model_path(region_key, scenario):
    return REGIONS[region_key]["model_template"].format(scenario=scenario)


def parse_summary(summary_file):
    with open(summary_file, "r", encoding="utf-8") as file:
        content = file.read()

    macro_f1_match = re.search(r"^Macro F1:\s*([\d.]+)\s*$", content, flags=re.MULTILINE)
    if not macro_f1_match:
        raise ValueError(f"Gagal parse Macro F1 dari {summary_file}")

    seq_length_match = re.search(r"^Seq Length:\s*(\d+)\s*$", content, flags=re.MULTILINE)
    if not seq_length_match:
        raise ValueError(f"Gagal parse Seq Length dari {summary_file}")

    class_multipliers_match = re.search(r"^Class multipliers:\s*(\[[^\n]+\])\s*$", content, flags=re.MULTILINE)
    if not class_multipliers_match:
        raise ValueError(f"Gagal parse class multipliers dari {summary_file}")

    selected_features = []
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "Selected features:":
            continue
        for feature_line in lines[index + 1 :]:
            if not feature_line.startswith("  "):
                break
            stripped = feature_line.strip()
            if stripped:
                selected_features.append(stripped)
        break

    if not selected_features:
        raise ValueError(f"Gagal parse selected features dari {summary_file}")

    per_class = {}
    for label, score in re.findall(r"^\s+(None|D0|D1|D2|D3|D4):\s*([\d.]+)\s*$", content, flags=re.MULTILINE):
        per_class[label] = float(score)

    report_match = re.search(r"Classification Report:\s*(.*)$", content, flags=re.DOTALL)
    if not report_match:
        raise ValueError(f"Gagal parse Classification Report dari {summary_file}")

    class_rows = re.findall(
        r"^\s*(None|D0|D1|D2|D3|D4)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$",
        report_match.group(1),
        flags=re.MULTILINE,
    )
    if len(class_rows) != len(CLASSES):
        raise ValueError(f"Gagal parse baris kelas dari {summary_file}")

    precision = []
    recall = []
    support = []
    for label in CLASSES:
        row = next((r for r in class_rows if r[0] == label), None)
        if row is None:
            raise ValueError(f"Kelas {label} tidak ditemukan di {summary_file}")
        precision.append(float(row[1]))
        recall.append(float(row[2]))
        support.append(int(row[4]))

    return {
        "summary_title": content.splitlines()[0].strip(),
        "macro_f1": float(macro_f1_match.group(1)),
        "seq_length": int(seq_length_match.group(1)),
        "class_multipliers": ast.literal_eval(class_multipliers_match.group(1)),
        "selected_features": selected_features,
        "per_class_f1": [per_class[label] for label in CLASSES],
        "precision": precision,
        "recall": recall,
        "support": support,
        "total": sum(support),
    }


def compute_predicted_counts(precision_arr, recall_arr, support_arr):
    predicted = []
    for precision, recall, support in zip(precision_arr, recall_arr, support_arr):
        if precision == 0.0:
            predicted.append(0)
            continue
        true_positive = recall * support
        predicted_total = true_positive / precision
        predicted.append(int(round(predicted_total)))
    return predicted


def load_scenario_data(scenario):
    scenario_data = {}
    for region_key in REGIONS:
        parsed = parse_summary(summary_path(region_key, scenario))
        parsed["predicted"] = compute_predicted_counts(
            parsed["precision"],
            parsed["recall"],
            parsed["support"],
        )
        scenario_data[region_key] = parsed
    return scenario_data


def require_plotting_libraries():
    if not HAS_PLOTTING_LIBRARIES:
        raise ImportError("matplotlib tidak terinstall sehingga plot tidak bisa dibuat.")


def require_core_data_libraries():
    if not HAS_CORE_DATA_LIBRARIES:
        raise ImportError("numpy/pandas tidak terinstall sehingga evaluasi data tidak bisa dijalankan.")


def decumulate_drought(row):
    pmf_d4 = row["D4"]
    pmf_d3 = max(0.0, row["D3"] - row["D4"])
    pmf_d2 = max(0.0, row["D2"] - row["D3"])
    pmf_d1 = max(0.0, row["D1"] - row["D2"])
    pmf_d0 = max(0.0, row["D0"] - row["D1"])
    pmf_none = max(0.0, row["None"])
    return pd.Series([pmf_none, pmf_d0, pmf_d1, pmf_d2, pmf_d3, pmf_d4])


def engineer_features(df):
    df_fe = df.copy()

    for lag in [1, 2, 4, 8]:
        df_fe[f"PREC_lag{lag}"] = df_fe.groupby("FIPS")["PRECTOTCORR"].shift(lag)
        df_fe[f"T2M_lag{lag}"] = df_fe.groupby("FIPS")["T2M"].shift(lag)
        df_fe[f"RH2M_lag{lag}"] = df_fe.groupby("FIPS")["RH2M"].shift(lag)

    for window in [4, 12]:
        df_fe[f"PREC_roll{window}_mean"] = df_fe.groupby("FIPS")["PRECTOTCORR"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df_fe[f"PREC_roll{window}_std"] = df_fe.groupby("FIPS")["PRECTOTCORR"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0.0)
        )
        df_fe[f"T2M_roll{window}_mean"] = df_fe.groupby("FIPS")["T2M"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    iso_week = df_fe["week_start"].dt.isocalendar().week.astype(int)
    df_fe["week_sin"] = np.sin(2 * np.pi * iso_week / 52.0)
    df_fe["week_cos"] = np.cos(2 * np.pi * iso_week / 52.0)

    for col in CLASSES:
        df_fe[f"{col}_lag1"] = df_fe.groupby("FIPS")[col].shift(1)
        df_fe[f"{col}_lag2"] = df_fe.groupby("FIPS")[col].shift(2)

    return df_fe


def prepare_feature_frame(region_key, selected_features):
    require_core_data_libraries()
    if not HAS_ML_LIBRARIES:
        raise ImportError("sklearn/tensorflow tidak terinstall.")

    df = pd.read_csv(DATASET_PATHS[region_key])
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["ValidEnd"] = pd.to_datetime(df["ValidEnd"])
    df = df.sort_values(["FIPS", "week_start"]).reset_index(drop=True)
    df[PMF_COLS] = df.apply(decumulate_drought, axis=1)
    df["Label"] = df[PMF_COLS].idxmax(axis=1).apply(lambda x: PMF_COLS.index(x))

    df_fe = engineer_features(df)
    df_fe = df_fe.dropna(subset=selected_features + ["Label"]).reset_index(drop=True)
    train_df = df_fe[df_fe["week_start"] <= TRAIN_END_DATE].copy()

    scaler = MinMaxScaler()
    scaler.fit(train_df[selected_features])
    df_fe.loc[:, selected_features] = scaler.transform(df_fe[selected_features])
    return df_fe


def create_sequences_from_df(df_input, feature_columns, label_col, seq_length, id_col="FIPS", start_date=None):
    require_core_data_libraries()
    x_values, y_values = [], []
    for _, group in df_input.groupby(id_col):
        group = group.sort_values("week_start")
        features = group[feature_columns].values
        labels = group[label_col].values
        dates = group["week_start"].values
        if len(group) < seq_length:
            continue
        for index in range(seq_length - 1, len(group)):
            target_date = pd.Timestamp(dates[index])
            if start_date is not None and target_date < pd.Timestamp(start_date):
                continue
            x_values.append(features[index - seq_length + 1 : index + 1])
            y_values.append(labels[index])
    return np.array(x_values), np.array(y_values)


def evaluate_model_dynamically(region_key, scenario, summary_data):
    require_core_data_libraries()
    if not HAS_ML_LIBRARIES:
        raise ImportError("sklearn/tensorflow tidak terinstall.")

    selected_features = summary_data["selected_features"]
    class_multipliers = np.array(summary_data["class_multipliers"], dtype=float)
    seq_length = summary_data["seq_length"]

    print(f"\n--- Evaluasi Dinamis {REGIONS[region_key]['label']} Scenario {scenario} ---")
    print(f"Membaca data: {DATASET_PATHS[region_key]}")
    df_fe = prepare_feature_frame(region_key, selected_features)

    print("Membuat sekuens data uji...")
    x_test_seq, y_test = create_sequences_from_df(
        df_fe,
        selected_features,
        "Label",
        seq_length=seq_length,
        start_date=TEST_START_DATE,
    )

    model_file = model_path(region_key, scenario)
    print(f"Memuat model: {model_file}")
    model = tf.keras.models.load_model(model_file, compile=False)

    print("Melakukan prediksi...")
    y_pred_prob = model.predict(x_test_seq, verbose=0)
    y_pred = np.argmax(y_pred_prob * class_multipliers, axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASSES))))
    support = np.bincount(y_test, minlength=len(CLASSES)).tolist()
    if support != summary_data["support"]:
        raise ValueError(f"Support evaluasi {support} tidak sama dengan summary {summary_data['support']}")

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"Evaluasi sukses. Macro F1 = {macro_f1:.4f}")
    return cm


def plot_perclass_f1_comparison(scenario, scenario_data, output_dir):
    require_plotting_libraries()
    style = SCENARIO_STYLES[scenario]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor("#F7F9FC")
    x = np.arange(len(CLASSES))

    for ax, region_key in zip(axes, ["kansas", "nebraska"]):
        region_meta = REGIONS[region_key]
        data = scenario_data[region_key]
        bars = ax.bar(
            x,
            data["per_class_f1"],
            color=style["color"],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=f"Macro F1 = {data['macro_f1']:.4f}",
        )

        for bar in bars:
            height = bar.get_height()
            if height > 0.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#333333",
                )

        ax.axhline(data["macro_f1"], color=style["color"], linestyle="--", linewidth=1.1, alpha=0.65)
        ax.set_facecolor("#FFFFFF")
        ax.set_title(region_meta["label"], fontweight="bold", pad=10)
        ax.set_xlabel("Kelas Kekeringan", labelpad=8)
        ax.set_ylabel("F1-Score", labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES)
        ax.set_ylim(0, 0.72)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#CCCCCC")
        ax.legend(loc="upper right", frameon=True, facecolor="#FAFAFA", edgecolor="#CCCCCC")

    plt.suptitle(
        f"Perbandingan F1-Score Per Kelas Skenario {scenario}\n{style['label']} - Kansas vs Nebraska",
        fontsize=14,
        fontweight="bold",
        color="#1A237E",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.93])

    output_path = os.path.join(output_dir, f"scenario{scenario}_perclass_f1.png")
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[1/3] Tersimpan: {output_path}")


def plot_distribution_shift(scenario, region_key, region_data, output_dir):
    require_plotting_libraries()
    style = SCENARIO_STYLES[scenario]
    region_meta = REGIONS[region_key]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        gridspec_kw={"height_ratios": [2.2, 1]},
        facecolor="#F7F9FC",
    )
    fig.patch.set_facecolor("#F7F9FC")

    x = np.arange(len(CLASSES))
    bar_width = 0.35
    actual_counts = region_data["support"]
    predicted_counts = region_data["predicted"]
    total = region_data["total"]

    ax_top.set_facecolor("#FFFFFF")
    ax_top.grid(axis="y", linestyle="--", alpha=0.4, color="#CCCCCC", zorder=0)

    bars_actual = ax_top.bar(
        x - bar_width / 2,
        actual_counts,
        bar_width,
        label="Distribusi Aktual (Ground Truth)",
        color=region_meta["actual_color"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    bars_pred = ax_top.bar(
        x + bar_width / 2,
        predicted_counts,
        bar_width,
        label="Distribusi Prediksi Model",
        color=region_meta["pred_color"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    for bars, counts in [(bars_actual, actual_counts), (bars_pred, predicted_counts)]:
        for bar, count in zip(bars, counts):
            if count <= 0:
                continue
            pct = count / total * 100
            ax_top.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 15,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#333333",
            )

    ax_top.set_title(
        f"Pergeseran Distribusi Prediksi vs Aktual - Skenario {scenario} ({style['label']})\n"
        f"Wilayah {region_meta['label']} | Total Sampel Uji: {total:,}",
        fontweight="bold",
        color="#1A237E",
        pad=12,
    )
    ax_top.set_ylabel("Jumlah Sampel", labelpad=8, fontweight="bold")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(CLASSES, fontsize=10)
    ax_top.legend(loc="upper right", frameon=True, facecolor="#FAFAFA", edgecolor="#CCCCCC", fontsize=9.5)
    ax_top.set_xlim(-0.6, len(CLASSES) - 0.4)

    ax_bottom.set_facecolor("#FFFFFF")
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.4, color="#CCCCCC", zorder=0)

    deltas = [pred - actual for pred, actual in zip(predicted_counts, actual_counts)]
    delta_colors = ["#2E7D32" if delta >= 0 else "#C62828" for delta in deltas]
    bars_delta = ax_bottom.bar(
        x,
        deltas,
        0.55,
        color=delta_colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    for bar, delta in zip(bars_delta, deltas):
        sign = "+" if delta >= 0 else ""
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2,
            delta + (20 if delta >= 0 else -40),
            f"{sign}{delta}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#2E7D32" if delta >= 0 else "#C62828",
        )

    ax_bottom.axhline(0, color="#333333", linewidth=1.2)
    ax_bottom.set_title("Delta Prediksi - Aktual", fontsize=10, color="#444444", pad=8)
    ax_bottom.set_ylabel("Delta Count", labelpad=8, fontweight="bold")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(CLASSES, fontsize=10)
    ax_bottom.set_xlim(-0.6, len(CLASSES) - 0.4)

    plt.tight_layout(rect=[0, 0.01, 1, 0.99])

    output_path = os.path.join(output_dir, f"scenario{scenario}_dist_shift_{region_key}.png")
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Tersimpan: {output_path}")


def plot_confusion_matrices(scenario, scenario_data, confusion_matrices, output_dir):
    require_plotting_libraries()
    if sns is None:
        raise ImportError("seaborn tidak terinstall sehingga heatmap confusion matrix tidak bisa dibuat.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.patch.set_facecolor("#F8F9FA")

    for axis, region_key in zip(axes, ["kansas", "nebraska"]):
        region_meta = REGIONS[region_key]
        cm = confusion_matrices[region_key]
        cm_norm = cm.astype(float) / np.maximum(1.0, cm.sum(axis=1, keepdims=True))
        cm_norm = np.nan_to_num(cm_norm)
        labels = np.empty_like(cm, dtype=object)
        for row_index in range(cm.shape[0]):
            for col_index in range(cm.shape[1]):
                labels[row_index, col_index] = f"{cm[row_index, col_index]}\n({cm_norm[row_index, col_index]:.1%})"

        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap=region_meta["cm_cmap"],
            cbar=True,
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            ax=axis,
            annot_kws={"size": 10.5, "weight": "bold"},
            linewidths=0.5,
            linecolor="#EEEEEE",
        )
        axis.set_title(
            f"{region_meta['label']} Scenario {scenario} - Confusion Matrix (Test Data)\n"
            f"Macro F1 Summary: {scenario_data[region_key]['macro_f1']:.4f}",
            fontweight="bold",
            pad=12,
        )
        axis.set_xlabel("Predicted Class", fontweight="bold", labelpad=8)
        axis.set_ylabel("Actual Class", fontweight="bold", labelpad=8)

    plt.suptitle(
        f"Heatmap Confusion Matrix Data Uji Skenario {scenario}\n"
        f"{SCENARIO_STYLES[scenario]['label']} - Kansas dan Nebraska",
        fontsize=15,
        fontweight="bold",
        y=0.96,
        color="#1A237E",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])

    output_path = os.path.join(output_dir, f"scenario{scenario}_confusion_matrices.png")
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Tersimpan: {output_path}")


def run_scenario_analysis(scenario, script_dir):
    if scenario not in SCENARIO_STYLES:
        raise ValueError(f"Skenario {scenario} belum didukung.")

    style = SCENARIO_STYLES[scenario]
    scenario_data = load_scenario_data(scenario)

    print("=" * 65)
    print(f"  Visualisasi Skenario {scenario} - {style['label']}")
    print("=" * 65)

    plot_perclass_f1_comparison(scenario, scenario_data, script_dir)

    for region_key in ["kansas", "nebraska"]:
        region_label = REGIONS[region_key]["label"]
        region_data = scenario_data[region_key]
        print(f"\n{region_label} Sc{scenario} - Distribusi Aktual  : {dict(zip(CLASSES, region_data['support']))}")
        print(f"{region_label} Sc{scenario} - Distribusi Prediksi: {dict(zip(CLASSES, region_data['predicted']))}")
        plot_distribution_shift(scenario, region_key, region_data, script_dir)

    if HAS_CORE_DATA_LIBRARIES and HAS_ML_LIBRARIES and HAS_PLOTTING_LIBRARIES and sns is not None:
        confusion_matrices = {
            region_key: evaluate_model_dynamically(region_key, scenario, region_data)
            for region_key, region_data in scenario_data.items()
        }
        plot_confusion_matrices(scenario, scenario_data, confusion_matrices, script_dir)
    else:
        print("\nConfusion matrix dilewati karena dependency plotting/ML belum lengkap di environment ini.")

    print("\nSelesai! Semua file berhasil disimpan di folder visualization/.")
