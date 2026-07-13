from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    missing_module = getattr(exc, "name", "dependency")
    raise SystemExit(
        "Dependensi visualisasi belum terpasang. "
        f"Install dulu modul `{missing_module}` lalu jalankan ulang script ini."
    ) from exc

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


CORRELATION_FILES = {
    "feature_feature_abs_correlation_all.csv": "All Features Absolute Correlation",
    "feature_feature_abs_correlation_selected.csv": "Selected Features Absolute Correlation",
}

MI_FILE = "feature_model_mutual_information.csv"


def load_correlation_matrix(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def plot_correlation_heatmap(csv_path: Path, title: str) -> Path:
    corr_df = load_correlation_matrix(csv_path)
    n_features = max(len(corr_df.columns), 1)
    width = max(10, min(28, n_features * 0.55))
    height = max(8, min(24, n_features * 0.5))

    plt.figure(figsize=(width, height))
    if sns is not None:
        sns.heatmap(
            corr_df,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            square=False,
            cbar_kws={"label": "Absolute correlation"},
        )
    else:  # pragma: no cover
        plt.imshow(corr_df, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(label="Absolute correlation")
        plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
        plt.yticks(range(len(corr_df.index)), corr_df.index)

    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = csv_path.with_name(f"{csv_path.stem}_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_mutual_information(csv_path: Path, top_n: int) -> Path:
    mi_df = pd.read_csv(csv_path)
    mi_df["mutual_information_score"] = pd.to_numeric(
        mi_df["mutual_information_score"], errors="coerce"
    )
    mi_df = mi_df.dropna(subset=["feature", "mutual_information_score"])
    mi_df = mi_df.sort_values("mutual_information_score", ascending=False).head(top_n)
    mi_df = mi_df.iloc[::-1]

    height = max(6, min(16, len(mi_df) * 0.45))
    plt.figure(figsize=(12, height))
    colors = ["#1f77b4"] * len(mi_df)

    plt.barh(mi_df["feature"], mi_df["mutual_information_score"], color=colors)
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Feature")
    plt.title(f"Top {len(mi_df)} Feature-Model Mutual Information")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_path = csv_path.with_name(f"{csv_path.stem}_bar.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize feature-feature correlation and feature-model mutual information CSV files."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path.cwd(),
        help="Folder containing scenario2 feature CSV exports.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum number of features to show in the mutual information plot.",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    created_files: list[Path] = []

    for filename, title in CORRELATION_FILES.items():
        csv_path = folder / filename
        if csv_path.exists():
            created_files.append(plot_correlation_heatmap(csv_path, title))
        else:
            print(f"Skip: {csv_path.name} tidak ditemukan di {folder}")

    mi_csv_path = folder / MI_FILE
    if mi_csv_path.exists():
        created_files.append(plot_mutual_information(mi_csv_path, args.top_n))
    else:
        print(f"Skip: {mi_csv_path.name} tidak ditemukan di {folder}")

    if not created_files:
        raise FileNotFoundError(f"Tidak ada file CSV target ditemukan di {folder}")

    print("Visualisasi berhasil dibuat:")
    for created_file in created_files:
        print(f"- {created_file}")


if __name__ == "__main__":
    main()
