from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def decumulate_drought(row):
    pmf_d4 = row['D4']
    pmf_d3 = max(0.0, row['D3'] - row['D4'])
    pmf_d2 = max(0.0, row['D2'] - row['D3'])
    pmf_d1 = max(0.0, row['D1'] - row['D2'])
    pmf_d0 = max(0.0, row['D0'] - row['D1'])
    pmf_none = max(0.0, row['None'])
    return pd.Series([pmf_none, pmf_d0, pmf_d1, pmf_d2, pmf_d3, pmf_d4])


def build_parser():
    parser = argparse.ArgumentParser(description='Run Kansas EDA and save visualization outputs.')
    parser.add_argument(
        '--data-path',
        type=Path,
        default=None,
        help='Path to Integrated_weekly_KAN_20counties.csv (defaults to repo root).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory to save EDA figures (defaults to kansas/EDA).',
    )
    return parser


def main(data_path=None, output_dir=None):
    sns.set_theme(style='whitegrid', context='talk')
    plt.rcParams.update({
        'figure.dpi': 140,
        'savefig.dpi': 180,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parents[1]

    if data_path is not None:
        data_path = Path(data_path)
    else:
        candidate_paths = [
            project_root / 'Integrated_weekly_KAN_20counties.csv',
            base_dir.parent / 'Integrated_weekly_KAN_20counties.csv',
            base_dir / 'Integrated_weekly_KAN_20counties.csv',
        ]
        data_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    output_dir = Path(output_dir) if output_dir is not None else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    eda_fig_path = output_dir / 'kansas_eda.png'
    weekly_fig_path = output_dir / 'kansas_weekly_majority_share.png'

    df = pd.read_csv(data_path)
    df['week_start'] = pd.to_datetime(df['week_start'])
    df['ValidEnd'] = pd.to_datetime(df['ValidEnd'])
    df = df.sort_values(['FIPS', 'week_start']).reset_index(drop=True)

    pmf_cols = ['PMF_None', 'PMF_D0', 'PMF_D1', 'PMF_D2', 'PMF_D3', 'PMF_D4']
    df[pmf_cols] = df.apply(decumulate_drought, axis=1)
    df['Label'] = df[pmf_cols].idxmax(axis=1).apply(lambda x: pmf_cols.index(x))
    label_map = {0: 'None', 1: 'D0', 2: 'D1', 3: 'D2', 4: 'D3', 5: 'D4'}

    weather_cols = ['ALLSKY_SFC_SW_DWN', 'PRECTOTCORR', 'PS', 'RH2M', 'T2M', 'WS2M']
    drought_cols = ['None', 'D0', 'D1', 'D2', 'D3', 'D4']
    eda_cols = drought_cols + weather_cols

    print('=== Missing Values ===')
    missing = df[eda_cols + ['FIPS', 'week_start', 'ValidEnd']].isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print('No missing values in selected EDA columns.')
    else:
        display(missing.to_frame(name='missing_count'))

    print('\n=== Summary Statistics (mean/std/min/max) ===')
    eda_summary = df[eda_cols].agg(['mean', 'std', 'min', 'max']).T
    display(eda_summary.round(3))

    print('\n=== Class Imbalance ===')
    label_counts = df['Label'].value_counts().sort_index()
    label_df = pd.DataFrame({
        'label': [label_map[i] for i in label_counts.index],
        'count': label_counts.values,
        'share_pct': (label_counts.values / label_counts.sum() * 100).round(2),
    })
    display(label_df)

    majority_label = int(label_counts.idxmax())
    minority_label = int(label_counts.idxmin())
    majority_name = label_map[majority_label]
    minority_name = label_map[minority_label]

    fig, axes = plt.subplots(3, 2, figsize=(18, 18), constrained_layout=True)
    fig.suptitle('Kansas Weekly EDA', fontsize=18, fontweight='bold')

    label_palette = sns.color_palette('deep', n_colors=len(label_counts))
    sns.barplot(x=[label_map[i] for i in label_counts.index], y=label_counts.values, ax=axes[0, 0], palette=label_palette)
    axes[0, 0].set_title('Drought Label Distribution')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=0)
    axes[0, 0].grid(alpha=0.2, axis='y')

    feature_means = df[weather_cols].mean().sort_values(ascending=False)
    sns.barplot(x=feature_means.index, y=feature_means.values, ax=axes[0, 1], color='#4C72B0')
    axes[0, 1].set_title('Mean of Core Weather Features')
    axes[0, 1].set_xlabel('Feature')
    axes[0, 1].set_ylabel('Mean')
    axes[0, 1].tick_params(axis='x', rotation=30)
    axes[0, 1].grid(alpha=0.2, axis='y')

    feature_std = df[weather_cols].std().sort_values(ascending=False)
    sns.barplot(x=feature_std.index, y=feature_std.values, ax=axes[1, 0], color='#C44E52')
    axes[1, 0].set_title('Standard Deviation of Core Weather Features')
    axes[1, 0].set_xlabel('Feature')
    axes[1, 0].set_ylabel('Std Dev')
    axes[1, 0].tick_params(axis='x', rotation=30)
    axes[1, 0].grid(alpha=0.2, axis='y')

    corr = df[weather_cols + drought_cols].corr()
    sns.heatmap(corr, cmap='vlag', center=0, vmin=-1, vmax=1, ax=axes[1, 1], square=False, cbar_kws={'label': 'Correlation'})
    axes[1, 1].set_title('Correlation Heatmap')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].tick_params(axis='y', rotation=0)

    majority_mask = df['Label'] == majority_label
    minority_mask = df['Label'] == minority_label
    sns.kdeplot(data=df.loc[majority_mask, 'T2M'], ax=axes[2, 0], label=f'Majority: {majority_name}', color='#4C72B0', fill=True, alpha=0.25, linewidth=2)
    sns.kdeplot(data=df.loc[minority_mask, 'T2M'], ax=axes[2, 0], label=f'Minority: {minority_name}', color='#DD8452', fill=True, alpha=0.25, linewidth=2)
    axes[2, 0].set_title('T2M Distribution: Majority vs Minority')
    axes[2, 0].set_xlabel('T2M')
    axes[2, 0].set_ylabel('Density')
    axes[2, 0].legend(frameon=True)
    axes[2, 0].grid(alpha=0.2)

    box_df = df[df['Label'].isin([majority_label, minority_label])].copy()
    box_df['Class'] = box_df['Label'].map(label_map)
    sns.boxplot(data=box_df, x='Class', y='PRECTOTCORR', palette=['#4C72B0', '#DD8452'], ax=axes[2, 1])
    axes[2, 1].set_title('Precipitation by Majority vs Minority Class')
    axes[2, 1].set_xlabel('Class')
    axes[2, 1].set_ylabel('PRECTOTCORR')
    axes[2, 1].grid(alpha=0.2, axis='y')

    fig.savefig(eda_fig_path, bbox_inches='tight')
    plt.show()

    weekly_counts = df.groupby(['week_start', 'Label']).size().unstack(fill_value=0).sort_index()
    weekly_majority_share = (weekly_counts.get(majority_label, 0) / weekly_counts.sum(axis=1)).fillna(0)
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(weekly_majority_share.index, weekly_majority_share.values, color='#4C72B0', linewidth=1.8, label=f'Majority share: {majority_name}')
    ax.axhline(0.5, color='#666666', linestyle='--', linewidth=1, alpha=0.7, label='50% threshold')
    minority_periods = weekly_majority_share.index[weekly_majority_share.values < 0.5]
    for ts in minority_periods:
        ax.axvspan(ts, ts + pd.Timedelta(days=7), color='#DD8452', alpha=0.12)
    ax.set_title('Kansas Weekly Majority Share with Minority-Dominant Periods Highlighted')
    ax.set_xlabel('Week Start')
    ax.set_ylabel(f'Share of {majority_name}')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(weekly_fig_path, bbox_inches='tight')
    plt.show()

    print(f'Saved EDA figure to: {eda_fig_path}')
    print(f'Saved weekly share figure to: {weekly_fig_path}')


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(data_path=args.data_path, output_dir=args.output_dir)
