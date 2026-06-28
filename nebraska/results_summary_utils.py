def _write_section(handle, title, lines):
    handle.write(f'{title}\n')
    handle.write(f'{"=" * len(title)}\n')
    for line in lines:
        handle.write(f'{line}\n')
    handle.write('\n')


def _metric_lines(metrics):
    return [
        f'Accuracy: {metrics["accuracy"]:.4f}',
        f'Macro F1: {metrics["macro_f1"]:.4f}',
        f'Weighted F1: {metrics["weighted_f1"]:.4f}',
    ]


def _per_class_lines(scores, label_map):
    return [f'{label_map[i]}: {scores[i]:.4f}' for i in range(len(scores))]


def write_results_summary(
    summary_path,
    scenario_name,
    selected_features,
    best_trial,
    seq_length,
    best_val_macro_f1,
    raw_val_macro_f1,
    tuned_val_macro_f1,
    class_multipliers,
    raw_metrics,
    tuned_metrics,
    per_class_f1_raw,
    per_class_f1_tuned,
    report_raw,
    report_tuned,
    trial_results,
    label_map,
    extra_sections=None,
):
    with open(summary_path, 'w', encoding='utf-8') as handle:
        handle.write(f'{scenario_name}\n')
        handle.write(f'{"=" * len(scenario_name)}\n\n')
        handle.write(f'Best trial: {best_trial["name"]}\n')
        handle.write(f'Best trial config: {best_trial}\n')
        handle.write(f'Seq Length: {seq_length}\n')
        handle.write(f'Val Macro F1 (best trial): {best_val_macro_f1:.4f}\n')
        handle.write(f'Val Macro F1 (raw): {raw_val_macro_f1:.4f}\n')
        handle.write(f'Val Macro F1 (tuned): {tuned_val_macro_f1:.4f}\n')
        handle.write(f'Class multipliers: {class_multipliers.tolist()}\n\n')

        _write_section(handle, 'Features Used', selected_features)

        for title, lines in extra_sections or []:
            _write_section(handle, title, lines)

        leaderboard_lines = [
            f'{row["name"]}: {row["val_macro_f1"]:.4f} (balancer={row["balancer"]})'
            for row in sorted(trial_results, key=lambda item: item['val_macro_f1'], reverse=True)
        ]
        _write_section(handle, 'Trial Leaderboard', leaderboard_lines)
        _write_section(handle, 'Raw Results', _metric_lines(raw_metrics))
        _write_section(handle, 'Tuned Results', _metric_lines(tuned_metrics))
        _write_section(handle, 'Per-Class F1 Raw', _per_class_lines(per_class_f1_raw, label_map))
        _write_section(handle, 'Per-Class F1 Tuned', _per_class_lines(per_class_f1_tuned, label_map))
        _write_section(handle, 'Classification Report Raw', report_raw.rstrip().splitlines())
        _write_section(handle, 'Classification Report Tuned', report_tuned.rstrip().splitlines())
