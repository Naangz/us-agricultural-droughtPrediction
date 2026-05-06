**Detailed Results Analysis (from results_summary.txt files)**

**Overview**
- This document compiles key metrics and observations extracted from each `results_summary.txt` produced by the scenario scripts. For each scenario I list the best trial, key validation/test metrics (Val Macro F1, Accuracy, Macro F1, Weighted F1), per-class performance highlights, selected features (when present), and short recommendations.

---

**Kansas — Aggregated Scenarios**

- `output_weekly_kansas_20counties/results_summary.txt` (Tuned baseline)
  - Best trial: `ros_focal_no_cw_96x48`
  - Val Macro F1 (best): 0.8514 (raw/tuned 0.8514 / 0.8553)
  - Accuracy: 0.8067 | Macro F1: 0.8039 | Weighted F1: 0.8061
  - Per-class F1 (highlights): None 0.8977, D0 0.7614, D1 0.7435, D4 0.8835
  - Notes: Strong overall performance; class D2 and D3 slightly weaker than top classes.

- `kansas/output_weekly_kansas_clean/results_summary.txt` (Clean baseline)
  - Balancing: ROS
  - Accuracy: 0.6968 | Macro F1: 0.7327 | Weighted F1: 0.6970
  - Per-class F1: D4 extremely strong (0.9154); other classes variable.
  - Notes: Baseline for larger multi-county runs; consider tuning to close gap to tuned runs.

- `kansas/output_weekly_kansas_clean_expB/results_summary.txt` (clean expB)
  - Accuracy: 0.6774 | Macro F1: 0.7303 | Weighted F1: 0.6818
  - Notes: Similar to clean baseline; Inspect differences in preprocessing for expB.

- `kansas/output_weekly_kansas_scenario2/results_summary.txt` (Correlation top-15)
  - Val Macro F1 (best): 0.6092 (raw/tuned 0.6092 / 0.6805)
  - Accuracy: 0.6007 | Macro F1: 0.5262 | Weighted F1: 0.6124
  - Per-class F1: None 0.9164, D0 0.7099, D1 0.6740, D2 0.4914
  - Selected features: precipitation and RH/T2M derived stats + drought carryover
  - Notes: Correlation-based reduction to 15 features yields mixed gains; D2/D3 underperform.

- `kansas/output_weekly_kansas_scenario2A/results_summary.txt` (Correlation top-20)
  - Val Macro F1 (best): 0.6663 (raw/tuned 0.6663 / 0.7818)
  - Accuracy: 0.5313 | Macro F1: 0.4376 | Weighted F1: 0.5536
  - Notes: Tuned improvement large vs raw; however overall accuracy remains modest — class imbalance effects present.

- `kansas/output_weekly_kansas_scenario2B/results_summary.txt` (Correlation top-25)
  - Val Macro F1 (best): 0.8429 (raw/tuned 0.8429 / 0.8458)
  - Accuracy: 0.7940 | Macro F1: 0.8014 | Weighted F1: 0.7932
  - Notes: Top-25 selection performs similarly to tuned baseline — good tradeoff between parsimony and performance.

- `kansas/output_weekly_kansas_scenario2C/results_summary.txt` (Correlation top-30)
  - Val Macro F1 (best): 0.8429
  - Accuracy: 0.7940 | Macro F1: 0.8014 | Weighted F1: 0.7932
  - Notes: Comparable to 2B; selected features include a balance of lagged drought indicators and precipitation stats.

- `kansas/output_weekly_kansas_scenario3/results_summary.txt` (F1-driven selection)
  - Val Macro F1 (best): 0.8320 (raw/tuned 0.8320 / 0.8491)
  - Accuracy: 0.7536 | Macro F1: 0.6710 | Weighted F1: 0.7491
  - Selected subset: Top-10 features (lags and carryover terms) produced best validation macro-F1.
  - Notes: Compact 10-feature set retained most predictive power; consider using for lightweight models.

- `kansas/output_weekly_kansas_scenario4/results_summary.txt` (Weather only)
  - Val Macro F1 (best): 0.1746 (raw/tuned 0.1746 / 0.2060) — very low
  - Accuracy: 0.1438 | Macro F1: 0.1185 | Weighted F1: 0.1201
  - Notes: Weather-only models underperform strongly compared to drought-history or combined inputs.

- `kansas/output_weekly_kansas_scenario5/results_summary.txt` (Weather lag only)
  - Val Macro F1 (best): 0.2510 (raw/tuned 0.2510 / 0.2570)
  - Accuracy: 0.2134 | Macro F1: 0.1991 | Weighted F1: 0.2232
  - Notes: Lagged weather alone gives small signal but insufficient for robust classification.

- `kansas/output_weekly_kansas_scenario6/results_summary.txt` (No drought history)
  - Val Macro F1 (best): 0.2510 / 0.2068 variants across configs — generally low
  - Notes: Removing drought history degrades performance significantly — drought-history is a key predictor.

- `kansas/output_weekly_kansas_scenario7/results_summary.txt` (Drought history only)
  - Val Macro F1 (best): 0.8436
  - Accuracy: 0.8167 | Macro F1: 0.8085 | Weighted F1: 0.8159
  - Notes: Drought-history-only models are strong; confirms drought history holds substantial predictive power.

---

**Nebraska — Aggregated Scenarios**

- `nebraska/output_weekly_nebraska_20counties/results_summary.txt` (Tuned baseline)
  - Best trial: `ros_focal_no_cw_96x48`
  - Val Macro F1 (best): 0.8657 (raw/tuned 0.8657 / 0.8724)
  - Accuracy: 0.8285 | Macro F1: 0.8230 | Weighted F1: 0.8292
  - Per-class F1: None 0.9285, D1 0.7941, D4 0.8418
  - Notes: Strong performance across classes; balanced and tuned approach yields reliability.

- `nebraska/output_weekly_nebraska_scenario2/results_summary.txt` (Correlation top-15)
  - Val Macro F1 (best): 0.6092
  - Accuracy: 0.6007 | Macro F1: 0.5262 | Weighted F1: 0.6124
  - Notes: Similar trends to Kansas: moderate performance with reduced feature sets.

- `nebraska/output_weekly_nebraska_scenario2A/results_summary.txt` (Top-20)
  - Val Macro F1 (best): 0.6040 (raw/tuned 0.6040 / 0.6424)
  - Accuracy: 0.5878 | Macro F1: 0.4942 | Weighted F1: 0.5830

- `nebraska/output_weekly_nebraska_scenario2B/results_summary.txt` (Top-25)
  - Val Macro F1 (best): 0.8510 (raw/tuned 0.8510 / 0.8606)
  - Accuracy: 0.7689 | Macro F1: 0.7517 | Weighted F1: 0.7752

- `nebraska/output_weekly_nebraska_scenario2C/results_summary.txt` (Top-26/30)
  - Val Macro F1 (best): 0.8558 (raw/tuned 0.8558 / 0.8658)
  - Accuracy: 0.7782 | Macro F1: 0.7603 | Weighted F1: 0.7825
  - Notes: Correlation-aware selections around 25–30 features provide near-baseline performance.

- `nebraska/output_weekly_nebraska_scenario3/results_summary.txt` (F1-driven selection)
  - Val Macro F1 (best): 0.7835 (raw/tuned 0.7835 / 0.7980)
  - Accuracy: 0.7627 | Macro F1: 0.7476 | Weighted F1: 0.7636
  - Selected features: Top-15 chosen by permutation importance performed best.

- `nebraska/output_weekly_nebraska_scenario4/results_summary.txt` (Weather only)
  - Val Macro F1 (best): 0.3156
  - Accuracy: 0.3383 | Macro F1: 0.2709 | Weighted F1: 0.3013
  - Notes: Weather-only has a stronger signal here than Kansas weather-only runs, but still far below history-inclusive runs.

- `nebraska/output_weekly_nebraska_scenario5/results_summary.txt` (Weather lag only)
  - Val Macro F1 (best): 0.3240
  - Accuracy: 0.3344 | Macro F1: 0.2725 | Weighted F1: 0.2966

- `nebraska/output_weekly_nebraska_scenario6/results_summary.txt` (No drought history)
  - Val Macro F1 (best): 0.1520 (raw/tuned improved to 0.2774 in one config)
  - Accuracy: 0.1990 | Macro F1: 0.1516 | Weighted F1: 0.1455
  - Notes: Removing drought-history severely degrades results.

- `nebraska/output_weekly_nebraska_scenario7/results_summary.txt` (Drought history only)
  - Val Macro F1 (best): 0.8657
  - Accuracy: 0.8285 | Macro F1: 0.8230 | Weighted F1: 0.8292
  - Notes: Drought-history-only models perform very similarly to full tuned models.

---

**Cross-scenario Observations**

- Drought-history features are consistently the most predictive: scenarios that include drought-history (or use drought-history-only) reach the highest macro-F1 and accuracy. When drought history is removed, macro-F1 and accuracy drop sharply.
- Correlation-aware feature selection (25–30 features) often preserves nearly baseline performance while reducing feature count; the 20–25 feature range is a good balance.
- F1-driven selection can yield compact feature sets (10–15) that retain strong macro-F1 in several Kansas/Nebraska runs — useful for smaller models or faster training.
- Weather-only and weather-lag-only models perform substantially worse than combinations including drought history; Nebraska weather-only results are somewhat stronger than Kansas but still below history-inclusive runs.
- Class imbalance remains an issue: some scenarios show very low performance on rare classes (D4 in some subsets, or D3 depending on config). The tuned balancing strategies (ROS, focal loss, class weights) improved macro-F1 in many trials.

**Recommendations**

- Use drought-history features as a required input in production models; if resources are limited, prefer the drought-history-only or drought-history+small feature set (Top-10/Top-15) discovered by F1-driven selection.
- For feature reduction aim for 20–30 features (correlation-aware top-25/30) to preserve baseline performance while simplifying models.
- Continue using ROS + focal loss configurations (e.g., `ros_focal_no_cw_96x48`) as they appear frequently as best trials across scenarios.
- For rare-class improvements, consider additional targeted balancing or data augmentation and review per-class confusion matrices to identify systematic errors.

---

If you'd like, I can now:
- (A) Update `ANALYSIS_SCENARIOS.md` with these details inline, or
- (B) Generate a CSV or markdown table comparing key metrics across scenarios, or
- (C) Produce per-county breakdowns and visualizations from the per-scenario outputs.

Tell me which next step you prefer and I'll continue.
