Detailed cross-scenario analysis is below.

**1) Important comparability note**
- The clean runs use a different test size and class supports (14694 total) than your 20-county/scenario runs (4180 total).  
- So clean vs 20-county should be interpreted as directional only, not strict apples-to-apples.
- Within the 20-county family (baseline + scenarios 2/3/4/5/6/7), comparison is valid.

**2) Performance ranking on the common 20-county benchmark (Test Macro F1)**
| Rank | Experiment | Val Macro F1 | Test Macro F1 | Test Accuracy | Notes |
|---|---:|---:|---:|---:|---|
| 1 | Scenario 7 (Drought history only) | 0.8436 | 0.8085 | 0.8167 | Best test macro among scenarios |
| 2 | Baseline 20 counties | 0.8514 | 0.8039 | 0.8067 | Strongest validation, near-best test |
| 3 | Scenario 2B | 0.8429 | 0.8014 | 0.7940 | Best reduced-feature model |
| 3 | Scenario 2C | 0.8429 | 0.8014 | 0.7940 | Exactly same as 2B |
| 5 | Scenario 3 (Top-10 by permutation) | 0.8320 | 0.6710 | 0.7536 | High accuracy, much lower macro |
| 6 | Scenario 2A | 0.6663 | 0.4376 | 0.5313 | Moderate collapse |
| 7 | Scenario 2 | 0.5638 | 0.4165 | 0.5433 | More collapse |
| 8 | Scenario 6 (No drought history) | 0.2510 | 0.1991 | 0.2134 | Severe degradation |
| 9 | Scenario 4 (Weather only) | 0.1746 | 0.1185 | 0.1438 | Very poor |
| 10 | Scenario 5 (Weather + lag only) | 0.2068 | 0.1152 | 0.1715 | Very poor |

**3) Core modeling conclusion**
- Drought-memory features are the dominant predictive signal in your setup.
- Evidence:
  - Scenario 7 (only drought history) is best overall on test macro F1.
  - Removing drought history (Scenario 6) drops from around 0.80 macro F1 to around 0.20.
  - Weather-only and weather+lag-only are near-collapse.
- This indicates the model is primarily learning drought persistence dynamics, with weather acting as a secondary enhancer.

**4) Feature-selection scenarios (2/2A/2B/2C/3)**
- Scenario 2B is the strongest feature-selection tradeoff:
  - Keeps much of baseline quality with fewer features.
  - Good all-class balance compared with other reduced sets.
- Scenario 2C appears to be a duplicate outcome of 2B:
  - Same selected feature count (25), same selected feature list, and identical metrics in results_summary.txt and results_summary.txt.
  - This likely means your top-30 target was constrained by correlation pruning and ended up identical to top-25.
- Scenario 3 (Top-10) over-compresses:
  - Validation looks strong, but test macro drops hard.
  - Suggests underrepresentation of minority/rare class structure after aggressive pruning.

**5) Generalization gaps (Val best vs Test macro F1)**
Smaller gap usually means better transfer from validation to test:
- Scenario 7: about -0.035 (best stability)
- Baseline: about -0.048
- Scenario 2B/2C: about -0.042
- Scenario 6: about -0.052 (still poor absolute score)
- Scenario 4: about -0.056
- Scenario 5: about -0.092
- Scenario 2: about -0.147
- Scenario 3: about -0.161
- Scenario 2A: about -0.229 (worst)

Interpretation:
- Best absolute models also generalize relatively well (baseline, scenario 7, scenario 2B).
- Mid/low-performing reduced-feature setups show much larger validation-to-test mismatch.

**6) Per-class behavior trends**
- Best balanced class performance:
  - Baseline and Scenario 7.
  - Both maintain strong D4 and reasonable D3.
- Scenario 2B/2C:
  - Surprisingly robust across classes; D4 is very strong.
- Scenario 3:
  - D4 collapses badly (F1 around 0.315) despite decent overall accuracy.
- Scenario 2 and 2A:
  - D2/D3/D4 instability; minority classes become weak.
- Scenario 4 and 5:
  - Degenerate class behavior.
  - Scenario 4 predicts D4 heavily (high D4 recall, near-zero D3).
  - Scenario 5 has D0 and D4 F1 equal to 0, with D3 recall very high but precision very low.
- Scenario 6:
  - All classes weak; confirms drought-history removal is highly destructive.

**7) About the clean baselines**
- results_summary.txt: macro F1 0.7327
- results_summary.txt: macro F1 0.7303
- These are consistent with each other and stable, but because supports differ substantially from the 20-county runs, treat them as a separate benchmark track.

**8) Final takeaway**
- If your objective is strongest drought-class macro performance on this 20-county benchmark, current ordering is:
  1. Scenario 7
  2. Baseline 20 counties
  3. Scenario 2B/2C
- If you want feature reduction with minimal loss, Scenario 2B is the practical winner.
- If you want physics interpretability around weather-only causality, current results say weather-only signal is insufficient in this pipeline without drought-memory variables.


====================================================================

Great request. Here is a full scenario-by-scenario change log relative to the main 20-county baseline pipeline.

Baseline reference script:
BiLSTM_Weekly_Kansas_Clean.py

Primary baseline output:
results_summary.txt

**1. Baseline (20 counties)**
Script: BiLSTM_Weekly_Kansas_Clean.py  
Output: results_summary.txt

What is used:
- Full engineered feature set (41 features).
- No feature-selection step.
- Original trial configs (two ROS trials without class weight, one NONE trial with class weight).

What this means:
- This is your full-information reference model for 20-county experiments.

---

**2. Scenario 2 (Correlation-aware feature selection, Top-15 target)**
Script: BiLSTM_Scenario2_Correlation.py  
Output: results_summary.txt

What changed:
- Added feature selection block between split and scaling.
- Method: mutual information ranking + correlation pruning.
- Correlation threshold: 0.9.
- Target feature count: 15.

Final selected features:
- drought_carryover_lag1
- D0_lag1
- RH2M_lag8
- T2M
- RH2M
- RH2M_lag4
- PREC_roll12_std
- PREC_roll4_mean
- RH2M_lag2
- T2M_roll12_mean
- ALLSKY_SFC_SW_DWN
- RH2M_lag1
- PREC_roll12_mean
- PRECTOTCORR
- PREC_lag2

---

**3. Scenario 2A (Correlation-aware feature selection, Top-20 target)**
Script: BiLSTM_Scenario2A_Correlation.py  
Output: results_summary.txt

What changed:
- Same method as Scenario 2.
- Correlation threshold: 0.9.
- Target feature count: 20.

Final selected features:
- drought_carryover_lag1
- D0_lag1
- RH2M_lag8
- T2M
- RH2M
- RH2M_lag4
- PREC_roll12_std
- PREC_roll4_mean
- RH2M_lag2
- T2M_roll12_mean
- ALLSKY_SFC_SW_DWN
- RH2M_lag1
- PREC_roll12_mean
- PRECTOTCORR
- PREC_lag2
- PREC_lag1
- PREC_lag4
- PREC_roll4_std
- WS2M
- PS

---

**4. Scenario 2B (Correlation-aware feature selection, Top-25 target)**
Script: BiLSTM_Scenario2B_Correlation.py  
Output: results_summary.txt

What changed:
- Same method as Scenario 2/2A.
- Correlation threshold: 0.9.
- Target feature count: 25.

Final selected features:
- drought_carryover_lag1
- D0_lag1
- RH2M_lag8
- T2M
- RH2M
- RH2M_lag4
- PREC_roll12_std
- PREC_roll4_mean
- RH2M_lag2
- T2M_roll12_mean
- ALLSKY_SFC_SW_DWN
- RH2M_lag1
- PREC_roll12_mean
- PRECTOTCORR
- PREC_lag2
- PREC_lag1
- PREC_lag4
- PREC_roll4_std
- WS2M
- PS
- PREC_lag8
- D2_lag1
- severe_carryover_lag1
- D4_lag1
- week_sin

---

**5. Scenario 2C (Correlation-aware feature selection, Top-30 target)**
Script: BiLSTM_Scenario2C_Correlation.py  
Output: results_summary.txt

What changed:
- Intended target feature count: 30.
- Same selection method and threshold as 2/2A/2B.

Important outcome:
- Final selected feature count ended at 25 (not 30), same selected feature list as Scenario 2B.
- Results are effectively identical to Scenario 2B in your current outputs.

---

**6. Scenario 3 (Macro-F1 driven feature selection)**
Script: BiLSTM_Scenario3_F1Selection.py  
Output: results_summary.txt

What changed:
- Feature selection strategy changed completely.
- Method: permutation importance ranking using validation macro-F1.
- Subset sizes tested: 30, 25, 20, 15, 10.
- Best subset selected: Top-10 features.

Final selected Top-10 features:
- D3_lag1
- D2_lag1
- None_lag1
- D1_lag1
- D0_lag1
- severe_carryover_lag1
- D2_lag2
- PREC_lag1
- drought_carryover_lag1
- PREC_roll12_mean

---

**7. Scenario 4 (Weather only)**
Script: BiLSTM_Scenario4_WeatherOnly.py  
Output: results_summary.txt

What changed:
- Feature set replaced with only 6 raw weather variables.
- Added scenario name + feature logging print block.
- Output folder isolated to scenario-specific folder.
- Summary file now records selected features.
- Class weight usage set to true in every trial configuration.

Features used (6):
- ALLSKY_SFC_SW_DWN
- PRECTOTCORR
- PS
- RH2M
- T2M
- WS2M

---

**8. Scenario 5 (Weather + lag only)**
Script: BiLSTM_Scenario5_WeatherLagOnly.py  
Output: results_summary.txt

What changed:
- Feature set replaced with raw weather + weather lags + seasonal sin/cos.
- No rolling weather stats.
- No drought history features.
- Added scenario logging block.
- Isolated output folder.
- Summary includes selected features.
- Class weight true in every trial.

Features used (20):
- Raw weather 6 features
- PREC_lag1, PREC_lag2, PREC_lag4, PREC_lag8
- T2M_lag1, T2M_lag2, T2M_lag4, T2M_lag8
- RH2M_lag1, RH2M_lag2, RH2M_lag4, RH2M_lag8
- week_sin, week_cos

---

**9. Scenario 6 (No drought history)**
Script: BiLSTM_Scenario6_NoDroughtHistory.py  
Output: results_summary.txt

What changed:
- Uses weather raw + lag + rolling + seasonal + heat_dry_stress.
- Explicitly removes all drought history lag/carryover features.
- Added scenario logging block.
- Isolated output folder.
- Summary includes selected features.
- Class weight true in every trial.

Features used (27):
- Raw weather 6
- Weather lags 12
- Rolling stats 6
- week_sin, week_cos
- heat_dry_stress

---

**10. Scenario 7 (Drought history only)**
Script: BiLSTM_Scenario7_DroughtHistoryOnly.py  
Output: results_summary.txt

What changed:
- Removes all weather features.
- Keeps only drought persistence signals.
- Added scenario logging block.
- Isolated output folder.
- Summary includes selected features.
- Class weight true in every trial.

Features used (14):
- None_lag1, D0_lag1, D1_lag1, D2_lag1, D3_lag1, D4_lag1
- None_lag2, D0_lag2, D1_lag2, D2_lag2, D3_lag2, D4_lag2
- drought_carryover_lag1
- severe_carryover_lag1

---

**11. Clean baseline track (different dataset/split context)**
Outputs:

What changed:
- These are separate clean-track runs (different sample/support profile from 20-county scenario runs).
- Useful as another baseline family, but not directly one-to-one comparable with the 4180-support scenario outputs.

---

====================================================================

Perfect. Here is the consolidated scenario matrix you asked for.

**A. Change Matrix (relative to full 20-county baseline)**

| Scenario | Feature selection method | Raw weather | Weather lags | Rolling weather stats | Seasonal (`week_sin/cos`) | Drought lags (`*_lag1/2`) | Carryover drought (`drought_carryover_lag1`, `severe_carryover_lag1`) | Heat-dry stress | Class-weight policy |
|---|---|---|---|---|---|---|---|---|---|
| Baseline 20counties | None (full feature set) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Mixed by trial (not always true) |
| Scenario 2 | MI + corr pruning (target 15) | Partial | Partial | Partial | No | Very limited | Yes | No | Mixed by trial (best used class weight) |
| Scenario 2A | MI + corr pruning (target 20) | Yes | Partial | Yes | No | Very limited | Yes | No | Mixed by trial (best used class weight) |
| Scenario 2B | MI + corr pruning (target 25) | Yes | Yes | Yes | Partial (`week_sin` only) | Some | Yes | No | Mixed by trial (best did not use class weight) |
| Scenario 2C | MI + corr pruning (target 30 requested) | Same as 2B | Same as 2B | Same as 2B | Same as 2B | Same as 2B | Same as 2B | Same as 2B | Same as 2B |
| Scenario 3 | Permutation macro-F1 subset search (Top-10 selected) | Very limited | Minimal | Minimal | No | Yes | Yes | No | Mixed by trial (best did not use class weight) |
| Scenario 4 | Manual feature ablation | Yes | No | No | No | No | No | No | Always true in all trials |
| Scenario 5 | Manual feature ablation | Yes | Yes | No | Yes | No | No | No | Always true in all trials |
| Scenario 6 | Manual feature ablation | Yes | Yes | Yes | Yes | No | No | Yes | Always true in all trials |
| Scenario 7 | Manual feature ablation | No | No | No | No | Yes | Yes | No | Always true in all trials |

---

**B. Exact feature sets for the manual ablation scenarios**

**Scenario 4 (Weather only, 6 features)**  
- ALLSKY_SFC_SW_DWN  
- PRECTOTCORR  
- PS  
- RH2M  
- T2M  
- WS2M

**Scenario 5 (Weather + lag only, 20 features)**  
- Raw weather 6  
- PREC_lag1, PREC_lag2, PREC_lag4, PREC_lag8  
- T2M_lag1, T2M_lag2, T2M_lag4, T2M_lag8  
- RH2M_lag1, RH2M_lag2, RH2M_lag4, RH2M_lag8  
- week_sin, week_cos

**Scenario 6 (No drought history, 27 features)**  
- Raw weather 6  
- Weather lags 12  
- Rolling: PREC_roll4_mean, PREC_roll4_std, PREC_roll12_mean, PREC_roll12_std, T2M_roll4_mean, T2M_roll12_mean  
- week_sin, week_cos  
- heat_dry_stress

**Scenario 7 (Drought history only, 14 features)**  
- None_lag1, D0_lag1, D1_lag1, D2_lag1, D3_lag1, D4_lag1  
- None_lag2, D0_lag2, D1_lag2, D2_lag2, D3_lag2, D4_lag2  
- drought_carryover_lag1  
- severe_carryover_lag1

---

**C. Output/reporting changes in your new manual scenarios (4–7)**

For each of:
- BiLSTM_Scenario4_WeatherOnly.py
- BiLSTM_Scenario5_WeatherLagOnly.py
- BiLSTM_Scenario6_NoDroughtHistory.py
- BiLSTM_Scenario7_DroughtHistoryOnly.py

you changed:
- `SCENARIO_NAME` added
- scenario-specific `OUTPUT_FOLDER` used  
- feature logging block added (prints feature count + full list)
- results_summary.txt now writes selected feature list
- `use_class_weight = True` forced in every trial config

---

**D. Short “what differs most” summary**
- Largest structural differences are whether drought history exists.
- Scenario 7 keeps only drought persistence memory.
- Scenario 6 removes all drought memory but keeps weather dynamics.
- Scenarios 4/5 are weather-centric with progressively more temporal weather context.
- Scenarios 2/2A/2B/2C/3 are algorithmic feature-reduction variants from the full baseline rather than pure domain ablations.


