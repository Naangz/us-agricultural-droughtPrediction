# Kansas Refresh Outputs Design

**Date:** 2026-06-27

**Goal**

Add one Python utility in `kansas/` that regenerates `results_summary.txt`, `confusion_matrix.png`, and `per_class_f1.png` for one or more Kansas scenarios by loading the existing `best_model.keras` files, without retraining.

**Chosen Approach**

Use approach 2: reuse the existing scenario scripts as the source of truth for preprocessing, sequence creation, validation calibration, and report formatting, but stop before training and replace the training section with model loading plus inference-only evaluation.

**Why This Approach**

- It preserves scenario-specific feature engineering with minimal behavior drift.
- It avoids rebuilding a full abstraction layer for all scenarios.
- It keeps regenerated outputs aligned with the format and evaluation logic already present in the training scripts.

**Scope**

- Create a new script at `kansas/update_outputs_from_best_model.py`.
- Support refreshing a single scenario or multiple scenarios from the command line.
- Regenerate only:
  - `results_summary.txt`
  - `confusion_matrix.png`
  - `per_class_f1.png`
- Do not retrain models.
- Do not overwrite `training_history.png`.

**Inputs and Outputs**

**Inputs**

- Kansas integrated weekly dataset at the same path used by the scenario scripts.
- Scenario training scripts in `kansas/` as the reference for preprocessing rules.
- Existing scenario output folders containing `best_model.keras`.

**Outputs**

- Rewritten `results_summary.txt` in each target scenario output folder.
- Rewritten `confusion_matrix.png` in each target scenario output folder.
- Rewritten `per_class_f1.png` in each target scenario output folder.

**Command-Line Interface**

The new script should accept:

- `--scenario <name>` for a single scenario such as `1`, `2A`, `2B`, `2C`, `2`, `3`, `4`, `5`, `6`, `7`
- `--scenario all` to process every supported Kansas scenario

Optional flags are not required unless needed during implementation for safety or diagnostics.

**Architecture**

The utility should be organized around a small scenario registry. Each registry entry defines:

- Human-readable scenario name
- Path to the existing scenario script
- Path to the scenario output folder
- A callable or extraction strategy that recreates the scenario-specific data preparation and evaluation configuration

The main flow should be:

1. Resolve the requested scenario list.
2. For each scenario, reproduce the preprocessing and split logic used by the original script.
3. Recreate validation and test sequences exactly as before.
4. Load `best_model.keras` with `compile=False`.
5. Run validation predictions to recompute class multipliers using the existing tuning logic.
6. Run test predictions and compute raw and tuned metrics.
7. Rewrite the summary file and regenerate the confusion matrix and per-class F1 plots.

**Implementation Strategy**

The script should favor direct reuse of code patterns from the existing scenario files rather than inventing new modeling logic. Because the scenario scripts are not currently structured as importable modules, the utility may need to duplicate focused inference-time sections in a controlled way:

- shared drought label derivation
- shared temporal split logic
- scenario-specific feature engineering
- sequence generation
- class multiplier tuning
- summary and plot generation

The script should not copy training loops, callbacks, oversampling, or checkpointing behavior.

**Scenario Coverage**

The first implementation should support the Kansas scenarios currently present in the repository:

- Scenario 1 baseline
- Scenario 2
- Scenario 2A
- Scenario 2B
- Scenario 2C
- Scenario 3
- Scenario 4
- Scenario 5
- Scenario 6
- Scenario 7

Each scenario may differ in feature selection. The new script must preserve those differences so that loaded models receive the same input shape and feature ordering they were trained with.

**Error Handling**

The utility should fail clearly when:

- `best_model.keras` is missing for a requested scenario
- the recreated feature list does not match the loaded model input width
- required source data is missing
- a scenario key is unsupported

Errors should name the scenario and the missing or mismatched artifact to make reruns easy.

**Testing Strategy**

Testing should focus on inference behavior rather than training:

- scenario resolution works for single values and `all`
- regenerated sequence feature width matches model input width
- summary file is written for a representative scenario
- plot files are regenerated for a representative scenario

Where practical, tests should target helper functions rather than full TensorFlow-heavy end-to-end runs. A small integration test can still validate one scenario path if the environment permits it.

**Risks and Mitigations**

**Risk:** Scenario scripts contain duplicated but slightly different logic.

**Mitigation:** Lift only the inference-time pieces that are required and keep scenario-specific feature lists explicit.

**Risk:** Some output folders do not include scaler or config artifacts.

**Mitigation:** Recompute preprocessing from source data using the original script logic instead of depending on saved preprocessing artifacts.

**Risk:** Recomputed summary formatting may drift from existing files.

**Mitigation:** Reuse the existing section ordering, labels, and metric naming from the scenario scripts.

**Non-Goals**

- No retraining
- No hyperparameter search
- No regeneration of `training_history.png`
- No refactor of the original scenario training files unless a tiny extraction is required to support testing
