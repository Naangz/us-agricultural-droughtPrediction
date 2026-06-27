# Kansas Refresh Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Kansas utility that reloads existing `best_model.keras` files and regenerates `results_summary.txt`, `confusion_matrix.png`, and `per_class_f1.png` without retraining.

**Architecture:** Add one script in `kansas/` with small helper functions for scenario resolution, feature preparation, selected-feature loading, sequence generation, and output writing. Reuse scenario-specific preprocessing rules from the existing training scripts, but replace all training with inference-only evaluation from the saved best model.

**Tech Stack:** Python, pandas, numpy, matplotlib, seaborn, scikit-learn, TensorFlow, pytest

---

### Task 1: Add test coverage for scenario parsing and feature selection loading

**Files:**
- Create: `tests/test_update_outputs_from_best_model.py`
- Test: `tests/test_update_outputs_from_best_model.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from kansas.update_outputs_from_best_model import (
    normalize_scenario_key,
    load_selected_features_from_csv,
    scenario_output_dir_name,
)


def test_normalize_scenario_key_accepts_aliases():
    assert normalize_scenario_key("1") == "1"
    assert normalize_scenario_key("scenario3") == "3"
    assert normalize_scenario_key("2b") == "2B"


def test_scenario_output_dir_name_maps_baseline_and_regular_scenarios():
    assert scenario_output_dir_name("1") == "output_weekly_kansas_20counties"
    assert scenario_output_dir_name("5") == "output_weekly_kansas_scenario5"


def test_load_selected_features_from_csv_reads_ranked_features(tmp_path: Path):
    csv_path = tmp_path / "selected_feature_ranking.csv"
    csv_path.write_text(
        "rank,feature,mutual_information_score\\n1,D3_lag1,0.1\\n2,D2_lag1,0.09\\n",
        encoding="utf-8",
    )
    assert load_selected_features_from_csv(csv_path) == ["D3_lag1", "D2_lag1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_update_outputs_from_best_model.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors because the utility file does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
SCENARIO_ALIASES = {"1": "1", "SCENARIO3": "3", "2B": "2B"}

def normalize_scenario_key(value: str) -> str:
    return SCENARIO_ALIASES[value.strip().upper()]

def scenario_output_dir_name(key: str) -> str:
    if key == "1":
        return "output_weekly_kansas_20counties"
    return f"output_weekly_kansas_scenario{key}"

def load_selected_features_from_csv(csv_path):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_update_outputs_from_best_model.py -q`
Expected: PASS

### Task 2: Implement inference-only Kansas output refresh utility

**Files:**
- Create: `kansas/update_outputs_from_best_model.py`
- Modify: `tests/test_update_outputs_from_best_model.py`
- Test: `tests/test_update_outputs_from_best_model.py`

- [ ] **Step 1: Extend the tests for feature preparation helpers**

```python
from kansas.update_outputs_from_best_model import (
    build_base_feature_frame,
    get_feature_columns_for_scenario,
)


def test_get_feature_columns_for_scenario_uses_csv_for_selected_scenarios(tmp_path: Path):
    output_dir = tmp_path / "output_weekly_kansas_scenario2"
    output_dir.mkdir()
    (output_dir / "selected_feature_ranking.csv").write_text(
        "rank,feature,mutual_information_score\\n1,D3_lag1,0.1\\n2,D2_lag1,0.09\\n",
        encoding="utf-8",
    )
    feature_cols = get_feature_columns_for_scenario("2", output_dir)
    assert feature_cols == ["D3_lag1", "D2_lag1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_update_outputs_from_best_model.py::test_get_feature_columns_for_scenario_uses_csv_for_selected_scenarios -q`
Expected: FAIL because the helper is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```python
def get_feature_columns_for_scenario(scenario_key: str, output_dir: Path) -> list[str]:
    if scenario_key in {"2", "2A", "2B", "2C"}:
        return load_selected_features_from_csv(output_dir / "selected_feature_ranking.csv")
    if scenario_key == "3":
        return load_feature_list_from_summary(output_dir / "results_summary.txt")
    return SCENARIO_FEATURES[scenario_key]
```

Add the rest of the utility:
- shared drought label derivation
- base feature engineering
- train-only scaling
- validation/test sequence generation
- `tf.keras.models.load_model(..., compile=False)`
- class multiplier tuning copied from the scenario scripts
- summary writing and plot generation
- CLI entry point for `--scenario`

- [ ] **Step 4: Run focused tests and one real scenario refresh**

Run: `pytest tests/test_update_outputs_from_best_model.py -q`
Expected: PASS

Run: `python kansas/update_outputs_from_best_model.py --scenario 3`
Expected: exit code 0 and refreshed files in `kansas/output_weekly_kansas_scenario3`
