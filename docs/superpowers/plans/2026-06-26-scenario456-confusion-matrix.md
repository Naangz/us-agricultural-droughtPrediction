# Scenario 4-6 Confusion Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menambahkan evaluasi confusion matrix dinamis ke helper visualisasi skenario 4, 5, dan 6.

**Architecture:** Perluasan dilakukan di `visualization/scenario_analysis_common.py` agar parser summary, rekonstruksi fitur, evaluasi model, dan plotting heatmap berada dalam satu helper bersama. Tiga script skenario tetap tipis dan hanya memanggil entrypoint helper, sementara pengujian difokuskan pada parsing summary dan konfigurasi evaluasi yang tidak bergantung pada backend plotting.

**Tech Stack:** Python, unittest, pandas, numpy, scikit-learn, TensorFlow, matplotlib, seaborn

---

### Task 1: Tambah pengujian parser metadata skenario 4-6

**Files:**
- Create: `tests/test_scenario_analysis_common.py`
- Modify: `visualization/scenario_analysis_common.py`

- [ ] **Step 1: Write the failing test**

```python
def test_parse_summary_extracts_seq_length_and_class_multipliers():
    parsed = scenario_common.parse_summary(summary_path)
    assert parsed["seq_length"] == 52
    assert parsed["class_multipliers"] == [1.2, 0.8, 1.0, 0.9, 1.1, 0.7]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_scenario_analysis_common -v`
Expected: FAIL karena key `seq_length` dan `class_multipliers` belum tersedia dari `parse_summary`

- [ ] **Step 3: Write minimal implementation**

```python
seq_length_match = re.search(r"^Seq Length:\s*(\d+)\s*$", content, flags=re.MULTILINE)
class_multipliers_match = re.search(r"^Class multipliers:\s*(\[[^\n]+\])\s*$", content, flags=re.MULTILINE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_scenario_analysis_common -v`
Expected: PASS untuk test parser baru

### Task 2: Tambah pipeline evaluasi confusion matrix skenario 4-6

**Files:**
- Modify: `visualization/scenario_analysis_common.py`
- Test: `tests/test_scenario_analysis_common.py`

- [ ] **Step 1: Write the failing test**

```python
def test_get_model_and_summary_paths_follow_scenario_number():
    assert scenario_common.model_path("kansas", 5).endswith("output_weekly_kansas_scenario5\\best_model.keras")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_scenario_analysis_common -v`
Expected: FAIL karena helper model path belum ada

- [ ] **Step 3: Write minimal implementation**

```python
def model_path(region_key, scenario):
    return REGIONS[region_key]["model_template"].format(scenario=scenario)
```

- [ ] **Step 4: Extend evaluator and plotting**

```python
def evaluate_model_dynamically(region_key, scenario, summary_data):
    ...
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASSES))))
    return cm
```

- [ ] **Step 5: Run targeted tests**

Run: `python -m unittest tests.test_scenario_analysis_common -v`
Expected: PASS untuk helper path dan parser

### Task 3: Sambungkan heatmap confusion matrix ke entrypoint skenario

**Files:**
- Modify: `visualization/scenario_analysis_common.py`
- Modify: `visualization/plot_scenario4_analysis.py`
- Modify: `visualization/plot_scenario5_analysis.py`
- Modify: `visualization/plot_scenario6_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_scenario_analysis_can_skip_confusion_matrix_when_ml_libs_missing():
    with mock.patch.object(scenario_common, "HAS_ML_LIBRARIES", False):
        scenario_common.run_scenario_analysis(4, script_dir=".")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_scenario_analysis_common -v`
Expected: FAIL jika helper memaksa evaluasi ML tanpa fallback yang jelas

- [ ] **Step 3: Write minimal implementation**

```python
if HAS_ML_LIBRARIES:
    plot_confusion_matrices(...)
else:
    print("Confusion matrix dilewati karena dependency ML tidak tersedia.")
```

- [ ] **Step 4: Run verification commands**

Run: `python -m py_compile visualization/scenario_analysis_common.py visualization/plot_scenario4_analysis.py visualization/plot_scenario5_analysis.py visualization/plot_scenario6_analysis.py tests/test_scenario_analysis_common.py`
Expected: exit code 0

- [ ] **Step 5: Run full scenario scripts if dependencies tersedia**

Run: `python visualization/plot_scenario4_analysis.py`
Run: `python visualization/plot_scenario5_analysis.py`
Run: `python visualization/plot_scenario6_analysis.py`
Expected: file `scenarioX_confusion_matrices.png` ikut terbuat bersama plot lainnya
