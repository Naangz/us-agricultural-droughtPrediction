# Hugging Face Space Kansas Scenario 7 UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the Hugging Face Space app into a guided, educational drought-prediction wizard for non-technical users.

**Architecture:** Keep the prediction pipeline and runtime assets intact, but extract pure UI/input helpers into a small module and rebuild the Gradio layout around a three-step wizard. The redesign keeps SHAP explanations available as secondary content while making the novice flow the only primary path.

**Tech Stack:** Python, Gradio, TensorFlow, NumPy, pandas, pytest

---

### Task 1: Extract and test wizard input helpers

**Files:**
- Create: `hf_space_kansas_scenario7/__init__.py`
- Create: `hf_space_kansas_scenario7/ui_helpers.py`
- Modify: `tests/test_hf_space_kansas_scenario7.py`

- [ ] **Step 1: Write the failing tests**

```python
from hf_space_kansas_scenario7.ui_helpers import (
    build_input_features,
    build_review_note,
    describe_coverage,
)


def test_build_input_features_maps_drought_answers_to_model_features():
    features = build_input_features("D2", 40, "D0", 10)
    assert features == [60.0, 40.0, 40.0, 40.0, 0.0, 0.0, 90.0, 10.0, 10.0, 0.0, 0.0, 0.0]


def test_build_review_note_flags_unusual_normal_selection():
    note = build_review_note("None", 85, "D0", 10)
    assert "Periksa kembali" in note


def test_describe_coverage_returns_plain_language_band():
    assert "setengah wilayah" in describe_coverage(50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hf_space_kansas_scenario7.py -v`
Expected: FAIL because `hf_space_kansas_scenario7.ui_helpers` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
LEVEL_ORDER = {"None": 0, "D0": 1, "D1": 2, "D2": 3, "D3": 4, "D4": 5}


def build_input_features(last_week_level, last_week_pct, two_weeks_level, two_weeks_pct):
    def encode(level, pct):
        severity = LEVEL_ORDER[level]
        pct = float(pct)
        return [100.0 - pct, pct, pct if severity >= 1 else 0.0, pct if severity >= 2 else 0.0, pct if severity >= 3 else 0.0, pct if severity >= 4 else 0.0]

    return encode(last_week_level, last_week_pct) + encode(two_weeks_level, two_weeks_pct)


def describe_coverage(percent):
    if percent <= 10:
        return "hanya sebagian kecil wilayah"
    if percent <= 35:
        return "sebagian wilayah"
    if percent <= 65:
        return "sekitar setengah wilayah"
    if percent <= 90:
        return "sebagian besar wilayah"
    return "hampir seluruh wilayah"


def build_review_note(last_week_level, last_week_pct, two_weeks_level, two_weeks_pct):
    if last_week_level == "None" and float(last_week_pct) >= 70:
        return "Periksa kembali: Anda memilih kondisi normal tetapi area terdampak sangat luas."
    if two_weeks_level == "None" and float(two_weeks_pct) >= 70:
        return "Periksa kembali: Dua minggu lalu dipilih normal, tetapi area terdampak sangat luas."
    return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hf_space_kansas_scenario7.py -v`
Expected: PASS for new helper tests and existing deploy package checks.

### Task 2: Rebuild the app as a guided wizard

**Files:**
- Modify: `hf_space_kansas_scenario7/app.py`
- Test: `tests/test_hf_space_kansas_scenario7.py`

- [ ] **Step 1: Write the failing UI structure tests**

```python
from pathlib import Path


def test_hf_space_app_contains_wizard_copy():
    app_text = Path("hf_space_kansas_scenario7/app.py").read_text(encoding="utf-8")
    assert "Langkah 1 dari 3" in app_text
    assert "Mulai Prediksi" in app_text
    assert "Mengapa model memprediksi ini?" in app_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hf_space_kansas_scenario7.py -v`
Expected: FAIL because the current app still uses mode toggles and lacks the wizard copy.

- [ ] **Step 3: Write minimal implementation**

```python
def set_step(step_number):
    return (
        gr.update(visible=step_number == 1),
        gr.update(visible=step_number == 2),
        gr.update(visible=step_number == 3),
        gr.update(value=render_progress(step_number)),
    )


with gr.Blocks(...) as demo:
    intro = gr.Markdown("## Prediksi kondisi kekeringan minggu depan")
    progress_html = gr.HTML(render_progress(1))
    with gr.Group(visible=True) as step_one:
        ...
    with gr.Group(visible=False) as step_two:
        ...
    with gr.Group(visible=False) as step_three:
        review_html = gr.HTML()
        predict_btn = gr.Button("Mulai Prediksi", variant="primary")
    with gr.Accordion("Mengapa model memprediksi ini?", open=False):
        output_plot = gr.Plot()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hf_space_kansas_scenario7.py -v`
Expected: PASS

### Task 3: Final app verification

**Files:**
- Modify: `hf_space_kansas_scenario7/app.py`
- Test: `tests/test_hf_space_kansas_scenario7.py`

- [ ] **Step 1: Verify the app compiles**

Run: `python -m py_compile hf_space_kansas_scenario7/app.py`
Expected: exit code 0

- [ ] **Step 2: Verify all targeted tests**

Run: `python -m pytest tests/test_hf_space_kansas_scenario7.py -v`
Expected: PASS

- [ ] **Step 3: Inspect the final package contents**

Run: `Get-ChildItem hf_space_kansas_scenario7`
Expected: deploy package still contains the required runtime files plus any new helper module files.
