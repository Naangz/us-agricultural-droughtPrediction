# Hugging Face Space Kansas Scenario 7 UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the existing novice wizard UI so it feels warmer, more educational, and more presentation-ready without changing the prediction flow.

**Architecture:** Keep the current three-step wizard and prediction logic intact, but enrich the Gradio layout, copywriting, and CSS hierarchy inside `app.py`. Protect the polish with lightweight source-based tests that assert the new guidance panels, richer result framing, and visual communication hooks remain present.

**Tech Stack:** Python, Gradio, pytest

---

### Task 1: Add failing tests for the polish requirements

**Files:**
- Modify: `tests/test_hf_space_kansas_scenario7.py`
- Test: `tests/test_hf_space_kansas_scenario7.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_hf_space_app_contains_support_panels():
    app_text = read_app_text()
    assert "Cocok untuk" in app_text
    assert "Waktu isi" in app_text
    assert "Siapkan info ini" in app_text


def test_hf_space_app_contains_next_attention_guidance():
    app_text = read_app_text()
    assert "Hal yang perlu diperhatikan setelah ini" in app_text


def test_hf_space_app_contains_risk_badges():
    app_text = read_app_text()
    assert "badge-aman" in app_text
    assert "badge-waspada" in app_text
    assert "badge-serius" in app_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hf_space_kansas_scenario7.py -v`
Expected: FAIL because the current UI does not include the new polish copy and badge hooks.

- [ ] **Step 3: Write minimal implementation**

```python
gr.HTML(
    """
    <section class="support-grid">
        <div class="support-card"><div class="support-label">Cocok untuk</div>...</div>
        <div class="support-card"><div class="support-label">Siapkan info ini</div>...</div>
        <div class="support-card"><div class="support-label">Waktu isi</div>...</div>
    </section>
    """
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hf_space_kansas_scenario7.py -v`
Expected: PASS

### Task 2: Apply the warm-educational UI polish

**Files:**
- Modify: `hf_space_kansas_scenario7/app.py`
- Modify: `tests/test_hf_space_kansas_scenario7.py`

- [ ] **Step 1: Polish hero and support panels**

Add a more welcoming hero, plus a three-card support row for audience, what to prepare, and completion time.

- [ ] **Step 2: Polish step communication**

Add stronger step badges, richer helper copy, and visual cue badges such as safe, watchful, and serious for the drought choices.

- [ ] **Step 3: Polish result framing**

Add a clearer interpreted-outcome feel with a “what this means” emphasis and a “Hal yang perlu diperhatikan setelah ini” guidance block.

- [ ] **Step 4: Polish CSS hierarchy**

Improve card styling, spacing, color rhythm, and support-panel presentation while preserving mobile readability.

- [ ] **Step 5: Run verification**

Run: `python -m pytest tests/test_hf_space_kansas_scenario7.py -v`
Expected: PASS

Run: `python -m py_compile hf_space_kansas_scenario7/app.py`
Expected: exit code 0
