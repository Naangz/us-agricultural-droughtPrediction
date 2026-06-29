# Hugging Face Space Kansas Scenario 7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a root-level Hugging Face Spaces deployment package for the Kansas Scenario 7 inference app.

**Architecture:** Build a standalone deploy directory that copies only runtime artifacts from the existing experiment output folder. Keep the original folder intact and verify the new package with a repository test.

**Tech Stack:** Python, Gradio, TensorFlow, pytest, Hugging Face Spaces

---

### Task 1: Add a failing regression test

**Files:**
- Create: `tests/test_hf_space_kansas_scenario7.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT / "hf_space_kansas_scenario7"


def test_hf_space_kansas_scenario7_contains_runtime_files():
    required_files = [
        "README.md",
        "app.py",
        "best_model.keras",
        "config.json",
        "requirements.txt",
        "shap_background.npy",
        "tokenizer.pkl",
    ]

    assert DEPLOY_DIR.is_dir(), f"Missing deploy directory: {DEPLOY_DIR}"
    missing_files = [name for name in required_files if not (DEPLOY_DIR / name).is_file()]
    assert not missing_files, f"Missing runtime files: {missing_files}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hf_space_kansas_scenario7.py -v`
Expected: FAIL because `hf_space_kansas_scenario7` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `hf_space_kansas_scenario7/` and add the required runtime files copied or adapted from `kansas/output_weekly_kansas_scenario7/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hf_space_kansas_scenario7.py -v`
Expected: PASS

### Task 2: Make the deploy app Spaces-ready

**Files:**
- Create: `hf_space_kansas_scenario7/README.md`
- Create: `hf_space_kansas_scenario7/.gitignore`
- Create: `hf_space_kansas_scenario7/app.py`
- Create: `hf_space_kansas_scenario7/requirements.txt`
- Create: `hf_space_kansas_scenario7/best_model.keras`
- Create: `hf_space_kansas_scenario7/config.json`
- Create: `hf_space_kansas_scenario7/tokenizer.pkl`
- Create: `hf_space_kansas_scenario7/shap_background.npy`

- [ ] **Step 1: Add Hugging Face Spaces metadata**

Write a `README.md` with YAML front matter for `sdk: gradio` and `app_file: app.py`.

- [ ] **Step 2: Keep app paths local to deploy folder**

Use `CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))` and join all runtime file paths from there.

- [ ] **Step 3: Configure Spaces-friendly launch**

Use `demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))`.

- [ ] **Step 4: Add lightweight runtime hygiene**

Add a `.gitignore` that excludes `__pycache__/` and local temporary files from the deploy folder.
