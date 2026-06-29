---
title: Kansas Scenario 7 Drought Forecast
sdk: gradio
app_file: app.py
pinned: false
---

# Kansas Scenario 7 Drought Forecast

This folder is a Hugging Face Spaces deployment package for the weekly Kansas Scenario 7 drought prediction app.

## Included runtime assets

- `app.py`
- `best_model.keras`
- `config.json`
- `tokenizer.pkl`
- `requirements.txt`

## Notes

- The original experiment outputs remain in `kansas/output_weekly_kansas_scenario7/`.
- This package is intended for inference only.
- Upload this folder as the root of a Gradio Space repository.

## Suggested repository contents

When creating a separate Hugging Face repository, keep these files at the repository root:

- `README.md`
- `app.py`
- `best_model.keras`
- `config.json`
- `tokenizer.pkl`
- `requirements.txt`
- `LICENSE`

## Local smoke run

```bash
python app.py
```
