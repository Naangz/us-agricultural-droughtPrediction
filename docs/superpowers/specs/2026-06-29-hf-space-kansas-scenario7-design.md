# Hugging Face Space Kansas Scenario 7 Design

## Goal

Create a clean deployment folder at the repository root for the Kansas Scenario 7 Gradio inference app so it can be uploaded to Hugging Face Spaces without mixing runtime assets and training artifacts.

## Scope

- Add a root-level deployment folder dedicated to Hugging Face Spaces.
- Include only runtime files required by the Gradio app.
- Keep the original training/output folder unchanged as the experiment archive.
- Add a lightweight repository test that verifies the deploy folder contains the required runtime files.

## Deployment Structure

The deployment package lives in `hf_space_kansas_scenario7/` and contains:

- `app.py` as the Spaces entrypoint
- `best_model.keras` as the primary model artifact
- `config.json` for feature and label metadata
- `tokenizer.pkl` because the current app loads this file as the scaler object
- `shap_background.npy` for SHAP explanations
- `requirements.txt` with runtime Python dependencies
- `README.md` with Hugging Face Spaces metadata
- `.gitignore` for local runtime byproducts

## App Behavior

The deploy app remains functionally aligned with `kansas/output_weekly_kansas_scenario7/app.py`, but paths resolve only within the new deploy folder and launch settings target Hugging Face Spaces compatibility.

## Testing

Add a pytest check that fails when the deploy folder or any required runtime file is missing. This gives the repository a simple regression guard for future cleanup work.
