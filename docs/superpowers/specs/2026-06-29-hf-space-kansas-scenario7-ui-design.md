# Hugging Face Space Kansas Scenario 7 UI Redesign

## Goal

Rework the Hugging Face Spaces app UI in `hf_space_kansas_scenario7/app.py` so non-technical users can predict weekly drought conditions through a guided, educational wizard instead of a technical form.

## User Experience Direction

The app should feel like a guided decision-support tool rather than a machine learning demo. The primary audience is non-expert users who may not understand USDM terminology, raw feature names, or model explanation charts.

The interface should:

- prioritize the novice flow only
- remove the advanced mode from the main UI
- present inputs in everyday language
- guide users step by step with a wizard flow
- explain drought severity levels in short, practical terms
- keep model explanation details secondary

## Interaction Flow

The page is organized as a three-step wizard with a friendly landing section above it.

### Landing Section

The top of the page should include:

- a clear title focused on weekly drought prediction
- a short explanation of what the app does
- a compact "what you need to know" panel explaining that users only need to estimate drought conditions from the last two weeks

### Step 1: Two Weeks Ago

Users choose:

- the main drought severity level from large selectable cards
- the estimated percentage of affected area with a simple slider

The step includes short educational help text describing what each severity level means in practical terms.

### Step 2: Last Week

Users repeat the same input pattern for the more recent week:

- severity selection with large cards
- affected-area percentage slider

This step should feel visually consistent with Step 1 and reinforce the same educational guidance.

### Step 3: Review and Predict

Users see:

- a summary card for conditions two weeks ago
- a summary card for conditions last week
- a short interpretation of what their answers mean
- a primary prediction button

This step lets users confirm their choices before running the model.

### Results

After prediction, the app should show:

- a dominant result card with the predicted drought level
- a plain-language explanation of what the prediction means
- a concise recap of the inputs used
- an optional secondary section for SHAP-based model explanation

The SHAP section should be hidden inside an accordion or secondary panel by default.

## Visual Design

The visual direction should be warm, clean, and clear.

- Use earthy greens, yellows, oranges, and reds as the core palette.
- Favor rounded cards and soft surfaces instead of dense controls.
- Use a clear step/progress indicator to orient users in the wizard.
- Keep the layout usable on mobile and desktop.
- Make the novice flow visually dominant and keep technical details visually de-emphasized.

## Input Model Mapping

The existing model input contract remains unchanged. The wizard continues to collect:

- drought severity level
- estimated affected-area percentage

for both time periods. The app should convert those answers to the same cumulative input features currently expected by the model.

This keeps the redesign focused on user experience without retraining or changing the model.

## Validation Behavior

Validation should help without becoming strict or intimidating.

- If a user selects a normal condition with a very high affected-area percentage, show a gentle review note.
- If inputs appear unusual, encourage review rather than blocking the user outright.
- Slider labels should help users interpret percentages in practical terms, such as small area, half the county, or nearly all areas.

## Content Strategy

The interface should replace technical labels with simple explanations whenever possible.

- Avoid exposing raw feature names in the main experience.
- Explain drought categories with practical field-level descriptions.
- Keep technical explanation content optional and secondary.

## Implementation Scope

The redesign is limited to the Hugging Face Spaces app in `hf_space_kansas_scenario7/app.py`.

Expected changes include:

- restructuring the Gradio layout into a wizard-style experience
- replacing dropdown-first inputs with card-based selections
- removing the advanced mode from the visible UI
- adding friendly summary and validation messaging
- repositioning SHAP output into a secondary explanation section

The underlying prediction logic and asset loading remain intact unless small refactors are needed to support the new UI.
