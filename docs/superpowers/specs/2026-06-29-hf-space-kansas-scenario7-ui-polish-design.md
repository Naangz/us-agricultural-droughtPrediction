# Hugging Face Space Kansas Scenario 7 UI Polish Design

## Goal

Polish the existing novice wizard UI in `hf_space_kansas_scenario7/app.py` so it feels warmer, more educational, and more presentation-ready for Hugging Face Spaces without changing the prediction flow or model contract.

## Experience Direction

The current wizard already simplifies the interaction. This polish pass should improve tone, hierarchy, and visual atmosphere so the app feels more like a public-facing guidance tool and less like a functional prototype.

The target feeling is:

- warm and approachable
- educational without being verbose
- visually clearer and more guided
- suitable for demos and public sharing

## Scope

This pass focuses on:

- hero presentation
- information panels
- step communication
- input guidance copy
- result presentation
- CSS hierarchy and visual emphasis

This pass does not change:

- model assets
- prediction pipeline
- wizard step order
- helper input mapping contract

## Hero Polish

The top section should feel more welcoming and informative.

Expected improvements:

- stronger title and subheading hierarchy
- small support panels such as who this tool is for, what needs to be prepared, and how long it takes
- visual framing that reassures users the process is short and manageable

## Informational Panels

The app should use compact educational panels to reduce user hesitation.

Expected improvements:

- a clearer "what you need to know" section
- short context labels or badges
- lightweight helper blocks that explain why the app asks about two prior weeks

These panels should be easy to scan and should not overwhelm the main task.

## Wizard Step Polish

Each step should feel more intentional and more supportive.

Expected improvements:

- clearer step badges and section hierarchy
- stronger supporting text under each step title
- visual grouping between instructional content and input controls
- risk-language cues such as safe, watchful, or serious where helpful

The severity cards should remain easy to compare and should feel more like guided choices than form controls.

## Input Guidance Copy

Copy should read like guidance from a helpful assistant.

Expected improvements:

- shorter and more natural instructional language
- practical examples instead of technical wording
- coverage guidance that helps users estimate area impact with confidence

All copy should stay in Indonesian and remain accessible to non-technical users.

## Result Polish

The result area should feel like an interpreted outcome rather than a raw model output.

Expected improvements:

- stronger main result card emphasis
- a clearer "what this means" explanation
- a better recap of the user's recent conditions
- an optional "what to pay attention to next" style guidance block

The SHAP explanation remains secondary and should continue to feel safely optional.

## Visual Direction

The visual style should be warm-educational and more illustrative than the current version.

Expected improvements:

- richer use of earthy greens, warm neutrals, soft yellows, and drought-warning accents
- more expressive card styling, spacing, and badge treatments
- improved contrast and grouping so scanning the screen feels easier
- better visual rhythm between hero, steps, review, and results

The interface should remain clean and readable on both desktop and mobile.

## Implementation Limits

Implementation should stay inside:

- `hf_space_kansas_scenario7/app.py`
- `tests/test_hf_space_kansas_scenario7.py`

Additional helper extraction is only acceptable if clearly necessary, but this polish pass should prefer incremental changes to the current structure.
