---
title: Kansas Monthly Drought Prediction Baseline
emoji: chart_with_upwards_trend
colorFrom: green
colorTo: lime
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
---

# Monthly Drought Prediction Dashboard (Kansas)

Model Scenario 1: Baseline (BiLSTM)

This Space runs a Bidirectional LSTM model trained to predict drought categories
(None, D0-D4) for 20 Kansas counties using monthly weather, seasonality, and
drought history inputs with a 4-month forecast horizon.
