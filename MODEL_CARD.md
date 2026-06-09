Model Card: BiLSTM Weekly Drought Classifier
============================================

Model Details
-------------
- Model name: BiLSTM Weekly Drought Classifier
- Version: 1.0 (undergraduate final project)
- Authors: Project author (add name/affiliation in README)
- License: MIT

Overview
--------
This model predicts weekly drought class (None, D0..D4) for county-level inputs using 52 weeks of historical weather and USDM-derived drought signals. The architecture is a bidirectional LSTM sequence classifier with configurable hidden sizes and training strategies (balancing, focal loss).

Intended Use
------------
- Educational and research purposes: reproducible case study in sequence modeling for drought classification.
- Prototype regional drought monitoring and exploratory transfer experiments across nearby regions (Kansas and Nebraska).

Out-of-Scope Uses
-----------------
- Operational decision-making without domain validation.
- Real-time hazard warning or emergency response systems.

Factors and Metrics
-------------------
- Primary metric: macro-F1 (validation and test). Latest reported raw test macro-F1: 0.8137.
- Other metrics: per-class F1, confusion matrices, and classification reports are produced by the training pipeline.

Training Data
-------------
- Sources: NASA POWER daily weather (aggregated to weekly) and USDM weekly drought coverage exports.
- Primary dataset: `Integrated_weekly_KAN_20counties.csv` (Kansas, 20 counties, 2009–2025 subset in repo).
- Nebraska datasets provided as secondary experiments.
- Notes: USDM-derived labels are aggregated coverage percentages and converted to categorical labels via decumulation and argmax. Label noise and aggregation artifacts are possible.

Evaluation Data
---------------
- Temporal split used: Train (<=2019-12-31), Validation (2020-2021), Test (>=2022-01-01).
- Sequences of length 52 built per county; targets are the final timestep class.

Architecture and Training
-------------------------
- Input: (52, n_features) where n_features = 41 engineered features.
- Two stacked Bidirectional LSTM layers with dropout/batchnorm, followed by Dense layers and softmax output (6 classes).
- Loss: categorical focal loss (configurable gamma) with optional class weights; Adam optimizer.
- Training artifacts: best model checkpoints, training history plots, and per-trial results in `output_*` folders.

Limitations and Biases
----------------------
- Geographic bias: trained and evaluated on a fixed set of Kansas counties; behavior outside this domain is untested.
- Labeling bias: labels derived from USDM coverage which may not perfectly represent local impacts.
- Class imbalance: drought categories are imbalanced; training strategies (ROS, focal loss) mitigate but do not eliminate effects.

Ethical Considerations and Safety
--------------------------------
- The model is not intended for automated operational decisions; human oversight is required.
- Users should be aware of potential harms from incorrect drought classification (resource misallocation, false reassurance).
- Before deployment, perform region-specific validation and consult domain experts.

Recommendations for Use
-----------------------
- Reproduce training using `BiLSTM_Weekly_Kansas_Clean.py` and the included datasets/scripts.
- If applying to new regions, retrain or fine-tune on local data and report per-class metrics.
- Add data provenance and a DOI if publishing datasets used for evaluation.
