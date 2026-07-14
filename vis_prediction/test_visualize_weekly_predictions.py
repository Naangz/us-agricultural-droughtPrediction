import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import visualize_weekly_predictions as vwp


class WeeklyPredictionUtilityTests(unittest.TestCase):
    def test_create_sequences_keeps_county_and_target_week_metadata(self):
        rows = []
        for fips in [31001, 31003]:
            for week in pd.date_range("2021-12-19", periods=5, freq="W-SUN"):
                rows.append(
                    {
                        "FIPS": fips,
                        "week_start": week,
                        "feature": float(fips % 100) + week.day,
                        "Label": week.month % 6,
                    }
                )
        df = pd.DataFrame(rows)

        x_seq, y_seq, meta = vwp.create_sequences_with_metadata(
            df,
            ["feature"],
            seq_length=3,
            start_date="2022-01-01",
        )

        self.assertEqual(x_seq.shape, (6, 3, 1))
        self.assertEqual(y_seq.tolist(), meta["actual_label"].tolist())
        self.assertEqual(meta.columns.tolist(), ["FIPS", "target_week", "actual_label"])
        self.assertEqual(meta["target_week"].min(), pd.Timestamp("2022-01-02"))
        self.assertEqual(meta["FIPS"].tolist(), [31001, 31001, 31001, 31003, 31003, 31003])

    def test_get_scenario_features_uses_saved_selection_csv_order(self):
        tmp_dir = Path(self._testMethodName)
        tmp_dir.mkdir(exist_ok=True)
        csv_path = tmp_dir / "selected_feature_ranking.csv"
        pd.DataFrame(
            {
                "rank": [2, 1],
                "feature": ["late_feature", "early_feature"],
            }
        ).to_csv(csv_path, index=False)

        try:
            features = vwp.get_scenario_features("scenario2", tmp_dir, model_feature_count=2)
        finally:
            csv_path.unlink()
            tmp_dir.rmdir()

        self.assertEqual(features, ["early_feature", "late_feature"])

    def test_build_prediction_frame_applies_class_multipliers_and_probabilities(self):
        meta = pd.DataFrame(
            {
                "FIPS": [1, 2],
                "target_week": pd.to_datetime(["2022-01-02", "2022-01-02"]),
                "actual_label": [0, 1],
            }
        )
        probs = np.array(
            [
                [0.40, 0.35, 0.10, 0.05, 0.05, 0.05],
                [0.10, 0.40, 0.30, 0.10, 0.05, 0.05],
            ]
        )

        frame = vwp.build_prediction_frame(meta, probs, multipliers=[1, 2, 1, 1, 1, 1])

        self.assertEqual(frame["pred_label"].tolist(), [1, 1])
        self.assertEqual(frame["pred_class"].tolist(), ["D0", "D0"])
        self.assertIn("prob_D4", frame.columns)
        self.assertEqual(frame["is_correct"].tolist(), [False, True])

    def test_scale_features_uses_train_split_and_handles_constant_columns(self):
        df = pd.DataFrame(
            {
                "FIPS": [1, 1, 1],
                "week_start": pd.to_datetime(["2019-01-06", "2019-01-13", "2022-01-02"]),
                "feature": [10.0, 20.0, 30.0],
                "constant": [5.0, 5.0, 5.0],
                "Label": [0, 1, 2],
            }
        )

        scaled = vwp.scale_features(df, ["feature", "constant"])

        self.assertEqual(scaled["feature"].round(6).tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(scaled["constant"].tolist(), [0.0, 0.0, 0.0])

    def test_plot_weekly_class_counts_accepts_datetime_weeks(self):
        predictions = pd.DataFrame(
            {
                "target_week": pd.to_datetime(["2022-01-02", "2022-01-02", "2022-01-09"]),
                "pred_class": ["None", "D0", "D0"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "weekly_counts.png"
            with patch.object(pd.DataFrame, "plot", side_effect=ValueError("Must supply freq for datetime value")):
                vwp.plot_weekly_class_counts(predictions, out_path, "Weekly counts")

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
