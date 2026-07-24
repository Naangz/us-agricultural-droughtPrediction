import unittest

import pandas as pd

from nebraska.BiLSTM_Scenario7_Monthly_4MonthForecast import (
    DROUGHT_COLS,
    SEQ_LENGTH,
    TEST_START_DATE,
    TRAIN_END_DATE,
    VAL_END_DATE,
    VAL_START_DATE,
    aggregate_weekly_to_monthly,
    create_sequences_from_df,
    prepare_monthly_drought_history_features,
)


class MonthlyNebraskaScenario7PreprocessingTests(unittest.TestCase):
    def test_monthly_sequence_length_matches_one_year_of_monthly_data(self):
        self.assertEqual(SEQ_LENGTH, 12)

    def test_split_dates_match_monthly_class_coverage_plan(self):
        self.assertEqual(TRAIN_END_DATE, "2020-12-31")
        self.assertEqual(VAL_START_DATE, "2021-01-01")
        self.assertEqual(VAL_END_DATE, "2022-12-31")
        self.assertEqual(TEST_START_DATE, "2023-01-01")

    def test_aggregate_weekly_to_monthly_averages_drought_state_per_county_month(self):
        df = pd.DataFrame(
            {
                "FIPS": [1, 1, 1, 1, 2, 2],
                "week_start": [
                    "2020-01-07",
                    "2020-01-28",
                    "2020-02-04",
                    "2020-02-25",
                    "2020-01-14",
                    "2020-01-21",
                ],
                "None": [80, 70, 60, 50, 90, 85],
                "D0": [20, 30, 40, 50, 10, 15],
                "D1": [0, 0, 0, 0, 0, 0],
                "D2": [0, 0, 0, 0, 0, 0],
                "D3": [0, 0, 0, 0, 0, 0],
                "D4": [0, 0, 0, 0, 0, 0],
            }
        )

        monthly = aggregate_weekly_to_monthly(df)

        self.assertEqual(list(monthly["month_start"].dt.strftime("%Y-%m-%d")), ["2020-01-01", "2020-02-01", "2020-01-01"])
        self.assertEqual(monthly["None"].tolist(), [75.0, 55.0, 87.5])
        self.assertEqual(monthly["D0"].tolist(), [25.0, 45.0, 12.5])

    def test_prepare_monthly_drought_history_features_uses_one_and_two_month_lags(self):
        rows = []
        for i, month in enumerate(pd.date_range("2020-01-01", periods=4, freq="MS")):
            row = {
                "FIPS": 1,
                "month_start": month,
                "Label": i,
            }
            for col in DROUGHT_COLS:
                row[col] = float(i)
            rows.append(row)
        df = pd.DataFrame(rows)

        df_fe, feature_cols = prepare_monthly_drought_history_features(df)

        self.assertEqual(
            feature_cols,
            [
                "None_lag1",
                "D0_lag1",
                "D1_lag1",
                "D2_lag1",
                "D3_lag1",
                "D4_lag1",
                "None_lag2",
                "D0_lag2",
                "D1_lag2",
                "D2_lag2",
                "D3_lag2",
                "D4_lag2",
            ],
        )
        self.assertEqual(df_fe["month_start"].dt.strftime("%Y-%m-%d").tolist(), ["2020-03-01", "2020-04-01"])
        self.assertEqual(df_fe["None_lag1"].tolist(), [1.0, 2.0])
        self.assertEqual(df_fe["None_lag2"].tolist(), [0.0, 1.0])

    def test_create_sequences_uses_label_four_months_after_input_window(self):
        rows = []
        for i, month in enumerate(pd.date_range("2020-01-01", periods=8, freq="MS")):
            rows.append(
                {
                    "FIPS": 1,
                    "month_start": month,
                    "feature": float(i),
                    "Label": i,
                }
            )
        df = pd.DataFrame(rows)

        X, y, input_end_dates, target_dates = create_sequences_from_df(
            df,
            ["feature"],
            "Label",
            seq_length=3,
            forecast_horizon=4,
            time_col="month_start",
            return_dates=True,
        )

        self.assertEqual(X.shape, (2, 3, 1))
        self.assertEqual(y.tolist(), [6, 7])
        self.assertEqual([d.strftime("%Y-%m-%d") for d in input_end_dates], ["2020-03-01", "2020-04-01"])
        self.assertEqual([d.strftime("%Y-%m-%d") for d in target_dates], ["2020-07-01", "2020-08-01"])


if __name__ == "__main__":
    unittest.main()
