import unittest

import pandas as pd

from kansas.BiLSTM_Scenario1_Monthly_4MonthForecast import (
    BASE_WEATHER,
    aggregate_weekly_to_monthly_baseline,
    create_sequences_from_df,
    prepare_monthly_baseline_features,
)


class MonthlyScenario1PreprocessingTests(unittest.TestCase):
    def test_aggregate_weekly_to_monthly_sums_precip_and_uses_last_drought_state(self):
        df = pd.DataFrame(
            {
                "FIPS": [1, 1],
                "week_start": ["2020-01-07", "2020-01-28"],
                "ValidEnd": ["2020-01-13", "2020-02-03"],
                "ALLSKY_SFC_SW_DWN": [10.0, 20.0],
                "PRECTOTCORR": [1.5, 2.5],
                "PS": [100.0, 104.0],
                "RH2M": [40.0, 60.0],
                "T2M": [5.0, 9.0],
                "WS2M": [2.0, 4.0],
                "None": [80.0, 70.0],
                "D0": [20.0, 30.0],
                "D1": [0.0, 0.0],
                "D2": [0.0, 0.0],
                "D3": [0.0, 0.0],
                "D4": [0.0, 0.0],
            }
        )

        monthly = aggregate_weekly_to_monthly_baseline(df)

        self.assertEqual(len(monthly), 1)
        self.assertEqual(monthly.loc[0, "PRECTOTCORR"], 4.0)
        self.assertEqual(monthly.loc[0, "T2M"], 7.0)
        self.assertEqual(monthly.loc[0, "None"], 70.0)
        self.assertEqual(monthly.loc[0, "D0"], 30.0)
        self.assertEqual(monthly.loc[0, "week_start"].strftime("%Y-%m-%d"), "2020-01-28")

    def test_prepare_monthly_baseline_features_creates_monthly_feature_set(self):
        rows = []
        for i, month in enumerate(pd.date_range("2020-01-01", periods=14, freq="MS")):
            row = {
                "FIPS": 1,
                "month_start": month,
                "week_start": month,
                "ValidEnd": month,
                "None": 100.0 - i,
                "D0": float(i),
                "D1": 0.0,
                "D2": 0.0,
                "D3": 0.0,
                "D4": 0.0,
                "Label": i % 6,
            }
            for col in BASE_WEATHER:
                row[col] = float(i + 1)
            rows.append(row)
        df = pd.DataFrame(rows)

        df_fe, feature_cols = prepare_monthly_baseline_features(df)

        self.assertIn("month_sin", feature_cols)
        self.assertIn("PREC_lag8", feature_cols)
        self.assertIn("None_lag2", feature_cols)
        self.assertEqual(len(feature_cols), 38)
        self.assertEqual(df_fe["month_start"].min().strftime("%Y-%m-%d"), "2020-09-01")

    def test_create_sequences_uses_four_month_forecast_horizon(self):
        rows = []
        for i, month in enumerate(pd.date_range("2020-01-01", periods=8, freq="MS")):
            rows.append({"FIPS": 1, "month_start": month, "feature": float(i), "Label": i})
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
