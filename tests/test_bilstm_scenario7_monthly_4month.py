import unittest

import pandas as pd

from kansas.BiLSTM_Scenario7_Monthly_4MonthForecast import (
    aggregate_weekly_to_monthly,
    create_sequences_from_df,
)


class MonthlyScenario7PreprocessingTests(unittest.TestCase):
    def test_aggregate_weekly_to_monthly_keeps_last_record_per_county_month(self):
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

        self.assertEqual(list(monthly["week_start"].dt.strftime("%Y-%m-%d")), ["2020-01-28", "2020-02-25", "2020-01-21"])
        self.assertEqual(list(monthly["month_start"].dt.strftime("%Y-%m-%d")), ["2020-01-01", "2020-02-01", "2020-01-01"])

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
