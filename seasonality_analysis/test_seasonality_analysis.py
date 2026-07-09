import numpy as np
import pandas as pd

from analyze_seasonality import (
    aggregate_weekly_region,
    compute_acf_values,
    interpret_acf_at_lag,
)


def test_aggregate_weekly_region_averages_counties_by_week():
    df = pd.DataFrame(
        {
            "FIPS": [1, 2, 1, 2],
            "week_start": ["2020-01-07", "2020-01-07", "2020-01-14", "2020-01-14"],
            "PRECTOTCORR": [1.0, 3.0, 5.0, 7.0],
            "T2M": [10.0, 14.0, 20.0, 24.0],
            "RH2M": [50.0, 70.0, 60.0, 80.0],
        }
    )

    weekly = aggregate_weekly_region(df, ["PRECTOTCORR", "T2M", "RH2M"])

    assert weekly["PRECTOTCORR"].tolist() == [2.0, 6.0]
    assert weekly["T2M"].tolist() == [12.0, 22.0]
    assert weekly["RH2M"].tolist() == [60.0, 70.0]


def test_compute_acf_values_detects_52_week_cycle():
    weeks = np.arange(156)
    seasonal = np.sin(2 * np.pi * weeks / 52.0)

    acf = compute_acf_values(seasonal, nlags=104)

    assert acf[0] == 1.0
    assert acf[52] > 0.55


def test_interpret_acf_at_lag_marks_significant_positive_lag():
    result = interpret_acf_at_lag(acf_value=0.45, ci_half_width=0.2, local_values=[0.1, 0.2, 0.45, 0.15])

    assert result["is_significant"] is True
    assert result["is_positive"] is True
    assert result["is_local_peak"] is True
