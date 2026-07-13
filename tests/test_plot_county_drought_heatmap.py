import unittest
from pathlib import Path

import pandas as pd

from visualization.dataset import plot_county_drought_heatmap as plot_module


class CountyDroughtHeatmapDataTests(unittest.TestCase):
    def test_prepare_heatmap_data_filters_selection_and_keeps_alphabetical_county_order(self):
        usdm = pd.DataFrame(
            {
                "FIPS": [20001, 20003, 20005, 20001, 20003, 20005],
                "County": [
                    "Zephyr County",
                    "Alpha County",
                    "Atchison County",
                    "Zephyr County",
                    "Alpha County",
                    "Atchison County",
                ],
                "D0": [30.0, 10.0, 90.0, 40.0, 20.0, 100.0],
                "ValidStart": [
                    "2020-01-14",
                    "2020-01-14",
                    "2020-01-14",
                    "2020-01-07",
                    "2020-01-07",
                    "2020-01-07",
                ],
            }
        )
        selection = pd.DataFrame(
            {
                "FIPS": [20001, 20003],
                "County": ["Zephyr County", "Alpha County"],
            }
        )

        matrix = plot_module.prepare_heatmap_matrix(usdm, selection)

        self.assertEqual(list(matrix.index), ["Alpha County", "Zephyr County"])
        self.assertEqual(
            list(matrix.columns),
            [pd.Timestamp("2020-01-07"), pd.Timestamp("2020-01-14")],
        )
        self.assertEqual(matrix.loc["Alpha County", pd.Timestamp("2020-01-07")], 20.0)
        self.assertEqual(matrix.loc["Zephyr County", pd.Timestamp("2020-01-14")], 30.0)

    def test_prepare_heatmap_data_rejects_missing_required_columns(self):
        usdm = pd.DataFrame({"FIPS": [20001], "D0": [10.0]})
        selection = pd.DataFrame({"FIPS": [20001], "County": ["Allen County"]})

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            plot_module.prepare_heatmap_matrix(usdm, selection)


class AxisTickTests(unittest.TestCase):
    def test_year_tick_positions_returns_one_midpoint_tick_per_year(self):
        dates = pd.to_datetime(
            ["2020-01-07", "2020-01-14", "2020-12-29", "2021-01-05", "2021-01-12"]
        )

        positions, labels = plot_module.year_tick_positions(pd.DatetimeIndex(dates))

        self.assertEqual(positions, [1.5, 4.0])
        self.assertEqual(labels, ["2020", "2021"])

    def test_year_tick_positions_returns_2009_to_2025_once_for_usdm_range(self):
        dates = pd.date_range("2009-12-29", "2025-12-30", freq="7D")

        positions, labels = plot_module.year_tick_positions(pd.DatetimeIndex(dates))

        self.assertEqual(labels, [str(year) for year in range(2009, 2026)])
        self.assertEqual(len(labels), len(set(labels)))
        self.assertGreater(positions[1] - positions[0], 20)


class SummaryTests(unittest.TestCase):
    def test_county_summary_computes_descriptive_metrics(self):
        matrix = pd.DataFrame(
            {
                pd.Timestamp("2020-01-07"): [0.0, 100.0],
                pd.Timestamp("2020-01-14"): [50.0, 100.0],
                pd.Timestamp("2020-01-21"): [100.0, 0.0],
                pd.Timestamp("2020-01-28"): [0.0, 0.0],
            },
            index=["Allen County", "Anderson County"],
        )

        summary = plot_module.compute_county_summary(matrix)

        self.assertEqual(
            list(summary.columns),
            [
                "county",
                "mean_d0_plus",
                "median_d0_plus",
                "pct_weeks_d0_plus_gt_0",
                "pct_weeks_d0_plus_eq_100",
            ],
        )
        allen = summary.loc[summary["county"] == "Allen County"].iloc[0]
        self.assertAlmostEqual(allen["mean_d0_plus"], 37.5)
        self.assertAlmostEqual(allen["median_d0_plus"], 25.0)
        self.assertAlmostEqual(allen["pct_weeks_d0_plus_gt_0"], 50.0)
        self.assertAlmostEqual(allen["pct_weeks_d0_plus_eq_100"], 25.0)

    def test_year_summary_computes_county_week_metrics(self):
        matrix = pd.DataFrame(
            {
                pd.Timestamp("2020-12-29"): [100.0, 0.0],
                pd.Timestamp("2021-01-05"): [50.0, 100.0],
                pd.Timestamp("2021-01-12"): [0.0, 100.0],
            },
            index=["Allen County", "Anderson County"],
        )

        summary = plot_module.compute_year_summary(matrix)

        self.assertEqual(
            list(summary.columns),
            ["year", "mean_d0_plus", "median_d0_plus", "pct_county_weeks_d0_plus_eq_100"],
        )
        row_2021 = summary.loc[summary["year"] == 2021].iloc[0]
        self.assertAlmostEqual(row_2021["mean_d0_plus"], 62.5)
        self.assertAlmostEqual(row_2021["median_d0_plus"], 75.0)
        self.assertAlmostEqual(row_2021["pct_county_weeks_d0_plus_eq_100"], 50.0)


class LabelTests(unittest.TestCase):
    def test_region_titles_and_colorbar_label_match_requested_text(self):
        self.assertEqual(
            plot_module.region_title("kansas"),
            "Persentase Wilayah County pada Kondisi D0 atau Lebih Parah di Kansas",
        )
        self.assertEqual(
            plot_module.region_title("nebraska"),
            "Persentase Wilayah County pada Kondisi D0 atau Lebih Parah di Nebraska",
        )
        self.assertEqual(
            plot_module.COLORBAR_LABEL,
            "Persentase Wilayah D0 atau Lebih Parah (%)",
        )


class OutputPathTests(unittest.TestCase):
    def test_region_output_path_uses_requested_filenames(self):
        output_dir = Path("visualization/dataset")

        self.assertEqual(
            plot_module.region_output_path(output_dir, "kansas"),
            output_dir / "heatmap_county_drought_kansas.png",
        )
        self.assertEqual(
            plot_module.region_output_path(output_dir, "nebraska"),
            output_dir / "heatmap_county_drought_nebraska.png",
        )

    def test_summary_output_paths_use_region_specific_csv_names(self):
        output_dir = Path("visualization/dataset")

        self.assertEqual(
            plot_module.county_summary_output_path(output_dir, "kansas"),
            output_dir / "summary_county_drought_kansas.csv",
        )
        self.assertEqual(
            plot_module.year_summary_output_path(output_dir, "nebraska"),
            output_dir / "summary_year_drought_nebraska.csv",
        )


if __name__ == "__main__":
    unittest.main()
