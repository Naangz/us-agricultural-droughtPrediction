import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import visualization.plot_macro_f1_raw_vs_tuned as plot_module


class ParseMacroF1SummaryTests(unittest.TestCase):
    def test_parse_summary_extracts_raw_and_tuned_macro_f1(self):
        summary_content = textwrap.dedent(
            """
            BiLSTM Weekly Kansas - Scenario 7
            Best trial: ros_focal_inv_no_cw_128x64

            Macro F1 (raw): 0.8202
            Accuracy (raw): 0.8211

            Macro F1: 0.8289
            Accuracy: 0.8246
            """
        ).strip()

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "results_summary.txt"
            summary_path.write_text(summary_content, encoding="utf-8")

            parsed = plot_module.parse_macro_f1_summary(summary_path)

        self.assertEqual(parsed["title"], "BiLSTM Weekly Kansas - Scenario 7")
        self.assertEqual(parsed["best_trial"], "ros_focal_inv_no_cw_128x64")
        self.assertAlmostEqual(parsed["raw_macro_f1"], 0.8202)
        self.assertAlmostEqual(parsed["tuned_macro_f1"], 0.8289)

    def test_parse_summary_uses_last_tuned_macro_f1_value(self):
        summary_content = textwrap.dedent(
            """
            Example Summary
            Macro F1: 0.7000
            Something else
            Macro F1 (raw): 0.6500
            Macro F1: 0.7100
            """
        ).strip()

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "results_summary.txt"
            summary_path.write_text(summary_content, encoding="utf-8")

            parsed = plot_module.parse_macro_f1_summary(summary_path)

        self.assertAlmostEqual(parsed["raw_macro_f1"], 0.6500)
        self.assertAlmostEqual(parsed["tuned_macro_f1"], 0.7100)


class ScenarioPathTests(unittest.TestCase):
    def test_summary_path_maps_baseline_and_all_scenarios(self):
        expected = {
            ("kansas", "Scenario 1"): Path("kansas/output_weekly_kansas_20counties/results_summary.txt"),
            ("kansas", "Scenario 2"): Path("kansas/output_weekly_kansas_scenario2/results_summary.txt"),
            ("kansas", "Scenario 2A"): Path("kansas/output_weekly_kansas_scenario2A/results_summary.txt"),
            ("kansas", "Scenario 2B"): Path("kansas/output_weekly_kansas_scenario2B/results_summary.txt"),
            ("kansas", "Scenario 2C"): Path("kansas/output_weekly_kansas_scenario2C/results_summary.txt"),
            ("kansas", "Scenario 3"): Path("kansas/output_weekly_kansas_scenario3/results_summary.txt"),
            ("kansas", "Scenario 4"): Path("kansas/output_weekly_kansas_scenario4/results_summary.txt"),
            ("kansas", "Scenario 5"): Path("kansas/output_weekly_kansas_scenario5/results_summary.txt"),
            ("kansas", "Scenario 6"): Path("kansas/output_weekly_kansas_scenario6/results_summary.txt"),
            ("kansas", "Scenario 7"): Path("kansas/output_weekly_kansas_scenario7/results_summary.txt"),
            ("nebraska", "Scenario 1"): Path("nebraska/output_weekly_nebraska_20counties/results_summary.txt"),
            ("nebraska", "Scenario 2"): Path("nebraska/output_weekly_nebraska_scenario2/results_summary.txt"),
            ("nebraska", "Scenario 2A"): Path("nebraska/output_weekly_nebraska_scenario2A/results_summary.txt"),
            ("nebraska", "Scenario 2B"): Path("nebraska/output_weekly_nebraska_scenario2B/results_summary.txt"),
            ("nebraska", "Scenario 2C"): Path("nebraska/output_weekly_nebraska_scenario2C/results_summary.txt"),
            ("nebraska", "Scenario 3"): Path("nebraska/output_weekly_nebraska_scenario3/results_summary.txt"),
            ("nebraska", "Scenario 4"): Path("nebraska/output_weekly_nebraska_scenario4/results_summary.txt"),
            ("nebraska", "Scenario 5"): Path("nebraska/output_weekly_nebraska_scenario5/results_summary.txt"),
            ("nebraska", "Scenario 6"): Path("nebraska/output_weekly_nebraska_scenario6/results_summary.txt"),
            ("nebraska", "Scenario 7"): Path("nebraska/output_weekly_nebraska_scenario7/results_summary.txt"),
        }

        for (region, scenario), summary_path in expected.items():
            with self.subTest(region=region, scenario=scenario):
                self.assertEqual(plot_module.summary_path(region, scenario), summary_path)


class LabelOrientationTests(unittest.TestCase):
    def test_add_value_labels_rotates_annotations_vertically(self):
        axis = mock.Mock()
        bar = mock.Mock()
        bar.get_height.return_value = 0.8289
        bar.get_x.return_value = 1.0
        bar.get_width.return_value = 0.36

        plot_module.add_value_labels(axis, [bar])

        _, kwargs = axis.annotate.call_args
        self.assertEqual(kwargs["rotation"], 90)


if __name__ == "__main__":
    unittest.main()
