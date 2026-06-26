import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import visualization.scenario_analysis_common as scenario_common


class ParseSummaryTests(unittest.TestCase):
    def test_parse_summary_extracts_confusion_matrix_metadata(self):
        summary_content = textwrap.dedent(
            """
            BiLSTM Weekly Kansas Scenario 5
            Best trial: none_focal_inv_cw_64x32
            Seq Length: 52
            Class multipliers: [1.2, 0.8, 1.0, 0.9, 1.1, 0.7]
            Selected features:
              ALLSKY_SFC_SW_DWN
              PRECTOTCORR
              week_sin
            Macro F1: 0.1152

            Per-class F1:
              None: 0.4184
              D0: 0.0000
              D1: 0.1209
              D2: 0.0034
              D3: 0.1485
              D4: 0.0000

            Classification Report:
                          precision    recall  f1-score   support

                    None     0.5529    0.3365    0.4184      1150
                      D0     0.0000    0.0000    0.0000       975
                      D1     0.2808    0.0770    0.1209       948
                      D2     0.0500    0.0018    0.0034       560
                      D3     0.0810    0.8889    0.1485       288
                      D4     0.0000    0.0000    0.0000       259
            """
        ).strip()

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "results_summary.txt"
            summary_path.write_text(summary_content, encoding="utf-8")

            parsed = scenario_common.parse_summary(str(summary_path))

        self.assertEqual(parsed["seq_length"], 52)
        self.assertEqual(parsed["class_multipliers"], [1.2, 0.8, 1.0, 0.9, 1.1, 0.7])
        self.assertEqual(parsed["selected_features"], ["ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "week_sin"])
        self.assertEqual(parsed["support"], [1150, 975, 948, 560, 288, 259])


class PathTemplateTests(unittest.TestCase):
    def test_model_path_uses_requested_scenario_number(self):
        self.assertEqual(
            scenario_common.model_path("kansas", 5),
            "kansas\\output_weekly_kansas_scenario5\\best_model.keras",
        )
        self.assertEqual(
            scenario_common.summary_path("nebraska", 6),
            "nebraska\\output_weekly_nebraska_scenario6\\results_summary.txt",
        )


class RunScenarioAnalysisTests(unittest.TestCase):
    def test_run_scenario_analysis_supports_scenario7(self):
        fake_data = {
            "kansas": {"support": [1, 2, 3, 4, 5, 6], "predicted": [1, 2, 3, 4, 5, 6]},
            "nebraska": {"support": [6, 5, 4, 3, 2, 1], "predicted": [6, 5, 4, 3, 2, 1]},
        }

        with mock.patch.object(scenario_common, "load_scenario_data", return_value=fake_data) as load_mock, \
            mock.patch.object(scenario_common, "plot_perclass_f1_comparison") as perclass_mock, \
            mock.patch.object(scenario_common, "plot_distribution_shift") as shift_mock, \
            mock.patch.object(scenario_common, "HAS_CORE_DATA_LIBRARIES", False), \
            mock.patch.object(scenario_common, "HAS_ML_LIBRARIES", False), \
            mock.patch.object(scenario_common, "HAS_PLOTTING_LIBRARIES", False), \
            mock.patch.object(scenario_common, "sns", None):
            scenario_common.run_scenario_analysis(7, script_dir="visualization")

        load_mock.assert_called_once_with(7)
        perclass_mock.assert_called_once_with(7, fake_data, "visualization")
        self.assertEqual(shift_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
