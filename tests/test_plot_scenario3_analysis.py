import tempfile
import textwrap
import unittest
from pathlib import Path

import visualization.plot_scenario3_analysis as scenario3_analysis


class ParseScenario3SummaryTests(unittest.TestCase):
    def test_parse_scenario3_summary_extracts_dynamic_metadata(self):
        summary_content = textwrap.dedent(
            """
            BiLSTM Weekly Kansas - Scenario 3: Macro F1-driven Feature Selection
            Best trial: ros_focal_no_cw_96x48
            Best trial config: {'name': 'ros_focal_no_cw_96x48'}
            Seq Length: 52

            --- FEATURE SELECTION ---
            Baseline feature count: 41
            Subset sizes evaluated: [30, 25, 20, 15, 10]

            Subset evaluation results:
              Top-20: Validation Macro F1 = 0.8106
              Top-15: Validation Macro F1 = 0.8062
              Top-25: Validation Macro F1 = 0.7607
              Top-10: Validation Macro F1 = 0.7388
              Top-30: Validation Macro F1 = 0.6328

            Best subset selected: Top-20 (20 features) with val macro-F1 = 0.8106
            Selected features: ['D3_lag1', 'D2_lag1', 'D1_lag1']

            --- RESULTS ---
            Val Macro F1 (best trial): 0.8486
            Val Macro F1 (raw/tuned): 0.8486 / 0.8516
            Class multipliers: [1.0772, 1.1422, 1.1386, 0.7460, 0.6103, 1.0819]
            Macro F1: 0.7940

            Classification Report:
                          precision    recall  f1-score   support

                    None     0.9100    0.9139    0.9119      1150
                      D0     0.7738    0.7928    0.7832       975
                      D1     0.7940    0.7806    0.7872       948
                      D2     0.7815    0.7089    0.7434       560
                      D3     0.7019    0.6458    0.6727       288
                      D4     0.7819    0.9691    0.8655       259
            """
        ).strip()

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "results_summary.txt"
            summary_path.write_text(summary_content, encoding="utf-8")

            parsed = scenario3_analysis.parse_scenario3_summary(str(summary_path))

        self.assertEqual(parsed["summary_title"], "BiLSTM Weekly Kansas - Scenario 3: Macro F1-driven Feature Selection")
        self.assertEqual(parsed["seq_length"], 52)
        self.assertEqual(parsed["best_subset_size"], 20)
        self.assertEqual(parsed["selected_features"], ['D3_lag1', 'D2_lag1', 'D1_lag1'])
        self.assertEqual(parsed["class_multipliers"], [1.0772, 1.1422, 1.1386, 0.7460, 0.6103, 1.0819])
        self.assertEqual(parsed["validation_trend"][10], 0.7388)
        self.assertEqual(parsed["validation_trend"][20], 0.8106)
        self.assertEqual(parsed["support"], [1150, 975, 948, 560, 288, 259])


if __name__ == "__main__":
    unittest.main()
