import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import visualization.plot_scenario1_analysis as scenario1_analysis


class ParseScenario1SummaryTests(unittest.TestCase):
    def test_uses_final_classification_report_when_raw_report_is_present(self):
        summary_content = textwrap.dedent(
            """
            BiLSTM Weekly Kansas Tuned (20 counties)
            Best trial: ros_focal_no_cw_96x48
            Seq Length: 52
            Class multipliers: [1.1, 0.9, 1.0, 1.0, 1.0, 1.0]
            Macro F1: 0.7970

            Per-class F1:
              None: 0.9002
              D0: 0.7836
              D1: 0.7725
              D2: 0.7520
              D3: 0.6954
              D4: 0.8781

            ======================================================================
            CLASSIFICATION REPORT (RAW)
            ======================================================================
                          precision    recall  f1-score   support

                    None     0.9345    0.8678    0.8999      1150
                      D0     0.7431    0.8246    0.7817       975
                      D1     0.8126    0.7363    0.7726       948
                      D2     0.7408    0.7554    0.7480       560
                      D3     0.6901    0.7500    0.7188       288
                      D4     0.8502    0.9421    0.8938       259

            ======================================================================
            CLASSIFICATION REPORT
            ======================================================================
                          precision    recall  f1-score   support

                    None     0.9320    0.8704    0.9002      1150
                      D0     0.7391    0.8338    0.7836       975
                      D1     0.8216    0.7289    0.7725       948
                      D2     0.7435    0.7607    0.7520       560
                      D3     0.6894    0.7014    0.6954       288
                      D4     0.8194    0.9459    0.8781       259

            ======================================================================
            Accuracy:    0.8081
            Macro F1:    0.7970
            Weighted F1: 0.8087
            ======================================================================
            """
        ).strip()

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "results_summary.txt"
            summary_path.write_text(summary_content, encoding="utf-8")

            parsed = scenario1_analysis.parse_scenario1_summary(str(summary_path))

        self.assertEqual(parsed["support"], [1150, 975, 948, 560, 288, 259])
        self.assertEqual(parsed["f1_scores"], [0.9002, 0.7836, 0.7725, 0.7520, 0.6954, 0.8781])
        self.assertEqual(parsed["predicted"], [1074, 1100, 841, 573, 293, 299])
        self.assertEqual(parsed["summary_title"], "BiLSTM Weekly Kansas Tuned (20 counties)")
        self.assertEqual(parsed["best_trial"], "ros_focal_no_cw_96x48")
        self.assertEqual(parsed["seq_length"], 52)
        self.assertEqual(parsed["class_multipliers"], [1.1, 0.9, 1.0, 1.0, 1.0, 1.0])


class SummaryPathSelectionTests(unittest.TestCase):
    def test_get_summary_path_prefers_most_recent_candidate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_summary = Path(tmp_dir) / "old_results_summary.txt"
            new_summary = Path(tmp_dir) / "new_results_summary.txt"
            old_summary.write_text("old", encoding="utf-8")
            new_summary.write_text("new", encoding="utf-8")

            with mock.patch.dict(
                scenario1_analysis.REGION_CONFIGS,
                {
                    "kansas": {
                        **scenario1_analysis.REGION_CONFIGS["kansas"],
                        "summary_candidates": [str(old_summary), str(new_summary)],
                    }
                },
                clear=False,
            ):
                with mock.patch("os.path.getmtime", side_effect=[10, 20]):
                    selected = scenario1_analysis.get_summary_path("kansas")

        self.assertEqual(selected, str(new_summary))


if __name__ == "__main__":
    unittest.main()
