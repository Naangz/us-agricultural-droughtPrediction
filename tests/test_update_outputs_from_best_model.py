from pathlib import Path

from kansas.update_outputs_from_best_model import (
    get_feature_columns_for_scenario,
    load_selected_features_from_csv,
    normalize_scenario_key,
    scenario_output_dir_name,
)


def test_normalize_scenario_key_accepts_aliases():
    assert normalize_scenario_key("1") == "1"
    assert normalize_scenario_key("scenario3") == "3"
    assert normalize_scenario_key("2b") == "2B"


def test_scenario_output_dir_name_maps_baseline_and_regular_scenarios():
    assert scenario_output_dir_name("1") == "output_weekly_kansas_20counties"
    assert scenario_output_dir_name("5") == "output_weekly_kansas_scenario5"


def test_load_selected_features_from_csv_reads_ranked_features(tmp_path: Path):
    csv_path = tmp_path / "selected_feature_ranking.csv"
    csv_path.write_text(
        "rank,feature,mutual_information_score\n1,D3_lag1,0.1\n2,D2_lag1,0.09\n",
        encoding="utf-8",
    )

    assert load_selected_features_from_csv(csv_path) == ["D3_lag1", "D2_lag1"]


def test_get_feature_columns_for_scenario_uses_csv_for_selected_scenarios(tmp_path: Path):
    output_dir = tmp_path / "output_weekly_kansas_scenario2"
    output_dir.mkdir()
    (output_dir / "selected_feature_ranking.csv").write_text(
        "rank,feature,mutual_information_score\n1,D3_lag1,0.1\n2,D2_lag1,0.09\n",
        encoding="utf-8",
    )

    feature_cols = get_feature_columns_for_scenario("2", output_dir)

    assert feature_cols == ["D3_lag1", "D2_lag1"]
