import ast
import os
from pathlib import Path

from hf_space_kansas_scenario7.ui_helpers import (
    build_input_features,
    build_review_note,
    describe_coverage,
)


ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT / "hf_space_kansas_scenario7"
APP_PATH = DEPLOY_DIR / "app.py"


def read_app_text():
    return APP_PATH.read_text(encoding="utf-8")


def load_app_function(name):
    module_ast = ast.parse(read_app_text(), filename=str(APP_PATH))
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            function_module = ast.Module(body=[node], type_ignores=[])
            namespace = {}
            exec(compile(function_module, filename=str(APP_PATH), mode="exec"), namespace)
            return namespace[name]
    raise AssertionError(f"Function {name} not found in {APP_PATH}")


def load_app_symbols(*names):
    module_ast = ast.parse(read_app_text(), filename=str(APP_PATH))
    selected_nodes = []
    remaining = set(names)

    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            target_names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if target_names & remaining:
                selected_nodes.append(node)
                remaining -= target_names
        elif isinstance(node, ast.FunctionDef) and node.name in remaining:
            selected_nodes.append(node)
            remaining.remove(node.name)

    if remaining:
        raise AssertionError(f"Symbols not found in {APP_PATH}: {sorted(remaining)}")

    namespace = {"os": os}
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), filename=str(APP_PATH), mode="exec"), namespace)
    return namespace


def test_hf_space_kansas_scenario7_contains_runtime_files():
    required_files = [
        ".gitignore",
        "LICENSE",
        "README.md",
        "app.py",
        "best_model.keras",
        "config.json",
        "requirements.txt",
        "tokenizer.pkl",
    ]

    assert DEPLOY_DIR.is_dir(), f"Missing deploy directory: {DEPLOY_DIR}"

    missing_files = [name for name in required_files if not (DEPLOY_DIR / name).is_file()]
    assert not missing_files, f"Missing runtime files: {missing_files}"


def test_hf_space_kansas_scenario7_readme_contains_spaces_metadata():
    readme_text = (DEPLOY_DIR / "README.md").read_text(encoding="utf-8")

    assert readme_text.startswith("---\n"), "README should start with YAML front matter"
    assert "sdk: gradio" in readme_text
    assert "app_file: app.py" in readme_text
    assert "shap_background.npy" not in readme_text


def test_hf_space_app_contains_wizard_copy():
    app_text = read_app_text()

    assert "Langkah 1 dari 3" in app_text
    assert "Mulai Prediksi" in app_text
    assert "Mengapa model memprediksi ini?" not in app_text


def test_hf_space_app_contains_info_panel_and_educational_copy():
    app_text = read_app_text()

    assert "Yang perlu Anda tahu" in app_text
    assert "dua minggu terakhir" in app_text
    assert "Normal berarti air masih cukup" in app_text
    assert "D0 sampai D1 menandakan tanah mulai mengering" in app_text
    assert ".info-panel, .info-panel * {" in app_text
    assert ".progress-card, .progress-card * {" in app_text
    assert ".step-shell .gr-markdown, .step-shell .gr-markdown * {" in app_text
    assert ".step-shell .prose, .step-shell .prose * {" in app_text
    assert ".step-shell strong {" in app_text
    assert "color: #183223 !important;" in app_text


def test_hf_space_app_contains_richer_hero_support_panels():
    app_text = read_app_text()

    assert "Cocok untuk" in app_text
    assert "Siapkan info ini" in app_text
    assert "Waktu isi" in app_text


def test_render_risk_guidance_outputs_structured_badge_hooks():
    namespace = load_app_symbols("RISK_GUIDANCE_ITEMS", "BADGE_HOOK_BY_LABEL", "render_risk_guidance")
    render_risk_guidance = namespace["render_risk_guidance"]

    html = render_risk_guidance()

    assert 'class="risk-grid"' in html
    assert html.count('class="risk-card ') == 3
    assert 'class="risk-badge badge-aman"' in html
    assert 'class="risk-badge badge-waspada"' in html
    assert 'class="risk-badge badge-serius"' in html
    assert "Aman" in html
    assert "Perlu waspada" in html
    assert "Risiko serius" in html
    assert html.count('class="risk-copy"') == 3


def test_hf_space_app_forces_readable_text_in_risk_cards():
    app_text = read_app_text()

    assert ".risk-card, .risk-card * {" in app_text
    assert ".step-kicker {" in app_text
    assert "color: #183223 !important;" in app_text


def test_hf_space_app_uses_sliders_for_affected_area_inputs():
    app_text = read_app_text()

    assert "two_weeks_pct = gr.Slider(" in app_text
    assert "last_week_pct = gr.Slider(" in app_text
    assert "two_weeks_pct = gr.Radio(" not in app_text
    assert "last_week_pct = gr.Radio(" not in app_text


def test_hf_space_app_forces_readable_text_in_wizard_cards():
    app_text = read_app_text()

    assert ".wizard-card-group label {" in app_text
    assert "color: #183223 !important;" in app_text
    assert ".wizard-card-group label span {" in app_text


def test_hf_space_app_keeps_results_simple():
    app_text = read_app_text()

    assert "Perkiraan keyakinan model" not in app_text
    assert "Ringkasan kemungkinan kategori" not in app_text
    assert "Ringkasan input Anda" in app_text


def test_hf_space_app_forces_readable_placeholder_text_in_result_cards():
    app_text = read_app_text()

    assert ".review-card {" in app_text
    assert "color: #33463a;" in app_text


def test_render_result_card_includes_guidance_block_and_serious_badge_hook():
    namespace = load_app_symbols("drought_info", "RESULT_TONE_BY_CODE", "render_result_card")
    render_result_card = namespace["render_result_card"]

    html = render_result_card("D2", "<div>Ringkasan</div>")

    assert 'class="result-badge serious badge-serius"' in html
    assert 'class="guidance-grid"' in html
    assert html.count('class="guidance-card"') == 2
    assert "Cara membaca hasil ini" in html
    assert "Hal yang perlu diperhatikan setelah ini" in html
    assert "<div>Ringkasan</div>" in html


def test_render_result_card_uses_expected_badge_hooks_for_all_tones():
    namespace = load_app_symbols("drought_info", "RESULT_TONE_BY_CODE", "render_result_card")
    render_result_card = namespace["render_result_card"]

    assert 'class="result-badge safe badge-aman"' in render_result_card("None", "")
    assert 'class="result-badge watchful badge-waspada"' in render_result_card("D0", "")
    assert 'class="result-badge serious badge-serius"' in render_result_card("D4", "")


def test_hf_space_app_no_longer_contains_shap_pipeline():
    app_text = read_app_text()

    assert "shap_explainer" not in app_text
    assert "shap.summary_plot" not in app_text
    assert "output_plot" not in app_text
    assert "Penjelasan faktor model" not in app_text


def test_get_launch_kwargs_prefers_localhost_outside_hf_spaces():
    namespace = load_app_symbols("get_launch_kwargs")
    get_launch_kwargs = namespace["get_launch_kwargs"]

    original_space_id = os.environ.pop("SPACE_ID", None)
    original_hf_space_id = os.environ.pop("HF_SPACE_ID", None)
    try:
        launch_kwargs = get_launch_kwargs()
    finally:
        if original_space_id is not None:
            os.environ["SPACE_ID"] = original_space_id
        if original_hf_space_id is not None:
            os.environ["HF_SPACE_ID"] = original_hf_space_id

    assert launch_kwargs["server_name"] == "127.0.0.1"


def test_get_launch_kwargs_uses_all_interfaces_inside_hf_spaces():
    namespace = load_app_symbols("get_launch_kwargs")
    get_launch_kwargs = namespace["get_launch_kwargs"]

    original_space_id = os.environ.get("SPACE_ID")
    original_hf_space_id = os.environ.get("HF_SPACE_ID")
    os.environ["SPACE_ID"] = "demo-space"
    try:
        launch_kwargs = get_launch_kwargs()
    finally:
        if original_space_id is None:
            os.environ.pop("SPACE_ID", None)
        else:
            os.environ["SPACE_ID"] = original_space_id

        if original_hf_space_id is None:
            os.environ.pop("HF_SPACE_ID", None)
        else:
            os.environ["HF_SPACE_ID"] = original_hf_space_id

    assert launch_kwargs["server_name"] == "0.0.0.0"


def test_predict_button_now_targets_html_output_only():
    app_text = read_app_text()

    assert "outputs=output_html" in app_text
    assert "outputs=[output_html, output_plot]" not in app_text


def test_build_input_features_maps_drought_answers_to_model_features():
    features = build_input_features("D2", 40, "D0", 10)

    assert features == [60.0, 40.0, 40.0, 40.0, 40.0, 0.0, 90.0, 10.0, 10.0, 0.0, 0.0, 0.0]


def test_build_input_features_preserves_legacy_none_mapping():
    features = build_input_features("None", 85, "None", 5)

    assert features == [15.0, 85.0, 0.0, 0.0, 0.0, 0.0, 95.0, 5.0, 0.0, 0.0, 0.0, 0.0]


def test_build_review_note_flags_unusual_normal_selection():
    note = build_review_note("None", 85, "D0", 10)

    assert "Periksa kembali" in note


def test_describe_coverage_returns_plain_language_band():
    assert "setengah wilayah" in describe_coverage(50)
