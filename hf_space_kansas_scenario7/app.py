import json
import os
import pickle

import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from ui_helpers import build_input_features, build_review_note, describe_coverage
except ImportError:
    from hf_space_kansas_scenario7.ui_helpers import (
        build_input_features,
        build_review_note,
        describe_coverage,
    )


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "best_model.keras")
SCALER_PATH = os.path.join(CURRENT_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")


try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    print("Config loaded successfully.")
except Exception as exc:
    print(f"Error loading config: {exc}")
    config = {}

feature_cols = config.get(
    "feature_cols",
    [
        "None_lag1",
        "D0_lag1",
        "D1_lag1",
        "D2_lag1",
        "D3_lag1",
        "D4_lag1",
        "None_lag2",
        "D0_lag2",
        "D1_lag2",
        "D2_lag2",
        "D3_lag2",
        "D4_lag2",
    ],
)
label_map = config.get(
    "label_map",
    {"0": "None", "1": "D0", "2": "D1", "3": "D2", "4": "D3", "5": "D4"},
)
label_map = {int(key): value for key, value in label_map.items()}
class_multipliers = np.array(config.get("class_multipliers", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
SEQ_LENGTH = config.get("max_sequence_length", 52)


try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as exc:
    print(f"Error loading model: {exc}")
    model = None

try:
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
    print("Scaler loaded successfully.")
except Exception as exc:
    print(f"Error loading scaler: {exc}")
    scaler = None


drought_info = {
    "None": {
        "title": "Normal",
        "headline": "Kondisi minggu depan diperkirakan tetap aman.",
        "color": "#1f6f43",
        "bg": "#ebf7ef",
        "desc": "Kelembaban tanah dan ketersediaan air masih cukup. Aktivitas harian dan pertanian biasanya tetap berjalan normal.",
    },
    "D0": {
        "title": "Mulai Kering (D0)",
        "headline": "Ada tanda awal kekeringan yang perlu dipantau.",
        "color": "#9a6b00",
        "bg": "#fff7dc",
        "desc": "Sebagian tanah mulai mengering. Pengairan dan pemantauan kondisi lahan sebaiknya mulai diperketat.",
    },
    "D1": {
        "title": "Kekeringan Sedang (D1)",
        "headline": "Beberapa dampak kekeringan mulai terasa.",
        "color": "#c96b00",
        "bg": "#fff1e5",
        "desc": "Debit air dan kondisi tanaman bisa mulai terpengaruh. Penghematan air dan pengawasan lapangan perlu ditingkatkan.",
    },
    "D2": {
        "title": "Kekeringan Parah (D2)",
        "headline": "Risiko gangguan air dan pertanian cukup tinggi.",
        "color": "#c0392b",
        "bg": "#fdeeee",
        "desc": "Kekurangan air mulai meluas dan kerusakan pertanian bisa menjadi signifikan. Rencana mitigasi perlu disiapkan.",
    },
    "D3": {
        "title": "Kekeringan Ekstrem (D3)",
        "headline": "Dampak kekeringan diperkirakan sangat berat.",
        "color": "#9c255f",
        "bg": "#fcecf4",
        "desc": "Pasokan air dapat turun tajam dan risiko gagal panen meningkat. Langkah penanganan darurat mungkin dibutuhkan.",
    },
    "D4": {
        "title": "Kekeringan Luar Biasa (D4)",
        "headline": "Situasi sangat serius dan butuh kewaspadaan penuh.",
        "color": "#5d2c91",
        "bg": "#f2ecfb",
        "desc": "Kelangkaan air dan dampak lingkungan berpotensi sangat berat. Koordinasi tanggap darurat menjadi sangat penting.",
    },
}

friendly_feature_names = [
    "Bebas kekeringan minggu lalu",
    "D0 atau lebih parah minggu lalu",
    "D1 atau lebih parah minggu lalu",
    "D2 atau lebih parah minggu lalu",
    "D3 atau lebih parah minggu lalu",
    "D4 minggu lalu",
    "Bebas kekeringan dua minggu lalu",
    "D0 atau lebih parah dua minggu lalu",
    "D1 atau lebih parah dua minggu lalu",
    "D2 atau lebih parah dua minggu lalu",
    "D3 atau lebih parah dua minggu lalu",
    "D4 dua minggu lalu",
]

LEVEL_CHOICES = [
    "Normal atau aman",
    "Mulai kering (D0)",
    "Kekeringan sedang (D1)",
    "Kekeringan parah (D2)",
    "Kekeringan ekstrem (D3)",
    "Kekeringan luar biasa (D4)",
]
LEVEL_VALUE_BY_LABEL = {
    "Normal atau aman": "None",
    "Mulai kering (D0)": "D0",
    "Kekeringan sedang (D1)": "D1",
    "Kekeringan parah (D2)": "D2",
    "Kekeringan ekstrem (D3)": "D3",
    "Kekeringan luar biasa (D4)": "D4",
}
LEVEL_DESCRIPTION_BY_CODE = {
    "None": "Air masih cukup dan kondisi wilayah terlihat stabil.",
    "D0": "Ada gejala awal wilayah mulai mengering.",
    "D1": "Kekeringan mulai terasa dan perlu dipantau ketat.",
    "D2": "Risiko terhadap air dan pertanian cukup jelas.",
    "D3": "Dampaknya berat dan perlu respons cepat.",
    "D4": "Situasi sangat serius dengan dampak paling berat.",
}

COVERAGE_CHOICES = [
    "0% - Belum terlihat",
    "10% - Hanya sedikit wilayah",
    "25% - Sebagian kecil wilayah",
    "50% - Sekitar setengah wilayah",
    "75% - Sebagian besar wilayah",
    "100% - Hampir seluruh wilayah",
]
COVERAGE_VALUE_BY_LABEL = {
    "0% - Belum terlihat": 0,
    "10% - Hanya sedikit wilayah": 10,
    "25% - Sebagian kecil wilayah": 25,
    "50% - Sekitar setengah wilayah": 50,
    "75% - Sebagian besar wilayah": 75,
    "100% - Hampir seluruh wilayah": 100,
}

STEP_LABELS = {
    1: "Langkah 1 dari 3",
    2: "Langkah 2 dari 3",
    3: "Langkah 3 dari 3",
}

RISK_GUIDANCE_ITEMS = [
    ("Aman", "Kondisi masih stabil dan biasanya cukup dipantau rutin."),
    ("Perlu waspada", "Ada perubahan awal yang layak diperiksa lebih dekat."),
    ("Risiko serius", "Dampak terhadap air dan pertanian bisa butuh respons cepat."),
]
BADGE_HOOK_BY_LABEL = {
    "Aman": "badge-aman",
    "Perlu waspada": "badge-waspada",
    "Risiko serius": "badge-serius",
}
RESULT_TONE_BY_CODE = {
    "None": {
        "label": "Aman",
        "class_name": "safe",
        "badge_hook": "badge-aman",
        "interpretation": "Model membaca kondisi minggu depan cenderung tetap stabil, jadi fokus utamanya adalah menjaga pemantauan rutin agar perubahan baru tidak terlewat.",
        "follow_up": "Tetap cek laporan lapangan dan curah hujan setempat. Bila area terdampak mulai bertambah, ulangi penilaian agar langkah antisipasi bisa disiapkan lebih awal.",
    },
    "D0": {
        "label": "Perlu waspada",
        "class_name": "watchful",
        "badge_hook": "badge-waspada",
        "interpretation": "Model melihat tanda awal wilayah mulai mengering. Ini belum selalu berarti gangguan berat, tetapi cukup penting untuk dipantau lebih rapat daripada kondisi normal.",
        "follow_up": "Perhatikan perubahan pada tanah, kebutuhan irigasi, dan area yang mulai lebih cepat kering. Catatan lapangan mingguan akan membantu memastikan apakah kondisi membaik atau memburuk.",
    },
    "D1": {
        "label": "Perlu waspada",
        "class_name": "watchful",
        "badge_hook": "badge-waspada",
        "interpretation": "Model menilai kekeringan sudah mulai terasa sehingga pengamatan lapangan dan pengelolaan air perlu lebih disiplin dibanding minggu biasa.",
        "follow_up": "Pantau sumber air, kebutuhan tanaman, dan wilayah yang paling cepat menunjukkan stres kekeringan. Jika cakupan terdampak bertambah, siapkan langkah penghematan air lebih dini.",
    },
    "D2": {
        "label": "Risiko serius",
        "class_name": "serious",
        "badge_hook": "badge-serius",
        "interpretation": "Model menangkap sinyal bahwa gangguan terhadap air dan pertanian bisa cukup nyata minggu depan. Hasil ini layak dibaca sebagai tanda perlunya perhatian operasional yang lebih kuat.",
        "follow_up": "Prioritaskan area yang paling rentan, cek kesiapan irigasi, dan komunikasikan risiko ke pihak lapangan yang relevan. Bila data terbaru menunjukkan cakupan meluas, pertimbangkan langkah mitigasi tambahan.",
    },
    "D3": {
        "label": "Risiko serius",
        "class_name": "serious",
        "badge_hook": "badge-serius",
        "interpretation": "Model memperkirakan dampak kekeringan yang berat sehingga keputusan pemantauan, alokasi air, dan koordinasi lapangan perlu bergerak lebih cepat.",
        "follow_up": "Tinjau rencana kontingensi, fokus pada wilayah dengan tekanan air tertinggi, dan koordinasikan respons dengan pihak yang menangani kebutuhan air dan pertanian setempat.",
    },
    "D4": {
        "label": "Risiko serius",
        "class_name": "serious",
        "badge_hook": "badge-serius",
        "interpretation": "Model membaca situasi yang sangat berat. Hasil ini sebaiknya dipahami sebagai sinyal untuk meningkatkan kewaspadaan penuh dan menyiapkan respons lintas pihak.",
        "follow_up": "Pastikan informasi lapangan terbaru terkumpul, komunikasikan dampak potensial secara cepat, dan arahkan perhatian pada kebutuhan air kritis serta perlindungan area yang paling terdampak.",
    },
}


def normalize_level(choice):
    return LEVEL_VALUE_BY_LABEL.get(choice, "None")


def normalize_coverage(choice):
    if isinstance(choice, (int, float)):
        return float(choice)
    return float(COVERAGE_VALUE_BY_LABEL.get(choice, 0))


def render_progress(step_number):
    progress_items = [
        (1, "Dua minggu lalu"),
        (2, "Minggu lalu"),
        (3, "Tinjau dan prediksi"),
    ]
    cards = []
    for number, label in progress_items:
        state = "current" if step_number == number else "done" if step_number > number else "upcoming"
        cards.append(
            f"""
            <div class="progress-card {state}">
                <div class="progress-step">{STEP_LABELS[number]}</div>
                <div class="progress-title">{label}</div>
            </div>
            """
        )
    return f'<div class="progress-grid">{"".join(cards)}</div>'


def render_risk_guidance():
    cards = []
    for title, copy in RISK_GUIDANCE_ITEMS:
        tone_class = "safe" if title == "Aman" else "watchful" if title == "Perlu waspada" else "serious"
        badge_hook = BADGE_HOOK_BY_LABEL[title]
        cards.append(
            f"""
            <div class="risk-card {tone_class}">
                <div class="risk-badge {badge_hook}">{title}</div>
                <div class="risk-copy">{copy}</div>
            </div>
            """
        )
    return f'<div class="risk-grid">{"".join(cards)}</div>'


def render_review_summary(two_weeks_level_choice, two_weeks_pct_choice, last_week_level_choice, last_week_pct_choice):
    two_weeks_level = normalize_level(two_weeks_level_choice)
    two_weeks_pct = normalize_coverage(two_weeks_pct_choice)
    last_week_level = normalize_level(last_week_level_choice)
    last_week_pct = normalize_coverage(last_week_pct_choice)
    note = build_review_note(last_week_level, last_week_pct, two_weeks_level, two_weeks_pct)

    def build_period_card(period_title, level_code, pct_value):
        coverage_text = describe_coverage(pct_value)
        detail = LEVEL_DESCRIPTION_BY_CODE[level_code]
        title = drought_info[level_code]["title"]
        return f"""
        <div class="review-card">
            <div class="review-period">{period_title}</div>
            <div class="review-level">{title}</div>
            <div class="review-copy">{coverage_text.capitalize()} terdampak, sekitar {int(pct_value)}% wilayah.</div>
            <div class="review-detail">{detail}</div>
        </div>
        """

    note_html = ""
    if note:
        note_html = f'<div class="review-note">{note}</div>'

    return f"""
    <div class="review-shell">
        <p class="review-intro">Cek lagi jawaban Anda sebelum model menghitung prediksi minggu depan.</p>
        <div class="review-grid">
            {build_period_card("Dua minggu lalu", two_weeks_level, two_weeks_pct)}
            {build_period_card("Minggu lalu", last_week_level, last_week_pct)}
        </div>
        {note_html}
    </div>
    """


def render_input_recap(two_weeks_level_code, two_weeks_pct, last_week_level_code, last_week_pct):
    return f"""
    <div class="input-recap">
        <div class="recap-title">Ringkasan input Anda</div>
        <div class="recap-copy">Dua minggu lalu: {drought_info[two_weeks_level_code]['title']} di sekitar {int(two_weeks_pct)}% wilayah.</div>
        <div class="recap-copy">Minggu lalu: {drought_info[last_week_level_code]['title']} di sekitar {int(last_week_pct)}% wilayah.</div>
    </div>
    """


def render_result_card(predicted_class, recap_html):
    info = drought_info[predicted_class]
    tone = RESULT_TONE_BY_CODE[predicted_class]
    return f"""
    <div class="result-card" style="background: {info['bg']}; border-color: {info['color']};">
        <div class="result-header">
            <div class="result-kicker">Prediksi minggu depan</div>
            <div class="result-badge {tone['class_name']} {tone['badge_hook']}">{tone['label']}</div>
        </div>
        <h2 style="color: {info['color']};">{info['title']}</h2>
        <p class="result-headline">{info['headline']}</p>
        <p class="result-copy">{info['desc']}</p>
        <div class="guidance-grid">
            <div class="guidance-card">
                <div class="guidance-title">Cara membaca hasil ini</div>
                <p>{tone['interpretation']}</p>
            </div>
            <div class="guidance-card">
                <div class="guidance-title">Hal yang perlu diperhatikan setelah ini</div>
                <p>{tone['follow_up']}</p>
            </div>
        </div>
        {recap_html}
    </div>
    """


def get_launch_kwargs():
    is_hf_space = bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))
    return {
        "server_name": "0.0.0.0" if is_hf_space else "127.0.0.1",
        "server_port": int(os.environ.get("PORT", 7860)),
        "share": False,
    }


def set_step(step_number):
    return (
        gr.update(visible=step_number == 1),
        gr.update(visible=step_number == 2),
        gr.update(visible=step_number == 3),
        gr.update(value=render_progress(step_number)),
    )


def show_review(two_weeks_level_choice, two_weeks_pct_choice, last_week_level_choice, last_week_pct_choice):
    return (
        *set_step(3),
        gr.update(
            value=render_review_summary(
                two_weeks_level_choice,
                two_weeks_pct_choice,
                last_week_level_choice,
                last_week_pct_choice,
            )
        ),
    )


def predict_from_wizard(two_weeks_level_choice, two_weeks_pct_choice, last_week_level_choice, last_week_pct_choice):
    if model is None:
        return "<div class='error-card'>Model tidak berhasil dimuat.</div>"
    if scaler is None:
        return "<div class='error-card'>Scaler tidak berhasil dimuat.</div>"

    try:
        two_weeks_level = normalize_level(two_weeks_level_choice)
        two_weeks_pct = normalize_coverage(two_weeks_pct_choice)
        last_week_level = normalize_level(last_week_level_choice)
        last_week_pct = normalize_coverage(last_week_pct_choice)

        input_features = build_input_features(last_week_level, last_week_pct, two_weeks_level, two_weeks_pct)
        raw_input_2d = np.array([input_features], dtype=float)
        input_df = pd.DataFrame(raw_input_2d, columns=feature_cols)
        scaled_input = scaler.transform(input_df)
        sequence_input = np.tile(scaled_input, (1, SEQ_LENGTH, 1))

        y_pred_prob = model.predict(sequence_input, verbose=0)
        tuned_probs = y_pred_prob[0] * class_multipliers
        pred_idx = int(np.argmax(tuned_probs))
        predicted_class = label_map[pred_idx]
        return render_result_card(
            predicted_class,
            render_input_recap(two_weeks_level, two_weeks_pct, last_week_level, last_week_pct),
        )
    except Exception as exc:
        return f"<div class='error-card'>Terjadi kesalahan saat menghitung prediksi: {exc}</div>"


theme = gr.themes.Soft(primary_hue="green", secondary_hue="emerald")

custom_css = """
body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(180deg, #f6fbf7 0%, #eef5ff 100%);
}
.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
}
.hero-shell {
    padding: 28px;
    border-radius: 24px;
    background: linear-gradient(135deg, #0f5132 0%, #2d7a58 45%, #dcefe4 100%);
    color: #ffffff;
    margin-bottom: 20px;
}
.hero-shell h1 {
    margin: 0 0 12px 0;
    font-size: 2rem;
}
.hero-shell p {
    margin: 0;
    max-width: 760px;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.92);
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 16px 0 18px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.18);
    font-size: 0.92rem;
    font-weight: 600;
}
.hero-support-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 12px;
    margin-top: 18px;
}
.hero-support-card {
    border-radius: 20px;
    padding: 16px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.22);
    backdrop-filter: blur(6px);
}
.hero-support-title {
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    margin-bottom: 8px;
    color: rgba(255, 255, 255, 0.8);
}
.hero-support-copy {
    line-height: 1.6;
    color: #ffffff;
}
.info-panel {
    margin: 0 0 18px 0;
    padding: 16px 18px;
    border-radius: 18px;
    background: #ffffff;
    border: 1px solid #dce8df;
}
.info-panel, .info-panel * {
    color: #415748 !important;
}
.info-panel-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #183223;
    margin-bottom: 6px;
}
.info-panel p {
    margin: 0;
    color: #415748;
    line-height: 1.55;
}
.progress-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 10px 0 24px;
}
.progress-card {
    border-radius: 18px;
    padding: 16px;
    border: 1px solid #dce8df;
    background: #ffffff;
}
.progress-card, .progress-card * {
    color: #183223 !important;
}
.progress-card.current {
    border-color: #1f6f43;
    box-shadow: 0 0 0 2px rgba(31, 111, 67, 0.12);
}
.progress-card.done {
    background: #edf7ef;
}
.progress-step {
    font-size: 0.82rem;
    font-weight: 700;
    color: #53725d;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.progress-title {
    font-size: 1rem;
    font-weight: 700;
    color: #183223;
    margin-top: 6px;
}
.step-shell {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #dde8df;
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 16px;
}
.step-shell .gr-markdown, .step-shell .gr-markdown * {
    color: #415748 !important;
}
.step-shell .prose, .step-shell .prose * {
    color: #415748 !important;
}
.step-shell h3 {
    margin-top: 0;
    margin-bottom: 8px;
    color: #183223;
}
.step-shell p {
    color: #415748;
    line-height: 1.6;
}
.step-shell strong {
    color: #183223 !important;
    font-weight: 700;
}
.step-kicker {
    display: inline-flex;
    align-items: center;
    padding: 7px 12px;
    border-radius: 999px;
    background: #edf7ef;
    color: #1f6f43 !important;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.step-support {
    margin: 16px 0 18px;
}
.support-lead {
    color: #33463a;
    margin-bottom: 12px;
    line-height: 1.6;
}
.risk-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
}
.risk-card {
    border-radius: 18px;
    padding: 16px;
    border: 1px solid #dce7de;
    background: #ffffff;
}
.risk-card, .risk-card * {
    color: #183223 !important;
}
.risk-card.safe {
    background: #edf7ef;
    border-color: #bed8c5;
}
.risk-card.watchful {
    background: #fff7dc;
    border-color: #f0d26b;
}
.risk-card.serious {
    background: #fdeeee;
    border-color: #efc1bc;
}
.risk-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(24, 50, 35, 0.08);
    color: #183223;
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.risk-copy {
    color: #415748;
    line-height: 1.55;
}
.wizard-card-group .wrap {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
}
.wizard-card-group label {
    border: 1px solid #d7e5d9 !important;
    border-radius: 18px !important;
    background: #ffffff !important;
    color: #183223 !important;
    padding: 14px !important;
    min-height: 88px;
    align-items: flex-start !important;
}
.wizard-card-group label span {
    color: #183223 !important;
}
.wizard-card-group label:has(input:checked) {
    border-color: #1f6f43 !important;
    background: #edf7ef !important;
    box-shadow: 0 0 0 2px rgba(31, 111, 67, 0.14);
}
.wizard-card-group span {
    line-height: 1.45;
}
.review-shell {
    display: grid;
    gap: 16px;
}
.review-intro {
    margin: 0;
    color: #415748;
}
.review-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
}
.review-card {
    border-radius: 20px;
    border: 1px solid #dce7de;
    background: #ffffff;
    color: #33463a;
    padding: 18px;
}
.review-period {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #53725d;
    font-weight: 700;
}
.review-level {
    font-size: 1.12rem;
    color: #183223;
    font-weight: 700;
    margin: 8px 0;
}
.review-copy,
.review-detail {
    color: #415748;
    line-height: 1.55;
}
.review-note {
    border-radius: 16px;
    padding: 14px 16px;
    background: #fff8dc;
    border: 1px solid #f0d26b;
    color: #6f5600;
}
.result-card {
    border-left: 8px solid;
    border-radius: 24px;
    padding: 24px;
}
.result-card h2 {
    margin: 8px 0 10px;
}
.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.result-kicker {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    color: #53725d;
}
.result-badge {
    display: inline-flex;
    align-items: center;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
}
.result-badge.safe {
    background: #dbeee2;
    color: #1f6f43;
}
.result-badge.watchful {
    background: #fff0c9;
    color: #9a6b00;
}
.result-badge.serious {
    background: #f8d7d3;
    color: #a12b1e;
}
.result-headline {
    font-size: 1.08rem;
    font-weight: 700;
    color: #183223;
    margin: 0 0 10px;
}
.result-copy {
    line-height: 1.65;
    color: #33463a;
    margin-bottom: 16px;
}
.guidance-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
}
.guidance-card {
    border-radius: 18px;
    padding: 16px;
    background: rgba(255, 255, 255, 0.64);
    border: 1px solid rgba(24, 50, 35, 0.12);
}
.guidance-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #183223;
    margin-bottom: 8px;
}
.guidance-card p {
    margin: 0;
    color: #33463a;
    line-height: 1.6;
}
.input-recap {
    border-top: 1px solid rgba(24, 50, 35, 0.12);
    padding-top: 14px;
}
.recap-title {
    font-weight: 700;
    color: #183223;
    margin-bottom: 6px;
}
.recap-copy {
    color: #33463a;
    line-height: 1.55;
}
.secondary-copy {
    color: #4a6351;
}
.error-card {
    color: #8b1f1f;
    background: #fdecec;
    border: 1px solid #f1c4c4;
    border-radius: 16px;
    padding: 16px;
}
@media (max-width: 760px) {
    .progress-grid,
    .review-grid,
    .guidance-grid {
        grid-template-columns: 1fr;
    }
    .result-header {
        align-items: flex-start;
        flex-direction: column;
    }
}
"""


with gr.Blocks(theme=theme, css=custom_css, title="Prediksi Kekeringan Mingguan County Kansas") as demo:
    gr.HTML(
        """
        <section class="hero-shell">
            <h1>Prediksi kekeringan mingguan untuk county di Kansas</h1>
            <p>
                Wizard ini membantu Anda memasukkan kondisi dua minggu terakhir dengan bahasa yang sederhana.
                Ikuti tiga langkah singkat, lalu model akan memberi perkiraan kondisi minggu depan dan alasan umumnya.
            </p>
            <div class="hero-badges">
                <div class="hero-badge">3 langkah sederhana</div>
                <div class="hero-badge">Bahasa ramah pemula</div>
                <div class="hero-badge">Penjelasan hasil tetap singkat</div>
            </div>
            <div class="hero-support-grid">
                <div class="hero-support-card">
                    <div class="hero-support-title">Cocok untuk</div>
                    <div class="hero-support-copy">Petugas lapangan, pendamping pertanian, atau siapa pun yang ingin meninjau tren kekeringan mingguan tanpa istilah teknis yang berat.</div>
                </div>
                <div class="hero-support-card">
                    <div class="hero-support-title">Siapkan info ini</div>
                    <div class="hero-support-copy">Ingat kondisi dua minggu lalu dan minggu lalu: tingkat kekeringan yang paling mendekati, plus perkiraan luas wilayah terdampak.</div>
                </div>
                <div class="hero-support-card">
                    <div class="hero-support-title">Waktu isi</div>
                    <div class="hero-support-copy">Sekitar 1-2 menit bila Anda sudah punya gambaran umum kondisi lapangan.</div>
                </div>
            </div>
        </section>
        """
    )
    gr.HTML(
        """
        <section class="info-panel">
            <div class="info-panel-title">Yang perlu Anda tahu</div>
            <p>
                Anda hanya perlu memperkirakan kondisi kekeringan dari dua minggu terakhir.
                Pilih tingkat kondisi yang paling mendekati situasi lapangan, lalu geser persentase area terdampak secara perkiraan.
            </p>
        </section>
        """
    )

    progress_html = gr.HTML(render_progress(1))

    with gr.Group(visible=True) as step_one:
        with gr.Column(elem_classes=["step-shell"]):
            gr.Markdown(
                """
                ### Langkah 1 dari 3
                **Dua minggu lalu**, seperti apa kondisi wilayah yang paling mendekati situasi di county Anda?

                Normal berarti air masih cukup dan belum tampak gangguan besar.
                D0 sampai D1 menandakan tanah mulai mengering dan tanaman perlu dipantau.
                D2 sampai D4 berarti dampaknya makin berat, dari kekurangan air hingga potensi krisis.
                """
            )
            gr.HTML(
                f"""
                <div class="step-support">
                    <div class="step-kicker">Baca cepat sebelum memilih</div>
                    <div class="support-lead">Gunakan patokan sederhana ini untuk memilih tingkat kondisi yang paling mendekati pengamatan Anda.</div>
                    {render_risk_guidance()}
                </div>
                """
            )
            two_weeks_level = gr.Radio(
                choices=LEVEL_CHOICES,
                value="Normal atau aman",
                label="Pilih tingkat kondisi",
                elem_classes=["wizard-card-group"],
            )
            two_weeks_pct = gr.Slider(
                minimum=0,
                maximum=100,
                step=5,
                value=0,
                label="Perkiraan luas wilayah terdampak (%)",
            )
            gr.Markdown(
                "Panduan cepat persentase: 0% belum terlihat, 25% sebagian kecil, 50% sekitar setengah, 75% sebagian besar, 100% hampir seluruh wilayah."
            )
            next_step_one = gr.Button("Lanjut ke minggu lalu", variant="primary")

    with gr.Group(visible=False) as step_two:
        with gr.Column(elem_classes=["step-shell"]):
            gr.Markdown(
                """
                ### Langkah 2 dari 3
                Sekarang pilih kondisi **minggu lalu** agar model bisa membaca perubahan yang terjadi.

                Normal berarti air masih cukup dan aktivitas biasanya tetap lancar.
                D0 sampai D1 menandakan tanah mulai mengering dan pengairan perlu dipantau lebih dekat.
                D2 sampai D4 menunjukkan dampak yang lebih berat terhadap air, pertanian, dan kebutuhan harian.
                """
            )
            gr.HTML(
                f"""
                <div class="step-support">
                    <div class="step-kicker">Bandingkan dengan minggu sebelumnya</div>
                    <div class="support-lead">Perubahan kecil pun penting. Apakah kondisinya masih aman, mulai perlu waspada, atau sudah menunjukkan risiko serius?</div>
                    {render_risk_guidance()}
                </div>
                """
            )
            last_week_level = gr.Radio(
                choices=LEVEL_CHOICES,
                value="Normal atau aman",
                label="Pilih tingkat kondisi",
                elem_classes=["wizard-card-group"],
            )
            last_week_pct = gr.Slider(
                minimum=0,
                maximum=100,
                step=5,
                value=0,
                label="Perkiraan luas wilayah terdampak (%)",
            )
            gr.Markdown(
                "Panduan cepat persentase: 0% belum terlihat, 25% sebagian kecil, 50% sekitar setengah, 75% sebagian besar, 100% hampir seluruh wilayah."
            )
            with gr.Row():
                back_step_two = gr.Button("Kembali", variant="secondary")
                next_step_two = gr.Button("Tinjau jawaban", variant="primary")

    with gr.Group(visible=False) as step_three:
        with gr.Column(elem_classes=["step-shell"]):
            gr.Markdown(
                """
                ### Langkah 3 dari 3
                Tinjau ringkasan jawaban Anda, lalu tekan **Mulai Prediksi** untuk melihat perkiraan kondisi minggu depan.
                """
            )
            review_html = gr.HTML(
                render_review_summary(
                    "Normal atau aman",
                    0,
                    "Normal atau aman",
                    0,
                )
            )
            with gr.Row():
                back_step_three = gr.Button("Ubah jawaban", variant="secondary")
                predict_btn = gr.Button("Mulai Prediksi", variant="primary")

            gr.Markdown("### Hasil prediksi", elem_classes=["secondary-copy"])
            output_html = gr.HTML(
                value="""
                <div class="review-card">
                    Hasil akan muncul di sini setelah Anda menekan tombol prediksi.
                </div>
                """
            )

    next_step_one.click(
        fn=lambda: set_step(2),
        outputs=[step_one, step_two, step_three, progress_html],
    )
    back_step_two.click(
        fn=lambda: set_step(1),
        outputs=[step_one, step_two, step_three, progress_html],
    )
    next_step_two.click(
        fn=show_review,
        inputs=[two_weeks_level, two_weeks_pct, last_week_level, last_week_pct],
        outputs=[step_one, step_two, step_three, progress_html, review_html],
    )
    back_step_three.click(
        fn=lambda: set_step(2),
        outputs=[step_one, step_two, step_three, progress_html],
    )
    predict_btn.click(
        fn=predict_from_wizard,
        inputs=[two_weeks_level, two_weeks_pct, last_week_level, last_week_pct],
        outputs=output_html,
    )


if __name__ == "__main__":
    demo.launch(**get_launch_kwargs())
