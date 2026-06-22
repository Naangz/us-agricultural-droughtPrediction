import os
import sys
import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf
import pickle
import json
import shap
import matplotlib.pyplot as plt

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'best_model.keras')
SCALER_PATH = os.path.join(CURRENT_DIR, 'tokenizer.pkl')
CONFIG_PATH = os.path.join(CURRENT_DIR, 'config.json')

# Load config
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    print("Config loaded successfully.")
except Exception as e:
    print(f"Error loading config: {e}")
    config = {}

feature_cols = config.get("feature_cols", [
    'None_lag1', 'D0_lag1', 'D1_lag1', 'D2_lag1', 'D3_lag1', 'D4_lag1',
    'None_lag2', 'D0_lag2', 'D1_lag2', 'D2_lag2', 'D3_lag2', 'D4_lag2'])
label_map = config.get("label_map", {
    "0": "None", "1": "D0", "2": "D1", "3": "D2", "4": "D3", "5": "D4"
})
label_map = {int(k): v for k, v in label_map.items()}
class_multipliers = np.array(config.get("class_multipliers", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
SEQ_LENGTH = config.get("max_sequence_length", 52)

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the scaler
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Prepare background data for SHAP using KernelExplainer
shap_explainer = None
shap_background = None
try:
    shap_bg_path = os.path.join(CURRENT_DIR, 'shap_background.npy')
    if os.path.exists(shap_bg_path):
        shap_background = np.load(shap_bg_path)
        if len(shap_background.shape) == 3:
            shap_background = shap_background[:, -1, :] # Convert 3D sequence to 2D
        
        if model is not None and scaler is not None:
            # Prediction function wrapper for KernelExplainer
            def predict_wrapper(x_2d):
                batch_size = x_2d.shape[0]
                # Convert back to DataFrame to prevent MinMaxScaler warnings
                x_df = pd.DataFrame(x_2d, columns=feature_cols)
                x_scaled = scaler.transform(x_df)
                x_3d = np.tile(x_scaled.reshape(batch_size, 1, -1), (1, SEQ_LENGTH, 1))
                preds = model.predict(x_3d, verbose=0)
                return preds * class_multipliers
            
            # Use 15 samples from background to keep SHAP calculations very fast (2-3s)
            shap_explainer = shap.KernelExplainer(predict_wrapper, shap_background[:15])
            print("SHAP KernelExplainer initialized successfully.")
except Exception as e:
    print(f"Error preparing SHAP: {e}")

# Info for drought categories
drought_info = {
    'None': {
        'title': 'Normal (Bebas Kekeringan)',
        'color': '#2e7d32', # Dark Green
        'bg': '#e8f5e9',
        'desc': 'Kondisi kelembaban tanah cukup, tidak ada kekeringan. Pasokan air tanah, waduk, dan aktivitas pertanian berjalan normal.'
    },
    'D0': {
        'title': 'Abnormally Dry (Sangat Kering / Awal Kekeringan)',
        'color': '#fbc02d', # Yellow
        'bg': '#fffde7',
        'desc': 'Tanah mulai kering dan berdebu. Pertumbuhan tanaman mulai sedikit terhambat. Disarankan untuk mulai memantau pengairan.'
    },
    'D1': {
        'title': 'Moderate Drought (Kekeringan Sedang)',
        'color': '#f57c00', # Orange
        'bg': '#fff3e0',
        'desc': 'Debit air sungai mulai menurun. Terjadi sedikit kerusakan pada hasil pertanian. Himbauan penghematan air mulai dilakukan.'
    },
    'D2': {
        'title': 'Severe Drought (Kekeringan Parah)',
        'color': '#d32f2f', # Red
        'bg': '#ffebee',
        'desc': 'Kekurangan air bersih mulai meluas. Kerusakan tanaman pertanian cukup signifikan. Pembatasan penggunaan air resmi dimulai.'
    },
    'D3': {
        'title': 'Extreme Drought (Kekeringan Ekstrim)',
        'color': '#c2185b', # Pink/Red
        'bg': '#fce4ec',
        'desc': 'Terjadi kegagalan panen skala besar. Pasokan air waduk dan sungai menyusut tajam. Pembatasan air secara ketat diterapkan.'
    },
    'D4': {
        'title': 'Exceptional Drought (Kekeringan Luar Biasa)',
        'color': '#6a1b9a', # Purple
        'bg': '#f3e5f5',
        'desc': 'Bencana kekeringan nasional. Terjadi kelangkaan air minum darurat di semua wilayah. Kerusakan ekosistem parah.'
    }
}

friendly_feature_names = [
    'Bebas Kekeringan (Minggu Lalu)',
    'Kekeringan Ringan D0 (Minggu Lalu)',
    'Kekeringan Sedang D1 (Minggu Lalu)',
    'Kekeringan Parah D2 (Minggu Lalu)',
    'Kekeringan Ekstrim D3 (Minggu Lalu)',
    'Kekeringan Luar Biasa D4 (Minggu Lalu)',
    'Bebas Kekeringan (2 Minggu Lalu)',
    'Kekeringan Ringan D0 (2 Minggu Lalu)',
    'Kekeringan Sedang D1 (2 Minggu Lalu)',
    'Kekeringan Parah D2 (2 Minggu Lalu)',
    'Kekeringan Ekstrim D3 (2 Minggu Lalu)',
    'Kekeringan Luar Biasa D4 (2 Minggu Lalu)',
]

def predict(mode, newbie_lvl_1, newbie_pct_1, newbie_lvl_2, newbie_pct_2,
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12):
    if model is None:
        return "<h3>Error: Model tidak termuat.</h3>", None
    if scaler is None:
        return "<h3>Error: Scaler tidak termuat.</h3>", None
        
    try:
        if mode == "Mode Sederhana":
            # Map levels to values
            lvl_map = {
                "Normal / Bebas Kekeringan (None)": 0,
                "Sangat Kering (D0)": 1,
                "Kekeringan Sedang (D1)": 2,
                "Kekeringan Parah (D2)": 3,
                "Kekeringan Ekstrim (D3)": 4,
                "Kekeringan Luar Biasa (D4)": 5
            }
            k1 = lvl_map[newbie_lvl_1]
            p1 = float(newbie_pct_1)
            
            k2 = lvl_map[newbie_lvl_2]
            p2 = float(newbie_pct_2)
            
            # Map to cumulative values (0 - 100)
            None_lag1 = 100.0 - p1
            D0_lag1 = p1
            D1_lag1 = p1 if k1 >= 1 else 0.0
            D2_lag1 = p1 if k1 >= 2 else 0.0
            D3_lag1 = p1 if k1 >= 3 else 0.0
            D4_lag1 = p1 if k1 >= 4 else 0.0
            
            None_lag2 = 100.0 - p2
            D0_lag2 = p2
            D1_lag2 = p2 if k2 >= 1 else 0.0
            D2_lag2 = p2 if k2 >= 2 else 0.0
            D3_lag2 = p2 if k2 >= 3 else 0.0
            D4_lag2 = p2 if k2 >= 4 else 0.0

            input_features = [
                None_lag1, D0_lag1, D1_lag1, D2_lag1, D3_lag1, D4_lag1,
                None_lag2, D0_lag2, D1_lag2, D2_lag2, D3_lag2, D4_lag2
            ]
        else:
            input_features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]

        # Convert to pandas DataFrame with column names to prevent scaler warnings
        input_df = pd.DataFrame([input_features], columns=feature_cols)
        
        # Scale the inputs
        scaled_input = scaler.transform(input_df)
        
        # Tile across 52 timesteps
        sequence_input = np.tile(scaled_input, (1, SEQ_LENGTH, 1))
        
        # Predict
        y_pred_prob = model.predict(sequence_input, verbose=0)
        
        # Apply class multipliers
        tuned_probs = y_pred_prob[0] * class_multipliers
        pred_idx = np.argmax(tuned_probs)
        predicted_class = label_map[pred_idx]
        
        # Form HTML response
        info = drought_info[predicted_class]
        prediction_html = f"""
        <div style="background-color: {info['bg']}; border-left: 8px solid {info['color']}; padding: 20px; border-radius: 8px; margin-top: 10px;">
            <h3 style="color: {info['color']}; margin-top: 0; font-size: 22px; font-weight: bold;">🎯 Prediksi Minggu Depan: {info['title']}</h3>
            <p style="font-size: 16px; color: #333; line-height: 1.5; margin-bottom: 0;">{info['desc']}</p>
        </div>
        """
        
        # Generate SHAP explanations using KernelExplainer outputs
        fig = None
        if shap_explainer is not None:
            try:
                # Calculate SHAP values for the 2D input (shape: (1, 14))
                shap_values = shap_explainer.shap_values(scaled_input, nsamples=80)
                
                # Align shap values to shape (1, 14) robustly
                if isinstance(shap_values, list):
                    class_shap_values = shap_values[pred_idx]
                else:
                    if len(shap_values.shape) == 3:
                        class_shap_values = shap_values[pred_idx]
                    else:
                        class_shap_values = shap_values
                
                # Make sure both class_shap_values and scaled_input are exactly 2D shapes: (1, 14)
                if len(class_shap_values.shape) == 1:
                    class_shap_values = class_shap_values.reshape(1, -1)
                elif len(class_shap_values.shape) == 3:
                    class_shap_values = class_shap_values[0]
                    
                if len(scaled_input.shape) == 1:
                    scaled_input = scaled_input.reshape(1, -1)
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    class_shap_values,
                    scaled_input,
                    feature_names=friendly_feature_names,
                    plot_type="bar",
                    show=False
                )
                fig = plt.gcf()
                plt.tight_layout()
            except Exception as e:
                print(f"SHAP Error: {e}")
                
        return prediction_html, fig

    except Exception as e:
        return f"<div style='color: red;'>Error during prediction: {e}</div>", None

# Build interface
theme = gr.themes.Soft(
    primary_hue="green", 
    secondary_hue="emerald"
)

# Custom CSS for modern typography and glassmorphism styling
custom_css = """
body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}
.title-desc {
    margin-bottom: 20px;
}
"""

with gr.Blocks(theme=theme, css=custom_css, title="Sistem Prediksi Kekeringan Mingguan County (Kansas)") as demo:
    gr.Markdown(
        """
        # 🌾 Sistem Prediksi Kekeringan Mingguan County (Kansas Scenario 7)
        *Mendeteksi dan memproyeksikan tingkat kekeringan daerah untuk 1 minggu ke depan di tingkat county (by state) menggunakan model kecerdasan buatan (BiLSTM).*
        
        Pilih **Mode Sederhana** untuk memasukkan kondisi kekeringan dengan mudah menggunakan istilah umum, atau **Mode Ahli** untuk memasukkan persentase area USDM secara manual.
        """,
        elem_classes=["title-desc"]
    )
    
    # Toggle Mode
    mode_toggle = gr.Radio(
        choices=["Mode Sederhana", "Mode Ahli (Advanced)"],
        value="Mode Sederhana",
        label="Pilih Mode Input UI",
        interactive=True
    )
    
    # Simple Inputs Box
    with gr.Group() as newbie_group:
        gr.Markdown(
            """
            ### 👤 Input Mode Sederhana
            **Panduan Pengisian:**
            1. **Tingkat Kekeringan Utama**: Pilih kategori keparahan kekeringan yang paling menggambarkan kondisi county Anda saat ini.
               - *Normal/None*: Kondisi air tanah cukup, aman.
               - *Sangat Kering (D0)*: Tanah mulai berdebu, pertumbuhan tanaman sedikit terhambat.
               - *Sedang (D1)* s.d. *Luar Biasa (D4)*: Kekeringan semakin parah, dari mulai rusaknya hasil panen hingga krisis air bersih nasional.
            2. **Persentase Area Terdampak (%)**: Tentukan seberapa luas cakupan wilayah county yang mengalami tingkat kekeringan tersebut (atau tingkat yang lebih buruk).
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📅 Kondisi Minggu Lalu (1 Week Ago)")
                newbie_lvl_1 = gr.Dropdown(
                    choices=[
                        "Normal / Bebas Kekeringan (None)",
                        "Sangat Kering (D0)",
                        "Kekeringan Sedang (D1)",
                        "Kekeringan Parah (D2)",
                        "Kekeringan Ekstrim (D3)",
                        "Kekeringan Luar Biasa (D4)"
                    ],
                    value="Normal / Bebas Kekeringan (None)",
                    label="Tingkat Kekeringan Utama"
                )
                newbie_pct_1 = gr.Slider(
                    minimum=0, maximum=100, step=5, value=0,
                    label="Persentase Area Terdampak (%)"
                )
                
            with gr.Column():
                gr.Markdown("#### 📅 Kondisi Dua Minggu Lalu (2 Weeks Ago)")
                newbie_lvl_2 = gr.Dropdown(
                    choices=[
                        "Normal / Bebas Kekeringan (None)",
                        "Sangat Kering (D0)",
                        "Kekeringan Sedang (D1)",
                        "Kekeringan Parah (D2)",
                        "Kekeringan Ekstrim (D3)",
                        "Kekeringan Luar Biasa (D4)"
                    ],
                    value="Normal / Bebas Kekeringan (None)",
                    label="Tingkat Kekeringan Utama"
                )
                newbie_pct_2 = gr.Slider(
                    minimum=0, maximum=100, step=5, value=0,
                    label="Persentase Area Terdampak (%)"
                )
                
    # Advanced Inputs Box
    with gr.Group(visible=False) as advanced_group:
        gr.Markdown(
            """
            ### ⚙️ Input Mode Ahli (Raw Features)
            **Panduan Pengisian:**
            Masukkan persentase cakupan area (0-100%) untuk masing-masing tingkat kekeringan kumulatif USDM (U.S. Drought Monitor) 
            *Catatan: Nilai kumulatif harus menurun atau sama seiring meningkatnya tingkat keparahan (misal: None + D0 = 100%, D0 >= D1 >= D2 >= D3 >= D4).*
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📅 Lags 1 (Minggu Lalu)")
                f1 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=100.0, 
                               label="None_lag1", 
                               info="Persentase area county yang bebas kekeringan minggu lalu")
                f2 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=0.0, 
                               label="D0_lag1", 
                               info="Persentase area berkategori Sangat Kering (D0) atau lebih parah minggu lalu")
                f3 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=0.0, 
                               label="D1_lag1", 
                               info="Persentase area berkategori Kekeringan Sedang (D1) atau lebih parah minggu lalu")
                f4 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=0.0, 
                               label="D2_lag1", 
                               info="Persentase area berkategori Kekeringan Parah (D2) atau lebih parah minggu lalu")
                f5 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=0.0, 
                               label="D3_lag1", 
                               info="Persentase area berkategori Kekeringan Ekstrim (D3) atau lebih parah minggu lalu")
                f6 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=0.0, 
                               label="D4_lag1", 
                               info="Persentase area berkategori Kekeringan Luar Biasa (D4) minggu lalu")

            with gr.Column():
                gr.Markdown("#### 📅 Lags 2 (Dua Minggu Lalu)")
                f7 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=100.0, 
                               label="None_lag2", 
                               info="Persentase area county yang bebas kekeringan dua minggu lalu")
                f8 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=0.0, 
                               label="D0_lag2", 
                               info="Persentase area berkategori Sangat Kering (D0) atau lebih parah dua minggu lalu")
                f9 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=0.0, 
                               label="D1_lag2", 
                               info="Persentase area berkategori Kekeringan Sedang (D1) atau lebih parah dua minggu lalu")
                f10 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=0.0, 
                                label="D2_lag2", 
                                info="Persentase area berkategori Kekeringan Parah (D2) atau lebih parah dua minggu lalu")
                f11 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=0.0, 
                                label="D3_lag2", 
                                info="Persentase area berkategori Kekeringan Ekstrim (D3) atau lebih parah dua minggu lalu")
                f12 = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, value=0.0, 
                                label="D4_lag2", 
                                info="Persentase area berkategori Kekeringan Luar Biasa (D4) dua minggu lalu")

    # Visibility Toggle Logic
    def toggle_ui_mode(choice):
        if choice == "Mode Sederhana":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    mode_toggle.change(
        fn=toggle_ui_mode,
        inputs=mode_toggle,
        outputs=[newbie_group, advanced_group]
    )
            
    predict_btn = gr.Button("🔮 Hitung Prediksi Kekeringan", variant="primary", scale=1)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Hasil Analisis Model")
            output_html = gr.HTML(value="<div style='color: gray; padding: 10px;'>Klik tombol di atas untuk melihat prediksi kekeringan.</div>")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Faktor Kontribusi (Penjelasan SHAP)")
            output_plot = gr.Plot(label="Faktor Utama yang Memengaruhi Model")

    # Connect components to prediction function
    predict_btn.click(
        fn=predict,
        inputs=[
            mode_toggle, newbie_lvl_1, newbie_pct_1, newbie_lvl_2, newbie_pct_2,
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12
        ],
        outputs=[output_html, output_plot]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
