import os
import sys
import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEBRASKA_DIR = os.path.join(ROOT_DIR, 'nebraska')
MODEL_PATH = os.path.join(NEBRASKA_DIR, 'output_weekly_nebraska_scenario7', 'best_model.keras')
DATA_PATH = os.path.join(ROOT_DIR, 'Integrated_weekly_NEB_20counties.csv')

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Scenario 7 Features
feature_cols = [
    'None_lag1', 'D0_lag1', 'D1_lag1', 'D2_lag1', 'D3_lag1', 'D4_lag1',
    'None_lag2', 'D0_lag2', 'D1_lag2', 'D2_lag2', 'D3_lag2', 'D4_lag2']

# Class Mapping and Multipliers
label_map = {0: 'None', 1: 'D0', 2: 'D1', 3: 'D2', 4: 'D3', 5: 'D4'}
class_multipliers = np.array([1.736397624015808, 1.3926482200622559, 0.8140379190444946, 0.8686022758483887, 0.6817439198493958, 1.2355276346206665])
SEQ_LENGTH = 52

# Initialize and fit the scaler using training data rules
scaler = MinMaxScaler()
shap_explainer = None
shap_background = None

def prepare_scaler_and_explainer():
    global scaler, shap_explainer, shap_background
    try:
        # Load data to fit scaler exactly like in training
        df = pd.read_csv(DATA_PATH)
        df['week_start'] = pd.to_datetime(df['week_start'])
        df = df.sort_values(['FIPS', 'week_start']).reset_index(drop=True)
        
        # Calculate PMF
        def decumulate_drought(row):
            pmf_d4 = row['D4']
            pmf_d3 = max(0.0, row['D3'] - row['D4'])
            pmf_d2 = max(0.0, row['D2'] - row['D3'])
            pmf_d1 = max(0.0, row['D1'] - row['D2'])
            pmf_d0 = max(0.0, row['D0'] - row['D1'])
            pmf_none = max(0.0, row['None'])
            return pd.Series([pmf_none, pmf_d0, pmf_d1, pmf_d2, pmf_d3, pmf_d4])
        
        pmf_cols = ['PMF_None', 'PMF_D0', 'PMF_D1', 'PMF_D2', 'PMF_D3', 'PMF_D4']
        df[pmf_cols] = df.apply(decumulate_drought, axis=1)
        
        df_fe = df.copy()
        
        # Calculate lags
        for col in ['None', 'D0', 'D1', 'D2', 'D3', 'D4']:
            df_fe[f'{col}_lag1'] = df_fe.groupby('FIPS')[col].shift(1)
            df_fe[f'{col}_lag2'] = df_fe.groupby('FIPS')[col].shift(2)

        df_fe = df_fe.dropna(subset=feature_cols).reset_index(drop=True)
        
        # Train split for scaler (up to 2019-12-31)
        TRAIN_END_DATE = '2019-12-31'
        train_df = df_fe[df_fe['week_start'] <= TRAIN_END_DATE].copy()
        
        scaler.fit(train_df[feature_cols])
        print("Scaler fitted successfully.")
        
        # Prepare background data for SHAP (sample a few sequences)
        scaled_train = scaler.transform(train_df[feature_cols])
        bg_seqs = []
        for i in range(min(50, len(scaled_train) - SEQ_LENGTH)):
            bg_seqs.append(scaled_train[i:i+SEQ_LENGTH])
        
        # We need 2D background for KernelExplainer
        shap_background = np.array(bg_seqs)
        if len(shap_background.shape) == 3:
            shap_background = shap_background[:, -1, :] # Convert 3D sequence to 2D
        
        # Initialize explainer
        if model is not None:
            # Prediction wrapper for KernelExplainer
            def predict_wrapper(x_2d):
                batch_size = x_2d.shape[0]
                # Convert back to DataFrame to prevent MinMaxScaler warnings
                x_df = pd.DataFrame(x_2d, columns=feature_cols)
                x_scaled = scaler.transform(x_df)
                x_3d = np.tile(x_scaled.reshape(batch_size, 1, -1), (1, SEQ_LENGTH, 1))
                preds = model.predict(x_3d, verbose=0)
                return preds * class_multipliers
            
            # Use 15 samples from background for performance
            shap_explainer = shap.KernelExplainer(predict_wrapper, shap_background[:15])
            print("SHAP KernelExplainer initialized successfully.")
            
    except Exception as e:
        print(f"Error preparing scaler or explainer: {e}")

prepare_scaler_and_explainer()

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
        
        # Generate SHAP explanations
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
    primary_hue="blue", 
    secondary_hue="indigo"
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

with gr.Blocks(theme=theme, css=custom_css, title="Sistem Prediksi Kekeringan Mingguan County (Nebraska)") as demo:
    gr.Markdown(
        """
        # 🌾 Sistem Prediksi Kekeringan Mingguan County (Nebraska Scenario 7)
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
