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
    'None_lag2', 'D0_lag2', 'D1_lag2', 'D2_lag2', 'D3_lag2', 'D4_lag2',
    'drought_carryover_lag1', 'severe_carryover_lag1'
]

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
            
        df_fe['drought_carryover_lag1'] = df_fe['D0_lag1'] + df_fe['D1_lag1'] + 0.5 * df_fe['D2_lag1']
        df_fe['severe_carryover_lag1'] = df_fe['D3_lag1'] + df_fe['D4_lag1']
        
        df_fe = df_fe.dropna(subset=feature_cols).reset_index(drop=True)
        
        # Train split for scaler (up to 2019-12-31)
        TRAIN_END_DATE = '2019-12-31'
        train_df = df_fe[df_fe['week_start'] <= TRAIN_END_DATE].copy()
        
        scaler.fit(train_df[feature_cols])
        print("Scaler fitted successfully.")
        
        # Prepare background data for SHAP (sample a few sequences)
        # Using a simplistic approach to get a background distribution
        scaled_train = scaler.transform(train_df[feature_cols])
        # Create sequences of length SEQ_LENGTH
        bg_seqs = []
        for i in range(min(50, len(scaled_train) - SEQ_LENGTH)):
            bg_seqs.append(scaled_train[i:i+SEQ_LENGTH])
        shap_background = np.array(bg_seqs)
        
        # Initialize explainer
        if model is not None:
            # We use GradientExplainer for sequence models typically
            shap_explainer = shap.GradientExplainer(model, shap_background)
            print("SHAP explainer initialized.")
            
    except Exception as e:
        print(f"Error preparing scaler or explainer: {e}")

prepare_scaler_and_explainer()

def predict(*args):
    if model is None:
        return "Model not loaded. Please check the model path.", None
        
    try:
        # Args contains the 14 features in order
        input_data = np.array([list(args)])
        
        # Scale the inputs
        scaled_input = scaler.transform(input_data)
        
        # The model expects a sequence of length 52. 
        # For the dashboard demo, we tile this single timestep 52 times.
        # Alternatively, assume the user is providing the most recent timestep 
        # and previous timesteps are zeros or identical. We'll use identical for simplicity.
        sequence_input = np.tile(scaled_input, (1, SEQ_LENGTH, 1))
        
        # Predict
        y_pred_prob = model.predict(sequence_input, verbose=0)
        
        # Apply class multipliers from validation tuning
        tuned_probs = y_pred_prob[0] * class_multipliers
        pred_idx = np.argmax(tuned_probs)
        predicted_class = label_map[pred_idx]
        
        # Generate SHAP explanations
        fig = None
        if shap_explainer is not None:
            try:
                # Calculate SHAP values
                shap_values = shap_explainer.shap_values(sequence_input)
                # shap_values is a list of arrays (one for each class) 
                # Shape of each array: (1, 52, 14)
                
                # Get the shap values for the predicted class
                class_shap_values = shap_values[pred_idx][0] # shape: (52, 14)
                
                # Aggregate across the sequence dimension (sum over timesteps)
                agg_shap = np.sum(class_shap_values, axis=0) # shape: (14,)
                
                # Plot SHAP waterfall
                plt.figure(figsize=(10, 6))
                
                # Base value for the class
                expected_value = shap_explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    base_val = expected_value[pred_idx]
                else:
                    base_val = expected_value
                
                explanation = shap.Explanation(values=agg_shap, 
                                               base_values=base_val, 
                                               data=input_data[0], 
                                               feature_names=feature_cols)
                
                shap.waterfall_plot(explanation, show=False)
                fig = plt.gcf()
                plt.tight_layout()
            except Exception as e:
                print(f"SHAP Error: {e}")
                
        return f"Predicted Class: {predicted_class}", fig

    except Exception as e:
        return f"Error during prediction: {e}", None

# Create Gradio Interface
theme = gr.themes.Soft(
    primary_hue="blue", 
    secondary_hue="indigo"
)

with gr.Blocks(theme=theme, title="Drought Prediction Dashboard") as demo:
    gr.Markdown(
        """
        # 🌾 Weekly Drought Prediction Dashboard (Nebraska)
        **Model Scenario 7:** Drought History Only (BiLSTM)
        
        Enter the drought metrics below to predict the drought category for the current week. 
        For demonstration purposes, this single set of features is replicated across a 52-week sequence to emulate the model's sequence input.
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🕒 Lag 1 Features (Previous Week)")
            f1 = gr.Number(label="None_lag1", value=0.0)
            f2 = gr.Number(label="D0_lag1", value=0.0)
            f3 = gr.Number(label="D1_lag1", value=0.0)
            f4 = gr.Number(label="D2_lag1", value=0.0)
            f5 = gr.Number(label="D3_lag1", value=0.0)
            f6 = gr.Number(label="D4_lag1", value=0.0)
            f13 = gr.Number(label="drought_carryover_lag1", value=0.0)
            f14 = gr.Number(label="severe_carryover_lag1", value=0.0)
            
        with gr.Column():
            gr.Markdown("### 🕒 Lag 2 Features (Two Weeks Ago)")
            f7 = gr.Number(label="None_lag2", value=0.0)
            f8 = gr.Number(label="D0_lag2", value=0.0)
            f9 = gr.Number(label="D1_lag2", value=0.0)
            f10 = gr.Number(label="D2_lag2", value=0.0)
            f11 = gr.Number(label="D3_lag2", value=0.0)
            f12 = gr.Number(label="D4_lag2", value=0.0)
            
    predict_btn = gr.Button("🔮 Generate Prediction", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎯 Prediction Result")
            output_text = gr.Textbox(label="Main Prediction", lines=2)
            
        with gr.Column():
            gr.Markdown("### 📊 Explainability (SHAP)")
            output_plot = gr.Plot(label="SHAP Waterfall Plot")

    inputs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14]
    
    predict_btn.click(
        fn=predict,
        inputs=inputs,
        outputs=[output_text, output_plot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
