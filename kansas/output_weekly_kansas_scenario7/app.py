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
    'None_lag2', 'D0_lag2', 'D1_lag2', 'D2_lag2', 'D3_lag2', 'D4_lag2',
    'drought_carryover_lag1', 'severe_carryover_lag1'
])
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

# Prepare background data for SHAP
shap_explainer = None
shap_background = None
try:
    shap_bg_path = os.path.join(CURRENT_DIR, 'shap_background.npy')
    if os.path.exists(shap_bg_path):
        shap_background = np.load(shap_bg_path)
        if model is not None:
            shap_explainer = shap.GradientExplainer(model, shap_background)
            print("SHAP explainer loaded successfully.")
except Exception as e:
    print(f"Error preparing SHAP: {e}")

def predict(*args):
    if model is None:
        return "Model not loaded. Please check the model path.", None
    if scaler is None:
        return "Scaler not loaded. Please check the scaler path.", None
        
    try:
        # Args contains the 14 features in order
        input_data = np.array([list(args)])
        
        # Scale the inputs
        scaled_input = scaler.transform(input_data)
        
        # The model expects a sequence of length SEQ_LENGTH. 
        # Tile this single timestep SEQ_LENGTH times.
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
                # Get the shap values for the predicted class
                class_shap_values = shap_values[pred_idx][0] # shape: (SEQ_LENGTH, 14)
                
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
    primary_hue="green", 
    secondary_hue="emerald"
)

with gr.Blocks(theme=theme, title="Kansas Drought Prediction Dashboard") as demo:
    gr.Markdown(
        """
        # 🌾 Weekly Drought Prediction Dashboard (Kansas Scenario 7)
        **Model:** Bidirectional LSTM (Drought History Only)
        
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
