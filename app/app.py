
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
import matplotlib.pyplot as plt

PROJECT_DIR = "/content/drive/MyDrive/ECG_Assignment_3"
MODEL_PATH = PROJECT_DIR + "/models/ecgnet_best.pt"
TEST_PATH = PROJECT_DIR + "/data/mitbih_test.csv"

CLASS_NAMES = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown (Q)"
}

class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(128, 187) * np.sqrt(2/187))
        self.b1 = nn.Parameter(torch.zeros(128))
        self.W2 = nn.Parameter(torch.randn(64, 128) * np.sqrt(2/128))
        self.b2 = nn.Parameter(torch.zeros(64))
        self.W3 = nn.Parameter(torch.randn(5, 64) * np.sqrt(2/64))
        self.b3 = nn.Parameter(torch.zeros(5))

    def forward(self, x):
        a1 = torch.clamp(x @ self.W1.T + self.b1, min=0)
        a2 = torch.clamp(a1 @ self.W2.T + self.b2, min=0)
        return a2 @ self.W3.T + self.b3

model = ECGNet()
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

test_df = pd.read_csv(TEST_PATH, header=None)

def predict(row_number, uploaded_file):
    try:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, header=None).values.flatten()
            if len(data) != 187:
                raise gr.Error("CSV must contain exactly 187 ECG values.")
            signal = data.astype(np.float32)
        else:
            index = int(row_number)
            if index < 0 or index >= len(test_df):
                raise gr.Error(f"Choose a row between 0 and {len(test_df)-1}.")
            signal = test_df.iloc[index, :187].values.astype(np.float32)

        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)[0].numpy()

        prediction = int(np.argmax(probabilities))
        result = (
            f"### Prediction: {CLASS_NAMES[prediction]}\n"
            f"Confidence: **{probabilities[prediction]*100:.2f}%**"
        )

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal, linewidth=1.5)
        ax.set_title("ECG Heartbeat Signal")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        probability_df = pd.DataFrame({
            "Heartbeat Class": [CLASS_NAMES[i] for i in range(5)],
            "Probability (%)": probabilities * 100
        })

        return result, fig, probability_df

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(str(e))

css = """
.gradio-container {
    background: #171411 !important;
    color: #f5e6d0 !important;
}
h1, h2, h3 {
    color: #d9a45b !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="amber",
        neutral_hue="stone"
    ),
    css=css,
    title="PULSE | ECG Classifier"
) as demo:

    gr.Markdown("""
    # PULSE
    ### ECG Heartbeat Classification System

    Analyze an ECG signal and classify it into one of five heartbeat categories.
    """)

    with gr.Row():
        row_input = gr.Number(
            label="Test Dataset Row Number",
            value=0,
            precision=0
        )
        file_input = gr.File(
            label="Or Upload CSV (187 ECG values)",
            file_types=[".csv"],
            type="filepath"
        )

    predict_btn = gr.Button("Analyze Heartbeat", variant="primary")

    prediction_output = gr.Markdown()

    with gr.Row():
        ecg_plot = gr.Plot(label="ECG Waveform")
        probability_plot = gr.BarPlot(
            x="Heartbeat Class",
            y="Probability (%)",
            title="Class Probabilities",
            y_lim=[0, 100]
        )

    predict_btn.click(
        fn=predict,
        inputs=[row_input, file_input],
        outputs=[prediction_output, ecg_plot, probability_plot]
    )

if __name__ == "__main__":
    demo.launch(share=True)
