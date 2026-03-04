import gradio as gr
import numpy as np
from PIL import Image

WEIGHTS_PATH = "numpy_mlp_weights.npy"

# --- Simplified ScratchNet for Inference ---
class ScratchNetInference:
    def __init__(self, weights_path):
        # Load the dictionary of parameters: W1, b1, W2, b2, W3, b3
        self.params = np.load(weights_path, allow_pickle=True).item()

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        # Stability fix: subtract max
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward(self, X):
        # Layer 1
        Z1 = np.dot(self.params['W1'], X) + self.params['b1']
        A1 = self.relu(Z1)
        # Layer 2
        Z2 = np.dot(self.params['W2'], A1) + self.params['b2']
        A2 = self.relu(Z2)
        # Layer 3 (Output)
        Z3 = np.dot(self.params['W3'], A2) + self.params['b3']
        A3 = self.softmax(Z3)
        return A3

# Initialize Inference Engine
try:
    model = ScratchNetInference(WEIGHTS_PATH)
except FileNotFoundError:
    model = None
    print(f"Error: {WEIGHTS_PATH} not found. Please save your weights in the notebook first.")

# --- Prediction Logic ---
def predict(inp_img):
    if model is None:
        return "Weights file not found."
    
    # 1. Resize to CIFAR-10 dimensions (32x32)
    img_res = inp_img.resize((32, 32))
    
    # 2. Preprocess: Normalize and match your training shape
    # Training usually expects (Channels, Height, Width) flattened
    img_arr = np.array(img_res).astype(np.float32) / 255.0
    
    # Transpose from (H, W, C) to (C, H, W) and flatten to (3072, 1)
    img_flat = img_arr.transpose(2, 0, 1).reshape(3072, 1)
    
    # 3. Compute Forward Pass
    probs = model.forward(img_flat)
    probs = probs.flatten()  # Convert to 1D array for easier handling
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Convert probabilities to a dictionary for Gradio
    return {classes[i]: float(probs[i]) for i in range(10)}

# --- Gradio Interface ---
with gr.Blocks(title="NumPy ScratchNet") as demo:
    gr.Markdown("# Neural Network from Scratch")
    gr.Markdown("Inference performed using pure **NumPy** matrix operations.")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Input Image")
            run_btn = gr.Button("Calculate Forward Pass", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Class Probabilities")

    run_btn.click(fn=predict, inputs=img_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch()