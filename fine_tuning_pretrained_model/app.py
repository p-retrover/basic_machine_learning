import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Model Configuration & Loading
def load_model(model_path="resnet18_cifar10.pth"):
    # Reconstruct the architecture used during fine-tuning
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # Ensure this matches the number of classes in your training (CIFAR-10 = 10)
    model.fc = nn.Linear(num_ftrs, 10) 
    
    # Load weights onto CPU (best for deployment/testing)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None

# Initialize the model
model = load_model()

# 2. Image Preprocessing 
# These must be identical to your validation transforms
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Prediction Function
def predict(inp_img):
    if model is None:
        return "Error: resnet18_cifar10.pth not found in the directory."
    if inp_img is None:
        return None

    # Prepare image: Transform -> Add Batch Dim
    img_t = preprocess(inp_img).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(img_t)
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(prediction[0], dim=0)
    
    # CIFAR-10 Classes
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Return dictionary for Gradio Label component
    return {classes[i]: float(probs[i]) for i in range(10)}

# 4. Building the Interface
with gr.Blocks(title="CIFAR-10 Fine-Tuning Demo") as demo:
    gr.Markdown("# ResNet-18 Fine-Tuned on CIFAR-10")
    gr.Markdown(
        "Upload an image to test the fine-tuned model. This interface uses the weights "
        "saved from the transfer learning project."
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            submit_btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Top Predictions")

    submit_btn.click(fn=predict, inputs=image_input, outputs=label_output)

    # gr.Examples(
    #     examples=[],
    #     inputs=image_input
    # )

if __name__ == "__main__":
    demo.launch()