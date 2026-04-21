import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Classes (from your ImageFolder — alphabetical order)
classes = ["Parasitized", "Uninfected"]

# Rebuild model architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("best_resnet18_malaria.pth", map_location="cpu"))
model.eval()

# Same transform you used at test time
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(img):
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return {classes[i]: float(probs[i]) for i in range(2)}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Malaria Cell Detector",
    description="Upload a blood cell image. Model will classify it as Parasitized or Uninfected."
).launch()