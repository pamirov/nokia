import os
import pytest
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

# Absolute path to the dog.jpg
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "dog.jpg")

# Load the pre-trained ResNet50 model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Requiered preprocessing transformation of the pic
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to perform image classification
def classify_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image)
    image = torch.unsqueeze(image, 0) 
    with torch.no_grad():
        prediction = model(image)
    return prediction

# Function to test the classify_image functions
def test_classification():
    prediction = classify_image(image_path)

    with open(os.path.join(current_dir, "imagenet_classes.txt")) as f:
        labels = [line.strip() for line in f.readlines()]

    _, predicted_idx = torch.max(prediction, 1)

    assert predicted_idx < len(labels), "Predicted class index out of range."

    predicted_label = labels[predicted_idx]
    assert isinstance(predicted_label, str), "Predicted label is not a string."
