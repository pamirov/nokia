import os
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image #Imports Python Image Library that's used for image manipulation

# This code gets the absolute path to the dog.jpg
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "dog.jpg") #making sure dog.jpg is in the same dir as the script

# Loading the pre-trained ResNet50 model
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

# Usage
prediction = classify_image(image_path)

# Decode the prediction
with open(os.path.join(current_dir, "imagenet_classes.txt")) as f: #Opens the imagenet_classes.txt and assigns it to var f
    labels = [line.strip() for line in f.readlines()]

_, predicted_idx = torch.max(prediction, 1)

# Print predicted index and number of classes in label file for debugging
print("Predicted index:", predicted_idx)
print("Number of classes in label file:", len(labels))

# Check if the predicted index is within the range of the labels list
if predicted_idx < len(labels):
    predicted_label = labels[predicted_idx]
    print("Predicted class:", predicted_label)
else:
    print("Error: Predicted class index out of range.")
