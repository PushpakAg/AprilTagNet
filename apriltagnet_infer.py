import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import cv2

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading model...")
model = EfficientNet.from_name('efficientnet-b0')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 587)
model.load_state_dict(torch.load('model/apriltag_effnet_epoch_25.pth',weights_only=True))

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded and moved to {device}.")

def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0) 
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return pred.item()

image_path = "116_aug_2.jpg"
pred_image = predict_image(image_path, model)
print(f"Prediction for the image: {pred_image}")

