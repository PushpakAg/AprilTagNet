import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
from model import UNet
from efficientnet_pytorch import EfficientNet
import numpy as np
import time

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = "cuda"
print("Loading UNet model...")
unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load("model/unet_model_epoch_45.pth", weights_only=True))
unet_model.eval()
print("UNet model loaded and moved to device.")

print("Loading AprilTagtNet ...")
effnet_model = EfficientNet.from_name('efficientnet-b0')
num_ftrs = effnet_model._fc.in_features
effnet_model._fc = nn.Linear(num_ftrs, 587)
effnet_model.load_state_dict(torch.load('model/apriltag_effnet_epoch_25.pth', weights_only=True))
effnet_model.eval()
effnet_model = effnet_model.to(device)
print(f"AprilTagNet model loaded and moved to {device}.")

def predict_image(image, model):
    image = data_transforms(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return pred.item()

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

prev_time = time.time()
fps_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_output = unet_model(input_image)

    predicted_mask = mask_output.cpu().squeeze().numpy()
    binary_mask = (predicted_mask > 0.5).astype('uint8') * 255

    binary_mask_resized = cv2.resize(binary_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox_frame = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox_frame = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if bbox_frame is not None and w > 0 and h > 0:
            bbox_image = Image.fromarray(cv2.cvtColor(bbox_frame, cv2.COLOR_BGR2RGB))
            prediction = predict_image(bbox_image, effnet_model)
            cv2.putText(frame, f"Prediction: {prediction}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    binary_mask_3channel = cv2.cvtColor(binary_mask_resized, cv2.COLOR_GRAY2BGR)

    combined_frame = np.hstack((frame, binary_mask_3channel))

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    fps_list.append(fps)

    cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Combined Frame", combined_frame)

    if bbox_frame is not None:
        cv2.imshow("Bounding Box Area", bbox_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()