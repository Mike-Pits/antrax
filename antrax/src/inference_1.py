"""
Данный скрипт предназначен для выполнения инференса с визуализацией результатов.
Использует предобученную модель от Torchvision "deeplabv3_resnet50".
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

class InferenceConfig:
    MODEL_PATH = "./checkpoints/best_model.pth"
    TEST_DIR = "/home/mike-pi/Documents/coding/NU/Stazh/antrax/data/My First Project.v3i.png-mask-semantic/test/images"
    OUTPUT_DIR = "./inference_results"
    NUM_CLASSES = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = (640, 640)

def load_model():
    # Initialize with auxiliary classifier
    model = deeplabv3_resnet50(pretrained=False, aux_loss=True)
    
    # Modify both classifiers
    model.classifier[4] = nn.Conv2d(256, InferenceConfig.NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, InferenceConfig.NUM_CLASSES, kernel_size=1)
    
    # Load state dict
    state_dict = torch.load(InferenceConfig.MODEL_PATH, map_location=InferenceConfig.DEVICE)
    model.load_state_dict(state_dict, strict=False)  # strict=False handles any remaining mismatches
    
    model.to(InferenceConfig.DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(InferenceConfig.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(InferenceConfig.DEVICE)
    return image, image_tensor, original_size

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        # Use main classifier output (ignore aux output during inference)
        pred_mask = torch.argmax(output['out'], dim=1).squeeze().cpu().numpy()
    return pred_mask

def visualize_results(image, pred_mask, original_size, save_path=None):
    class_colors = {
        0: [0, 0, 0],      # Background
        1: [255, 0, 0],    # Class 1
        2: [0, 255, 0],    # Class 2
        3: [0, 0, 255]     # Class 3
    }
    
    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        colored_mask[pred_mask == class_idx] = color
    
    colored_mask = Image.fromarray(colored_mask).resize(original_size, Image.NEAREST)
    image = image.resize(original_size)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(colored_mask)
    plt.title("Prediction")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(colored_mask, alpha=0.5)
    plt.title("Overlay")
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def run_inference():
    model = load_model()
    os.makedirs(InferenceConfig.OUTPUT_DIR, exist_ok=True)
    
    for img_name in os.listdir(InferenceConfig.TEST_DIR):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(InferenceConfig.TEST_DIR, img_name)
            image, image_tensor, original_size = preprocess_image(img_path)
            pred_mask = predict(model, image_tensor)
            
            save_path = os.path.join(InferenceConfig.OUTPUT_DIR, 
                                   f"result_{os.path.splitext(img_name)[0]}.png")
            visualize_results(image, pred_mask, original_size, save_path)

if __name__ == "__main__":
    run_inference()