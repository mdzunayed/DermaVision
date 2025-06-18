# utils.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path='model/best_model_fold_4.pth'):
    """
    Loads the pre-trained Xception model and initializes GradCAM++.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.

    Returns:
        model (nn.Module): Loaded Xception model.
        cam (GradCAMPlusPlus): Grad-CAM++ instance targeting suitable layers.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model = timm.create_model(
            'xception',
            pretrained=True,
            num_classes=NUM_CLASSES
        )
        model.load_state_dict(checkpoint)

    model.to(DEVICE).eval()

    # Explicitly select the last convolutional layer for CAM
    target_layers = [model.conv4] if hasattr(model, 'conv4') else [
        layer for layer in model.modules() if isinstance(layer, nn.Conv2d)
    ][-1:]

    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    return model, cam


def generate_gradcam_visualization(model, cam, img_path, output_dir='static/cams', colormap=cv2.COLORMAP_JET):
    """
    Generates and saves a Grad-CAM visualization consisting of:
    1. Original image
    2. Grad-CAM heatmap
    3. Heatmap overlayed on original image.

    Args:
        model: The pre-trained model.
        cam: The initialized GradCAM++ object.
        img_path (str): Path to input image.
        output_dir (str): Directory to save the visualization.
        colormap: OpenCV colormap for heatmap.

    Returns:
        output_path (str): Path to the saved visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image at '{img_path}' not found")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img_rgb).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    target_class = np.argmax(probs)
    targets = [ClassifierOutputTarget(target_class)]

    # Grad-CAM++ Heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)[0]

    # Resize heatmap to original image dimensions
    heatmap_resized = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)

    # Concatenate images horizontally: original | heatmap | overlay
    combined_image = np.hstack((img_bgr, heatmap_color, overlay))

    # Save combined image
    filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, f'gradcam_{filename}')
    cv2.imwrite(output_path, combined_image)

    return output_path