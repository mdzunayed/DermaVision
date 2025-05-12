# utils.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

IMG_SIZE    = 224
NUM_CLASSES = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """
    Always builds Xception(num_classes=NUM_CLASSES) and loads your checkpoint.
    Hooks the last Conv2d layer for Grad-CAM.
    """
    ckpt_path = os.path.join('model', 'best_model_fold_4.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No model file at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    # If you saved the full model object, it will be an nn.Module
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    # Otherwise assume it's a state_dict → rebuild Xception
    else:
        model = timm.create_model(
            'xception',        # <–– only XCEPTION from TIMM
            pretrained=True,
            num_classes=NUM_CLASSES
        )
        model.load_state_dict(checkpoint)

    model.to(DEVICE).eval()

    # Hook the last Conv2d for Grad-CAM
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise RuntimeError("No Conv2d layers found in your model")
    cam = GradCAM(model=model, target_layers=[convs[-1]])
    return model, cam


def predict_with_cam(model, cam, img_path):
    """
    Runs inference on img_path, generates Grad-CAM overlay,
    saves it under static/cams/, and returns (probs, cam_path).
    """
    # Load and normalize image
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
    rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

    # Resize for model input
    rgb_resized = cv2.resize(rgb_orig, (IMG_SIZE, IMG_SIZE))

    # Preprocess
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    inp = preprocess((rgb_resized*255).astype(np.uint8)).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        out   = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    # Generate CAM
    gray_cam = cam(input_tensor=inp)[0]  # (IMG_SIZE, IMG_SIZE)

    # Upsample to original size
    heatmap_full = cv2.resize(gray_cam,
                              (rgb_orig.shape[1], rgb_orig.shape[0]))

    # Overlay
    overlay = show_cam_on_image(rgb_orig, heatmap_full, use_rgb=True)

    # Save overlay
    os.makedirs('static/cams', exist_ok=True)
    cam_fname = f"cam_{os.path.basename(img_path)}"
    cam_path  = os.path.join('static/cams', cam_fname)
    bgr_ov     = cv2.cvtColor((overlay*255).astype(np.uint8),
                              cv2.COLOR_RGB2BGR)
    cv2.imwrite(cam_path, bgr_ov)

    return probs, cam_path
