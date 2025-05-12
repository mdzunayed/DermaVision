# Cell: model + Grad-CAM loader and prediction
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
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2

def load_model():
    ckpt_path = os.path.join('model', 'best_model_fold_4.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No model file at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    # If state_dict, rebuild Xception; if full module, use directly
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model = timm.create_model('xception', pretrained=True, num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint)

    model.to(DEVICE).eval()

    # Hook the last Conv2d layer for CAM
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No Conv2d layers found in model")
    cam = GradCAM(model=model, target_layers=[convs[-1]])
    return model, cam

def predict_with_cam(model, cam, img_path):
    # 1) Load original and convert to RGB float32
    bgr_orig = cv2.imread(img_path)
    if bgr_orig is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    rgb_orig = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 2) Resize for network
    rgb_small = cv2.resize(rgb_orig, (IMG_SIZE, IMG_SIZE))

    # 3) Preprocess
    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    inp = prep((rgb_small * 255).astype(np.uint8)).unsqueeze(0).to(DEVICE)

    # 4) Inference
    with torch.no_grad():
        out   = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    # 5) Generate CAM on small image
    grayscale_cam = cam(input_tensor=inp)[0]  # (IMG_SIZE, IMG_SIZE)

    # 6) Resize heatmap to original size
    heatmap_full = cv2.resize(grayscale_cam,
                              (rgb_orig.shape[1], rgb_orig.shape[0]))

    # 7) Overlay on original
    overlay = show_cam_on_image(rgb_orig, heatmap_full, use_rgb=True)

    # 8) Save overlay
    os.makedirs('static/cams', exist_ok=True)
    cam_filename = f"cam_{os.path.basename(img_path)}"
    cam_path     = os.path.join('static/cams', cam_filename)
    cv2.imwrite(cam_path,
                cv2.cvtColor((overlay * 255).astype(np.uint8),
                             cv2.COLOR_RGB2BGR))

    return probs, cam_path
