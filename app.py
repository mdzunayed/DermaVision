from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from utils import load_model, generate_gradcam_visualization
import requests
def download_model(url):
    response = requests.get(url)
    with open("model/best_model_fold_4.pth", "wb") as f:
        f.write(response.content)
app = Flask(__name__)

# Folders for uploads and CAMs
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['CAM_FOLDER']    = os.path.join('static', 'cams')

# Your two classes
class_names = ['benign', 'malignant']

# Load model and CAM generator once
model, cam = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))

    if 'image' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    # Save uploaded file
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    # Generate Grad-CAM visualization and get prediction probabilities
    cam_path = generate_gradcam_visualization(model, cam, upload_path)

    # Predict class probabilities
    from torchvision import transforms
    import cv2
    import numpy as np
    import torch

    image = cv2.imread(upload_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image_rgb).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    predicted_idx = int(np.argmax(probs))
    predicted_class = class_names[predicted_idx]
    results = list(zip(class_names, probs))

    return render_template(
        'result.html',
        filename=filename,
        cam_filename=os.path.basename(cam_path),
        results=results,
        predicted_class=predicted_class
    )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CAM_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
