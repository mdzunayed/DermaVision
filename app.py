from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from utils import load_model, predict_with_cam

app = Flask(__name__)

# Folders for uploads and CAMs
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['CAM_FOLDER']    = os.path.join('static', 'cams')

# Your two classes
class_names = ['benign', 'malignant']

# Load your model + CAM once at startup
model, cam = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Redirect GET → home
    if request.method == 'GET':
        return redirect(url_for('home'))

    # Ensure file part is present
    if 'image' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    # Save upload
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename    = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    # Run inference + Grad-CAM
    probs, cam_path = predict_with_cam(model, cam, upload_path)

    # 1) Extract just the filename for the CAM overlay
    cam_filename = os.path.basename(cam_path)

    # 2) Determine which class has the highest probability
    predicted_idx   = int(probs.argmax())
    predicted_class = class_names[predicted_idx]

    # 3) Pair each class with its probability
    results = list(zip(class_names, probs))

    # 4) Render the results page with all the needed variables
    return render_template(
        'result.html',
        filename=filename,
        cam_filename=cam_filename,
        results=results,
        predicted_class=predicted_class
    )

if __name__ == '__main__':
    # Ensure folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CAM_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
