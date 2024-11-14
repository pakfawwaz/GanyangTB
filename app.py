from flask import Flask, request, render_template, redirect, url_for
import os
import torch
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier
from yolov5.models.common import DetectMultiBackend
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the model
device = select_device('cpu')  # Use 'cpu' if deploying on a CPU-only environment
model = DetectMultiBackend('yolov5/best.pt', device=device)  # Load your model weights

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the file is in the request
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Perform object detection
            result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + file.filename)
            detect_image(filepath, result_image_path)
            
            return render_template('result.html', uploaded_image=filepath, result_image=result_image_path)

    return render_template('index.html')

def detect_image(image_path, result_image_path):
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)

    # Preprocess image
    img = cv2.resize(img, (640, 640))  # YOLOv5 default input size is 640x640
    img = np.moveaxis(img, -1, 0)  # Change to (channels, height, width)
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)  # Normalize

    # Run inference
    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, 0.25, 0.45)  # Apply NMS

    # Draw bounding boxes on the image
    img = np.array(Image.open(image_path).convert('RGB'))
    for det in pred:  # detections per image
        if len(det):
            det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=(0, 255, 0), line_thickness=2)

    # Save the result
    result_img = Image.fromarray(img)
    result_img.save(result_image_path)

if __name__ == '__main__':
    port =int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)