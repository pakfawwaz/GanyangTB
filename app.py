from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
import os
from PIL import Image
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

# Load your trained model
model = YOLO('best.pt')

# Ensure upload and result folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Run object detection
            results = model.predict(source=filepath)
            
            # Save the detection results
            result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + file.filename)
            results[0].save(save_dir=app.config['RESULT_FOLDER'])
            
            return render_template('result.html', uploaded_image=filepath, result_image=result_image_path)

    return render_template('index.html')

if __name__ == '__main__':
    port =int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)