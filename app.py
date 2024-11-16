import os
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template, send_file
from io import BytesIO
from PIL import Image, ImageDraw

# Initialize Flask app
app = Flask(__name__)

# Roboflow API credentials
API_URL = "https://detect.roboflow.com"
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("ga nyambung woe API key nya")
MODEL_ID = "biofarma-x-mit-hacking-medicine-hackathon/1"

# Endpoint to render the upload form
@app.route('/')
def home():
    return render_template("index.html")

# Endpoint to handle image uploads and analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    # Get the uploaded image
    file = request.files['image']
    image = Image.open(file)

    # Convert the image to bytes for API request
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    # Send image to Roboflow API
    response = requests.post(
        f"{API_URL}/{MODEL_ID}",
        params={"api_key": API_KEY},
        files={"file": image_bytes}
    )

    if response.status_code != 200:
        return jsonify({"error": "Error calling Roboflow API", "details": response.text}), 500

    # Parse API response
    result = response.json()
    predictions = result.get("predictions", [])

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for pred in predictions:
        x, y = pred['x'], pred['y']
        width, height = pred['width'], pred['height']
        confidence = pred['confidence']

        # Calculate box coordinates
        xmin = x - width / 2
        ymin = y - height / 2
        xmax = x + width / 2
        ymax = y + height / 2

        # Draw the bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 10), f"{confidence:.2f}", fill="red")

    # Save the modified image to a buffer
    output_buffer = BytesIO()
    image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)

    return send_file(output_buffer, mimetype='image/jpeg')

if __name__ == '__main__':
    port =int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)