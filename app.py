from flask import Flask, render_template, request, jsonify, url_for
import os
import joblib
import cv2
import numpy as np
import webbrowser

# Initialize Flask app
app = Flask(__name__)

# Load the saved Voting Classifier model
voting_clf_loaded = joblib.load('model/voting_classifier_model.pkl')
IMG_SIZE = (128, 128)  # Image size for resizing

# Function to preprocess custom input images
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0  # Normalize pixel values
        return img.reshape(1, -1)  # Reshape to match the input shape
    else:
        raise ValueError(f"Image not found or invalid: {image_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded image
    upload_folder = 'static/uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    image_path = os.path.join(upload_folder, image_file.filename)
    image_file.save(image_path)

    try:
        # Preprocess and predict
        processed_img = preprocess_image(image_path)
        prediction = voting_clf_loaded.predict(processed_img)
        predicted_label = prediction[0]  # Direct output from the model

        return jsonify({
            'prediction': str(predicted_label),  # Convert to string for JSON response
            'image_url': url_for('static', filename=f'uploads/{image_file.filename}')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = 5000
    url = f"http://127.0.0.1:{port}/"
    webbrowser.open(url)
    app.run(debug=True, port=port)

