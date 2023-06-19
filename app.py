import os
import torch
from PIL import Image
import numpy as np
import cv2
from RealESRGAN import RealESRGAN
import imghdr
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/enhance', methods=['POST'])
def enhance_image():
    # Check if file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # Check if the file is a valid image
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'})

    try:
        # Save the image to a temporary location
        image_path = 'temp_image.png'
        file.save(image_path)

        # Load the model and perform image enhancement
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=2)
        model.load_weights('models/RealESRGAN_x2.pth')

        image = cv2.imread(image_path)
        preprocessed_image = preprocess_image(image)
        pil_image = Image.fromarray(preprocessed_image)

        sr_image = model.predict(pil_image)

        # Save the enhanced image
        enhanced_image_path = 'enhanced_image.png'
        sr_image.save(enhanced_image_path)

        # Convert the enhanced image to base64 string
        with open(enhanced_image_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')

        # Return the enhanced image as a response
        return jsonify({'enhanced_image': encoded_image})

    except Exception as e:
        return jsonify({'error': str(e)})


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    b, g, r = cv2.split(img)

    blurred_b = cv2.GaussianBlur(b, (0, 0), sigmaX=2)
    blurred_g = cv2.GaussianBlur(g, (0, 0), sigmaX=2)
    blurred_r = cv2.GaussianBlur(r, (0, 0), sigmaX=2)

    mask_b = b - blurred_b
    mask_g = g - blurred_g
    mask_r = r - blurred_r

    k = 0.8
    enhanced_b = b + k * mask_b
    enhanced_g = g + k * mask_g
    enhanced_r = r + k * mask_r

    enhanced_b = np.clip(enhanced_b, 0, 1)
    enhanced_g = np.clip(enhanced_g, 0, 1)
    enhanced_r = np.clip(enhanced_r, 0, 1)

    enhanced_image = cv2.merge([enhanced_b, enhanced_g, enhanced_r])

    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    return enhanced_image

def allowed_file(filename):
    # Check if the file has a valid image extension
    valid_extensions = ['jpg', 'jpeg', 'png']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in valid_extensions


if __name__ == '__main__':
    app.run()
