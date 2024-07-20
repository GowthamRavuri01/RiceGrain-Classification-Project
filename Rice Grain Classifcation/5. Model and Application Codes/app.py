from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the trained model
model = tf.keras.models.load_model('rice_grain_classifier.h5')

# Define the labels
class_labels = {
    0: 'arborio',
    1: 'basmati',
    2: 'ipsala',
    3: 'jasmine',
    4: 'karacadag'
}

# Function to preprocess uploaded image
def preprocess_image(image_path, target_size=(128, 128)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file uploaded')
        
        file = request.files['file']

        # If no file selected
        if file.filename == '':
            return render_template('index.html', message='No file selected')

        # If file exists and is valid
        if file:
            # Save the file to the uploads directory
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Preprocess the image
            img_array = preprocess_image(file_path)

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]

            # Render result template with predictions
            return render_template('results.html', prediction=predicted_class, image_file=file_path)

if __name__ == '__main__':
    app.run(debug=True)
