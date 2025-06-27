from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)
model = load_model("rice.h5")
classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded'


    file = request.files['file']
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img / 255.0, axis=0)

    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]

    return render_template('result.html', prediction=predicted_class, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
