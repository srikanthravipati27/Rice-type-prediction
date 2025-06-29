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
rice_details = {
    'Arborio': 'Arborio rice is a short-grain rice used primarily in risotto. It has a high starch content, making dishes creamy.',
    'Basmati': 'Basmati is a long-grain rice with a nutty flavor and fragrant aroma, popular in Indian and Middle Eastern cuisines.',
    'Ipsala': 'Ipsala rice is a Turkish variety known for its firm texture and suitability for pilaf dishes.',
    'Jasmine': 'Jasmine rice is a long-grain variety from Thailand, known for its floral aroma and soft texture.',
    'Karacadag': 'Karacadag rice is a traditional Turkish variety grown in the Karacadag region, known for its resilience and distinct taste.'
}


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

    description = rice_details.get(predicted_class, "No description available.")

    return render_template('result.html', prediction=predicted_class, image_path=file_path, description=description)

if __name__ == '__main__':
    app.run(debug=True)
