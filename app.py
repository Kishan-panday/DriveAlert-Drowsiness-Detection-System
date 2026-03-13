from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("drowsiness_model.h5")

classes = ['Closed','Open','no_yawn','yawn']


@app.route('/', methods=['GET','POST'])
def index():

    prediction = ""

    if request.method == 'POST':

        file = request.files['image']

        # Convert image from frontend to numpy
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = cv2.resize(img,(224,224))
        img = img/255.0
        img = np.reshape(img,[1,224,224,3])

        pred = model.predict(img)

        prediction = classes[np.argmax(pred)]

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)