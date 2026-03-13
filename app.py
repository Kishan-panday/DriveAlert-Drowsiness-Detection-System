from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Load model once when server starts
model = tf.keras.models.load_model("model")
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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)