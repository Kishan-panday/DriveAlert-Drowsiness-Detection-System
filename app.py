from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="drowsiness_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['Closed','Open','no_yawn','yawn']


@app.route('/', methods=['GET','POST'])
def index():

    prediction = ""

    if request.method == 'POST':

        file = request.files['image']

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = cv2.resize(img,(224,224))
        img = img/255.0
        img = np.reshape(img,[1,224,224,3]).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        pred = interpreter.get_tensor(output_details[0]['index'])

        prediction = classes[np.argmax(pred)]

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)