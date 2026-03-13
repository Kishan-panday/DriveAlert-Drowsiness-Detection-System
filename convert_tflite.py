import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("drowsiness_model.h5", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("drowsiness_model.tflite", "wb") as f:
    f.write(tflite_model)