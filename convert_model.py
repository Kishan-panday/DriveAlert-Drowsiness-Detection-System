import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("drowsiness_model.h5", compile=False)

model.export("model")