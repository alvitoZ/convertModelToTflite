import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import matplotlib.pyplot as plt 
import base64
from PIL import Image
import io
import math
from math import sqrt
import json
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %matplotlib inline

# shutil.unpack_archive(format="gzip", filename="model/saved_model.pb", extract_dir="./")

#--
export_dir = "./models/"
model = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")

#--
class ExampleModel(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
  def capture_fn(self, x):
    if not hasattr(self, 'weight'):
      self.weight = tf.Variable(5.0, name='weight')
    self.weight.assign_add(x * self.weight)
    return self.weight

  @tf.function
  def polymorphic_fn(self, x):
    return tf.constant(3.0) * x

#--
model = ExampleModel()
model.polymorphic_fn(tf.constant(4.0))
model.polymorphic_fn(tf.constant([1.0, 2.0, 3.0]))
tf.saved_model.save(
    model, "convert_model2", signatures={'serving_default': model.capture_fn})

#--
# model_path = "convert_model2"

#--
# modelss = tf.saved_model.load(model_path)

#--
# signa = list(modelss.signatures.keys())
# print(signa)

#--
# Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
# interpreter.allocate_tensors()

# Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# print(input_details)
# print(output_details)

#--
