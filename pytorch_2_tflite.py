import numpy as np
import os

import cv2

from utils import load_pretrained_weights
from unet_models import unet, uunet, wuunet

import torch
import onnx

import onnx_tf

import tensorflow as tf


NCLASS = 3
INPUT_SIZE = 224
COMB_TYPE = "ow"
MAIN_METRIC = "jaccard"
# MODEL_NAME = "unet_{}_n{}_{}_{}".format(INPUT_SIZE, NCLASS, COMB_TYPE, MAIN_METRIC)
MODEL_NAME = "unet_{}_n{}_{}_{}".format(INPUT_SIZE, NCLASS, "fs", MAIN_METRIC)
# MODEL_NAME = "uunet_{}_n{}_{}_{}".format(INPUT_SIZE, NCLASS, "fs", MAIN_METRIC)
# MODEL_NAME = "wuunet_{}_n{}_{}_{}".format(INPUT_SIZE, NCLASS, "fs", MAIN_METRIC)
unet_snapshot = "/home/vbochkov/workspace/development/wUUNet_fireSegm/pretrained/{}/snapshots/pretrained.pth.tar".format(MODEL_NAME)
output_dir = "output/translate"


def get_sample_input():
    image = cv2.imread(os.path.join('dataset/{}_{}/train_test/images/3/2.jpg'.format(COMB_TYPE, INPUT_SIZE))).astype(np.float32)
    return np.transpose(image.copy(), (2, 0, 1)) / 255.0


model = unet(NCLASS)
# model = uunet(NCLASS)
# model = wuunet(NCLASS)
# model = wuunet_inference(NCLASS)
load_pretrained_weights(model, unet_snapshot)
model.eval()


sample_input = torch.unsqueeze(torch.from_numpy(get_sample_input()), 0)
print('sample_input.shape = {}'.format(sample_input.shape))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

torch.onnx.export(
    model,  # PyTorch Model
    sample_input,  # Input tensor
    os.path.join(output_dir, MODEL_NAME + ".onnx"),  # Output file (eg. 'output_model.onnx')
    opset_version=12,  # Operator support version
    input_names=['input'],  # Input tensor name (arbitary)
    output_names=['output']  # Output tensor name (arbitary)
    # output_names=['output_bin', 'output_mult']  # Output tensor name (arbitary)
    # output_names=['', 'output_mult']  # Output tensor name (arbitary)
)

# Load the ONNX model
onnx_model = onnx.load(os.path.join(output_dir, MODEL_NAME + ".onnx"))

# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

# Print a Human readable representation of the graph
print("ONNX graph: " + onnx.helper.printable_graph(onnx_model.graph))

# ONNX -> TF
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph(os.path.join(output_dir, MODEL_NAME))

# TF -> TFLite
tf_model_path = os.path.join(output_dir, MODEL_NAME)
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the model
with open(os.path.join(output_dir, MODEL_NAME + ".tflite"), 'wb') as f:
    f.write(tflite_model)
