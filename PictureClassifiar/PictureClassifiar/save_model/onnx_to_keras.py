import onnx2keras
from onnx2keras import onnx_to_keras
import keras
import onnx

onnx_model = onnx.load('test.onnx')
k_model = onnx_to_keras(onnx_model, ['input'])

keras.models.save_model(k_model, 'test.h5', overwrite=True, include_optimizer=True)