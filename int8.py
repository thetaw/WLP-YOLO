import onnx
from onnxruntime.quantization import QuantType,quantize_dynamic
 
model_fp32 = "yolov8n-infer.onnx"
model_quant = "yolov8n.quant.onnx"


quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
 

