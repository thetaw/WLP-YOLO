import onnx
import onnxruntime
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# 加载测试图片并进行预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # 添加一个 batch 维度
    return image

# 加载 ONNX 模型
def load_onnx_model(onnx_path):
    return onnxruntime.InferenceSession(onnx_path)

# 进行模型推理
def inference_onnx_model(onnx_session, input_data):
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    return onnx_session.run([output_name], {input_name: input_data.numpy()})

# 量化 ONNX 模型
def quantize_onnx_model(onnx_model_path, quantized_model_path):
    model = onnx.load(onnx_model_path)
    quantized_model = onnxruntime.quantization.quantize_static(model, quantized_model_path)
    return quantized_model

# 测试图片路径
test_image_path = "test_image.jpg"
# 训练好的 ONNX 模型路径
onnx_model_path = "model.onnx"
# 量化后的模型保存路径
quantized_model_path = "quantized_model.onnx"

# 预处理测试图片
input_data = preprocess_image(test_image_path)

# 加载 ONNX 模型
onnx_session = load_onnx_model(onnx_model_path)

# 进行模型推理
predictions = inference_onnx_model(onnx_session, input_data)

# 量化模型
quantized_model = quantize_onnx_model(onnx_model_path, quantized_model_path)

print("Quantized model generated successfully!")
