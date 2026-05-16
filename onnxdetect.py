import onnxruntime
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

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

# 计算模型精度
def compute_accuracy(predictions, ground_truth):
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == ground_truth)
    return accuracy

# 测试图片路径
test_image_path = "test_image.jpg"
# ONNX 模型路径
onnx_model_path = "model.onnx"

# 预处理测试图片
input_data = preprocess_image(test_image_path)

# 加载 ONNX 模型
onnx_session = load_onnx_model(onnx_model_path)

# 进行模型推理
start_time = time.time()
predictions = inference_onnx_model(onnx_session, input_data)
end_time = time.time()
inference_time = end_time - start_time

# 计算 FPS
fps = 1.0 / inference_time

# 打印模型推理时间和 FPS
print("Inference Time: {:.4f} seconds".format(inference_time))
print("FPS: {:.2f}".format(fps))

# 如果有测试标签，则计算模型精度
ground_truth_labels = ...
accuracy = compute_accuracy(predictions, ground_truth_labels)
print("Accuracy: {:.2f}%".format(accuracy * 100))
