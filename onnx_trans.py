from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('v8.pt')

# Export the model to ONNX format
model.export(format='onnx')  # creates 'yolov8n.onnx'

# Load the exported ONNX model
# onnx_model = YOLO('yolov8n.onnx')

# Run inference
# results = onnx_model('')