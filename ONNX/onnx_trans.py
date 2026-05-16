from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Export the model to ONNX format
#model.export(format='onnx',half=True)  # creates 'yolov8n.onnx'

# Load the exported ONNX model
# onnx_model = YOLO('yolov8n.onnx')
# onnx_model = YOLO('yolov8n.onnx')
#torchscricp_model = YOLO('yolov8n.torchscript')

model.export(format='engine',device=0)

# Run inference
# results = onnx_model('')
# results = onnx_model('bus.jpg') 
# results = torchscricp_model('bus.jpg')
# engine_model = ('yolov8n.engine')
# result = torchscricp_model('bus.jpg')
# tensorrt_model = YOLO('yolov8n.engine')

# Run inference
# results = tensorrt_model('bus.jpg')
# Load the exported TensorRT model
tensorrt_model = YOLO('yolov8n.engine')

# Run inference
results = tensorrt_model('bus.jpg')