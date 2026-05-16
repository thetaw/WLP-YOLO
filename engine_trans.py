from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('GSA.pt')

# Export the model to TensorRT format
model.export(format='engine')  # creates 'yolov8n.engine'

# Load the exported TensorRT model
# tensorrt_model = YOLO('yolov8n.engine')

# Run inference
# results = tensorrt_model('https://ultralytics.com/images/bus.jpg')