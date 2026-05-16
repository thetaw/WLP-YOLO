from ultralytics import YOLO

# Load a model
model = YOLO(r"myyaml\SlimNeck\yolov8-SlimNeck-GhostHGNetV2.yaml")  # build a new model from scratch
model.info()