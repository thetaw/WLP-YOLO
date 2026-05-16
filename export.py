import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs\prune-walnut-new\yolov8-GSA-lamp-exp1-finetune2\weights\best.pt')
    # model.export(format ='onnx',opset=17)
    model.export(format ='engine')