import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(r'runs\VisDrone\yolov8-C2f-RFCAConv\weights\best.pt')
    model = YOLO(r'best.pt')
    model.val(data='dataset/walnut.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/walnut_new_val',
              name='exp',
              )