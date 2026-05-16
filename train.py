import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'myyaml\SlimNeck\yolov8n-SlimNeck-GhostHGNetV2.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset\walnut_new.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=10,
                workers=8,
                device='0',
                patience=0,
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/walnut_new_train',
                name='yolov8n-SlimNeck-GhostHGNetV2__', 
                )