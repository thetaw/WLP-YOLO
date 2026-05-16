import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-ASF-P2.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset\walnut_new.yaml',
                cache=True,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=20,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train_base_walnutnew',
                name='yolov8-ASF-P2',
                )