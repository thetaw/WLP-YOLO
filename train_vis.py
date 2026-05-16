import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'ultralytics\cfg\models\v8\yolov8n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset\VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=10,
                workers=8,
                device='0',
                patience=50,
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='VisDrone',
                name='',
                )