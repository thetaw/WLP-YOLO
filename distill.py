import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    # param_dict = {
    #     # origin
    #     'model': 'ultralytics/cfg/models/v8/yolov8n-fasternet.yaml',
    #     'data':'dataset/data.yaml',
    #     'imgsz': 640,
    #     'epochs': 250,
    #     'batch': 8,
    #     'workers': 8,
    #     'cache': True,
    #     'optimizer': 'SGD',
    #     'device': '0',
    #     'close_mosaic': 10,
    #     'project':'runs/distill',
    #     'name':'test',
        
    #     # distill
    #     'prune_model': False,
    #     'teacher_weights': 'ultralytics/cfg/models/v8/yolov8n-fasternet.yaml',
    #     'teacher_cfg': 'runs/train/yolov8n-fasternet/weights/best.pt',
    #     'kd_loss_type': 'feature',
    #     'kd_loss_decay': 'constant',
        
    #     'logical_loss_type': 'BCKD',
    #     'logical_loss_ratio': 0.4,
        
    #     'teacher_kd_layers': '0-1,0-2,0-3,0-4,4',
    #     'student_kd_layers': '0-1,0-2,0-3,0-4,4',
    #     'feature_loss_type': 'cwd',
    #     'feature_loss_ratio': 0.2
    # }


    param_dict = {
        # origin 
        # 'model': r'runs\prune-walnut-new\yolov8n-groupsl-exp1-prune\weights\prune.pt',
        'model':r'runs\prune-walnut-new\yolov8-GSA-lamp-exp1-finetune2\weights\best.pt',
        'data':'dataset/walnut_new.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 8,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'project':'runs/distill',
        'name':'GAS-prune(m)radio=0.001',
        
        
        # distill
        'prune_model': True,
        'teacher_weights': r'runs\walnut_new_train\yolov8-GhostHGNetV2-SlimNeck-ASF_M\weights\best.pt',
        'teacher_cfg': r'yolov8m-GhostHGNetV2-SlimNeck-ASF.yaml',
        'kd_loss_type': 'feature',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'l2',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '15,18,21',
        'student_kd_layers': '15,18,21',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 0.001    }
    
    model = DetectionDistiller(overrides=param_dict)
    model.distill()