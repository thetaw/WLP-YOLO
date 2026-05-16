import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune

def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    # param_dict = {
    #     # origin
    # #'model': r'runs\walnut_new_train\yolov8-GhostHGNetV2-SlimNeck-ASF\weights\best.pt', ## 需要剪枝的权重
    # #'data':r'dataset\walnut_new.yaml',
    # 'imgsz': 640,
    # 'epochs': 200,
    # 'batch': 16,
    # 'workers': 8,
    # 'cache': True,
    # 'optimizer': 'SGD',
    # 'device': '0',
    # 'close_mosaic': 20,
    # #'project':'runs/prune-walnut-new',
    # #'name':'yolov8-GSA-group_taylor-exp',
    
    #  # prune
    # 'prune_method':'group_taylor',
    # 'global_pruning': False,
    # 'speed_up': 2.0,
    # 'reg': 0.0005,
    # 'sl_epochs': 500,
    # 'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    # 'sl_model': None
    # }

    param_dict = {
    # origin
    'model': r'runs\walnut_new_train\yolov8-GhostHGNetV2-SlimNeck-ASF\weights\best.pt', ## 需要剪枝的权重
    'data':r'dataset\walnut_new.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune-walnut-new',
    'name':'yolov8-GSA-lamp-exp-spd=2.5_',
    
    
    # prune
    'prune_method':'lamp',
    'global_pruning': False,
    'speed_up': 2.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)