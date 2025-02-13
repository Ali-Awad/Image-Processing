# Add detectron2 to the Python path (optional, as the installation above already does this)
import sys
import os
sys.path.insert(0, os.path.abspath('Y:/Michigan_Tech_Courses/ALI_Work/Detectron/detectron2'))

import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
from detectron2.data.datasets import register_coco_instances

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

datasets = ["ACDC", "AutoEnhancer", "BayesianRetinex", "ICSP", "Original", 
            "PCDE", "Semi_UIR", "TEBCF", "TUDA", "USUIR"]


for dataset in datasets:
    print(f"Training on dataset: {dataset}")

    train_json = f"annotations/{dataset}/train.json"
    val_json = f"annotations/{dataset}/val.json"
    train_images = f"images/{dataset}/train"
    val_images = f"images/{dataset}/val"

    train_name = f"{dataset}_train"
    val_name = f"{dataset}_val"
    register_coco_instances(train_name, {}, train_json, train_images)
    register_coco_instances(val_name, {}, val_json, val_images)

    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0015
    cfg.SOLVER.MAX_ITER = 1100  
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.TEST.EVAL_PERIOD = 300 
    cfg.MODEL.DEVICE = "cuda"

    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  

    cfg.OUTPUT_DIR = f"output/{dataset}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()