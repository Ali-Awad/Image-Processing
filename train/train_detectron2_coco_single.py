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

# Paths to your COCO dataset
train_json = "annotations/train.json"
val_json = "annotations/val.json"
train_images = "images/train"
val_images = "images/val"

# Register the existing COCO dataset
register_coco_instances("my_train", {}, train_json, train_images)
register_coco_instances("my_val", {}, val_json, val_images)

# Detectron2 Configuration for Faster R-CNN R50-FPN
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_train",)
cfg.DATASETS.TEST = ("my_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0015
cfg.SOLVER.MAX_ITER = 2200 
cfg.SOLVER.WARMUP_ITERS = 500
cfg.TEST.EVAL_PERIOD = 300 
cfg.MODEL.DEVICE = "cuda"


# Set number of classes (update based on your dataset)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Modify if you have a different number of classes

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

