import sys
import os
sys.path.insert(0, os.path.abspath('detectron2'))

import os
from detectron2.engine import DefaultTrainer, hooks, launch
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import BestCheckpointer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

datasets = ["ACDC", "AutoEnhancer", "BayesianRetinex", "ICSP", "Original", 
            "PCDE", "Semi_UIR", "TEBCF", "TUDA", "USUIR"]


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        # Use COCO-style evaluation (mAP will be computed)
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        
        # Add BestCheckpointer hook to save the best model
        best_ckpt = BestCheckpointer(
            cfg.TEST.EVAL_PERIOD,
            self.checkpointer,
            "bbox/AP",  # Metric to track: COCO-style AP for bounding boxes
            mode="max",  # Higher is better for AP
            file_prefix="model_best"
        )
        hooks_list.append(best_ckpt)
        return hooks_list

for dataset in datasets:
    print(f"Training on dataset: {dataset}")

    train_json = f"Enhanced_RUOD_coco/{dataset}/annotations/train.json"
    val_json = f"Enhanced_RUOD_coco/{dataset}/annotations/val.json"
    train_images = f"Enhanced_RUOD_coco/{dataset}/images/train"
    val_images = f"Enhanced_RUOD_coco/{dataset}/images/val"

    train_name = f"{dataset}_train"
    val_name = f"{dataset}_val"
    register_coco_instances(train_name, {}, train_json, train_images)
    register_coco_instances(val_name, {}, val_json, val_images)

    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0015
    cfg.SOLVER.MAX_ITER = 1100 
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.TEST.EVAL_PERIOD = 300  # Evaluate every 300 iterations
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.RETINANET.NUM_CLASSES = 10  

    cfg.OUTPUT_DIR = f"output_retinanet/{dataset}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


