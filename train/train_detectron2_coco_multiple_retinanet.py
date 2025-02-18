import sys
import os
sys.path.insert(0, os.path.abspath('detectron2'))

from detectron2.engine import DefaultTrainer, hooks
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import BestCheckpointer
import detectron2.utils.comm as comm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

datasets = ["ACDC", "AutoEnhancer", "BayesianRetinex", "ICSP", "Original", 
            "PCDE", "Semi_UIR", "TEBCF", "TUDA", "USUIR"]

class EarlyStoppingHook(hooks.HookBase):
    def __init__(self, trainer, patience=5, metric="bbox/AP", mode="max"):
        """
        Implements early stopping for Detectron2 training.
        
        :param trainer: The trainer object.
        :param patience: Number of evaluation cycles to wait for improvement.
        :param metric: The metric to track (default is COCO bbox AP).
        :param mode: "max" means higher is better (AP metric).
        """
        self.trainer = trainer
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best_metric = None
        self.counter = 0

    def after_step(self):
        # Only check at evaluation steps
        if self.trainer.iter % self.trainer.cfg.TEST.EVAL_PERIOD == 0:
            eval_results = self.trainer.storage.latest()
            current_metric = eval_results.get(self.metric, None)

            if current_metric is None:
                return  # Skip if metric not found

            # Determine if the model has improved
            if self.best_metric is None or (
                (self.mode == "max" and current_metric > self.best_metric) or 
                (self.mode == "min" and current_metric < self.best_metric)
            ):
                self.best_metric = current_metric
                self.counter = 0  # Reset counter
            else:
                self.counter += 1  # No improvement

            # Stop training if patience is exceeded
            if self.counter >= self.patience:
                comm.synchronize()  # Ensure all workers are synchronized
                self.trainer.storage.put_scalar("early_stopping", 1)
                self.trainer.checkpointer.save("model_early_stop")
                print(f"Stopping training early after {self.counter} evaluations without improvement.")
                raise hooks.StopTraining()

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        
        # Add BestCheckpointer hook to save the best model
        best_ckpt = BestCheckpointer(
            self.cfg.TEST.EVAL_PERIOD,
            self.checkpointer,
            "bbox/AP",  # Metric to track: COCO-style AP for bounding boxes
            mode="max",
            file_prefix="model_best"
        )
        hooks_list.append(best_ckpt)

        # Add EarlyStoppingHook
        early_stop_hook = EarlyStoppingHook(self, patience=100, metric="bbox/AP", mode="max")
        hooks_list.append(early_stop_hook)
        
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
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.024
    cfg.SOLVER.MAX_ITER = 200000 
    cfg.SOLVER.WARMUP_ITERS = 10000
    cfg.TEST.EVAL_PERIOD = 2000  
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 5.0
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.RETINANET.NUM_CLASSES = 10  

    cfg.OUTPUT_DIR = f"output_retinanet/{dataset}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()