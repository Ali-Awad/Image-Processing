import os
import torch
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Define dataset names
datasets = ["ACDC", "AutoEnhancer", "BayesianRetinex", "ICSP", "Original", 
            "PCDE", "Semi_UIR", "TEBCF", "TUDA", "USUIR"]

# Set device for inference
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Path to trained models
CHECKPOINT_DIR = "output"

# Model configuration file
CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

# Store results
Results = {}

for dataset in datasets:
    print(f"Testing on dataset: {dataset}")

    val_json = f"annotations/{dataset}/val.json"
    val_images = f"images/{dataset}/val"

    val_name = f"{dataset}_val"
    
    # Register dataset if not already registered
    if val_name not in DatasetCatalog.list():
        register_coco_instances(val_name, {}, val_json, val_images)

    # Load configuration
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.DATASETS.TEST = (val_name,)
    cfg.MODEL.WEIGHTS = os.path.join(CHECKPOINT_DIR, dataset, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Create evaluator and test data loader
    evaluator = COCOEvaluator(val_name, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, val_name)

    # Run evaluation
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Extract metrics of interest
    if "bbox" in eval_results:
        metrics = eval_results["bbox"]
        ap50 = metrics["AP50"]
        ap75 = metrics["AP75"]
        ap = metrics["AP"]
        Results[dataset] = {"AP50": ap50, "AP75": ap75, "AP": ap}

# Convert results to a DataFrame
df = pd.DataFrame.from_dict(Results, orient="index")

# Save to CSV
df.to_csv("detection_results.csv")
print(df)
