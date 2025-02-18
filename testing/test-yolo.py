from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_val
from super_gradients.training import models
from super_gradients.training.metrics import (DetectionMetrics, DetectionMetrics_075, DetectionMetrics_050)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients import setup_device
from pathlib import Path
import os
import pandas as pd

#models_names = ["Original", "ACDC", "AutoEnhancer", "BayesianRetinex", "ICSP", "PCDE", "Semi_UIR", "TEBCF", "TUDA", "USUIR"]
#models_names = ["Original", "SRDRM_2X", "SRDRM_4X", "SRDRM-GAN_2X", "SRDRM-GAN_4X", "WaveNet_X2", "WaveNet_X3", "WaveNet_X4"]
models_names = ["Original"]
#models_names = ["50_Epochs", "100_Epochs", "150_Epochs", "200_Epochs", "250_Epochs", "300_Epochs", "Average_Epochs", "Best_Epochs",]
#DATA_DIRS = ['./data/Enhanced_Ours_yolo/', './data/Enhanced_RUOD_yolo/']
DATA_DIRS = ['./data/Enhanced_SISR/']
MODEL_ARCH = 'yolo_nas_l'
BATCH_SIZE = 16
NUM_WORKERS = 0
CHECKPOINT_DIR = './checkpoints'

DATA_DIRS[0] = Path(DATA_DIRS[0]).absolute().as_posix().__str__()
#DATA_DIRS[1] = Path(DATA_DIRS[1]).absolute().as_posix().__str__()
CHECKPOINT_DIR = Path(CHECKPOINT_DIR).absolute().as_posix().__str__()

CLASS = {}
#CLASS[DATA_DIRS[0]] = ["Bushy", "Leafy", "Tapey"]
CLASS[DATA_DIRS[0]] = ["holothurian", "echinus", "scallop", "starfish", "fish", "corals", "diver", "cuttlefish", "turtle", "jellyfish"]

#setup_device(device= 'cuda', multi_gpu='DDP', num_gpus=2)
setup_device(device= 'cuda')

Formatted = []
Results = {}


for data_dir in DATA_DIRS:
    CLASSES = CLASS[data_dir]
    Formatted.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
    for model in models_names:
        experiment = Path(data_dir).stem
        EXPERIMENT_NAME = f"{experiment}/{model}"
        print(data_dir, EXPERIMENT_NAME)
        LOCATION = os.path.join(data_dir, model)
        trainer = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=CHECKPOINT_DIR)
        
        dataset_params = {
            'data_dir': LOCATION,
            'train_images_dir':'train/images',
            'train_labels_dir':'train/labels',
            'val_images_dir':'val/images',
            'val_labels_dir':'val/labels',
            'test_images_dir':'test/images',
            'test_labels_dir':'test/labels',
            'classes': CLASSES
        }
        
        test_data = coco_detection_yolo_format_val(
            dataset_params={
                'data_dir': dataset_params['data_dir'],
                'images_dir': dataset_params['test_images_dir'],
                'labels_dir': dataset_params['test_labels_dir'],
                'classes': dataset_params['classes']
            },
            dataloader_params={
                'batch_size': BATCH_SIZE,
                'num_workers': NUM_WORKERS
            }
        )
        
        """## Load trained model"""	
        
        best_model = models.get(
            MODEL_ARCH,
            num_classes=len(dataset_params['classes']),
            #checkpoint_path=f"{CHECKPOINT_DIR}/{EXPERIMENT_NAME}/ckpt_best.pth")
            checkpoint_path=f"{CHECKPOINT_DIR}/Enhanced_SISR/Original/ckpt_best.pth") #always test with the original model⚠️
            #checkpoint_path=f"./checkpoints/coco2017_yolo_nas_l/average_model.pth")
        
        Results[model]=trainer.test(
            model=best_model,
            test_loader=test_data,
            test_metrics_list=[DetectionMetrics(
                include_classwise_ap=True,
                class_names=CLASSES,
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            # ),
            # DetectionMetrics_050(
            #     include_classwise_ap=True,
            #     class_names=CLASSES,
            #     score_thres=0.1,
            #     top_k_predictions=300,
            #     num_cls=len(dataset_params['classes']),
            #     normalize_targets=True,
            #     post_prediction_callback=PPYoloEPostPredictionCallback(
            #         score_threshold=0.01,
            #         nms_top_k=1000,
            #         max_predictions=300,
            #         nms_threshold=0.7
            #     )
            # ),
            # DetectionMetrics_075(
            #     include_classwise_ap=True,
            #     class_names=CLASSES,
            #     score_thres=0.1,
            #     top_k_predictions=300,
            #     num_cls=len(dataset_params['classes']),
            #     normalize_targets=True,
            #     post_prediction_callback=PPYoloEPostPredictionCallback(
            #         score_threshold=0.01,
            #         nms_top_k=1000,
            #         max_predictions=300,
            #         nms_threshold=0.7
            #     )
            # )]
        ])
        temp=[]
        for clas in CLASSES:
            #temp.append(Results[model][f"AP@0.50_{clas}"])
            #temp.append(Results[model][f"AP@0.75_{clas}"])
            temp.append(Results[model][f"AP@0.50:0.95_{clas}"])
        Formatted.append(temp)
final = pd.DataFrame(Formatted)