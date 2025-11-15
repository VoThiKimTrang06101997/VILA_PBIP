import os
import torch.distributed as dist
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.bcss import BCSSTrainingDataset, BCSSTestDataset, BCSSWSSSDataset
from medclip import MedCLIPModel, MedCLIPProcessor
import logging

logger = logging.getLogger(__name__)

def get_wsss_dataset(cfg):
    MEAN, STD = get_mean_std(cfg.dataset.name)
    mask_root = cfg.dataset.get('mask_root', 'mask')  # Default to 'mask' if not specified
    train_img_dir = cfg.dataset.train_root  # Direct path to training images
    val_img_dir = cfg.dataset.val_root  # Use val_root as the base, let BCSSTestDataset handle subdirectories

    # Check if directories exist
    if not os.path.exists(train_img_dir):
        logger.error(f"Training image directory {train_img_dir} does not exist. Please check config.dataset.train_root.")
        raise ValueError(f"Training image directory {train_img_dir} not found.")
    if not os.path.exists(val_img_dir):
        logger.error(f"Validation image directory {val_img_dir} does not exist. Please check config.dataset.val_root.")
        raise ValueError(f"Validation image directory {val_img_dir} not found.")

    transform = {
        "train": A.Compose([
            A.Normalize(MEAN, STD),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(),
            ToTensorV2(transpose_mask=True),
        ]),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ])
    }
    try:
        # Training dataset uses direct image directory
        train_dataset = BCSSTrainingDataset(train_img_dir, transform=transform["train"])
        # Validation dataset uses val_root as base, split="val" handles subdirectories
        val_dataset = BCSSTestDataset(val_img_dir, split="val", transform=transform["val"])
    except Exception as e:
        logger.error(f"Failed to load WSSS dataset: {str(e)}")
        raise
    return train_dataset, val_dataset

def get_cls_dataset(cfg, split="val", p=0.5, enable_rotation=True):
    MEAN, STD = get_mean_std(cfg.dataset.name)
    
    train_transforms = [
        A.Normalize(MEAN, STD),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
    ]
    
    if enable_rotation:
        train_transforms.append(A.RandomRotate90())
    
    train_transforms.append(ToTensorV2(transpose_mask=True))
    
    transform = {
        "train": A.Compose(train_transforms),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ]),
    }
    train_dataset = BCSSTrainingDataset(cfg.dataset.train_root, transform=transform["train"])
    val_dataset = BCSSTestDataset(cfg.dataset.val_root, split, transform=transform["val"])
    return train_dataset, val_dataset

def get_mean_std(dataset):
    norm = [[0.66791496, 0.47791372, 0.70623304], [0.1736589, 0.22564577, 0.19820057]]
    return norm[0], norm[1]

def all_reduced(x, n_gpus):
    dist.all_reduce(x)
    x /= n_gpus
    