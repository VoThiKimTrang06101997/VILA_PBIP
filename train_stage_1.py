import argparse
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.pyutils import set_seed, AverageMeter
from utils.trainutils import get_wsss_dataset
from utils.optimizer import PolyWarmupAdamW
from utils.evaluate import ConfusionMatrixAllClass
from utils.contrast_loss import InfoNCELossFG, InfoNCELossBG, diversity_loss_fn
from utils.hierarchical_utils import pair_features, merge_to_parent_predictions, merge_subclass_cams_to_parent, expand_parent_to_subclass_labels
from utils.validate import validate, generate_cam
from transformers import AutoModel, AutoProcessor
from model.model_utils import attention_diversity
from model.model import ClsNetwork
from datasets.bcss import BCSSTrainingDataset, BCSSTestDataset, BCSSWSSSDataset
import logging
import logging.handlers

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('nmslib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser(description="Train WSSS model with unified CAM")
parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
parser.add_argument("--batch-size", type=int, default=2, help="Batch size to reduce memory usage")
args = parser.parse_args()

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now().replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = time_now - time0
    eta = delta * scale
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def train(cfg):
    logger.warning("Initializing training process...")
    torch.backends.cudnn.benchmark = True
    num_workers = min(2, os.cpu_count())
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")
    set_seed(0)

    time0 = datetime.datetime.now().replace(microsecond=0)
    logger.warning("Preparing datasets...")

    transform_train = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.Normalize(mean=[0.66791496, 0.47791372, 0.70623304], std=[0.1736589, 0.22564577, 0.19820057]),
        ToTensorV2(transpose_mask=True),
    ])
    transform_val = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.66791496, 0.47791372, 0.70623304], std=[0.1736589, 0.22564577, 0.19820057]),
        ToTensorV2(transpose_mask=True),
    ])

    try:
        cfg.dataset.val_root = cfg.dataset.root_dir
        train_dataset, val_dataset = get_wsss_dataset(cfg)
    except ValueError as e:
        logger.error(f"Dataset loading failed: {str(e)}")
        return
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty")
        return
    if len(val_dataset) == 0:
        logger.error("Validation dataset is empty")
        return

    logger.warning(f"Full training dataset size: {len(train_dataset)}")
    logger.warning(f"Full validation dataset size: {len(val_dataset)}")

    max_samples = 100
    train_dataset = Subset(train_dataset, range(min(max_samples, len(train_dataset))))
    val_dataset = Subset(val_dataset, range(min(max_samples, len(val_dataset))))
    logger.warning(f"Training with {len(train_dataset)} samples (limited to {max_samples})")
    logger.warning(f"Validating with {len(val_dataset)} samples (limited to {max_samples})")

    # Validate dataset labels
    for dataset, name in [(train_dataset, "training"), (val_dataset, "validation")]:
        for idx in range(len(dataset)):
            img_name, _, cls_label, _ = dataset[idx]
            if not isinstance(cls_label, torch.Tensor) or cls_label.shape != torch.Size([4]):
                logger.error(f"Invalid cls_label shape in {name} dataset at index {idx} ({img_name}): {cls_label.shape if hasattr(cls_label, 'shape') else type(cls_label)}")
                return

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, cfg.train.samples_per_gpu),
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=1,
        persistent_workers=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False
    )

    num_epochs = cfg.train.max_epochs if hasattr(cfg.train, 'max_epochs') else 1
    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = num_epochs * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch
    config_k_list = cfg.model.k_list if hasattr(cfg.model, 'k_list') else [1, 1, 1, 1, 1]  # Match total_classes=5
    if OmegaConf.is_list(config_k_list):
        config_k_list = [int(k) for k in OmegaConf.to_container(config_k_list)]
    logger.warning(f"Config k_list: {config_k_list}")

    l_fea_path = cfg.model.get('l_fea_path', 'default_features')
    logger.warning(f"Using l_fea_path: {l_fea_path}")

    model = ClsNetwork(
        backbone=cfg.model.backbone.config,
        cls_num_classes=cfg.dataset.cls_num_classes,
        stride=cfg.model.backbone.stride,
        pretrained=cfg.train.pretrained,
        n_ratio=cfg.model.n_ratio,
        l_fea_path=l_fea_path,
        text_prompt=cfg.model.text_prompt,
        num_prototypes_per_class=cfg.model.num_prototypes_per_class,
        prototype_feature_dim=cfg.model.prototype_feature_dim,
        k_list=config_k_list  # Pass k_list from config
    )
    model.to(device)

    optimizer = PolyWarmupAdamW(
        params=model.get_param_groups(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    start_iter = 0
    best_fuse234_dice = 0.0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        logger.warning(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.warning("Optimizer state loaded successfully.")
        if 'iter' in checkpoint:
            start_iter = checkpoint['iter'] + 1
            start_epoch = start_iter // iters_per_epoch
            logger.warning(f"Resuming from iteration: {start_iter}, epoch: {start_epoch}")
        if 'best_mIoU' in checkpoint:
            best_fuse234_dice = checkpoint['best_mIoU']
            logger.warning(f"Loaded previous best mIoU: {best_fuse234_dice:.4f}")
    else:
        logger.warning("Starting training from scratch.")

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    loss_function = nn.BCEWithLogitsLoss().to(device)
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(device)

    csv_file = os.path.join(cfg.work_dir.ckpt_dir, "training_metrics.csv")
    csv_headers = ['Epoch', 'mIoU', 'Mean_Dice', 'FwIU', 'Best_mIoU']
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    logger.warning("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        train_loader_iter = iter(train_loader)
        epoch_loss = AverageMeter('loss')

        for n_iter in tqdm(range(iters_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            try:
                img_name, inputs, cls_labels, gt_label = next(train_loader_iter)
                if inputs is None or cls_labels is None:
                    logger.warning(f"Skipping batch with None values at iteration {n_iter}")
                    continue
                if not isinstance(inputs, torch.Tensor) or inputs.dim() != 4 or inputs.shape[1] != 3 or inputs.shape[2:] != (224, 224):
                    logger.error(f"Invalid inputs shape or type at iteration {n_iter}: {inputs.shape if hasattr(inputs, 'shape') else type(inputs)}")
                    continue
                if not isinstance(cls_labels, torch.Tensor) or cls_labels.dim() != 2 or cls_labels.shape[1] != 4:
                    logger.error(f"Invalid cls_labels shape or type at iteration {n_iter}: {cls_labels.shape if hasattr(cls_labels, 'shape') else type(cls_labels)}")
                    continue
            except StopIteration:
                train_loader_iter = iter(train_loader)
                img_name, inputs, cls_labels, gt_label = next(train_loader_iter)
                if inputs is None or cls_labels is None:
                    logger.warning(f"Skipping batch with None values after reset at iteration {n_iter}")
                    continue
                if not isinstance(inputs, torch.Tensor) or inputs.dim() != 4 or inputs.shape[1] != 3 or inputs.shape[2:] != (224, 224):
                    logger.error(f"Invalid inputs shape or type after reset at iteration {n_iter}: {inputs.shape if hasattr(inputs, 'shape') else type(inputs)}")
                    continue
                if not isinstance(cls_labels, torch.Tensor) or cls_labels.dim() != 2 or cls_labels.shape[1] != 4:
                    logger.error(f"Invalid cls_labels shape or type after reset at iteration {n_iter}: {cls_labels.shape if hasattr(cls_labels, 'shape') else type(cls_labels)}")
                    continue

            inputs = inputs.to(device).float()
            cls_labels = torch.clamp(torch.nan_to_num(cls_labels, nan=0.0, posinf=1.0, neginf=0.0), 0, 1).to(device).float()
            logger.debug(f"cls_labels shape: {cls_labels.shape}, cls_labels values: {cls_labels.unique()}")

            with torch.amp.autocast('cuda'):
                try:
                    (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, backbone_loss, fg_bg_prob, k_list, feature_map_for_diversity) = model(inputs, labels=cls_labels, cfg=cfg)
                except ValueError as e:
                    logger.error(f"Error unpacking model output: {str(e)}. Expected 13 values.")
                    continue

                if not k_list or not all(isinstance(k, (int, float)) for k in k_list) or sum(k_list) != 4:  # Changed from !=5 to !=4
                    logger.warning(f"Invalid k_list from model: {k_list}. Falling back to config k_list: {config_k_list}")
                    k_list = config_k_list
                elif k_list != config_k_list:  # Additional check to warn only if lists differ
                    logger.warning(f"k_list from model {k_list} differs from config {config_k_list}. Using config k_list.")
                    k_list = config_k_list
                logger.debug(f"Using k_list for merging: {k_list}")

                cls1_merge = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_train if cfg and hasattr(cfg.train, 'merge_train') else "mean")
                cls2_merge = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_train if cfg and hasattr(cfg.train, 'merge_train') else "mean")
                cls3_merge = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_train if cfg and hasattr(cfg.train, 'merge_train') else "mean")
                cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_train if cfg and hasattr(cfg.train, 'merge_train') else "mean")

                cls1_merge = torch.nan_to_num(cls1_merge.mean(dim=2) if cls1_merge.dim() > 2 else cls1_merge, nan=0.0)
                cls2_merge = torch.nan_to_num(cls2_merge.mean(dim=2) if cls2_merge.dim() > 2 else cls2_merge, nan=0.0)
                cls3_merge = torch.nan_to_num(cls3_merge.mean(dim=2) if cls3_merge.dim() > 2 else cls3_merge, nan=0.0)
                cls4_merge = torch.nan_to_num(cls4_merge.mean(dim=2) if cls4_merge.dim() > 2 else cls4_merge, nan=0.0)
                logger.debug(f"cls1_merge shape: {cls1_merge.shape}, cls_labels shape: {cls_labels.shape}")

                # Slice predictions to exclude background class (use first 4 classes)
                loss1 = loss_function(cls1_merge[:, :4], cls_labels)
                loss2 = loss_function(cls2_merge[:, :4], cls_labels)
                loss3 = loss_function(cls3_merge[:, :4], cls_labels)
                loss4 = loss_function(cls4_merge[:, :4], cls_labels)
                cls_loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4

                if n_iter >= cfg.train.warmup_iters:
                    try:
                        subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
                        cls4_expand = expand_parent_to_subclass_labels(cls4_merge, k_list)
                        if cls4.dim() != cls4_expand.dim() or cls4.size(0) != cls4_expand.size(0) or cls4.size(1) != cls4_expand.size(1):
                            logger.warning(f"Shape mismatch: cls4 {cls4.shape} vs cls4_expand {cls4_expand.shape}. Skipping comparison.")
                            cls4_bir = torch.zeros_like(cls4)
                        else:
                            cls4_bir = (cls4 > cls4_expand).float() * subclass_labels
                        batch_info = model.feature_extractor.process_batch(inputs, cam2, cls_labels)
                        if batch_info and batch_info['fg_img_features'] is not None:
                            fg_features, bg_features = batch_info['fg_img_features'], batch_info['bg_features']
                            set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
                            fg_features, bg_features, fg_pro, bg_pro = set_info['fg_features'], set_info['bg_features'], set_info['fg_text'], set_info['bg_text']
                            fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
                            bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)
                            contrastive_loss = fg_loss + bg_loss
                        else:
                            contrastive_loss = torch.tensor(0.0, device=device)

                        with torch.no_grad():
                            unified_cam_merged = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_train if cfg and hasattr(cfg.train, 'merge_train') else "mean")
                            cam_max, _ = torch.max(unified_cam_merged, dim=1, keepdim=True)
                            background_score = torch.full_like(cam_max, 0.2)
                            full_cam = torch.cat([background_score, unified_cam_merged], dim=1)
                            pseudo_mask = torch.argmax(full_cam, dim=1)
                        pseudo_mask_resized = F.interpolate(pseudo_mask.unsqueeze(1).long(), size=feature_map_for_diversity.shape[2:], mode='nearest').squeeze(1)

                        diversity_loss = diversity_loss_fn(feature_map_for_diversity, l_fea, pseudo_mask_resized)
                        loss = cls_loss + 0.25 * diversity_loss + 0.5 * contrastive_loss
                    except Exception as e:
                        logger.error(f"Error in warm-up phase: {str(e)}. Falling back to cls_loss.")
                        loss = cls_loss
                else:
                    loss = cls_loss

            epoch_loss.update(loss.item(), inputs.size(0))
            torch.cuda.empty_cache()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (n_iter + 1) % 100 == 0:
                delta, eta = cal_eta(time0, (epoch * iters_per_epoch) + n_iter + 1, cfg.train.max_iters)
                cur_lr = optimizer.param_groups[0]['lr']
                torch.cuda.synchronize()
                cls_pred4 = (torch.sigmoid(cls4_merge[:, :4]) > 0.5).float()
                all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().mean() * 100
                avg_cls_acc4 = ((cls_pred4 == cls_labels).float().mean(dim=0)).mean() * 100
                logger.warning(
                    f"Epoch {epoch+1}/{num_epochs}, Iter: {n_iter + 1}/{iters_per_epoch}; "
                    f"Elapsed: {delta}; ETA: {eta}; "
                    f"LR: {cur_lr:.3e}; Loss: {loss.item():.4f}; "
                    f"Acc4: {all_cls_acc4:.2f}/{avg_cls_acc4:.2f}"
                )

        try:
            val_mIoU, val_mean_dice, val_fw_iu, val_iu_per_class, val_dice_per_class = validate(
                model=model, data_loader=val_loader, cfg=cfg, cls_loss_func=loss_function
            )
            logger.warning(f"Epoch {epoch+1}/{num_epochs} Validation results (on {max_samples} samples):")
            logger.warning(f"Val mIoU: {val_mIoU:.4f}")
            logger.warning(f"Val Mean Dice: {val_mean_dice:.4f}")
            logger.warning(f"Val FwIU: {val_fw_iu:.4f}")
            current_miou = val_mIoU
            logger.warning(f"mIOU (for saving): {current_miou:.4f}")
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}. Skipping validation for this epoch.")
            val_mIoU, val_mean_dice, val_fw_iu, val_iu_per_class, val_dice_per_class = 0.0, 0.0, 0.0, [], []
            current_miou = 0.0

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, val_mIoU, val_mean_dice, val_fw_iu, best_fuse234_dice])

        save_path = os.path.join(cfg.work_dir.ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
        os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
        torch.save({
            "cfg": cfg,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mIoU": best_fuse234_dice
        }, save_path, _use_new_zipfile_serialization=True)
        logger.warning(f"Saved checkpoint to: {save_path}")

        if (epoch + 1) > (cfg.train.warmup_iters // iters_per_epoch + cfg.train.eval_iters // iters_per_epoch):
            if current_miou > best_fuse234_dice:
                best_fuse234_dice = current_miou
                best_save_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
                os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
                torch.save({
                    "cfg": cfg,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_mIoU": best_fuse234_dice
                }, best_save_path, _use_new_zipfile_serialization=True)
                logger.warning(f"Saved best model with mIoU: {best_fuse234_dice:.4f}")
        else:
            logger.warning(f"--- In warm-up or grace period (epoch: {epoch + 1}). Skipping best model check. ---")

    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - start_time
    logger.warning(f"Total training time: {total_training_time}")

    logger.warning("\n" + "="*80)
    logger.warning("POST-TRAINING EVALUATION AND CAM GENERATION")
    logger.warning("="*80)
    logger.warning("Preparing test dataset...")
    transform_test = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.66791496, 0.47791372, 0.70623304], std=[0.1736589, 0.22564577, 0.19820057]),
        ToTensorV2(transpose_mask=True),
    ])
    test_dataset, _ = get_wsss_dataset(cfg)
    logger.warning(f"Full test dataset size: {len(test_dataset)}")
    max_samples = 100
    test_dataset = Subset(test_dataset, range(min(max_samples, len(test_dataset))))
    logger.warning(f"Testing with {len(test_dataset)} samples (limited to {max_samples})")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False
    )
    logger.warning("1. Testing on test dataset...")
    logger.warning("-" * 100)

    test_mIoU, test_mean_dice, test_fw_iu, test_iu_per_class, test_dice_per_class = validate(
        model=model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function
    )
    logger.warning("Testing results:")
    logger.warning(f"Test mIoU: {test_mIoU:.4f}")
    logger.warning(f"Test Mean Dice: {test_mean_dice:.4f}")
    logger.warning(f"Test FwIU: {test_fw_iu:.4f}")
    logger.warning("\nPer-class IoU scores (FG classes + BG):")
    for i, score in enumerate(test_iu_per_class):
        label = f"Class {i}" if i < len(test_iu_per_class) - 1 else "Background"
        logger.warning(f" {label}: {score*100:.4f}")
    logger.warning("\nPer-class Dice scores (FG classes + BG):")
    for i, score in enumerate(test_dice_per_class):
        label = f"Class {i}" if i < len(test_dice_per_class) - 1 else "Background"
        logger.warning(f" {label}: {score*100:.4f}")

    logger.warning("2. Generating unified CAMs for 100 trained samples...")
    logger.warning("-" * 100)

    train_cam_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False
    )
    logger.warning(f"Generating unified CAMs for {len(train_dataset)} trained samples (max 100)...")
    results_dir = os.path.join(cfg.work_dir.dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    logger.warning(f"Output directory: {results_dir}")

    best_model_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
    if os.path.exists(best_model_path):
        logger.warning(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_iter = checkpoint.get("iter", "unknown")
        logger.warning(f"✓ Best model loaded successfully! (Saved at iteration: {best_iter})")
    else:
        logger.warning("⚠ Warning: Best model checkpoint not found, using current model state")
        logger.warning(f"Expected path: {best_model_path}")

    cam_heatmap_dir = os.path.join(results_dir, "CAM_heatmap")
    binary_mask_dir = os.path.join(results_dir, "binary_mask")
    os.makedirs(cam_heatmap_dir, exist_ok=True)
    os.makedirs(binary_mask_dir, exist_ok=True)

    generate_cam(model=model, data_loader=train_cam_loader, cfg=cfg)

    logger.warning("\nFiles generated:")
    logger.warning(f" • Training unified CAM visualizations: {results_dir}/CAM_heatmap/*.pth")
    logger.warning(f" • Unified binary masks: {results_dir}/binary_mask/*.pth")
    logger.warning(f" • Model checkpoint: {cfg.work_dir.ckpt_dir}/best_cam.pth")
    logger.warning("="*80)

if __name__ == "__main__":
    cfg = OmegaConf.load(args.config)
    cfg.work_dir.dir = os.path.dirname(args.config)
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    logger.warning(f'Args: {args}')
    logger.warning(f'Configs: {cfg}')
    set_seed(0)
    train(cfg=cfg)