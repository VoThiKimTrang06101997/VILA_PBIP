import argparse
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
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
import random  # Added for full seeding

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('nmslib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser(description="Train WSSS model with unified CAM")
parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
parser.add_argument("--batch-size", type=int, default=16, help="Batch size (increase for full data)")
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
    num_workers = min(8, os.cpu_count())  # Increase workers for full data
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")
    set_seed(0)

    time0 = datetime.datetime.now().replace(microsecond=0)
    logger.warning("Preparing datasets...")

    # Chú thích ko lật ngược rotate ảnh và chỉnh độ sáng ảnh gì hết
    transform_train = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.66791496, 0.47791372, 0.70623304], std=[0.1736589, 0.22564577, 0.19815057]),
        ToTensorV2(transpose_mask=True),
    ])
    transform_val = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.66791496, 0.47791372, 0.70623304], std=[0.1736589, 0.22564577, 0.19815057]),
        ToTensorV2(transpose_mask=True),
    ])

    try:
        cfg.dataset.val_root = cfg.dataset.root_dir
        train_dataset, val_dataset = get_wsss_dataset(cfg)
    except ValueError as e:
        logger.error(f"Dataset loading failed: {str(e)}")
        return

    # Removed max_samples to train on full dataset
    logger.warning(f"Training with FULL dataset: {len(train_dataset)} samples")
    logger.warning(f"Validating with FULL dataset: {len(val_dataset)} samples")

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
    config_k_list = cfg.model.k_list if hasattr(cfg.model, 'k_list') else [1, 1, 1, 1]
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
        k_list=config_k_list
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
        logger.warning(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'iter' in checkpoint:
            start_iter = checkpoint['iter'] + 1
            start_epoch = start_iter // iters_per_epoch
        if 'best_mIoU' in checkpoint:
            best_fuse234_dice = checkpoint['best_mIoU']

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    class_weights = torch.tensor([1.0, 1.0, 2.0, 1.5], device=device)  # Boost class 2 (LYM) and 3 (NEC)
    loss_function = nn.BCEWithLogitsLoss(weight=class_weights).to(device)
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(device)

    # === 2 file CSV ===
    csv_train = os.path.join(cfg.work_dir.ckpt_dir, "training_metrics.csv")
    csv_each_class = os.path.join(cfg.work_dir.ckpt_dir, "compute_each_classes.csv")

    # Header cho file training_metrics.csv (giữ nguyên như cũ)
    if not os.path.exists(csv_train):
        with open(csv_train, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'mIoU', 'Mean_Dice', 'FwIU', 'Best_mIoU'])

    # Header cho file compute_each_classes.csv
    if not os.path.exists(csv_each_class):
        with open(csv_each_class, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Epoch', 'Dataset', 'mIoU', 'Mean_Dice', 'FwIU',
                'TUM_IoU', 'STR_IoU', 'LYM_IoU', 'NEC_IoU',
                'TUM_Dice', 'STR_Dice', 'LYM_Dice', 'NEC_Dice'
            ])

    logger.warning("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        train_loader_iter = iter(train_loader)
        epoch_loss = AverageMeter('loss')

        for n_iter in tqdm(range(iters_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            try:
                img_name, inputs, cls_labels, gt_label = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                img_name, inputs, cls_labels, gt_label = next(train_loader_iter)

            if inputs is None or cls_labels is None:
                continue

            inputs = inputs.to(device).float()
            cls_labels = torch.clamp(torch.nan_to_num(cls_labels, nan=0.0, posinf=1.0, neginf=0.0), 0, 1).to(device).float()

            with torch.amp.autocast('cuda'):
                try:
                    (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, backbone_loss, fg_bg_prob, k_list, feature_map_for_diversity) = model(inputs, labels=cls_labels, cfg=cfg)
                except Exception as e:
                    logger.error(f"Model forward error: {e}")
                    continue

                k_list = config_k_list  # Dùng config để ổn định

                cls1_merge = merge_to_parent_predictions(cls1, k_list, method="mean")
                cls2_merge = merge_to_parent_predictions(cls2, k_list, method="mean")
                cls3_merge = merge_to_parent_predictions(cls3, k_list, method="mean")
                cls4_merge = merge_to_parent_predictions(cls4, k_list, method="mean")

                loss1 = loss_function(cls1_merge[:, :4], cls_labels)
                loss2 = loss_function(cls2_merge[:, :4], cls_labels)
                loss3 = loss_function(cls3_merge[:, :4], cls_labels)
                loss4 = loss_function(cls4_merge[:, :4], cls_labels)
                cls_loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4

                if n_iter >= cfg.train.warmup_iters:
                    try:
                        subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
                        cls4_expand = expand_parent_to_subclass_labels(cls4_merge, k_list)
                        batch_info = model.feature_extractor.process_batch(inputs, cam2, cls_labels)
                        if batch_info and batch_info['fg_img_features'] is not None:
                            fg_features, bg_features = batch_info['fg_img_features'], batch_info['bg_features']
                            set_info = pair_features(fg_features, bg_features, l_fea, cls_labels)
                            fg_features, bg_features, fg_pro, bg_pro = set_info['fg_features'], set_info['bg_features'], set_info['fg_text'], set_info['bg_text']
                            fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
                            bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)
                            contrastive_loss = fg_loss + bg_loss
                        else:
                            contrastive_loss = torch.tensor(0.0, device=device)

                        with torch.no_grad():
                            unified_cam_merged = merge_subclass_cams_to_parent(cam1, k_list, method="mean")
                            cam_max, _ = torch.max(unified_cam_merged, dim=1, keepdim=True)
                            background_score = torch.full_like(cam_max, 0.2)
                            full_cam = torch.cat([background_score, unified_cam_merged], dim=1)
                            pseudo_mask = torch.argmax(full_cam, dim=1)
                        pseudo_mask_resized = F.interpolate(pseudo_mask.unsqueeze(1).long(), size=feature_map_for_diversity.shape[2:], mode='nearest').squeeze(1)
                        diversity_loss = diversity_loss_fn(feature_map_for_diversity, l_fea, pseudo_mask_resized)
                        loss = cls_loss + 0.25 * diversity_loss + 0.5 * contrastive_loss
                    except Exception as e:
                        logger.error(f"Error in warm-up phase: {e}. Falling back to cls_loss.")
                        loss = cls_loss
                else:
                    loss = cls_loss

            epoch_loss.update(loss.item(), inputs.size(0))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # === Validation ===
        try:
            val_mIoU, val_mean_dice, val_fw_iu, val_iu_per_class, val_dice_per_class = validate(
                model=model, data_loader=val_loader, cfg=cfg, cls_loss_func=loss_function
            )
            current_miou = val_mIoU
        except Exception as e:
            logger.error(f"Validation failed: {e}. Skipping.")
            val_mIoU, val_mean_dice, val_fw_iu = 0.0, 0.0, 0.0
            val_iu_per_class = [0.0]*4
            val_dice_per_class = [0.0]*4
            current_miou = 0.0

        # Ghi vào training_metrics.csv
        with open(csv_train, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, val_mIoU, val_mean_dice, val_fw_iu, best_fuse234_dice])

        # Ghi vào compute_each_classes.csv (val set)
        with open(csv_each_class, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 'val',
                f"{val_mIoU:.4f}", f"{val_mean_dice:.4f}", f"{val_fw_iu:.4f}",
                f"{val_iu_per_class[0]*100:.4f}", f"{val_iu_per_class[1]*100:.4f}",
                f"{val_iu_per_class[2]*100:.4f}", f"{val_iu_per_class[3]*100:.4f}",
                f"{val_dice_per_class[0]*100:.4f}", f"{val_dice_per_class[1]*100:.4f}",
                f"{val_dice_per_class[2]*100:.4f}", f"{val_dice_per_class[3]*100:.4f}"
            ])

        # Save checkpoint + best model
        save_path = os.path.join(cfg.work_dir.ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            "cfg": cfg,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mIoU": best_fuse234_dice
        }, save_path)

        if current_miou > best_fuse234_dice:
            best_fuse234_dice = current_miou
            best_save_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
            torch.save({
                "cfg": cfg,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_fuse234_dice
            }, best_save_path)
            logger.warning(f"New best mIoU: {best_fuse234_dice:.4f}")

    # ===================================================================
    # POST-TRAINING: Test trên folder training + ghi chi tiết vào CSV
    # ===================================================================
    logger.warning("\n" + "="*80)
    logger.warning("POST-TRAINING EVALUATION ON TRAINING FOLDER")
    logger.warning("="*80)

    train_test_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_mIoU, test_mean_dice, test_fw_iu, test_iu_per_class, test_dice_per_class = validate(
        model=model, data_loader=train_test_loader, cfg=cfg, cls_loss_func=loss_function
    )

    logger.warning("Testing results on training folder:")
    logger.warning(f"Test mIoU: {test_mIoU:.4f}")
    logger.warning(f"Test Mean Dice: {test_mean_dice:.4f}")
    logger.warning(f"Test FwIU: {test_fw_iu:.4f}")
    logger.warning("\nPer-class IoU scores:")
    for i, score in enumerate(test_iu_per_class):
        label = f"Class {i} (Tumor):" if i == 0 else f"Class {i} (Stroma):" if i == 1 else f"Class {i} (Lymphocyte):" if i == 2 else f"Class {i} (Necrosis):"
        logger.warning(f" {label} {score*100:.4f}")
    logger.warning("\nPer-class Dice scores:")
    for i, score in enumerate(test_dice_per_class):
        label = f"Class {i} (Tumor):" if i == 0 else f"Class {i} (Stroma):" if i == 1 else f"Class {i} (Lymphocyte):" if i == 2 else f"Class {i} (Necrosis):"
        logger.warning(f" {label} {score*100:.4f}")

    # Ghi kết quả test vào compute_each_classes.csv
    with open(csv_each_class, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "FINAL", 'train_folder',
            f"{test_mIoU:.4f}", f"{test_mean_dice:.4f}", f"{test_fw_iu:.4f}",
            f"{test_iu_per_class[0]*100:.4f}", f"{test_iu_per_class[1]*100:.4f}",
            f"{test_iu_per_class[2]*100:.4f}", f"{test_iu_per_class[3]*100:.4f}",
            f"{test_dice_per_class[0]*100:.4f}", f"{test_dice_per_class[1]*100:.4f}",
            f"{test_dice_per_class[2]*100:.4f}", f"{test_dice_per_class[3]*100:.4f}"
        ])

    # Generate CAM
    generate_cam(model=model, data_loader=train_test_loader, cfg=cfg)

    logger.warning("Training và evaluation hoàn tất!")
    logger.warning(f"File CSV chi tiết: {csv_each_class}")
    logger.warning(f"File CSV tổng hợp: {csv_train}")

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
    