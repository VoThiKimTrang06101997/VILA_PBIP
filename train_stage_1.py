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
import random  # Added for full seeding

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   
os.environ["TORCH_USE_CUDA_DSA"] = "1"     

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


# HÀM SIÊU AN TOÀN – DÙNG CHO TẤT CẢ CLS_LABELS
def safe_cls_labels(labels):
    if labels is None:
        return None
    # Bắt buộc: chỉ được 0 hoặc 1
    labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
    labels = torch.clamp(labels, 0.0, 1.0)
    return labels

def train(cfg):
    logger.warning("Initializing training process...")
    torch.backends.cudnn.benchmark = True
    num_workers = min(1, os.cpu_count())  # Increase workers for full data
    # device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # logger.warning(f"Using device: {device}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.warning(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found! Running on CPU (extremely slow)")
    set_seed(0)

    time0 = datetime.datetime.now().replace(microsecond=0)
    logger.warning("Preparing datasets...")
    
    # Thêm dòng này để thấy progress trên Colab
    from tqdm import tqdm
    tqdm._instances.clear()  # Fix lỗi tqdm bị treo trên Colab

    try:
        cfg.dataset.val_root = cfg.dataset.root_dir
        train_dataset, val_dataset = get_wsss_dataset(cfg)
    except ValueError as e:
        logger.error(f"Dataset loading failed: {str(e)}")
        return

    # Thêm log để biết dataset có bị rỗng không
    if len(train_dataset) == 0:
        logger.error("TRAIN DATASET IS EMPTY! Check your data path and filename format [class_labels]")
        raise ValueError("No training samples found!")

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

    # max_samples = 200
    # train_dataset = Subset(train_dataset, range(min(max_samples, len(train_dataset))))
    # val_dataset = Subset(val_dataset, range(min(max_samples, len(val_dataset))))
    
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
    ).to(device).float()  

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

    class_weights = torch.tensor([1.0, 1.0, 2.0, 1.5], device=device, dtype=torch.float32)  # Boost class 2 (LYM) and 3 (NEC)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device) 
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
    
    for epoch in range(start_epoch, cfg.train.max_epochs):
        epoch_loss = AverageMeter('loss')
        step = 0

        for img_name, inputs, cls_labels_raw, gt_label in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.train.max_epochs}",
            leave=False,
            ncols=100
        ):
            step += 1
            global_step = epoch * len(train_loader) + step

            inputs = inputs.to(device, non_blocking=True)

            # =====================================================
            # FIX 1: LÀM SẠCH CLS_LABELS – SIÊU SIÊU AN TOÀN
            # =====================================================
            cls_labels = cls_labels_raw.float()
            cls_labels = torch.nan_to_num(cls_labels, nan=0.0, posinf=0.0, neginf=0.0)
            cls_labels = torch.clamp(cls_labels, 0.0, 1.0)
            cls_labels = cls_labels.to(device, non_blocking=True)

            # Binary version + CHẮN CHẮN KHÔNG ĐƯỢC >1
            cls_labels_binary = (cls_labels >= 0.5).float()
            cls_labels_binary = torch.clamp(cls_labels_binary, 0.0, 1.0)  # BẮT BUỘC

            # Làm sạch gt_label
            if gt_label is not None and gt_label.numel() > 0:
                gt_label = torch.where(gt_label == 255, torch.zeros_like(gt_label), gt_label)
                gt_label = torch.where(gt_label >= 4, torch.zeros_like(gt_label), gt_label)
                gt_label = torch.clamp(gt_label, 0, 3).long().to(device)

            optimizer.zero_grad(set_to_none=True)

            try:
                outputs = model(inputs, labels=cls_labels, cfg=cfg)
                (
                    cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4,
                    l_fea, backbone_loss, fg_bg_prob, k_list, feature_map_for_diversity
                ) = outputs

                # Clamp tất cả CAM về [0,1]
                cam1 = torch.clamp(cam1[:, :4], 0.0, 1.0)
                cam2 = torch.clamp(cam2[:, :4], 0.0, 1.0)
                cam3 = torch.clamp(cam3[:, :4], 0.0, 1.0)
                cam4 = torch.clamp(cam4[:, :4], 0.0, 1.0)

                # Merge
                from utils.hierarchical_utils import merge_to_parent_predictions
                cls1_merge = merge_to_parent_predictions(cls1, config_k_list, "mean")[:, :4]
                cls2_merge = merge_to_parent_predictions(cls2, config_k_list, "mean")[:, :4]
                cls3_merge = merge_to_parent_predictions(cls3, config_k_list, "mean")[:, :4]
                cls4_merge = merge_to_parent_predictions(cls4, config_k_list, "mean")[:, :4]

                # Loss
                l1 = loss_function(cls1_merge, cls_labels)
                l2 = loss_function(cls2_merge, cls_labels)
                l3 = loss_function(cls3_merge, cls_labels)
                l4 = loss_function(cls4_merge, cls_labels)
                cls_loss = cfg.train.l1 * l1 + cfg.train.l2 * l2 + cfg.train.l3 * l3 + cfg.train.l4 * l4
                total_loss = cls_loss

                # =====================================================
                # CONTRASTIVE + DIVERSITY: 1 cách bị lỗi Cuda Assertion 
                # =====================================================
                # if global_step > getattr(cfg.train, 'warmup_iters', 1000):
                #     try:
                #         # BƯỚC 1: CAM AN TOÀN
                #         cam2_safe = torch.clamp(cam2[:, :4], 0.0, 1.0)  # [B,4,H,W]

                #         # BƯỚC 2: CLS_LABELS_BINARY SIÊU SẠCH
                #         cls_labels_clean = torch.nan_to_num(cls_labels, nan=0.0, posinf=0.0, neginf=0.0)
                #         cls_labels_clean = torch.clamp(cls_labels_clean, 0.0, 1.0)
                #         cls_labels_binary = (cls_labels_clean >= 0.5).float()  # CHỈ 0 HOẶC 1

                #         batch_info = model.feature_extractor.process_batch(
                #             inputs, cam2_safe, cls_labels_binary
                #         )

                #         if (batch_info and 
                #             batch_info.get('fg_img_features') is not None and 
                #             batch_info['fg_img_features'].numel() > 0 and
                #             batch_info.get('bg_features') is not None and
                #             batch_info['bg_features'].numel() > 0):

                #             fg_f = batch_info['fg_img_features']   # [B, C]
                #             bg_f = batch_info['bg_features']       # [B, C]

                #             # FIX CUỐI CÙNG: KHÔNG DÙNG INDEX, DÙNG GATHER + MASK
                #             B, C = fg_f.shape
                #             device = fg_f.device

                #             # l_fea: [4, 512] → mở rộng thành [B, 4, 512]
                #             l_fea_exp = l_fea.unsqueeze(0).expand(B, -1, -1)  # [B,4,512]

                #             # Tạo index hợp lệ từ 0 đến 3
                #             class_indices = torch.arange(4, device=device).view(1, 4, 1)  # [1,4,1]
                #             batch_indices = torch.arange(B, device=device).view(B, 1, 1)   # [B,1,1]

                #             # Mask: [B,4,1] → chỉ lấy class hiện diện
                #             mask = cls_labels_binary.unsqueeze(2)  # [B,4,1]

                #             # Dùng gather để lấy prototype an toàn
                #             selected_proto = l_fea_exp.gather(1, class_indices.expand(B, 4, C) * mask)  # [B,4,512] chỉ giữ class có mặt

                #             # Trung bình các prototype có mặt
                #             fg_text = selected_proto.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B,512]
                #             bg_text = fg_text.clone()

                #             # Fallback nếu không có class nào (rất hiếm)
                #             no_class = (mask.sum(dim=1) == 0).squeeze(-1)
                #             if no_class.any():
                #                 mean_proto = l_fea.mean(dim=0, keepdim=True)
                #                 fg_text[no_class] = mean_proto
                #                 bg_text[no_class] = mean_proto

                #             # Normalize
                #             fg_text = F.normalize(fg_text, dim=-1)
                #             bg_text = F.normalize(bg_text, dim=-1)

                #             # Expand cho InfoNCE
                #             fg_text = fg_text.unsqueeze(1)
                #             bg_text = bg_text.unsqueeze(1)

                #             # Loss
                #             fg_loss = InfoNCELossFG(temperature=0.07)(fg_f, fg_text, bg_text)
                #             bg_loss = InfoNCELossBG(temperature=0.07)(bg_f, fg_text, bg_text)
                #             contrast_loss = fg_loss + bg_loss

                #             # Diversity loss
                #             div_loss = attention_diversity(
                #                 model.prototype_to_diversity(l_fea),
                #                 feature_map_for_diversity,
                #                 num_heads=8
                #             )

                #             total_loss = cls_loss + 0.25 * div_loss + 0.5 * contrast_loss

                #     except Exception as e:
                #         logger.debug(f"Step {global_step}: Contrastive skipped → {e}")
                #         total_loss = cls_loss
                
                
                # === THAY ĐOẠN DIVERSITY BẰNG ĐOẠN NÀY – VẪN DÙNG DIVERSITY LOSS: : 1 cách bị lỗi Cuda Assertion  ===
                # if global_step > getattr(cfg.train, 'warmup_iters', 1000):
                #     try:
                #         # BƯỚC 1: LÀM SẠCH l_fea TRƯỚC KHI DÙNG
                #         l_fea_clean = torch.nan_to_num(l_fea, nan=0.0, posinf=0.0, neginf=0.0)
                #         l_fea_clean = F.normalize(l_fea_clean, dim=-1)

                #         # BƯỚC 2: LÀM SẠCH feature_map_for_diversity
                #         feat_clean = torch.nan_to_num(feature_map_for_diversity, nan=0.0, posinf=0.0, neginf=0.0)
                #         feat_clean = F.normalize(feat_clean.flatten(1), dim=-1)  # [B, N*D]

                #         # BƯỚC 3: DÙNG DIVERSITY LOSS AN TOÀN
                #         div_loss = attention_diversity(
                #             model.prototype_to_diversity(l_fea_clean),
                #             feat_clean,
                #             num_heads=8
                #         )

                #         total_loss = cls_loss + 0.25 * div_loss  # VẪN DÙNG DIVERSITY!

                #     except Exception as e:
                #         logger.debug(f"Step {global_step}: Diversity skipped → {e}")
                #         total_loss = cls_loss
                # else:
                #     total_loss = cls_loss
                
                
                # === TKo dùng Diversity Loss + Constractive Loss luôn ===
                if global_step > getattr(cfg.train, 'warmup_iters', 1000):
                    try:
                        # TẮT HOÀN TOÀN DIVERSITY LOSS – CHỈ DÙNG CLS LOSS
                        total_loss = cls_loss  

                    except Exception as e:
                        logger.debug(f"Step {global_step}: All extra losses skipped → {e}")
                        total_loss = cls_loss
                else:
                    total_loss = cls_loss

                # if global_step > getattr(cfg.train, 'warmup_iters', 1000):
                #     try:
                #         # BƯỚC 1: LÀM SẠCH l_fea
                #         l_fea_clean = torch.nan_to_num(l_fea, nan=0.0)
                #         l_fea_clean = F.normalize(l_fea_clean, dim=-1)  # [4, D]

                #         # BƯỚC 2: DIVERSITY LOSS 
                #         # Tính cosine similarity giữa các prototype
                #         sim_matrix = torch.mm(l_fea_clean, l_fea_clean.t())  # [4,4]
                #         sim_matrix = sim_matrix.fill_diagonal_(0.0)         # bỏ tự giống
                        
                #         # Loss = trung bình cosine similarity → càng nhỏ càng đa dạng
                #         diversity_loss = sim_matrix.mean()
                        
                #         # Maximize diversity → trả về -diversity_loss
                #         total_loss = cls_loss - 0.25 * diversity_loss

                #     except Exception as e:
                #         logger.debug(f"Step {global_step}: Diversity skipped → {e}")
                #         total_loss = cls_loss
                # else:
                #     total_loss = cls_loss

                total_loss.backward()
                optimizer.step()
                epoch_loss.update(total_loss.item(), inputs.size(0))

            except Exception as e:
                logger.error(f"STEP {global_step} SKIPPED → {e}")
                torch.cuda.empty_cache()
                continue

    
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
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)  
    
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


