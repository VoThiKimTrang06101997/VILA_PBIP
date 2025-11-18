import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import logging
from utils.evaluate import ConfusionMatrixAllClass
from utils.pyutils import AverageMeter
from utils.hierarchical_utils import merge_subclass_cams_to_parent, merge_to_parent_predictions
from utils.crf import DenseCRF
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import ttach as tta
from PIL import Image
import torch.cuda.amp as amp
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold

logger = logging.getLogger(__name__)

# ===================================================================
# 1. get_seg_label: Tạo CAM hoàn chỉnh + pseudo-label từ 1 CAM thô
# ===================================================================
def get_seg_label(cams, inputs, cls_label, labels=None, feature_extractor=None):
    """
    Input:
        cams: [B, C, H, W] – CAM thô (có thể 4 hoặc 5 kênh)
        inputs: ảnh gốc
        cls_label: [B, 4] – nhãn class-level
    Output:
        cam_all: [B, 5, H, W] – CAM đã có background
        pred_labels: [B, H, W] – pseudo-label
    """
    with torch.no_grad():
        b, c, h, w = inputs.shape
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0) # Loại bỏ giá trị âm

        if cams.shape[1] == 0:
            logger.error(f"Empty cams channels: {cams.shape}. Returning default labels.")
            pred_labels = torch.zeros((b, h, w), dtype=torch.long)
            cam_all = torch.zeros((b, 5, h, w))
            return cam_all, pred_labels

        # Class-wise normalization
        cams_norm = np.zeros_like(cams)
        cls_label_np = cls_label.cpu().data.numpy()
        if cls_label_np.shape[1] != 4:
            logger.warning(f"cls_label channels ({cls_label_np.shape[1]}) should be 4. Adjusting.")
            cls_label_np = np.pad(cls_label_np, ((0, 0), (0, 4 - cls_label_np.shape[1])), mode='constant')
        num_classes = cls_label_np.shape[1]  # 4 foreground classes

        for i in range(b):
            for j in range(num_classes):
                if cls_label_np[i, j] > 0:  # Chỉ normalize class hiện diện
                    c_min = np.min(cams[i, j])
                    c_max = np.max(cams[i, j])
                    if c_max > c_min:
                        cams_norm[i, j] = (cams[i, j] - c_min) / (c_max - c_min)
                    else:
                        cams_norm[i, j] = cams[i, j]
                else:
                    cams_norm[i, j] = 0

        cams = cams_norm

        # ------------------- Thêm background channel -------------------
        bg_label = 1 - np.any(cls_label_np > 0, axis=1, keepdims=True)  # [B,1]
        cls_label_expanded = np.hstack((cls_label_np, np.ones((cls_label_np.shape[0], 1))))  # Include background
        logger.debug(f"cam.shape: {cams.shape}, cls_label_expanded.shape: {cls_label_expanded.shape}")

        if cams.shape[1] not in [4, 5]:
            logger.error(f"Unexpected cams channels: {cams.shape[1]}. Expected 4 or 5. Padding to 5.")
            cams = np.pad(cams, ((0, 0), (0, 5 - cams.shape[1]), (0, 0), (0, 0)), mode='constant')
        elif cams.shape[1] == 4:
            logger.debug(f"Expanding cams from {cams.shape} to include background channel")
            bg_channel = np.zeros((b, 1, cams.shape[2], cams.shape[3]))
            cams = np.concatenate([cams, bg_channel], axis=1)

        # Chỉ giữ lại class có nhãn
        cams = cams * cls_label_expanded[:, :, None, None]

        cams = torch.from_numpy(cams).float().to(inputs.device)
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        
        # Thêm background mạnh hơn: (1 - max_fg)^2
        cam_max = torch.max(cams[:, :-1], dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 2
        cam_all = torch.cat([cams[:, :-1], bg_cam], dim=1)

        pred_labels = torch.argmax(cam_all, dim=1).clamp(0, 4)

        # Optional: FG/BG feature extraction (dùng trong training)
        if feature_extractor and cls_label is not None:
            batch_info = feature_extractor.process_batch(inputs, cam_all, cls_label)
            if batch_info and batch_info['fg_img_features'] is not None:
                logger.debug(f"ResNet FG features shape: {batch_info['fg_img_features'].shape}")
                logger.debug(f"ResNet BG features shape: {batch_info['bg_features'].shape}")
                
        # Xử lý ground truth label nếu có
        if labels is not None:
            if labels.dim() <= 1:
                logger.warning(f"Invalid labels dimension: {labels.dim()}. Reshaping to [batch_size, height, width].")
                labels = torch.zeros(b, h, w, dtype=torch.long, device='cuda') # default background
                for i in range(b):
                    active_classes = cls_label[i] > 0
                    class_indices = torch.where(active_classes)[0].cpu().numpy()
                    if len(class_indices) > 0:
                        num_active = len(class_indices)
                        section_size = h // num_active if num_active > 0 else h
                        label_map = torch.zeros(h, w, device='cuda')
                        for idx, c in enumerate(class_indices):
                            if c >= num_classes:
                                logger.warning(f"Index {c} out of bounds for num_classes {num_classes}. Skipping.")
                                continue
                            start = idx * section_size
                            end = (idx + 1) * section_size if idx < num_active - 1 else h
                            label_map[start:end, :] = c
                        # Assign background where no foreground
                        label_map[torch.sum(label_map, dim=0) == 0] = 4
                        labels[i] = label_map.long()
                    else:
                        labels[i] = torch.full((h, w), 4, device='cuda', dtype=torch.long)
            elif labels.dim() == 4:
                if labels.size(1) == 1:
                    labels = labels.squeeze(1).long()
                else:
                    labels = labels.argmax(dim=1).long()
            elif labels.dim() == 3:
                labels = labels.long()
            elif labels.dim() == 2:
                labels = labels.unsqueeze(0) if b == 1 else labels
                if labels.shape[0] != b:
                    logger.error(f"Invalid labels shape after unsqueeze: {labels.shape}. Using default zero labels.")
                    labels = torch.zeros(b, h, w, dtype=torch.long, device='cuda')
            else:
                logger.error(f"Invalid labels shape: {labels.shape}. Using default zero labels.")
                labels = torch.zeros(b, h, w, dtype=torch.long, device='cuda')

            labels = labels.clamp(0, 4)
            if pred_labels.shape != labels.shape:
                logger.warning(f"Shape mismatch in get_seg_label: pred_labels {pred_labels.shape} vs labels {labels.shape}. Adjusting.")
                pred_labels = pred_labels.view(labels.shape)

        return cam_all, pred_labels

# ===================================================================
# 2. validate: Tính mIoU, Dice, FW-IoU trên validation set
# ===================================================================
def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    model.eval()
    avg_meter = AverageMeter('all_cls_acc4', 'avg_cls_acc4', 'cls_loss')
    fuse234_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.cls_num_classes + 1)
    if data_loader is None or len(data_loader) == 0:
        logger.error("Validation data_loader is None or empty")
        return 0.0, 0.0, 0.0, [0.0] * (cfg.dataset.cls_num_classes + 1), [0.0] * (cfg.dataset.cls_num_classes + 1)
    logger.warning(f"Validation data_loader length: {len(data_loader)}")
    
    # ------------------- Chuẩn hóa ảnh -------------------
    MEAN = [0.66791496, 0.47791372, 0.70623304]
    STD = [0.1736589, 0.22564577, 0.19820057]
    std_tensor = torch.tensor(STD, device='cuda').view(1, 3, 1, 1)
    mean_tensor = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1)

    # ------------------- TTA -------------------
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])
    
    # ------------------- CRF -------------------
    crf = DenseCRF()
    crf_config_path = os.path.join(cfg.work_dir.dir, "crf_config.npy")
    if not os.path.exists(crf_config_path):
        logger.warning(f"CRF config file {crf_config_path} does not exist - using defaults and creating default config")
        default_config = np.array([15, 30, 10, 20, 50, 10])
        np.save(crf_config_path, default_config)
    crf.load_config(crf_config_path)
    
    # ------------------- Feature Extractor -------------------
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=0.5)
    feature_extractor = FeatureExtractor(mask_adapter, clip_size=224, biomedclip_model=model.resnet_model)


    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            if inputs is None or cls_label is None or labels is None:
                logger.warning(f"Skipping invalid batch due to None values: {name}")
                continue
            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()
            if cls_label.shape[1] != 4:
                logger.error(f"Invalid cls_label shape: {cls_label.shape}. Expected [batch_size, 4]. Skipping batch.")
                continue
            # ------------------- Forward chính -------------------
            try:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs, labels=cls_label, cfg=cfg)
                    if len(outputs) != 13:
                        logger.error(f"Error unpacking model output: expected 13 values, got {len(outputs)}. Using default values.")
                        cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, loss, _, k_list, _x1 = [torch.zeros(1, device=inputs.device) if i == 0 else None for i in range(13)]
                    else:
                        cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, loss, _, k_list, _x1 = outputs
                    # Hierarchical merge
                    if isinstance(k_list, (list, ListConfig)):
                        k_list = [int(k) for k in k_list]
                    if sum(k_list) != cfg.dataset.cls_num_classes:
                        logger.warning(f"k_list sum {sum(k_list)} does not match cls_num_classes {cfg.dataset.cls_num_classes}. Using default k_list.")
                        k_list = [1] * cfg.dataset.cls_num_classes
                    cls1 = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")
                    cls4 = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")
                    logger.debug(f"cls1 shape: {cls1.shape}, cls_label shape: {cls_label.shape}")
                    cls_loss = cls_loss_func(cls1[:, :4], torch.clamp(cls_label, 0, 1))
                    cls4_acc_check = (torch.sigmoid(cls4[:, :4]) > 0.5).float()
                    # Compute per-class accuracy
                    all_cls_acc4 = ((cls4_acc_check == cls_label).float().mean(dim=0) * 100).mean()
                    avg_cls_acc4 = ((cls4_acc_check == cls_label).float().mean(dim=0) * 100).mean()
                    avg_meter.add({"all_cls_acc4": all_cls_acc4.item(), "avg_cls_acc4": avg_cls_acc4.item(), "cls_loss": cls_loss.item()})
                    logger.debug(f"Updated metrics: all_cls_acc4={all_cls_acc4.item()}, avg_cls_acc4={avg_cls_acc4.item()}, cls_loss={cls_loss.item()}")

                    cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")
                    cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")
                    cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")

                    # Non-TTA baseline
                    cam2_all_non_tta, _ = get_seg_label(cam2, inputs, cls_label, labels, feature_extractor)
                    cam2_all_non_tta = cam2_all_non_tta.cuda()

                    cam3_all_non_tta, _ = get_seg_label(cam3, inputs, cls_label, labels, feature_extractor)
                    cam3_all_non_tta = cam3_all_non_tta.cuda()

                    cam4_all_non_tta, _ = get_seg_label(cam4, inputs, cls_label, labels, feature_extractor)
                    cam4_all_non_tta = cam4_all_non_tta.cuda()

            except Exception as e:
                logger.error(f"Error processing batch {name}: {str(e)}")
                continue

            cams2 = []
            cams3 = []
            cams4 = []
            for tta_trans in tta_transform:
                augmented_tensor = tta_trans.augment_image(inputs)
                aug_b, aug_c, aug_h, aug_w = augmented_tensor.shape
                try:
                    with torch.amp.autocast('cuda'):
                        outputs = model(augmented_tensor, labels=cls_label, cfg=cfg)
                        if len(outputs) != 13:
                            logger.error(f"Error unpacking model output with TTA: expected 13 values, got {len(outputs)}. Using default CAMs.")
                            _, cam1, _, cam2, _, cam3, _, cam4, _, _, _, k_list, _ = [torch.zeros(1, device=inputs.device) if i % 2 == 1 else None for i in range(13)]
                        else:
                            _, cam1, _, cam2, _, cam3, _, cam4, _, _, _, k_list, _ = outputs
                        if isinstance(k_list, (list, ListConfig)):
                            k_list = [int(k) for k in k_list]
                        if sum(k_list) != cfg.dataset.cls_num_classes:
                            logger.warning(f"k_list sum {sum(k_list)} does not match cls_num_classes {cfg.dataset.cls_num_classes}. Using default k_list.")
                            k_list = [1] * cfg.dataset.cls_num_classes
                        cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")
                        cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")
                        cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")

                        cam2_all, pred_labels = get_seg_label(cam2, augmented_tensor, cls_label, labels, feature_extractor)
                        cam2_all = cam2_all.cuda()
                        cam2_all = F.interpolate(cam2_all, size=(h, w), mode='bilinear', align_corners=True)
                        cam2_all = tta_trans.deaugment_mask(cam2_all)
                        if cam2_all.dim() == 4 and cam2_all.size(1) == 5:
                            cams2.append(cam2_all)
                        else:
                            logger.warning(f"Invalid cam2_all shape: {cam2_all.shape}. Expected [batch_size, 5, height, width]. Skipping this TTA transformation.")

                        cam3_all, pred_labels = get_seg_label(cam3, augmented_tensor, cls_label, labels, feature_extractor)
                        cam3_all = cam3_all.cuda()
                        cam3_all = F.interpolate(cam3_all, size=(h, w), mode='bilinear', align_corners=True)
                        cam3_all = tta_trans.deaugment_mask(cam3_all)
                        if cam3_all.dim() == 4 and cam3_all.size(1) == 5:
                            cams3.append(cam3_all)
                        else:
                            logger.warning(f"Invalid cam3_all shape: {cam3_all.shape}. Expected [batch_size, 5, height, width]. Skipping this TTA transformation.")

                        cam4_all, pred_labels = get_seg_label(cam4, augmented_tensor, cls_label, labels, feature_extractor)
                        cam4_all = cam4_all.cuda()
                        cam4_all = F.interpolate(cam4_all, size=(h, w), mode='bilinear', align_corners=True)
                        cam4_all = tta_trans.deaugment_mask(cam4_all)
                        if cam4_all.dim() == 4 and cam4_all.size(1) == 5:
                            cams4.append(cam4_all)
                        else:
                            logger.warning(f"Invalid cam4_all shape: {cam4_all.shape}. Expected [batch_size, 5, height, width]. Skipping this TTA transformation.")
                except Exception as e:
                    logger.error(f"Error in model forward with TTA for batch {name}: {str(e)}. Skipping TTA for this batch.")
                    continue
            
            # Fallback nếu TTA fail
            if not cams2:
                logger.warning(f"TTA failed for CAM2 in batch {name}. Using non-TTA CAM2.")
                cams2 = [cam2_all_non_tta]
            if not cams3:
                logger.warning(f"TTA failed for CAM3 in batch {name}. Using non-TTA CAM3.")
                cams3 = [cam3_all_non_tta]
            if not cams4:
                logger.warning(f"TTA failed for CAM4 in batch {name}. Using non-TTA CAM4.")
                cams4 = [cam4_all_non_tta]

            # ------------------- Fuse multi-scale CAM -------------------
            try:
                cams2 = torch.stack(cams2, dim=0).mean(dim=0)
                cams3 = torch.stack(cams3, dim=0).mean(dim=0)
                cams4 = torch.stack(cams4, dim=0).mean(dim=0)

                fuse234 = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4
                logger.debug(f"Fuse234 shape: {fuse234.shape}, Fuse234 max: {fuse234.max()}, min: {fuse234.min()}")

                # ------------------- CRF Refinement -------------------
                inputs_np = (inputs * std_tensor + mean_tensor).clamp(0, 1).cpu().numpy()
                probs_np = F.softmax(fuse234, dim=1).cpu().numpy()
                crf_probs_list = []
                for i in range(b):
                    prob_sample = probs_np[i]  # Shape: [num_classes, height, width]
                    input_sample = inputs_np[i].transpose(1, 2, 0) * 255  # Shape: [height, width, 3]
                    if prob_sample.shape[1:] != (h, w) or input_sample.shape[:2] != (h, w):
                        logger.error(f"Spatial dimensions mismatch for sample {i} in batch {name}: probs {prob_sample.shape[1:]} vs images {input_sample.shape[:2]}")
                        continue
                    try:
                        maxconf_crf, crf_probs = crf.process(prob_sample[np.newaxis, :], input_sample[np.newaxis, :].astype(np.uint8))  # Ensure 4D input
                        crf_probs_list.append(crf_probs[0])  # Take the first (and only) batch element
                    except Exception as e:
                        logger.error(f"CRF processing failed for sample {i} in batch {name}: {str(e)}")
                        crf_probs_list.append(prob_sample)
                if not crf_probs_list:
                    logger.error(f"No valid CRF results for batch {name}. Using original probabilities.")
                    full_probs_tensor = F.softmax(fuse234, dim=1)
                else:
                    full_probs_tensor = torch.from_numpy(np.stack(crf_probs_list, axis=0)).cuda().float()
                    logger.debug(f"Stacked crf_probs_list shape: {full_probs_tensor.shape}")

                logger.warning(f"CRF processing applied or fallback used for batch {name}.")
                logger.debug(f"full_probs_tensor shape: {full_probs_tensor.shape}")
                if full_probs_tensor.dim() != 4 or full_probs_tensor.size(1) != 5:
                    logger.error(f"Invalid full_probs_tensor shape: {full_probs_tensor.shape}. Expected [batch_size, 5, height, width].")
                    continue
                full_probs_tensor = F.softmax(full_probs_tensor, dim=1)
                full_probs_tensor = torch.clamp(full_probs_tensor, min=1e-8, max=1.0 - 1e-8)

                # Generate smoother mask using probability map thresholding
                prob_map = full_probs_tensor.max(dim=1)[0]  # Max probability across classes
                # ------------------- Final prediction -------------------
                pred_labels = torch.argmax(full_probs_tensor, dim=1).long().clamp(0, 4).cuda()
                smooth_mask = (prob_map > 0.5).float() * pred_labels  # Threshold for smoothness
                logger.debug(f"pred_labels shape: {pred_labels.shape}, smooth_mask shape: {smooth_mask.shape}")

                if labels.dim() <= 1:
                    logger.warning(f"Invalid labels dimension: {labels.dim()}. Reshaping to [batch_size, height, width].")
                    labels = torch.zeros(b, h, w, dtype=torch.long, device='cuda')
                    for i in range(b):
                        active_classes = cls_label[i] > 0
                        class_indices = torch.where(active_classes)[0].cpu().numpy()
                        if len(class_indices) > 0:
                            num_active = len(class_indices)
                            section_size = h // num_active if num_active > 0 else h
                            label_map = torch.zeros(h, w, device='cuda')
                            for idx, c in enumerate(class_indices):
                                if c >= 4:
                                    logger.warning(f"Index {c} out of bounds for num_classes 4. Skipping.")
                                    continue
                                start = idx * section_size
                                end = (idx + 1) * section_size if idx < num_active - 1 else h
                                label_map[start:end, :] = c
                            # Assign background where no foreground
                            label_map[torch.sum(label_map, dim=0) == 0] = 4
                            labels[i] = label_map.long()
                        else:
                            labels[i] = torch.full((h, w), 4, device='cuda', dtype=torch.long)
                elif labels.dim() == 4:
                    if labels.size(1) == 1:
                        labels = labels.squeeze(1).long()
                    else:
                        labels = labels.argmax(dim=1).long()
                elif labels.dim() == 3:
                    labels = labels.long()
                elif labels.dim() == 2:
                    labels = labels.unsqueeze(0) if b == 1 else labels
                    if labels.shape[0] != b:
                        logger.error(f"Invalid labels shape after unsqueeze: {labels.shape}. Using default zero labels.")
                        labels = torch.zeros(b, h, w, dtype=torch.long, device='cuda')
                else:
                    logger.error(f"Invalid labels shape: {labels.shape}. Using default zero labels.")
                    labels = torch.zeros(b, h, w, dtype=torch.long, device='cuda')

                labels = labels.clamp(0, 4)
                if pred_labels.shape != labels.shape:
                    logger.error(f"Shape mismatch for batch {name}: pred_labels {pred_labels.shape} vs labels {labels.shape}. Skipping update.")
                    continue
                
                # Cập nhật metric
                fuse234_matrix.update(labels.detach().clone(), pred_labels.clone())

            except Exception as e:
                logger.error(f"Error processing CAMs for batch {name}: {str(e)}. Skipping batch.")
                continue

        all_cls_acc4 = avg_meter.pop('all_cls_acc4') if avg_meter.get('all_cls_acc4') is not None else 0.0
        avg_cls_acc4 = avg_meter.pop('avg_cls_acc4') if avg_meter.get('avg_cls_acc4') is not None else 0.0
        cls_loss = avg_meter.pop('cls_loss') if avg_meter.get('cls_loss') is not None else 0.0
        logger.warning(f"Validation metrics - all_cls_acc4: {all_cls_acc4:.2f}, avg_cls_acc4: {avg_cls_acc4:.2f}, cls_loss: {cls_loss}")

        if fuse234_matrix.mat1.sum() == 0 and fuse234_matrix.mat2.sum() == 0:
            logger.warning("Confusion matrix is empty. Possible issues with pred_labels or labels alignment.")
            logger.debug(f"Last pred_labels shape: {pred_labels.shape if 'pred_labels' in locals() else 'undefined'}, Last labels shape: {labels.shape}")
            return 0.0, 0.0, 0.0, [0.0] * (cfg.dataset.cls_num_classes + 1), [0.0] * (cfg.dataset.cls_num_classes + 1)

        # ------------------- Kết quả Metrics -------------------
        acc_global, acc, iu_per_class, dice_per_class, dice_bg_fg, fw_iu = fuse234_matrix.compute()
        logger.debug(f"Confusion matrix values: {fuse234_matrix.mat1}, acc_global: {acc_global}, iu_per_class: {iu_per_class}")

        mIoU = np.nan_to_num(iu_per_class.mean() * 100, nan=0.0)
        mean_dice = np.nan_to_num(dice_per_class.mean() * 100, nan=0.0)
        fw_iu = np.nan_to_num(fw_iu * 100, nan=0.0)

        model.train()
        return mIoU, mean_dice, fw_iu, iu_per_class, dice_per_class

def generate_cam(model=None, data_loader=None, cfg=None):
    model.eval()
    MEAN = [0.66791496, 0.47791372, 0.70623304]
    STD = [0.1736589, 0.22564577, 0.19820057]
    std = torch.tensor(STD, device='cuda').view(1, 3, 1, 1)
    mean = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1)
    if data_loader is None or len(data_loader) == 0:
        logger.error("Training data_loader is None or empty for CAM generation")
        return
    logger.warning(f"Training data_loader length for CAM: {len(data_loader)}")
    sample_count = 0
    torch.cuda.empty_cache()

    crf = DenseCRF()
    crf_config_path = os.path.join(cfg.work_dir.dir, "crf_config.npy")
    if not os.path.exists(crf_config_path):
        logger.warning(f"CRF config file {crf_config_path} does not exist - using defaults and creating default config")
        default_config = np.array([15, 30, 10, 20, 50, 10])
        np.save(crf_config_path, default_config)
    crf.load_config(crf_config_path)

    mask_adapter = MaskAdapter_DynamicThreshold(alpha=0.5)
    feature_extractor = FeatureExtractor(mask_adapter, clip_size=224, biomedclip_model=model.resnet_model)

    with torch.no_grad():
        for data in tqdm(data_loader, total=min(150, len(data_loader)), ncols=100, ascii=" >="):
            if sample_count >= 150:
                break
            name, inputs, cls_label, labels = data
            if inputs is None or cls_label is None or labels is None:
                logger.warning(f"Skipping invalid batch {name} due to None values")
                sample_count += 1
                continue
            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()
            if cls_label.shape[1] != 4:
                logger.error(f"Invalid cls_label shape: {cls_label.shape}. Expected [batch_size, 4]. Skipping batch.")
                continue

            try:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs, labels=cls_label, cfg=cfg)
                    if len(outputs) != 13:
                        logger.error(f"Error unpacking model output: expected 13 values, got {len(outputs)}. Using default values.")
                        cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, loss, _, k_list, _x1 = [torch.zeros(1, device=inputs.device) if i == 0 else None for i in range(13)]
                    else:
                        cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, loss, _, k_list, _x1 = outputs
                    if isinstance(k_list, (list, ListConfig)):
                        k_list = [int(k) for k in k_list]
                    if sum(k_list) != cfg.dataset.cls_num_classes:
                        logger.warning(f"k_list sum {sum(k_list)} does not match cls_num_classes {cfg.dataset.cls_num_classes}. Using default k_list.")
                        k_list = [1] * cfg.dataset.cls_num_classes
                    cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")
                    cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")
                    cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test if cfg and hasattr(cfg.train, 'merge_test') else "mean")

                cam2_all, pred_labels = get_seg_label(cam2, inputs, cls_label, labels, feature_extractor)
                cam2_all = cam2_all.cuda()
                cam3_all, pred_labels = get_seg_label(cam3, inputs, cls_label, labels, feature_extractor)
                cam3_all = cam3_all.cuda()
                cam4_all, pred_labels = get_seg_label(cam4, inputs, cls_label, labels, feature_extractor)
                cam4_all = cam4_all.cuda()

                fuse234 = 0.3 * cam2_all + 0.3 * cam3_all + 0.4 * cam4_all
                full_probs_tensor = F.softmax(fuse234, dim=1)
                inputs_np = (inputs * std + mean).clamp(0, 1).cpu().numpy()
                probs_np = full_probs_tensor.cpu().numpy()
                crf_probs_list = []
                for i in range(b):
                    prob_sample = probs_np[i]  # Shape: [num_classes, height, width]
                    input_sample = inputs_np[i].transpose(1, 2, 0) * 255  # Shape: [height, width, 3]
                    if prob_sample.shape[1:] != (h, w) or input_sample.shape[:2] != (h, w):
                        logger.error(f"Spatial dimensions mismatch for sample {i} in batch {name}: probs {prob_sample.shape[1:]} vs images {input_sample.shape[:2]}")
                        continue
                    try:
                        maxconf_crf, crf_probs = crf.process(prob_sample[np.newaxis, :], input_sample[np.newaxis, :].astype(np.uint8))
                        crf_probs_list.append(crf_probs[0])  # Take the first (and only) batch element
                    except Exception as e:
                        logger.error(f"CRF processing failed for sample {i} in batch {name}: {str(e)}")
                        crf_probs_list.append(prob_sample)
                if not crf_probs_list:
                    logger.error(f"No valid CRF results for batch {name}. Using original probabilities.")
                    full_probs_tensor = F.softmax(fuse234, dim=1)
                else:
                    full_probs_tensor = torch.from_numpy(np.stack(crf_probs_list, axis=0)).cuda().float()
                    logger.debug(f"Stacked crf_probs_list shape: {full_probs_tensor.shape}")

                logger.warning(f"CRF processing applied or fallback used for batch {name}.")
                logger.debug(f"full_probs_tensor shape: {full_probs_tensor.shape}")
                if full_probs_tensor.dim() != 4 or full_probs_tensor.size(1) != 5:
                    logger.error(f"Invalid full_probs_tensor shape: {full_probs_tensor.shape}. Expected [batch_size, 5, height, width].")
                    continue
                full_probs_tensor = F.softmax(full_probs_tensor, dim=1)
                full_probs_tensor = torch.clamp(full_probs_tensor, min=1e-8, max=1.0 - 1e-8)

                # Generate smoother mask using probability map
                prob_map = full_probs_tensor.max(dim=1)[0]  # Max probability across classes
                pred_labels = torch.argmax(full_probs_tensor, dim=1).long().clamp(0, 4).cuda()
                smooth_mask = F.interpolate(full_probs_tensor.argmax(dim=1, keepdim=True).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1)
                smooth_mask = (smooth_mask * 255).byte()  # Scale to 0-255 for visualization
                logger.debug(f"pred_labels shape: {pred_labels.shape}, smooth_mask shape: {smooth_mask.shape}")

                img_denorm_tensor = (inputs * std + mean).clamp(0, 1) * 255
                img_np = img_denorm_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

                                # === LƯU MASK ĐẸP + LOG KHI PHÁT HIỆN LYM/NEC ===
                PALETTE = [
                    255, 0, 0,      # 0: TUM - Đỏ rực
                    0, 255, 0,      # 1: STR - Xanh lá
                    0, 0, 255,      # 2: LYM - Xanh dương đậm (rõ nhất)
                    180, 0, 255,    # 3: NEC - Tím đậm (đẹp hơn 153,0,255)
                    255, 255, 255   # 4: Background - Trắng
                ]

                for i in range(b):
                    if sample_count >= 150:  
                        break
                    
                    pred_mask = pred_labels[i].cpu().numpy().astype(np.uint8)
    
                    # Tăng độ tương phản cho LYM (class 2) và NEC (class 3)
                    if 2 in np.unique(pred_mask):
                        logger.info(f"LYM (blue) detected in {name[i]} - beautiful!")
                    if 3 in np.unique(pred_mask):
                        logger.info(f"NEC (purple) detected in {name[i]}")

                    mask_pil = Image.fromarray(pred_mask).convert('P')
                    mask_pil.putpalette([
                        255, 0, 0,      # 0: TUM - đỏ
                        0, 255, 0,      # 1: STR - xanh lá
                        0, 0, 255,      # 2: LYM - xanh dương (rõ nhất)
                        180, 0, 255,    # 3: NEC - tím đậm (đẹp hơn 153,0,255)
                        255, 255, 255   # 4: Background - trắng
                    ])
                    mask_pil.save(os.path.join(cfg.work_dir.pred_dir, f"{name[i]}_mask.png"))

                    # pred_mask_np = pred_labels[i].cpu().numpy().astype(np.uint8)

                    # # Kiểm tra và log khi phát hiện LYM hoặc NEC
                    # unique_classes = np.unique(pred_mask_np)
                    # if 2 in unique_classes:
                    #     logger.info(f"LYM (xanh dương) detected in {name[i]} - BEAUTIFUL!")
                    # if 3 in unique_classes:
                    #     logger.info(f"NEC (tím) detected in {name[i]}")

                    # # Tạo và lưu mask màu chuẩn
                    # mask_pil = Image.fromarray(pred_mask_np).convert('P')
                    # mask_pil.putpalette(PALETTE)
                    # mask_path = os.path.join(cfg.work_dir.pred_dir, f"{name[i]}_mask.png")
                    # mask_pil.save(mask_path)

                    # === HEATMAP ĐẸP: JET + overlay lên ảnh gốc ===
                    cam_fg = full_probs_tensor[i, :4]  # Chỉ lấy 4 class foreground
                    cam_heatmap = cam_fg.max(dim=0)[0].cpu().numpy()
                    cam_heatmap = (cam_heatmap - cam_heatmap.min()) / (cam_heatmap.max() - cam_heatmap.min() + 1e-8)
                    cam_heatmap = (cam_heatmap * 255).astype(np.uint8)
                    cam_heatmap = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_JET)
                    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

                    # Overlay lên ảnh gốc
                    img_overlay = cv2.addWeighted(img_np[i], 0.6, cam_heatmap, 0.4, 0)
                    overlay_pil = Image.fromarray(img_overlay)
                    cam_path = os.path.join(cfg.work_dir.pred_dir, f"{name[i]}_cam.png")
                    overlay_pil.save(cam_path)

                    logger.info(f"Saved: {name[i]} → mask + beautiful JET overlay CAM")

                    sample_count += 1
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error generating CAM for {name[0]}: {str(e)}")
                sample_count += 1
                torch.cuda.empty_cache()
                continue

    model.train()
    torch.cuda.empty_cache()
    return
