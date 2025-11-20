import cv2
from omegaconf import ListConfig
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import logging
from utils.evaluate import ConfusionMatrixAllClass
from utils.pyutils import AverageMeter
from utils.hierarchical_utils import merge_subclass_cams_to_parent
from utils.crf import DenseCRF
import ttach as tta
from PIL import Image
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold

logger = logging.getLogger(__name__)

def get_seg_label(cams, inputs, cls_label):
    """
    Input: cams [B, >=4, H, W], inputs [B,3,H,W], cls_label [B,4]
    Output: cam_4ch [B,4,H,W], pred [B,H,W]
    """
    with torch.no_grad():
        b, _, h, w = inputs.shape
        device = inputs.device

        # Lấy 4 kênh đầu tiên
        if cams.size(1) >= 4:
            cams = cams[:, :4]
        else:
            pad = torch.zeros((cams.size(0), 4 - cams.size(1), cams.size(2), cams.size(3)), device=device)
            cams = torch.cat([cams, pad], dim=1)

        # Normalize từng class có nhãn
        cams = cams.clone()
        cls_label_np = cls_label.cpu().numpy()
        for i in range(b):
            for j in range(4):
                if cls_label_np[i, j] > 0:
                    c_min = cams[i, j].min()
                    c_max = cams[i, j].max()
                    if c_max > c_min:
                        cams[i, j] = (cams[i, j] - c_min) / (c_max - c_min)

        # Chỉ giữ class có nhãn
        cams = cams * cls_label.unsqueeze(2).unsqueeze(3)  # [B,4,1,1]

        # Resize về kích thước ảnh gốc
        cams = F.interpolate(cams, size=(h, w), mode='bilinear', align_corners=True)

        pred = torch.argmax(cams, dim=1)  # [B, H, W]
        return cams, pred


def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = ConfusionMatrixAllClass(num_classes=4)  # Chỉ 4 class

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data

            inputs = inputs.to(device).float()
            cls_label = cls_label.to(device).float()
            b, _, h, w = inputs.shape

            try:
                outputs = model(inputs, labels=cls_label, cfg=cfg)
                if len(outputs) != 13:
                    continue

                _, _, _, cam2, _, cam3, _, cam4, _, _, _, k_list, _ = outputs

                # Chỉ lấy 4 kênh
                cam2 = cam2[:, :4] if cam2.size(1) >= 4 else cam2
                cam3 = cam3[:, :4] if cam3.size(1) >= 4 else cam3
                cam4 = cam4[:, :4] if cam4.size(1) >= 4 else cam4

                # Tạo pseudo-label từ từng scale
                cam2, _ = get_seg_label(cam2, inputs, cls_label)
                cam3, _ = get_seg_label(cam3, inputs, cls_label)
                cam4, _ = get_seg_label(cam4, inputs, cls_label)

                # Fuse 3 scale
                fuse = 0.3 * cam2 + 0.3 * cam3 + 0.4 * cam4
                pred = torch.argmax(fuse, dim=1)  # [B, H, W]

                # Xử lý ground truth
                if labels.dim() > 3:
                    labels = labels.squeeze(1) if labels.size(1) == 1 else labels.argmax(1)
                elif labels.dim() == 2:
                    labels = labels.view(b, h, w)

                labels = labels.long().clamp(0, 3).to(device)

                # Cập nhật confusion matrix (batch-wise để tránh lỗi device)
                for i in range(b):
                    matrix.update(pred[i], labels[i])

            except Exception as e:
                logger.error(f"Validation error: {e}")
                continue

    # Tính metric
    _, _, iu, dice, _, _ = matrix.compute()
    mIoU = np.nan_to_num(iu.mean() * 100)
    mean_dice = np.nan_to_num(dice.mean() * 100)

    logger.warning(f"Validation → mIoU: {mIoU:.2f}% | Mean Dice: {mean_dice:.2f}%")

    model.train()
    return mIoU, mean_dice, 0.0, iu.tolist(), dice.tolist()


# ===================================================================
# generate_cam – 4 màu + JET overlay đẹp
# ===================================================================
def generate_cam(model=None, data_loader=None, cfg=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MEAN = [0.66791496, 0.47791372, 0.70623304]
    STD = [0.1736589, 0.22564577, 0.19820057]
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)

    crf = DenseCRF()
    crf_config_path = os.path.join(cfg.work_dir.dir, "crf_config.npy")
    if not os.path.exists(crf_config_path):
        np.save(crf_config_path, np.array([15, 30, 10, 20, 50, 10]))
    crf.load_config(crf_config_path)

    PALETTE = [255,0,0, 0,255,0, 0,0,255, 180,0,255]  # TUM, STR, LYM, NEC

    sample_count = 0
    with torch.no_grad():
<<<<<<< HEAD
        for data in tqdm(data_loader, total=min(1000, len(data_loader)), ncols=100):
            if sample_count >= 1000: break
            name, inputs, cls_label, _ = data
            if inputs is None: continue

            inputs = inputs.to(device).float()
            cls_label = cls_label.to(device).float()
            b, _, h, w = inputs.shape
=======
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
>>>>>>> 88f5325ab342253aaedf9e305b1b7bd67dc2eee4

            try:
                outputs = model(inputs, labels=cls_label, cfg=cfg)
                if len(outputs) != 13: continue
                _, _, _, cam2, _, cam3, _, cam4, _, _, _, k_list, _ = outputs

                cam2 = cam2[:, :4] if cam2.size(1) >= 4 else cam2
                cam3 = cam3[:, :4] if cam3.size(1) >= 4 else cam3
                cam4 = cam4[:, :4] if cam4.size(1) >= 4 else cam4

                c2, _ = get_seg_label(cam2, inputs, cls_label)
                c3, _ = get_seg_label(cam3, inputs, cls_label)
                c4, _ = get_seg_label(cam4, inputs, cls_label)

                fuse = 0.3*c2 + 0.3*c3 + 0.4*c4
                probs = F.softmax(fuse, dim=1)

                # CRF trên CPU (ổn định nhất)
                img_np = ((inputs * std + mean).clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
                probs_np = probs.cpu().numpy()
                crf_out = []
                for i in range(b):
                    p = probs_np[i]
                    img = img_np[i].transpose(1,2,0)
                    try:
                        _, refined = crf.process(p[np.newaxis, :4], img[np.newaxis, :])
                        crf_out.append(refined[0])
                    except:
                        crf_out.append(p[:4])

                refined_probs = torch.from_numpy(np.stack(crf_out)).to(device)
                refined_probs = F.softmax(refined_probs, dim=1)
                pred = torch.argmax(refined_probs, dim=1).cpu().numpy()

                img_denorm = ((inputs * std + mean).clamp(0,1).cpu().numpy() * 255).astype(np.uint8)

<<<<<<< HEAD
                for i in range(b):
                    if sample_count >= 1000: break

                    # Lưu mask 4 màu
                    mask = pred[i].astype(np.uint8)
                    mask_pil = Image.fromarray(mask).convert('P')
                    mask_pil.putpalette(PALETTE)
                    mask_pil.save(os.path.join(cfg.work_dir.pred_dir, f"{name[i]}_mask.png"))

                    if 2 in np.unique(mask):
                        logger.info(f"LYM (blue) detected in {name[i]} - BEAUTIFUL!")
                    if 3 in np.unique(mask):
                        logger.info(f"NEC (purple) detected in {name[i]}")
=======
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
>>>>>>> 88f5325ab342253aaedf9e305b1b7bd67dc2eee4

                    # Heatmap JET overlay
                    heatmap = refined_probs[i].max(0)[0].cpu().numpy()
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                    heatmap = (heatmap * 255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    overlay = cv2.addWeighted(img_denorm[i].transpose(1,2,0), 0.6, heatmap, 0.4, 0)
                    Image.fromarray(overlay).save(os.path.join(cfg.work_dir.pred_dir, f"{name[i]}_cam.png"))

                    logger.info(f"Saved: {name[i]} → 4-color mask + JET overlay")
                    sample_count += 1

            except Exception as e:
                logger.error(f"Generate cam error: {e}")
                continue

    model.train()
<<<<<<< HEAD
    return
=======
    torch.cuda.empty_cache()
    return
>>>>>>> 88f5325ab342253aaedf9e305b1b7bd67dc2eee4
