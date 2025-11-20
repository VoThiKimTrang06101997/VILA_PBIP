import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ResNetModel
import logging

logger = logging.getLogger(__name__)

class MaskAdapter_DynamicThreshold(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        logger.info(f"MaskAdapter_DynamicThreshold initialized with alpha={alpha}")

    def forward(self, x):
        """
        x: [B, C, H, W] – C thường là 4 (TUM, STR, LYM, NEC)
        Output: binary mask [B, 1, H, W]
        """
        B, C, H, W = x.shape
        binary_masks = []
        for i in range(B):
            # Tính threshold động cho từng sample
            cam_max = x[i].max()
            th = cam_max * self.alpha
            mask = (x[i] >= th).float()  # [C, H, W]
            mask = mask.sum(dim=0, keepdim=True) > 0  # [1, H, W] – pixel thuộc bất kỳ class nào
            binary_masks.append(mask.float())
        return torch.stack(binary_masks, dim=0)  # [B, 1, H, W]


class FeatureExtractor:
    def __init__(self, mask_adapter, clip_size=224, biomedclip_model=None):
        self.mask_adapter = mask_adapter
        self.clip_size = clip_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet_model = biomedclip_model if biomedclip_model else \
            ResNetModel.from_pretrained("microsoft/resnet-50").to(self.device)
        logger.info("FeatureExtractor initialized with ResNet-50")

    def prepare_cam_mask(self, cam):
        """
        Input: cam [B, >=4, H, W]
        Output: cam_224 [B, 4, 224, 224], mask [B, 1, 224, 224]
        """
        if cam is None or cam.numel() == 0:
            B = 1
            cam = torch.zeros((B, 4, self.clip_size, self.clip_size), device=self.device)
        else:
            B = cam.size(0)
            # Chỉ lấy 4 kênh đầu tiên
            if cam.size(1) >= 4:
                cam = cam[:, :4]
            else:
                pad = torch.zeros((cam.size(0), 4 - cam.size(1), cam.size(2), cam.size(3)), device=cam.device)
                cam = torch.cat([cam, pad], dim=1)

        # Resize về 224x224
        cam_224 = F.interpolate(cam, size=(self.clip_size, self.clip_size),
                                mode='bilinear', align_corners=True)  # [B, 4, 224, 224]

        # Tạo mask nhị phân từ 4 class
        cam_224_mask = self.mask_adapter(cam_224)  # [B, 1, 224, 224]

        return cam_224, cam_224_mask

    def prepare_image(self, img):
        return F.interpolate(img, size=(self.clip_size, self.clip_size),
                             mode='bilinear', align_corners=True)

    @torch.no_grad()
    def process_batch(self, inputs, cam, cls_label=None):
        """
        inputs: [B, 3, H, W]
        cam: [B, 4, H, W] hoặc hơn
        cls_label: [B, 4] (optional)
        """
        if inputs is None or cam is None:
            return None

        B, _, H, W = inputs.shape
        device = inputs.device

        # 1. Chuẩn bị CAM + mask
        cam_224, cam_mask = self.prepare_cam_mask(cam)  # [B,4,224,224], [B,1,224,224]

        # 2. Resize ảnh về 224x224
        img_224 = self.prepare_image(inputs)  # [B,3,224,224]

        # 3. Forward ResNet lấy feature map
        resnet_out = self.resnet_model(img_224)
        feat = resnet_out.last_hidden_state  # [B, 2048, 7, 7] hoặc [B, 1024, 14, 14]

        if feat.dim() != 4:
            logger.error(f"Invalid feature map from ResNet: {feat.shape}")
            return None

        _, C, h, w = feat.shape
        feat_flat = feat.view(B, C, -1)  # [B, C, N]

        # 4. Resize mask về kích thước feature map (7x7 hoặc 14x14)
        mask_resized = F.interpolate(
            cam_mask,
            size=(h, w),
            mode='bilinear',
            align_corners=True
        )  # [B, 1, h, w]

        mask_resized = (mask_resized > 0.5).float()
        mask_flat = mask_resized.view(B, -1)  # [B, N]

        # 5. Lấy FG/BG features
        fg_features_list = []
        bg_features_list = []

        for i in range(B):
            fg_idx = (mask_flat[i] > 0.5).nonzero(as_tuple=True)[0]
            bg_idx = (mask_flat[i] <= 0.5).nonzero(as_tuple=True)[0]

            # Nếu không có FG → lấy top-k từ CAM
            if fg_idx.numel() == 0:
                scores = cam_224[i].sum(0).flatten()
                _, topk_idx = torch.topk(scores, k=min(50, scores.numel()))
                fg_idx = topk_idx

            # Nếu không có BG → lấy vài điểm
            if bg_idx.numel() == 0:
                bg_idx = torch.tensor([0], device=device)

            # FG feature
            fg_feat = feat_flat[i:i+1, :, fg_idx]  # [1, C, N_fg]
            fg_feat = fg_feat.mean(dim=2) if fg_feat.size(2) > 0 else torch.zeros(1, C, device=device)

            # BG feature
            bg_feat = feat_flat[i:i+1, :, bg_idx]
            bg_feat = bg_feat.mean(dim=2) if bg_feat.size(2) > 0 else torch.zeros(1, C, device=device)

            fg_features_list.append(fg_feat)
            bg_features_list.append(bg_feat)

        fg_img_features = torch.cat(fg_features_list, dim=0)  # [B, C]
        bg_img_features = torch.cat(bg_features_list, dim=0)  # [B, C]

        return {
            'fg_img_features': fg_img_features,      # [B, C]
            'bg_features': bg_img_features,         # [B, C]
            'cam_224': cam_224,
            'cam_224_mask': cam_mask
        }

    def print_debug_info(self, batch_info):
        if batch_info is None:
            logger.info("No foreground samples in batch")
            return
        logger.info("\n=== FeatureExtractor Debug Info ===")
        logger.info(f"FG features shape: {batch_info['fg_img_features'].shape}")
        logger.info(f"BG features shape: {batch_info['bg_features'].shape}")
        logger.info(f"CAM shape: {batch_info['cam_224'].shape}")
        logger.info(f"Mask shape: {batch_info['cam_224_mask'].shape}")
        logger.info(f"Mask mean: {batch_info['cam_224_mask'].mean().item():.4f}")
        