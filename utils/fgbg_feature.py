import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import ResNetModel
import logging

logger = logging.getLogger(__name__)

class MaskAdapter_DynamicThreshold(nn.Module):
    def __init__(self, alpha, mask_cam=False):
        super(MaskAdapter_DynamicThreshold, self).__init__()
        self.alpha = alpha
        self.mask_cam = mask_cam
        logger.info(f"MaskAdapter_DynamicThreshold initialized: alpha={alpha}, mask_cam={mask_cam}")

    def forward(self, x):
        binary_mask = []
        for i in range(x.shape[0]):
            th = torch.max(x[i]) * self.alpha
            binary_mask.append(
                torch.where(x[i] >= th, torch.ones_like(x[i]), torch.zeros_like(x[i]))
            )
        binary_mask = torch.stack(binary_mask, dim=0)
        if self.mask_cam:
            return x * binary_mask
        return binary_mask

class FeatureExtractor:
    def __init__(self, mask_adapter, clip_size=224, biomedclip_model=None):
        self.mask_adapter = mask_adapter
        self.clip_size = clip_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet_model = biomedclip_model if biomedclip_model else ResNetModel.from_pretrained("microsoft/resnet-50").to(self.device)
        logger.info("Successfully loaded ResNet-50 model")

    def prepare_cam_mask(self, cam, N):
        if cam.size(1) != 5:
            logger.warning(f"CAM has {cam.size(1)} channels, expected 5. Adjusting.")
            if cam.size(1) == 4:
                cam_max = torch.max(cam, dim=1, keepdim=True)[0]
                bg_cam = (1 - cam_max) ** 10
                cam = torch.cat([cam, bg_cam], dim=1)
            else:
                logger.error(f"Cannot adjust CAM with {cam.size(1)} channels. Returning None.")
                return None, None
        cam_224 = F.interpolate(cam, (self.clip_size, self.clip_size), mode="bilinear", align_corners=True)
        cam_224_mask = self.mask_adapter(cam_224)
        return cam_224, cam_224_mask

    def prepare_image(self, img):
        return F.interpolate(img, (self.clip_size, self.clip_size), mode="bilinear", align_corners=True)

    @torch.amp.autocast('cuda')
    def extract_features(self, img_224, cam_224, cam_224_mask, label):
        batch_indices, class_indices = torch.where(label == 1)
        if len(batch_indices) == 0 or len(class_indices) == 0:
            logger.warning("No foreground samples in batch.")
            return None, None, None, None

        # Ensure img_224 is [batch_size, 3, H, W]
        if img_224.dim() != 4 or img_224.size(1) != 3:
            logger.error(f"Invalid img_224 shape: {img_224.shape}. Expected [batch_size, 3, H, W].")
            return None, None, None, None

        # Select images and CAMs for foreground classes
        img_selected = img_224[batch_indices]  # Shape: [num_samples, 3, H, W]
        cam_selected = cam_224[batch_indices, class_indices]  # Shape: [num_samples, H, W]
        mask_selected = cam_224_mask[batch_indices, class_indices]  # Shape: [num_samples, H, W]

        # Expand CAM and mask to match image channels
        cam_expanded = cam_selected.unsqueeze(1)  # Shape: [num_samples, 1, H, W]
        mask_expanded = mask_selected.unsqueeze(1)  # Shape: [num_samples, 1, H, W]

        # Compute foreground and background features
        fg_features = cam_expanded * img_selected  # Shape: [num_samples, 3, H, W]
        bg_features = (1 - cam_expanded) * img_selected  # Shape: [num_samples, 3, H, W]

        fg_masks = mask_expanded  # Shape: [num_samples, 1, H, W]
        bg_masks = 1 - mask_expanded  # Shape: [num_samples, 1, H, W]

        return fg_features, bg_features, fg_masks, bg_masks

    @torch.amp.autocast('cuda')
    def get_masked_features(self, fg_features, bg_features, fg_masks, bg_masks):
        if fg_features is None or bg_features is None:
            logger.error("Invalid features. Returning None.")
            return None, None

        # Validate shapes
        if fg_features.dim() != 4 or fg_features.size(1) != 3:
            logger.error(f"Invalid fg_features shape: {fg_features.shape}. Expected [num_samples, 3, H, W].")
            return None, None
        if bg_features.dim() != 4 or bg_features.size(1) != 3:
            logger.error(f"Invalid bg_features shape: {bg_features.shape}. Expected [num_samples, 3, H, W].")
            return None, None

        # Normalize features
        fg_min = fg_features.amin(dim=(2, 3), keepdim=True)
        fg_max = fg_features.amax(dim=(2, 3), keepdim=True)
        normalized_fg_features = (fg_features - fg_min) / (fg_max - fg_min + 1e-8)

        bg_min = bg_features.amin(dim=(2, 3), keepdim=True)
        bg_max = bg_features.amax(dim=(2, 3), keepdim=True)
        normalized_bg_features = (bg_features - bg_min) / (bg_max - bg_min + 1e-8)

        # Apply masks and ensure correct shape
        fg_input = normalized_fg_features * fg_masks  # Shape: [num_samples, 3, H, W]
        bg_input = normalized_bg_features * bg_masks  # Shape: [num_samples, 3, H, W]

        # Process through ResNet
        fg_outputs = self.resnet_model(fg_input)  # Expects [num_samples, 3, H, W]
        fg_img_features = fg_outputs.last_hidden_state.mean(dim=1)

        bg_outputs = self.resnet_model(bg_input)  # Expects [num_samples, 3, H, W]
        bg_img_features = bg_outputs.last_hidden_state.mean(dim=1)

        return fg_img_features, bg_img_features

    @torch.amp.autocast('cuda')
    def process_batch(self, inputs, cam, label, attention_weights=None):
        if not torch.any(label == 1):
            logger.warning("No foreground samples in batch.")
            return None

        N = inputs.size(0)
        cam_224, cam_224_mask = self.prepare_cam_mask(cam, N)
        if cam_224 is None or cam_224_mask is None:
            logger.error("Failed to prepare CAM mask. Returning None.")
            return None

        img_224 = self.prepare_image(inputs)
        fg_features, bg_features, fg_masks, bg_masks = self.extract_features(img_224, cam_224, cam_224_mask, label)
        if fg_features is None:
            logger.error("Failed to extract features. Returning None.")
            return None

        fg_img_features, bg_img_features = self.get_masked_features(fg_features, bg_features, fg_masks, bg_masks)
        if fg_img_features is None:
            logger.error("Failed to get masked features. Returning None.")
            return None

        return {
            'fg_features': fg_features,
            'bg_features': bg_features,
            'fg_masks': fg_masks,
            'bg_masks': bg_masks,
            'fg_img_features': fg_img_features,
            'bg_img_features': bg_img_features,
            'cam_224': cam_224,
            'cam_224_mask': cam_224_mask
        }

    def print_debug_info(self, batch_info):
        if batch_info is None:
            logger.info("No foreground samples in batch")
            return

        logger.info("\nFeature extraction debug info:")
        logger.info(f"Number of foreground samples: {batch_info['fg_masks'].shape[0]}")
        logger.info(f"Foreground features shape: {batch_info['fg_features'].shape}")
        logger.info(f"Background features shape: {batch_info['bg_features'].shape}")
        logger.info(f"ResNet FG features shape: {batch_info['fg_img_features'].shape}")
        logger.info(f"ResNet BG features shape: {batch_info['bg_img_features'].shape}")
        logger.info(f"CAM shape: {batch_info['cam_224'].shape}")
        logger.info(f"Mask shape: {batch_info['cam_224_mask'].shape}")
        logger.info(f"Foreground mask mean: {batch_info['fg_masks'].mean().item():.4f}")
        logger.info(f"Background mask mean: {batch_info['bg_masks'].mean().item():.4f}")
        