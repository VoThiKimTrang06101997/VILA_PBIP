import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

def pair_features(fg_features, bg_features, l_fea, cls_label):
    fg_text, bg_text = [], []
    for i in range(fg_features.shape[0]):
        active_indices = cls_label[i].nonzero().squeeze(1)
        if active_indices.numel() == 0:
            logger.warning(f"No active classes for sample {i}. Using default zero tensor.")
            fg_text.append(torch.zeros(1, l_fea.shape[1], device=l_fea.device))
            bg_text.append(torch.zeros(1, l_fea.shape[1], device=l_fea.device))
        else:
            fg_text.append(l_fea[active_indices])
            bg_text.append(l_fea[active_indices])
    fg_text = torch.cat(fg_text, dim=0)
    bg_text = torch.cat(bg_text, dim=0)
    return {
        'fg_features': fg_features,
        'bg_features': bg_features,
        'fg_text': fg_text,
        'bg_text': bg_text
    }

def merge_subclass_cams_to_parent(cams, k_list, method='max'):
    if cams.size(1) == 5:  # Already includes background channel
        logger.debug("CAM already has 5 channels, returning unchanged.")
        return cams
    if not k_list or sum(k_list) != cams.shape[1]:
        logger.warning(f"Invalid k_list: {k_list}, cams shape: {cams.shape}. Returning cams unchanged.")
        return cams
    cumsum_k = np.cumsum([0] + k_list)
    parent_cams = []
    for i in range(len(k_list)):
        start_idx = cumsum_k[i]
        end_idx = cumsum_k[i + 1]
        if start_idx >= cams.shape[1] or end_idx > cams.shape[1]:
            logger.error(f"Invalid indices: start_idx={start_idx}, end_idx={end_idx}, cams channels={cams.shape[1]}. Returning empty CAM.")
            return torch.zeros_like(cams[:, :1])
        sub_cams = cams[:, start_idx:end_idx]
        if sub_cams.shape[1] == 0:
            logger.warning(f"Empty sub_cams for class {i}. Using zero tensor.")
            parent_cams.append(torch.zeros((cams.shape[0], 1, cams.shape[2], cams.shape[3]), device=cams.device))
        elif method == 'max':
            parent_cams.append(torch.max(sub_cams, dim=1, keepdim=True)[0])
        elif method == 'mean':
            parent_cams.append(torch.mean(sub_cams, dim=1, keepdim=True))
        else:
            raise ValueError(f"Unknown merge method: {method}")
    parent_cams = torch.cat(parent_cams, dim=1)
    if parent_cams.size(1) == 4:  # Add background channel if not present
        cam_max = torch.max(parent_cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        parent_cams = torch.cat([parent_cams, bg_cam], dim=1)
    return parent_cams

def merge_to_parent_predictions(predictions, k_list, method='max'):
    if not k_list or sum(k_list) != predictions.shape[1]:
        logger.warning(f"Invalid k_list: {k_list}, predictions shape: {predictions.shape}. Returning predictions unchanged.")
        return predictions
    cumsum_k = np.cumsum([0] + k_list)
    parent_preds = []
    for i in range(len(k_list)):
        start_idx = cumsum_k[i]
        end_idx = cumsum_k[i + 1]
        if end_idx > predictions.shape[1]:
            logger.error(f"Index out of bounds: k_list={k_list}, start_idx={start_idx}, end_idx={end_idx}, predictions shape={predictions.shape}")
            return predictions
        sub_preds = predictions[:, start_idx:end_idx]
        if method == 'max':
            parent_preds.append(torch.max(sub_preds, dim=1, keepdim=True)[0])
        elif method == 'mean':
            parent_preds.append(torch.mean(sub_preds, dim=1, keepdim=True))
        else:
            raise ValueError(f"Unknown merge method: {method}")
    return torch.cat(parent_preds, dim=1)

def expand_parent_to_subclass_labels(parent_labels, k_list):
    cumsum_k = np.cumsum([0] + k_list)
    subclass_labels = []
    for i in range(len(k_list)):
        n_subclasses = k_list[i]
        for _ in range(n_subclasses):
            subclass_labels.append(parent_labels[:, i:i+1])
    return torch.cat(subclass_labels, dim=1)
