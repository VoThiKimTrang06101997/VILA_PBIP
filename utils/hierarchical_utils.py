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
    if cams is None or cams.numel() == 0:
        return torch.zeros((1, 4, 224, 224), device='cuda' if torch.cuda.is_available() else 'cpu')

    if cams.size(1) <= 4:
        return cams[:, :4]

    if not k_list or sum(k_list) != cams.shape[1]:
        return cams[:, :4]

    try:
        cumsum_k = np.cumsum([0] + k_list)
        parent_cams = []
        for i in range(len(k_list)):
            s, e = cumsum_k[i], min(cumsum_k[i+1], cams.shape[1])
            sub = cams[:, s:e]
            if method == 'max':
                parent_cams.append(torch.max(sub, dim=1, keepdim=True)[0])
            else:
                parent_cams.append(torch.mean(sub, dim=1, keepdim=True))
        return torch.cat(parent_cams, dim=1)[:, :4]
    except:
        return cams[:, :4]


def merge_to_parent_predictions(predictions, k_list, method='max'):
    if predictions is None or predictions.numel() == 0:
        logger.warning("Predictions is None or empty. Returning zero tensor.")
        return torch.zeros((1, 4), device='cuda' if torch.cuda.is_available() else 'cpu')

    if predictions.dim() < 2:
        logger.warning(f"Invalid predictions dim: {predictions.dim()}. Returning first 4.")
        return predictions[:, :4] if predictions.shape[0] > 4 else predictions

    if predictions.shape[1] <= 4:
        return predictions[:, :4]

    if not k_list or sum(k_list) != predictions.shape[1]:
        logger.warning(f"k_list mismatch. Using first 4 channels.")
        return predictions[:, :4]

    try:
        cumsum_k = np.cumsum([0] + k_list)
        parent_preds = []
        for i in range(len(k_list)):
            s, e = cumsum_k[i], min(cumsum_k[i+1], predictions.shape[1])
            sub = predictions[:, s:e]
            if method == 'max':
                parent_preds.append(torch.max(sub, dim=1, keepdim=True)[0])
            else:
                parent_preds.append(torch.mean(sub, dim=1, keepdim=True))
        return torch.cat(parent_preds, dim=1)[:, :4]
    except Exception as e:
        logger.error(f"Error in merge_to_parent_predictions: {e}. Returning first 4.")
        return predictions[:, :4]
    

def expand_parent_to_subclass_labels(parent_labels, k_list):
    cumsum_k = np.cumsum([0] + k_list)
    subclass_labels = []
    for i in range(len(k_list)):
        n_subclasses = k_list[i]
        for _ in range(n_subclasses):
            subclass_labels.append(parent_labels[:, i:i+1])
    return torch.cat(subclass_labels, dim=1)
