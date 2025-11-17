import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InfoNCELossFG(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossFG with temperature: {temperature}')

    def forward(self, fg_img_feature, fg_pro_feature, bg_pro_feature):
        positive_sims = torch.tensor(0., requires_grad=True, device=fg_img_feature.device)
        negative_sims = torch.tensor(0., requires_grad=True, device=fg_img_feature.device)

        fg_img_feature = fg_img_feature / fg_img_feature.norm(dim=-1, keepdim=True)  # [N, D]
        batch_size = fg_img_feature.shape[0]
        for i in range(batch_size):
            curr_fg_img = fg_img_feature[i:i+1]  # [1, D]
            curr_fg_pro = fg_pro_feature[i:i+1]  # [1, D]
            curr_bg_pro = bg_pro_feature[i]  # [L, D]
            fg_img_fg_pro_logits = curr_fg_img @ curr_fg_pro.t()  # [1, 1]
            fg_img_bg_pro_logits = curr_fg_img @ curr_bg_pro.t()  # [1, L]
            positive_sims = positive_sims + torch.exp(fg_img_fg_pro_logits / self.temperature).sum()
            negative_sims = negative_sims + torch.exp(fg_img_fg_pro_logits / self.temperature).sum() + torch.exp(fg_img_bg_pro_logits / self.temperature).sum()
        loss = -torch.log(positive_sims / negative_sims + 1e-8)  # Add small epsilon to avoid log(0)
        return loss

class InfoNCELossBG(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossBG with temperature: {temperature}')

    def forward(self, bg_img_feature, fg_pro_feature, bg_pro_feature):
        positive_sims = torch.tensor(0., requires_grad=True, device=bg_img_feature.device)
        negative_sims = torch.tensor(0., requires_grad=True, device=bg_img_feature.device)

        bg_img_feature = bg_img_feature / bg_img_feature.norm(dim=-1, keepdim=True)  # [N, D]
        batch_size = bg_img_feature.shape[0]
        for i in range(batch_size):
            curr_bg_img = bg_img_feature[i:i+1]  # [1, D]
            curr_fg_pro = fg_pro_feature[i:i+1]  # [1, D]
            curr_bg_pro = bg_pro_feature[i]  # [L, D]
            bg_img_bg_pro_logits = curr_bg_img @ curr_bg_pro.t()  # [1, L]
            bg_img_fg_pro_logits = curr_bg_img @ curr_fg_pro.t()  # [1, 1]
            positive_sims = positive_sims + torch.exp(bg_img_bg_pro_logits / self.temperature).mean()
            negative_sims = negative_sims + torch.exp(bg_img_bg_pro_logits / self.temperature).mean() + torch.exp(bg_img_fg_pro_logits / self.temperature).sum()
        loss = -torch.log(positive_sims / negative_sims + 1e-8)  # Add small epsilon to avoid log(0)
        return loss

def diversity_loss_fn(feature_map, l_fea, pseudo_mask):
    """
    Compute diversity loss to encourage diverse feature representations.
    Args:
          feature_map (torch.Tensor): Feature map from the backbone, e.g., [batch_size, channels, H, W]
          l_fea (torch.Tensor): Learned feature embeddings, e.g., [num_classes, feature_dim]
          pseudo_mask (torch.Tensor): Pseudo mask, e.g., [batch_size, H, W]
    Returns:
          torch.Tensor: Diversity loss scalar
    """
    batch_size, channels, h, w = feature_map.shape
    num_classes = l_fea.shape[0]

    # Flatten feature map and mask for processing
    feature_map_flat = feature_map.view(batch_size, channels, -1)  # [batch_size, channels, H*W]
    pseudo_mask_flat = pseudo_mask.view(batch_size, -1)  # [batch_size, H*W]

    # Compute per-class features weighted by pseudo mask
    class_features = []
    for c in range(num_classes):
        mask_c = (pseudo_mask_flat == c).float()  # Binary mask for class c
        if torch.sum(mask_c) == 0:  # Avoid division by zero
            continue
        weighted_features = torch.sum(feature_map_flat * mask_c.unsqueeze(1), dim=-1) / (torch.sum(mask_c) + 1e-8)
        class_features.append(weighted_features)
    if not class_features:
        return torch.tensor(0.0, device=feature_map.device)
    class_features = torch.stack(class_features)  # [num_classes, batch_size, channels]

    # Compute pairwise cosine similarity
    class_features = F.normalize(class_features, dim=-1)
    similarity_matrix = torch.bmm(class_features, class_features.transpose(1, 2))  # [num_classes, batch_size, batch_size]
    # Penalize high similarity (encourage diversity)
    diversity_loss = torch.mean(torch.relu(similarity_matrix - 0.1))  # Threshold at 0.1

    return diversity_loss
