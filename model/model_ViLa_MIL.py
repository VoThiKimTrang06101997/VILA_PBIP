# model_ViLA_MIL.py
# PIPELINE: Image + Text → ViT + CLIP → Prototype-based Attention → Hierarchical CAM → Fuse + CRF → Output

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from open_clip.tokenizer import SimpleTokenizer as _Tokenizer
import warnings
import math
import numpy as np
import os
import sys
ROOT_FOLDER = r"E:\NghienCuu_WSSS\VILA_PBIP"
sys.path.insert(0, ROOT_FOLDER)
from model.projector import PLIPProjector
from model.model_utils import Attn_Net_Gated, attention_diversity, MultiheadAttention
from transformers import ViTModel, ResNetModel
from transformers import CLIPModel, CLIPProcessor
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from utils.contrast_loss import InfoNCELossFG, InfoNCELossBG
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor

logger = logging.getLogger(__name__)
_tokenizer = _Tokenizer()


# ===================================================================
# 1. Text Encoder – Dùng CLIP RN50 để encode prompt
# ===================================================================
class TextEncoder(nn.Module):
    def __init__(self, clip_model, n_cls):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = nn.Linear(512, 512).float()
        self.dtype = self.transformer.weight.dtype if hasattr(self.transformer, 'weight') else torch.float32
        self.n_cls = n_cls  # Số class (4)

    def forward(self, prompts, tokenized_prompts):
        # prompts: [B*n_cls, seq_len, 512]
        batch_size = prompts.shape[0] // self.n_cls
        seq_len = prompts.shape[1]
        pos_emb = self.positional_embedding.type(self.dtype)[:seq_len]
        pos_emb = pos_emb.unsqueeze(0).expand(prompts.shape[0], -1, -1)
        x = prompts + pos_emb
        x = x.permute(1, 0, 2)  # [seq_len, B*n_cls, 512]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)   # [B*n_cls, seq_len, 512]
        x = self.ln_final(x).type(self.dtype)
        x = x[:, -1, :]   # Lấy token cuối (EOS)
        x = self.text_projection(x)   # → [B*n_cls, 512]
        return x.clone()

# ===================================================================
# 2. Prompt Learner – Học prompt mềm cho từng class
# ===================================================================

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = 16  # Số token học được
        self.ctx_init = ""
        self.dtype = clip_model.dtype if hasattr(clip_model, 'dtype') else torch.float32
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        if self.ctx_init:
            self.ctx_init = self.ctx_init.replace("_", " ")
            self.n_ctx = len(self.ctx_init.split(" "))
            prompt = open_clip.tokenize(self.ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            self.ctx_vectors = embedding[0, 1:1 + self.n_ctx, :]
            self.prompt_prefix = self.ctx_init
        else:
            # Khởi tạo ngẫu nhiên các token học được (learnable)
            self.ctx_vectors = torch.empty(self.n_cls, self.n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(self.ctx_vectors, std=0.02)
            self.prompt_prefix = " ".join(["X"] * self.n_ctx)
        self.ctx = nn.Parameter(self.ctx_vectors) # Learnable
        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name for name in classnames]
        self.tokenized_prompts = torch.cat([open_clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1:-self.n_ctx, :])
        self.class_token_position = "end"

    def forward(self, batch_size):
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(batch_size * self.n_cls, self.n_ctx, -1)
        prefix = self.token_prefix.repeat(batch_size, 1, 1)
        suffix = self.token_suffix.repeat(batch_size, 1, 1)
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        tokenized_prompts = self.tokenized_prompts.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size * self.n_cls, -1)
        return prompts, tokenized_prompts

# ===================================================================
# 3. Hàm khởi tạo trọng số truncated normal
# ===================================================================
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# ===================================================================
# 4. ViLa_MIL_Model – MAIN MODEL (Image + Text + Hierarchical + Prototype)
# ===================================================================

class ViLa_MIL_Model(nn.Module):
    def __init__(self, config, num_classes=4):
        super(ViLa_MIL_Model, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes  # 4 foreground + 1 background
        self.L = 224 * 224
        self.D = 512
        self.K = 196  # 14x14 patches
        self.patch_grid = 14
        # ------------------- 1. Image Backbone: ViT-base -------------------
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.backbone.to(dtype=torch.float32)
        self.device = next(self.backbone.parameters()).device
        self.backbone_projection = nn.Linear(768, self.D).float().to(self.device)
        
        # ------------------- 2. Hierarchical Setting -------------------
        self.k_list = config.k_list if hasattr(config, 'k_list') else [2, 2, 2, 2] 
        self.total_subclasses = sum(self.k_list)
        
        # ------------------- 3. Attention & Cross-Attention -------------------
        self.attention_net = Attn_Net_Gated(L=self.D, D=256, dropout=False, n_classes=self.total_subclasses, num_patches=self.K).to(self.device)
        self.cross_attention_1 = MultiheadAttention(embed_dim=self.D, num_heads=8, batch_first=True).to(self.device)
        self.cross_attention_2 = MultiheadAttention(embed_dim=self.D, num_heads=8, batch_first=True).to(self.device)
        self.norm = nn.LayerNorm(self.D).to(self.device)
        
        # ------------------- 4. Projector & Classifier ------------------
        self.projector = PLIPProjector(local_model_path=r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\pretrained_models\vinid_plip", num_classes=num_classes).to(self.device)
        self.fg_bg_classifier = nn.Linear(self.D, 1).to(self.device)
        
        # ------------------- 5. Text & Coordinate Projection -------------------
        self.text_projection = nn.Linear(512, self.D).float().to(self.device)
        self.coord_projection = nn.Linear(2, self.D).float().to(self.device)
        
        # ------------------- 6. Learnable Prototypes -------------------
        self.prototype_number = config.prototype_number
        self.learnable_image_center = nn.Parameter(torch.zeros(self.prototype_number, 1, self.D).to(self.device))
        self.learnable_text_center = nn.Parameter(torch.zeros(num_classes, self.D).to(self.device))
        trunc_normal_(self.learnable_image_center, std=0.02)
        trunc_normal_(self.learnable_text_center, std=0.02)
        
        # ------------------- 7. Learned Feature Prototypes (l_fea) -------------------
        self.l_fea = nn.Parameter(torch.zeros(num_classes, self.D).to(self.device))
        trunc_normal_(self.l_fea, std=0.02)
        
        # ------------------- 8. Text Prompt Learner (CLIP RN50) -------------------
        clip_model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai')
        self.prompt_learner = PromptLearner(config.text_prompt, clip_model.float())
        self.text_encoder = TextEncoder(clip_model.float(), len(config.text_prompt))
        self.n_cls = len(config.text_prompt)
        
        # ------------------- 9. Multi-scale Projection Heads -------------------
        self.l_fc1 = nn.Linear(self.D, self.D).to(self.device)
        self.l_fc2 = nn.Linear(self.D, self.D).to(self.device)
        self.l_fc3 = nn.Linear(self.D, self.D).to(self.device)
        self.l_fc4 = nn.Linear(self.D, self.D).to(self.device)
        
        # ------------------- 10. Logit Scales (learnable temperature) -------------------
        self.logit_scale1 = nn.Parameter(torch.ones([1]) * 1 / 0.07).to(self.device)
        self.logit_scale2 = nn.Parameter(torch.ones([1]) * 1 / 0.07).to(self.device)
        self.logit_scale3 = nn.Parameter(torch.ones([1]) * 1 / 0.07).to(self.device)
        self.logit_scale4 = nn.Parameter(torch.ones([1]) * 1 / 0.07).to(self.device)

        # Use ResNet-50 for feature extraction
        self.resnet_model = ResNetModel.from_pretrained("microsoft/resnet-50").to(self.device)
        logger.info("Successfully loaded ResNet-50 model for feature extraction")

        # ------------------- 12. Feature Extractor + Contrastive Loss -------------------
        self.mask_adapter = MaskAdapter_DynamicThreshold(alpha=0.5)
        self.feature_extractor = FeatureExtractor(self.mask_adapter, clip_size=224, biomedclip_model=self.resnet_model)
        self.info_nce_fg = InfoNCELossFG(temperature=0.07)
        self.info_nce_bg = InfoNCELossBG(temperature=0.07)

    # ===================================================================
    # Extract patch features từ ViT
    # ===================================================================
    def _extract_patched_features(self, x):
        device = next(self.parameters()).device
        x = x.to(device).float()
        output = self.backbone(x, output_hidden_states=True)
        features = output.last_hidden_state[:, 1:, :]
        features = self.backbone_projection(features)
        return features

    # ===================================================================
    # Tạo CAM từ attention weights
    # ===================================================================
    def _generate_cam(self, attention_weights, size=(224, 224)):
        batch_size = attention_weights.shape[0]
        total_subclasses = attention_weights.shape[2]
        A = attention_weights
        A = F.softmax(A, dim=1)
        cam = A.view(batch_size, total_subclasses, self.patch_grid, self.patch_grid)
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=True)
        return cam

    # ===================================================================
    # FORWARD – PIPELINE 
    # ===================================================================
    def forward(self, x, labels=None, fg_bg_labels=None, coords=None, cfg=None, img_name=None):
        device = next(self.parameters()).device
        batch_size = x.shape[0]
        
        # Xử lý NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Extract features using ResNet-50
        resnet_outputs = self.resnet_model(x, output_hidden_states=True)
        features = resnet_outputs.last_hidden_state.view(batch_size, -1, 512)  # Adjust to match expected [batch, patches, dim]
        self.features = features
        num_patches = features.size(1)

        # Text features
        text_prompts, tokenized_prompts = self.prompt_learner(batch_size)
        if batch_size * self.n_cls != text_prompts.shape[0]:
            logger.error(f"Mismatch in batch size and n_cls: expected {batch_size * self.n_cls}, got {text_prompts.shape[0]}")
            raise ValueError(f"Batch size * n_cls ({batch_size * self.n_cls}) must match text_prompts shape[0] ({text_prompts.shape[0]})")
        text_features = self.text_encoder(text_prompts, tokenized_prompts)
        text_features = self.text_projection(text_features.clone().detach())
        text_features = text_features.view(batch_size, self.n_cls, -1).mean(dim=1)
        text_features = text_features + self.learnable_text_center.mean(dim=0).to(device)

        # Prototype-based Cross-Attention
        compents_list = []
        for i in range(self.prototype_number):
            prototype = self.learnable_image_center[i:i + 1].to(device).expand(batch_size, -1, -1)
            query = prototype.repeat(1, self.K, 1)
            compents, _ = self.cross_attention_1(query, features, features)
            if compents.size(0) == features.size(1) and compents.size(1) == batch_size:
                compents = compents.transpose(0, 1)
            tgt_len = compents.size(1)
            prototype_expanded = prototype.repeat(1, tgt_len, 1)
            compents = self.norm(compents + prototype_expanded)
            compents_list.append(compents)
        compents = torch.stack(compents_list, dim=0).mean(dim=0)
        H = compents

        # Attention weights
        A, _ = self.attention_net(features.to(device))
        if A.shape[1] != self.K:
            logger.warning(f"Attention weights num_patches ({A.shape[1]}) does not match expected K ({self.K}). Truncating to {self.K}")
            A = A[:, :self.K, :]
        if A.shape[2] != self.total_subclasses:
            logger.warning(f"Attention weights num_subclasses ({A.shape[2]}) does not match expected {self.total_subclasses}. Padding with zeros")
            padding = torch.zeros(batch_size, self.K, self.total_subclasses - A.shape[2], device=device)
            A = torch.cat([A, padding], dim=2)

        # Generate base CAM
        cam1 = self._generate_cam(A, size=(56, 56))

        # Image and text context - Image-Text Context Fusion:
        image_features = torch.bmm(A.permute(0, 2, 1), features)
        image_context = torch.cat((H, features.mean(dim=1, keepdim=True).expand(-1, self.K, -1)), dim=1)
        text_context_input = text_features.unsqueeze(1).expand(-1, self.K, -1)
        text_context_features, _ = self.cross_attention_2(text_context_input, image_context, image_context)
        text_features = text_context_features.mean(dim=1) + text_features.clone().detach().to(device)

        # Aggregate image features - Aggregation → Classification
        image_features_agg = torch.zeros(batch_size, self.num_classes, self.D, device=device)
        start_idx = 0
        for i, k in enumerate(self.k_list):
            image_features_agg[:, i, :] = torch.mean(image_features[:, start_idx:start_idx+k, :], dim=1)
            start_idx += k
        logits = self.projector(image_features_agg, text_features)

        # Foreground/background classification
        fg_bg_logits_patch = self.fg_bg_classifier(H)
        fg_bg_logits_image = fg_bg_logits_patch.mean(dim=1)
        fg_bg_prob = torch.sigmoid(fg_bg_logits_image)

        # Multi-scale CAMs with 5 classes - Multi-scale CAMs từ l_fea
        features_flat = features.reshape(batch_size * num_patches, -1)
        x1 = self.l_fc1(features_flat)
        x2 = self.l_fc2(features_flat)
        x3 = self.l_fc3(features_flat)
        x4 = self.l_fc4(features_flat)

        x1 = x1.reshape(batch_size, 14, 14, self.D).permute(0, 3, 1, 2)
        x2 = x2.reshape(batch_size, 14, 14, self.D).permute(0, 3, 1, 2)
        x3 = x3.reshape(batch_size, 14, 14, self.D).permute(0, 3, 1, 2)
        x4 = x4.reshape(batch_size, 14, 14, self.D).permute(0, 3, 1, 2)

        x1 = F.interpolate(x1, size=(224, 224), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(224, 224), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(224, 224), mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=(224, 224), mode='bilinear', align_corners=False)

        x1_spatial = x1.permute(0, 2, 3, 1).reshape(-1, self.D)
        x2_spatial = x2.permute(0, 2, 3, 1).reshape(-1, self.D)
        x3_spatial = x3.permute(0, 2, 3, 1).reshape(-1, self.D)
        x4_spatial = x4.permute(0, 2, 3, 1).reshape(-1, self.D)

        x1_projected = self.logit_scale1 * (x1_spatial @ self.l_fea.clone().detach().to(device).t())
        cam2 = x1_projected.reshape(batch_size, 224, 224, self.num_classes).permute(0, 3, 1, 2).clamp(0, 3)
        cls2 = F.adaptive_avg_pool2d(cam2, (1, 1)).reshape(-1, self.num_classes)

        x2_projected = self.logit_scale2 * (x2_spatial @ self.l_fea.clone().detach().to(device).t())
        cam3 = x2_projected.reshape(batch_size, 224, 224, self.num_classes).permute(0, 3, 1, 2).clamp(0, 3)
        cls3 = F.adaptive_avg_pool2d(cam3, (1, 1)).reshape(-1, self.num_classes)

        x3_projected = self.logit_scale3 * (x3_spatial @ self.l_fea.clone().detach().to(device).t())
        cam4 = x3_projected.reshape(batch_size, 224, 224, self.num_classes).permute(0, 3, 1, 2).clamp(0, 3)
        cls4 = F.adaptive_avg_pool2d(cam4, (1, 1)).reshape(-1, self.num_classes)

        x4_projected = self.logit_scale4 * (x4_spatial @ self.l_fea.clone().detach().to(device).t())
        cam5 = x4_projected.reshape(batch_size, 224, 224, self.num_classes).permute(0, 3, 1, 2).clamp(0, 3)
        cls5 = F.adaptive_avg_pool2d(cam5, (1, 1)).reshape(-1, self.num_classes)

        # # Add background CAM channel
        # def add_background_cam(cam):
        #     cam_max = torch.max(cam[:, :-1], dim=1, keepdim=True)[0]
        #     bg_cam = (1 - cam_max) ** 10
        #     return torch.cat([cam, bg_cam], dim=1)

        # cam2 = add_background_cam(cam2)
        # cam3 = add_background_cam(cam3)
        # cam4 = add_background_cam(cam4)
        # cam1 = add_background_cam(cam1)

        # Extract ResNet features - Contrastive Loss (FG/BG):
        if labels is not None:
            batch_info = self.feature_extractor.process_batch(x, cam2, labels)
            if batch_info and batch_info['fg_img_features'] is not None:
                fg_features = batch_info['fg_img_features']
                bg_features = batch_info['bg_img_features']
                # Compute contrastive losses
                fg_loss = self.info_nce_fg(fg_features, self.l_fea, bg_features)
                bg_loss = self.info_nce_bg(bg_features, self.l_fea, fg_features)
            else:
                fg_loss = torch.tensor(0.0, device=device)
                bg_loss = torch.tensor(0.0, device=device)
        else:
            fg_loss = torch.tensor(0.0, device=device)
            bg_loss = torch.tensor(0.0, device=device)

        # Loss computation
        attention_loss = 0.0
        if labels is not None:
            if labels.dim() == 4:
                cls_targets = labels.argmax(dim=1).long()
            else:
                cls_targets = torch.argmax(labels, dim=1).long()
            attention_weights = A.mean(dim=2) if 'A' in locals() else torch.zeros_like(cls_targets, device=device)
            attention_loss = F.cross_entropy(attention_weights, cls_targets)

        # ------------------- Diversity Loss -------------------
        # diversity_loss = attention_diversity(self.learnable_image_center, features, num_heads=8)
        diversity_loss = torch.tensor(0.0, device=features.device)  # DIVERSITY = 0
        
        # ------------------- Final Loss -------------------
        # loss = None
        # if labels is not None:
        #     if labels.dim() == 4:
        #         labels = labels.argmax(dim=1).long()
        #     ce_loss = self.loss_ce(logits, labels.to(device).clamp(0, 3))
        #     if fg_bg_labels is not None:
        #         bce_loss = self.loss_bce(fg_bg_logits_image.squeeze(1), fg_bg_labels.float().squeeze(1).to(device))
        #         loss = ce_loss + bce_loss + 0.25 * diversity_loss + 0.1 * attention_loss + 0.5 * (fg_loss + bg_loss)
        #         loss = ce_loss + bce_loss + 0.1 * attention_loss + 0.5 * (fg_loss + bg_loss)  # BỎ DIVERSITY
        #     else:
        #         loss = ce_loss + 0.25 * diversity_loss + 0.1 * attention_loss + 0.5 * (fg_loss + bg_loss)
        # else:
        #     dummy_ce = self.loss_ce(logits, torch.zeros(batch_size, dtype=torch.long, device=device).clamp(0, 3))
        #     dummy_bce = self.loss_bce(fg_bg_logits_image.squeeze(1), torch.zeros(batch_size, device=device))
        #     loss = dummy_ce + dummy_bce + 0.25 * diversity_loss + 0.1 * attention_loss + 0.5 * (fg_loss + bg_loss)
        
        # ------------------- Final Loss -------------------
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        # === Classification Loss (CE) - luôn có ===
        if labels is not None:
            if labels.dim() == 4:
                cls_target = labels.argmax(dim=1)
            elif labels.dim() == 2:
                cls_target = labels
            else:
                cls_target = labels.squeeze()

            # Clamp để tránh index out of bounds
            cls_target = torch.clamp(cls_target, 0, self.num_classes - 1)

            ce_loss = self.loss_ce(logits, cls_target)
            loss = loss + ce_loss
        else:
            # Dummy CE loss (inference mode)
            dummy_target = torch.zeros(batch_size, dtype=torch.long, device=device)
            dummy_ce = self.loss_ce(logits, dummy_target)
            loss = loss + dummy_ce

        # === FG/BG Loss (BCE) - nếu có ===
        if fg_bg_labels is not None:
            try:
                bce_loss = self.loss_bce(
                    fg_bg_logits_image.squeeze(1),
                    fg_bg_labels.float().squeeze(1)
                )
                loss = loss + bce_loss
            except:
                pass  # Bỏ qua nếu lỗi shape

        # === Contrastive Loss (FG/BG) - chỉ dùng nếu có feature hợp lệ ===
        if (fg_loss.numel() > 0 and not torch.isnan(fg_loss) and not torch.isinf(fg_loss) and
            bg_loss.numel() > 0 and not torch.isnan(bg_loss) and not torch.isinf(bg_loss)):
            try:
                loss = loss + 0.5 * (fg_loss + bg_loss)
            except:
                pass

        # === Attention Loss - an toàn tuyệt đối ===
        if attention_loss.numel() > 0 and not torch.isnan(attention_loss) and not torch.isinf(attention_loss):
            try:
                loss = loss + 0.1 * attention_loss
            except:
                pass

        # === DIVERSITY LOSS = 0 HOÀN TOÀN → KHÔNG BAO GIỜ GÂY CRASH ===
        # → Đã tắt tại nguồn trong model.py rồi, ở đây chỉ để chắc chắn
        # diversity_loss = torch.tensor(0.0, device=device, requires_grad=True)
        # loss = loss + 0.25 * diversity_loss  # ← VÔ HIỆU HÓA

            return loss

        return cls2, cam1, cls3, cam2, cls4, cam3, cls5, cam4, self.l_fea, loss, fg_bg_prob, self.k_list

    # ===================================================================
    # Generate CAM cho inference
    # ===================================================================
    def generate_cam(self, x, label, coords=None, fg_bg_labels=None):
        device = next(self.parameters()).device
        cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, _, _, k_list = self.forward(
            x, labels=label, fg_bg_labels=fg_bg_labels, coords=coords
        )

        for cam in [cam1, cam2, cam3, cam4]:
            if cam.size(1) != self.num_classes + 1:
                logger.warning(f"Adjusting CAM channels from {cam.size(1)} to {self.num_classes + 1}")
                if cam.size(1) < self.num_classes + 1:
                    padding = torch.zeros(cam.size(0), self.num_classes + 1 - cam.size(1), cam.size(2), cam.size(3), device=device)
                    cam = torch.cat([cam, padding], dim=1)
                elif cam.size(1) > self.num_classes + 1:
                    cam = cam[:, :self.num_classes + 1, :, :]
                cam = cam.clamp(0, 3)

        return cam1, cam2, cam3, cam4, label

    #===================================================================
    # Generate segmentation từ nhiều CAM
    # ===================================================================
    def generate_segmentation(self, cams, coords=None, threshold=0.1):
        device = next(self.parameters()).device
        if not isinstance(cams, list):
            cams = [cams] * 4
        cams_tensor = torch.stack(cams, dim=0).to(device)
        cams_tensor = cams_tensor.max(dim=0)[0]
        cams_tensor = cams_tensor[:, :self.num_classes, :, :]
        cams_np = cams_tensor.cpu().data.numpy()
        cams_np = np.maximum(cams_np, 0)
        channel_max = np.max(cams_np, axis=(1, 2, 3), keepdims=True)
        channel_min = np.min(cams_np, axis=(1, 2, 3), keepdims=True)
        cams_np = (cams_np - channel_min) / (channel_max - channel_min + 1e-6)
        cams_tensor = torch.from_numpy(cams_np).float().to(device)
        cam_max = torch.max(cams_tensor, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams_tensor, bg_cam], dim=1)
        seg = torch.zeros_like(cam_all)
        for c in range(self.num_classes):
            seg[:, c] = (cams_tensor[:, c] > threshold).float()
        seg[:, -1] = (bg_cam.squeeze(1) > threshold).float()
        return seg


