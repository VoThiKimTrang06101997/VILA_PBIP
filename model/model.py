# Pipeline: Image + Text → SegFormer → Multi-scale Attention → Learned Prototypes → Hierarchical CAM → Fuse + CRF → _cam.png + _mask.png

import pickle as pkl
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from model.model_ViLa_MIL import PromptLearner, TextEncoder # Text branch (CLIP RN50)
from model.segform import mix_transformer   # SegFormer encoder
from model.model_utils import Attn_Net_Gated, attention_diversity # Attention + diversity loss
from model.projector import PLIPProjector
import open_clip
import numpy as np
import sys
import os
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor, SegformerForSemanticSegmentation
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from utils.hierarchical_utils import merge_subclass_cams_to_parent, merge_to_parent_predictions
from transformers import ResNetModel

ROOT_FOLDER = r"E:\NghienCuu_WSSS\VILA_PBIP"
sys.path.insert(0, ROOT_FOLDER)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================================================================
# 1. Adaptive Layer – Dùng để project l_fea về không gian của từng scale
# ===================================================================

class AdaptiveLayer(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ===================================================================
# 2. ClsNetwork – MAIN MODEL (Image + Text Branch + Hierarchical CAM)
# ===================================================================
class ClsNetwork(nn.Module):
    def __init__(
        self,
        backbone='mit_b1',
        cls_num_classes=4,  # 4 foreground classes (TUM, STR, LYM, NEC)
        stride=[4, 2, 2, 1],
        pretrained=True,
        n_ratio=0.5,
        l_fea_path=None,
        text_prompt=None,
        num_prototypes_per_class=10,
        prototype_feature_dim=512,
        k_list=None
    ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.stride = stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims
        if pretrained:
            try:
                # ------------------- 1. Image Backbone: SegFormer -------------------
                segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b1")
                state_dict = segformer_model.segformer.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if 'decode_head' not in k and k in self.encoder.state_dict()}
                self.encoder.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded SegFormer pretrained weights from Hugging Face")
            except Exception as e:
                logger.error(f"Failed to load SegFormer pretrained weights from Hugging Face: {str(e)}. Using untrained encoder.")
        self.pooling = F.adaptive_avg_pool2d

        # Feature Extraction dùng Resnet-50
        self.resnet_model = ResNetModel.from_pretrained("microsoft/resnet-50").to(self.device)
        logger.info("Successfully loaded ResNet-50 model for feature extraction")

        self.mask_adapter = MaskAdapter_DynamicThreshold(alpha=0.5)
        self.feature_extractor = FeatureExtractor(self.mask_adapter, clip_size=224, biomedclip_model=self.resnet_model)

        # ------------------- 5. Adaptive projection cho từng scale -------------------
        self.l_fc1 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[3])

        self.l_fea = None
        # ------------------- 3. Hierarchical Setting -------------------
        self.k_list = k_list if k_list is not None else [1, 1, 1, 1]  # Mỗi class có 1 subclass
        if sum(self.k_list) != cls_num_classes:
            logger.warning(f"k_list sum {sum(self.k_list)} does not match cls_num_classes {cls_num_classes}. Adjusting k_list.")
            self.k_list = [1] * cls_num_classes
        self.cumsum_k = np.cumsum(self.k_list)  # = số channel trong attention

        
        # ------------------- 4. Learned Prototypes (l_fea) -------------------
        if l_fea_path:
            try:
                with open(f"./features/image_features/{l_fea_path}.pkl", "rb") as lf:
                    info = pkl.load(lf)
                    prototype_features = info['features'].to(self.device)
                    if prototype_features.shape[1] != prototype_feature_dim:
                        logger.warning(f"Prototype feature dim ({prototype_features.shape[1]}) does not match config ({prototype_feature_dim}). Reshaping.")
                        prototype_features = F.interpolate(prototype_features.unsqueeze(0), size=(prototype_feature_dim, prototype_features.shape[2]), mode='linear', align_corners=False).squeeze(0)
                    self.l_fea = nn.Parameter(prototype_features, requires_grad=True)
                    if 'k_list' in info and info['k_list'] != self.k_list:
                        logger.warning(f"Loaded k_list {info['k_list']} from {l_fea_path} does not match config k_list {self.k_list}. Using config k_list.")
                    self.cumsum_k = np.cumsum(self.k_list)
            except FileNotFoundError:
                logger.error(f"Prototype file not found at {l_fea_path}. Proceeding without prototypes.")
            except Exception as e:
                logger.error(f"Error loading prototype file {l_fea_path}: {str(e)}. Proceeding without prototypes.")
        else:
            logger.warning("l_fea_path not provided. Initializing empty prototypes.")
            self.l_fea = nn.Parameter(torch.zeros(self.cls_num_classes, prototype_feature_dim), requires_grad=True)
            self.cumsum_k = np.cumsum(self.k_list)

        self.total_classes = sum(self.k_list)

        # ------------------- 2. Text Branch (CLIP RN50) -------------------
        self.text_prompt = text_prompt if text_prompt is not None else [
            "A WSI of Tumor with irregular shapes, dense cellularity, and heterogeneous staining.",
            "A WSI of Stroma with fibrous tissue and low cellularity.",
            "A WSI of Lymphocyte with small dark clusters and speckled appearance.",
            "A WSI of Necrosis with pale amorphous zones and loss of structure."
        ]
        clip_model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai')
        self.prompt_learner = PromptLearner(self.text_prompt, clip_model.float())
        self.text_encoder = TextEncoder(clip_model.float(), len(self.text_prompt))
        self.text_projection = nn.Linear(512, prototype_feature_dim).float()

        # ------------------- 6. Attention Nets cho 4 scale -------------------
        self.attention_net1 = Attn_Net_Gated(in_channels=self.in_channels[0], D=256, dropout=False, n_classes=self.total_classes, num_patches=3136).to(self.device)
        self.attention_net2 = Attn_Net_Gated(in_channels=self.in_channels[1], D=256, dropout=False, n_classes=self.total_classes, num_patches=784).to(self.device)
        self.attention_net3 = Attn_Net_Gated(in_channels=self.in_channels[2], D=256, dropout=False, n_classes=self.total_classes, num_patches=196).to(self.device)
        self.attention_net4 = Attn_Net_Gated(in_channels=self.in_channels[3], D=256, dropout=False, n_classes=self.total_classes, num_patches=196).to(self.device)

        # ------------------- 7. Logit scales (learnable temperature) -------------------
        # Dùng để scale cosine similarity → giống CLIP
        self.logit_scale1 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale2 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale3 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale4 = nn.Parameter(torch.ones([1]) * 1 / 0.07)

        # Khởi tạo trọng số
        self.apply(self._init_weights)
        
        # ------------------- 8. Loss functions -------------------
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()

    # ===================================================================
    # Hàm hỗ trợ: Tính số patch từ feature map shape
    # ===================================================================
    def _get_num_patches(self, feature_shape):
        if len(feature_shape) == 4:
            _, _, h, w = feature_shape
            return h * w
        logger.warning(f"Invalid feature shape {feature_shape}. Using default num_patches=3136.")
        return 3136

    # ===================================================================
    # Khởi tạo trọng số
    # ===================================================================
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ===================================================================
    # Tạo param groups cho optimizer (weight decay riêng cho bias)
    # ===================================================================
    def get_param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    # ===================================================================
    # FORWARD – PIPELINE (Image + Hierarchical + CAM + Fuse)
    # ===================================================================
    def forward(self, x, labels=None, cfg=None):
        batch_size = x.shape[0]
        device = x.device

        # ------------------- 1. SegFormer Encoder → 4 scale features -------------------
        _x, _attns = self.encoder(x)  # List 4 tensor
        imshape = [_.shape for _ in _x]  # [B,C,H,W] mỗi scale
        image_features = [_.permute(0, 2, 3, 1).reshape(batch_size, -1, _.shape[1]) for _ in _x]
        _x1, _x2, _x3, _x4 = image_features  # [B, N, D] mỗi scale

        # ------------------- 2. Tự động điều chỉnh num_patches -------------------
        num_patches1 = self._get_num_patches(imshape[0])
        num_patches2 = self._get_num_patches(imshape[1])
        num_patches3 = self._get_num_patches(imshape[2])
        num_patches4 = self._get_num_patches(imshape[3])

        if self.attention_net1.num_patches != num_patches1:
            logger.warning(f"Adjusting num_patches for attention_net1 from {self.attention_net1.num_patches} to {num_patches1}")
            self.attention_net1 = Attn_Net_Gated(in_channels=self.in_channels[0], D=256, dropout=False, n_classes=self.total_classes, num_patches=num_patches1).to(self.device)
        if self.attention_net2.num_patches != num_patches2:
            logger.warning(f"Adjusting num_patches for attention_net2 from {self.attention_net2.num_patches} to {num_patches2}")
            self.attention_net2 = Attn_Net_Gated(in_channels=self.in_channels[1], D=256, dropout=False, n_classes=self.total_classes, num_patches=num_patches2).to(self.device)
        if self.attention_net3.num_patches != num_patches3:
            logger.warning(f"Adjusting num_patches for attention_net3 from {self.attention_net3.num_patches} to {num_patches3}")
            self.attention_net3 = Attn_Net_Gated(in_channels=self.in_channels[2], D=256, dropout=False, n_classes=self.total_classes, num_patches=num_patches3).to(self.device)
        if self.attention_net4.num_patches != num_patches4:
            logger.warning(f"Adjusting num_patches for attention_net4 from {self.attention_net4.num_patches} to {num_patches4}")
            self.attention_net4 = Attn_Net_Gated(in_channels=self.in_channels[3], D=256, dropout=False, n_classes=self.total_classes, num_patches=num_patches4).to(self.device)

        # ------------------- 3. Attention weights A1–A4 -------------------
        A1, x1 = self.attention_net1(_x1)  # [B, N, total_subclasses]
        A2, x2 = self.attention_net2(_x2)
        A3, x3 = self.attention_net3(_x3)
        A4, x4 = self.attention_net4(_x4)

        A1 = F.softmax(A1, dim=1)
        A2 = F.softmax(A2, dim=1)
        A3 = F.softmax(A3, dim=1)
        A4 = F.softmax(A4, dim=1)

        logger.debug(f"A1 shape: {A1.shape}, _x1 shape: {_x1.shape}, num_patches1: {num_patches1}")
        logger.debug(f"A2 shape: {A2.shape}, _x2 shape: {_x2.shape}, num_patches2: {num_patches2}")
        logger.debug(f"A3 shape: {A3.shape}, _x3 shape: {_x3.shape}, num_patches3: {num_patches3}")
        logger.debug(f"A4 shape: {A4.shape}, _x4 shape: {_x4.shape}, num_patches4: {num_patches4}")

        # ------------------- 4. Weighted features -------------------
        _x1_weighted = torch.bmm(A1.transpose(1, 2), _x1)
        _x2_weighted = torch.bmm(A2.transpose(1, 2), _x2)
        _x3_weighted = torch.bmm(A3.transpose(1, 2), _x3)
        _x4_weighted = torch.bmm(A4.transpose(1, 2), _x4)

        # ------------------- 5. CAM từ attention -------------------
        cam1 = A1.permute(0, 2, 1).view(batch_size, self.total_classes, imshape[0][2], imshape[0][3])
        cam2 = A2.permute(0, 2, 1).view(batch_size, self.total_classes, imshape[1][2], imshape[1][3])
        cam3 = A3.permute(0, 2, 1).view(batch_size, self.total_classes, imshape[2][2], imshape[2][3])
        cam4 = A4.permute(0, 2, 1).view(batch_size, self.total_classes, imshape[3][2], imshape[3][3])

        cam1 = F.interpolate(cam1, size=(224, 224), mode='bilinear', align_corners=True)
        cam2 = F.interpolate(cam2, size=(224, 224), mode='bilinear', align_corners=True)
        cam3 = F.interpolate(cam3, size=(224, 224), mode='bilinear', align_corners=True)
        cam4 = F.interpolate(cam4, size=(224, 224), mode='bilinear', align_corners=True)

        # ------------------- 6. Add background channel (1 - max)^10 -------------------
        def add_background_cam(cam):
            if cam.size(1) != self.cls_num_classes:
                logger.warning(f"CAM has {cam.size(1)} channels, expected {self.cls_num_classes}. Adjusting.")
                if cam.size(1) < self.cls_num_classes:
                    padding = torch.zeros(cam.size(0), self.cls_num_classes - cam.size(1), cam.size(2), cam.size(3), device=cam.device)
                    cam = torch.cat([cam, padding], dim=1)
                elif cam.size(1) > self.cls_num_classes:
                    cam = cam[:, :self.cls_num_classes]
            cam_max = torch.max(cam, dim=1, keepdim=True)[0]
            bg_cam = (1 - cam_max) ** 10
            return torch.cat([cam, bg_cam], dim=1)   # → 5 channels (4 FG + 1 BG)

        cam1 = add_background_cam(cam1)
        cam2 = add_background_cam(cam2)
        cam3 = add_background_cam(cam3)
        cam4 = add_background_cam(cam4)

        if cam1.size(1) != 5 or cam2.size(1) != 5 or cam3.size(1) != 5 or cam4.size(1) != 5:
            logger.error(f"Invalid CAM channels: cam1={cam1.size(1)}, cam2={cam2.size(1)}, cam3={cam3.size(1)}, cam4={cam4.size(1)}. Expected 5.")
            return [torch.zeros(1, device=device)] * 13

        batch_info1 = self.feature_extractor.process_batch(x, cam1, labels) if self.l_fea is not None else None
        batch_info2 = self.feature_extractor.process_batch(x, cam2, labels) if self.l_fea is not None else None
        batch_info3 = self.feature_extractor.process_batch(x, cam3, labels) if self.l_fea is not None else None
        batch_info4 = self.feature_extractor.process_batch(x, cam4, labels) if self.l_fea is not None else None

        
        # ------------------- 7. Project weighted features → classification score -------------------
        l_fea1 = self.l_fc1(self.l_fea) if self.l_fea is not None else torch.zeros(batch_size, self.in_channels[0], device=device)
        l_fea2 = self.l_fc2(self.l_fea) if self.l_fea is not None else torch.zeros(batch_size, self.in_channels[1], device=device)
        l_fea3 = self.l_fc3(self.l_fea) if self.l_fea is not None else torch.zeros(batch_size, self.in_channels[2], device=device)
        l_fea4 = self.l_fc4(self.l_fea) if self.l_fea is not None else torch.zeros(batch_size, self.in_channels[3], device=device)

        _x1_weighted = _x1_weighted / (_x1_weighted.norm(dim=-1, keepdim=True) + 1e-8)
        logits_per_image1 = self.logit_scale1 * _x1_weighted @ l_fea1.t().float()
        cls1 = logits_per_image1.mean(dim=2)

        _x2_weighted = _x2_weighted / (_x2_weighted.norm(dim=-1, keepdim=True) + 1e-8)
        logits_per_image2 = self.logit_scale2 * _x2_weighted @ l_fea2.t().float()
        cls2 = logits_per_image2.mean(dim=2)

        _x3_weighted = _x3_weighted / (_x3_weighted.norm(dim=-1, keepdim=True) + 1e-8)
        logits_per_image3 = self.logit_scale3 * _x3_weighted @ l_fea3.t().float()
        cls3 = logits_per_image3.mean(dim=2)

        _x4_weighted = _x4_weighted / (_x4_weighted.norm(dim=-1, keepdim=True) + 1e-8)
        logits_per_image4 = self.logit_scale4 * _x4_weighted @ l_fea4.t().float()
        cls4 = logits_per_image4.mean(dim=2)

        # ------------------- 8. Diversity loss -------------------
        diversity_loss = attention_diversity(self.l_fea, _x1, A1) if self.l_fea is not None else torch.tensor(0.0, device=device)

        # ------------------- 9. Hierarchical merge (subclass → parent) -------------------
        merge_method = cfg.train.merge_train if cfg and hasattr(cfg.train, 'merge_train') else "mean"
        cls1_merge = merge_to_parent_predictions(cls1, self.k_list, method=merge_method)
        cls2_merge = merge_to_parent_predictions(cls2, self.k_list, method=merge_method)
        cls3_merge = merge_to_parent_predictions(cls3, self.k_list, method=merge_method)
        cls4_merge = merge_to_parent_predictions(cls4, self.k_list, method=merge_method)

        cam1_merge = merge_subclass_cams_to_parent(cam1, self.k_list, method=merge_method)
        cam2_merge = merge_subclass_cams_to_parent(cam2, self.k_list, method=merge_method)
        cam3_merge = merge_subclass_cams_to_parent(cam3, self.k_list, method=merge_method)
        cam4_merge = merge_subclass_cams_to_parent(cam4, self.k_list, method=merge_method)

        # ------------------- 10. Loss -------------------
        loss = self.loss_bce(cls4_merge, labels) + 0.25 * diversity_loss if labels is not None and self.l_fea is not None else torch.tensor(0.0, device=device)

        k_list = self.k_list
        return cls1_merge, cam1_merge, cls2_merge, cam2_merge, cls3_merge, cam3_merge, cls4_merge, cam4_merge, self.l_fea, loss, None, k_list, _x1
    