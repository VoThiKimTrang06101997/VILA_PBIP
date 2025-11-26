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
# Adaptive Layer – Dùng để project l_fea về không gian của từng scale
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
# ClsNetwork – MAIN MODEL (Image + Text Branch + Hierarchical CAM)
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
                # ------------------- Image Backbone: SegFormer -------------------
                segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b1")
                state_dict = segformer_model.segformer.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if 'decode_head' not in k and k in self.encoder.state_dict()}
                self.encoder.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded SegFormer pretrained weights from Hugging Face")
            except Exception as e:
                logger.error(f"Failed to load SegFormer pretrained weights from Hugging Face: {str(e)}. Using untrained encoder.")
        self.pooling = F.adaptive_avg_pool2d

        # Feature Extraction dùng Resnet-50/ ResNet for FG/BG contrastive
        self.resnet_model = ResNetModel.from_pretrained("microsoft/resnet-50").to(self.device)
        logger.info("Successfully loaded ResNet-50 model for feature extraction")

        self.mask_adapter = MaskAdapter_DynamicThreshold(alpha=0.5)
        self.feature_extractor = FeatureExtractor(self.mask_adapter, clip_size=224, biomedclip_model=self.resnet_model)

        # ------------------- Adaptive projection cho từng scale -------------------
        self.l_fc1 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[3])


        # ============== Learnable Text Prototypes (Centers) ==============
        self.num_text_prototypes = cls_num_classes  # 1 vector trung tâm cho mỗi class
        self.learnable_text_centers = nn.Parameter(
            torch.randn(self.num_text_prototypes, prototype_feature_dim)
        )
        
        trunc_normal_(self.learnable_text_centers, std=0.02)
        

        # ------------------- Text Branch (CLIP RN50): PromptLearner + TextEncoder (CLIP RN50) -------------------
        self.text_prompt = text_prompt if text_prompt is not None else [
            "A WSI of Tumor with irregular shapes, dense cellularity, and heterogeneous staining.",
            "A WSI of Stroma with fibrous tissue and low cellularity.",
            "A WSI of Lymphocyte with small dark clusters and speckled appearance.",
            "A WSI of Necrosis with pale amorphous zones and loss of structure."
            
            # "A WSI of Tumor with visually descriptive characteristics of irregularly shaped regions, dense cellularity, heterogeneous staining, and distortion of adjacent structures due to growth, as well as atypical cells, enlarged nuclei, prominent nucleoli, high nuclear-to-cytoplasmic ratio, and mitotic figures."
            # "A WSI of Stroma with visually descriptive characteristics of fibrous connective tissue, lighter staining, low cellular density, and surrounding or infiltrating tumor areas, as well as elongated fibroblasts, collagen bundles, eosinophilic matrix, blood vessels, and occasional inflammatory cells."
            # "A WSI of Lymphocyte with visually descriptive characteristics of small dark clusters or infiltrates, often at tumor-stroma interfaces, appearing as speckled blue-purple areas, as well as small round cells, hyperchromatic nuclei, scant cytoplasm, and clustering in immune responses."
            # "A WSI of Necrosis with visually descriptive characteristics of pale amorphous zones, loss of structure, hypoeosinophilic appearance, and contrast with viable tissue, as well as cellular debris, karyorrhectic nuclei, cytoplasmic remnants, and infiltration by inflammatory cells."
        ]
        clip_model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai')
        self.prompt_learner = PromptLearner(self.text_prompt, clip_model.float())
        self.text_encoder = TextEncoder(clip_model.float(), len(self.text_prompt))
        self.text_projection = nn.Linear(512, prototype_feature_dim).float()
        
        
        # ============== Image Prototypes (l_fea) – sẽ được refine bằng text ==============
        self.l_fea = None
        # ------------------- Hierarchical Setting -------------------
        self.k_list = k_list if k_list is not None else [1, 1, 1, 1]  # Mỗi class có 1 subclass -> 4 classes trong Dataset
        if sum(self.k_list) != cls_num_classes:
            logger.warning(f"k_list sum {sum(self.k_list)} does not match cls_num_classes {cls_num_classes}. Adjusting k_list.")
            self.k_list = [1] * cls_num_classes
        self.cumsum_k = np.cumsum(self.k_list)  # = số channel trong attention
        
        
        # Giải thích:  file bcss_label_fea_pro_3333.pkl có shape [12, 512] (có thể là 4 class × 3 subclass mỗi class) nhưng model đang dùng cls_num_classes = 4 classes
        
        # ------------------- Learned Prototypes (l_fea) – LOAD + MERGE SUBCLASS -------------------
        self.prototype_feature_dim = prototype_feature_dim
        self.cls_num_classes = cls_num_classes

        # Khởi tạo
        self.l_fea = nn.Parameter(
            torch.randn(self.cls_num_classes, self.prototype_feature_dim) * 0.02,
            requires_grad=True
        )
        trunc_normal_(self.l_fea, std=0.02)
        logger.info(f"Initialized random l_fea [{self.cls_num_classes} x {self.prototype_feature_dim}]")
        
        
        # ==================== PROJECTOR CHO DIVERSITY LOSS (512 → 64) ====================
        self.prototype_to_diversity = nn.Linear(prototype_feature_dim, self.in_channels[0])  # 512 → 64
        nn.init.normal_(self.prototype_to_diversity.weight, std=0.02)
        nn.init.constant_(self.prototype_to_diversity.bias, 0)
        logger.info("Added prototype_to_diversity: 512→64 for diversity loss")

        # ------------------- Learned Prototypes (l_fea) – LOAD ĐÚNG FILE + MERGE 12→4 -------------------
        self.l_fea = nn.Parameter(torch.randn(self.cls_num_classes, prototype_feature_dim) * 0.02, requires_grad=True)
        trunc_normal_(self.l_fea, std=0.02)

        proto_path = r"E:\NghienCuu_WSSS\VILA_PBIP\features\image_features\bcss_label_fea_pro_3333.pkl"
        if os.path.exists(proto_path):
            try:
                info = pkl.load(open(proto_path, "rb"))
                feats = info['features'].to(self.device)

                if feats.shape[0] == 12:  # Auto merge 12 subclass → 4 class
                    merged = torch.stack([
                        feats[0:3].mean(0),
                        feats[3:6].mean(0),
                        feats[6:9].mean(0),
                        feats[9:12].mean(0)
                    ])
                    self.l_fea.data.copy_(merged.data)
                    logger.info("LOADED & MERGED bcss_label_fea_pro_3333.pkl → perfect prototypes!")
                elif feats.shape == (4, prototype_feature_dim):
                    self.l_fea.data.copy_(feats.data)
                    logger.info("LOADED perfect 4-class prototypes!")
                else:
                    logger.warning("Prototype shape wrong → keep random")
            except Exception as e:
                logger.warning(f"Load prototype error: {e}")
        else:
            logger.warning("Prototype file not found → using random")
            
            
        
        self.total_classes = sum(k_list or [1]*cls_num_classes)
        self.k_list = k_list or [1] * cls_num_classes
        self.cumsum_k = np.cumsum(self.k_list)

        # ------------------- Attention Nets cho 4 scale -------------------
        self.attention_net1 = Attn_Net_Gated(in_channels=self.in_channels[0], D=256, dropout=False, n_classes=self.total_classes, num_patches=3136).to(self.device)
        self.attention_net2 = Attn_Net_Gated(in_channels=self.in_channels[1], D=256, dropout=False, n_classes=self.total_classes, num_patches=784).to(self.device)
        self.attention_net3 = Attn_Net_Gated(in_channels=self.in_channels[2], D=256, dropout=False, n_classes=self.total_classes, num_patches=196).to(self.device)
        self.attention_net4 = Attn_Net_Gated(in_channels=self.in_channels[3], D=256, dropout=False, n_classes=self.total_classes, num_patches=196).to(self.device)

        # ------------------- 7. Logit scales (learnable temperature) -------------------
        # Dùng để scale cosine similarity → giống CLIP        
        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale3 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale4 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Khởi tạo trọng số
        self.apply(self._init_weights)
        
        # ------------------- Loss functions -------------------
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

        # ------------------- SegFormer features: Encoder → 4 scale features -------------------
        _x, _attns = self.encoder(x)  # List 4 tensor
        imshape = [_.shape for _ in _x]  # [B,C,H,W] mỗi scale
        image_features = [_.permute(0, 2, 3, 1).reshape(batch_size, -1, _.shape[1]) for _ in _x]
        _x1, _x2, _x3, _x4 = image_features  # [B, N, D] mỗi scale
        
        # Text features từ CLIP + PromptLearner → refine text centers
        text_prompts, tokenized_prompts = self.prompt_learner(batch_size)
        text_feats_raw = self.text_encoder(text_prompts, tokenized_prompts)  # [B*4, 512]
        text_feats = self.text_projection(text_feats_raw)  # [B*4, 512]
        text_feats = text_feats.view(batch_size, self.cls_num_classes, -1)

        # Refine learnable text centers bằng CLIP knowledge
        with torch.no_grad():
            text_feats_norm = F.normalize(text_feats.mean(dim=0), dim=-1)  # [4, 512]
        refined_text_centers = F.normalize(self.learnable_text_centers, dim=-1)
        refined_text_centers = 0.9 * refined_text_centers + 0.1 * text_feats_norm  # moving average style

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

        # -------------------Attention Weights: Attention + Weighted features A1–A4 -------------------
        # A1, x1 = self.attention_net1(_x1)  # [B, N, total_subclasses]
        # A2, x2 = self.attention_net2(_x2)
        # A3, x3 = self.attention_net3(_x3)
        # A4, x4 = self.attention_net4(_x4)
       
        A1 = torch.ones(batch_size, _x1.size(1), self.total_classes, device=device) / self.total_classes
        A2 = torch.ones(batch_size, _x2.size(1), self.total_classes, device=device) / self.total_classes
        A3 = torch.ones(batch_size, _x3.size(1), self.total_classes, device=device) / self.total_classes
        A4 = torch.ones(batch_size, _x4.size(1), self.total_classes, device=device) / self.total_classes
        
        A1, A2, A3, A4 = [F.softmax(A, dim=1) for A in [A1, A2, A3, A4]]

        # ------------------- 4. Weighted features -------------------
        # _x1_weighted = torch.bmm(A1.transpose(1, 2), _x1)
        # _x2_weighted = torch.bmm(A2.transpose(1, 2), _x2)
        # _x3_weighted = torch.bmm(A3.transpose(1, 2), _x3)
        # _x4_weighted = torch.bmm(A4.transpose(1, 2), _x4)
        
        # Và x1, x2, x3, x4 = weighted features → thay bằng mean pooling
        _x1_weighted = _x1.mean(dim=1, keepdim=True).expand(-1, _x1.size(1), -1)
        _x2_weighted = _x2.mean(dim=1, keepdim=True).expand(-1, _x2.size(1), -1)
        _x3_weighted = _x3.mean(dim=1, keepdim=True).expand(-1, _x3.size(1), -1)
        _x4_weighted = _x4.mean(dim=1, keepdim=True).expand(-1, _x4.size(1), -1)
        
        logger.debug(f"A1 shape: {A1.shape}, _x1 shape: {_x1.shape}, num_patches1: {num_patches1}")
        logger.debug(f"A2 shape: {A2.shape}, _x2 shape: {_x2.shape}, num_patches2: {num_patches2}")
        logger.debug(f"A3 shape: {A3.shape}, _x3 shape: {_x3.shape}, num_patches3: {num_patches3}")
        logger.debug(f"A4 shape: {A4.shape}, _x4 shape: {_x4.shape}, num_patches4: {num_patches4}")

        # ------------------- 5. CAM từ attention -------------------
        cam1 = A1.permute(0, 2, 1).view(batch_size, self.total_classes, imshape[0][2], imshape[0][3])
        cam2 = A2.permute(0, 2, 1).view(batch_size, self.total_classes, imshape[1][2], imshape[1][3])
        cam3 = A3.permute(0, 2, 1).view(batch_size, self.total_classes, imshape[2][2], imshape[2][3])
        cam4 = A4.permute(0, 2, 1).view(batch_size, self.total_classes, imshape[3][2], imshape[3][3])

        for cam in [cam1, cam2, cam3, cam4]:
            cam = F.interpolate(cam, (224,224), mode='bilinear', align_corners=True)

        # # ------------------- Add background channel (1 - max)^10-------------------     
        # def add_background_cam(cam):
        #     # Chỉ lấy 4 class foreground
        #     fg_cam = cam[:, :4, :, :]  # [B, 4, H, W]
        #     cam_max = torch.max(fg_cam, dim=1, keepdim=True)[0]  # max trong 4 class
            
        #     # Background yếu hơn rất nhiều (dùng **20 hoặc clamp)
        #     bg_score = torch.clamp(1.0 - cam_max, max=0.3)  # background max 0.3
        #     bg_cam = bg_score ** 20  # cực yếu
            
        #     # Ghép lại: 4 FG + 1 BG
        #     return torch.cat([fg_cam, bg_cam], dim=1) 
        
        # Trong hàm add_background_cam → BỎ HOÀN TOÀN BACKGROUND
        def add_background_cam(cam):
            # Không thêm background → chỉ giữ 4 class
            return cam[:, :4, :, :]  # Chỉ trả về 4 kênh
        
        
        # ------------------- FG/BG feature extraction-------------------
        batch_info1 = self.feature_extractor.process_batch(x, cam1, labels) if self.l_fea is not None else None
        batch_info2 = self.feature_extractor.process_batch(x, cam2, labels) if self.l_fea is not None else None
        batch_info3 = self.feature_extractor.process_batch(x, cam3, labels) if self.l_fea is not None else None
        batch_info4 = self.feature_extractor.process_batch(x, cam4, labels) if self.l_fea is not None else None

        
        # --------- Text Branch: PromptLearner → CLIP RN50 → refine text centers -------------------
        text_prompts, tokenized_prompts = self.prompt_learner(batch_size)
        text_feats_raw = self.text_encoder(text_prompts, tokenized_prompts)           # [B*4, 512]
        text_feats = self.text_projection(text_feats_raw)                            # [B*4, 512]
        text_feats = text_feats.view(batch_size, self.cls_num_classes, -1)           # [B, 4, 512]

        # Lấy trung bình theo batch → đại diện text cho từng class (dùng để guide)
        text_feats_mean = text_feats.mean(dim=0)                                     # [4, 512]
        text_feats_norm = F.normalize(text_feats_mean, dim=-1)

        # Learnable Text Centers + Online Refinement từ CLIP knowledge
        refined_text_centers = F.normalize(self.learnable_text_centers, dim=-1)
        refined_text_centers = 0.9 * refined_text_centers + 0.1 * text_feats_norm    # EMA-style update (no grad)

        # ------------------- Image-Text Alignment để guide prototype (l_fea) -------------------
        # Weighted image features tại scale 4 (mạnh nhất) → đại diện toàn ảnh
        image_prototype = _x4_weighted.mean(dim=1)                                   # [B, D]
        image_prototype_norm = F.normalize(image_prototype, dim=-1)

        # Cosine similarity giữa image và text centers → dùng làm soft label
        sim_i2t = image_prototype_norm @ refined_text_centers.t()                    # [B, 4]
        sim_i2t = sim_i2t * self.logit_scale4.exp()

        # Refine l_fea bằng text guidance (chỉ khi có label)
        l_fea_final = F.normalize(self.l_fea, dim=-1)
        if labels is not None:
            l_fea_updated = l_fea_final.clone()
            for b in range(batch_size):
                pos_cls = labels[b].nonzero(as_tuple=True)[0]
                if len(pos_cls) > 0:
                    # Pull l_fea của class hiện diện về phía text center tương ứng
                    alpha = 0.3
                    l_fea_updated[pos_cls] = F.normalize(
                        (1 - alpha) * l_fea_updated[pos_cls] + alpha * refined_text_centers[pos_cls],
                        dim=-1
                    )
            l_fea_final = l_fea_updated

        # ------------------- Project l_fea_final → các scale -------------------
        l_fea1 = self.l_fc1(l_fea_final) if self.l_fea is not None else torch.zeros(batch_size, self.in_channels[0], device=device)
        l_fea2 = self.l_fc2(l_fea_final) if self.l_fea is not None else torch.zeros(batch_size, self.in_channels[1], device=device)
        l_fea3 = self.l_fc3(l_fea_final) if self.l_fea is not None else torch.zeros(batch_size, self.in_channels[2], device=device)
        l_fea4 = self.l_fc4(l_fea_final) if self.l_fea is not None else torch.zeros(batch_size, self.in_channels[3], device=device)

        # ------------------- Classification scores từ weighted features -------------------
        _x1_weighted = F.normalize(_x1_weighted, dim=-1)
        _x2_weighted = F.normalize(_x2_weighted, dim=-1)
        _x3_weighted = F.normalize(_x3_weighted, dim=-1)
        _x4_weighted = F.normalize(_x4_weighted, dim=-1)

        logits1 = self.logit_scale1.exp() * (_x1_weighted @ l_fea1.t())
        logits2 = self.logit_scale2.exp() * (_x2_weighted @ l_fea2.t())
        logits3 = self.logit_scale3.exp() * (_x3_weighted @ l_fea3.t())
        logits4 = self.logit_scale4.exp() * (_x4_weighted @ l_fea4.t())

        cls1 = logits1.mean(dim=1)   # [B, total_subclasses]
        cls2 = logits2.mean(dim=1)
        cls3 = logits3.mean(dim=1)
        cls4 = logits4.mean(dim=1)

        # ------------------- Diversity loss trên l_fea_final -------------------
        # diversity_loss = attention_diversity(l_fea_final.unsqueeze(0).expand(batch_size, -1, -1), _x1, num_heads=8)
        
        # Project l_fea_final về không gian của _x1 (scale 1, channel=64)
        l_fea_for_diversity = self.prototype_to_diversity(l_fea_final)  # [4, 512] → [4, 64]
        # diversity_loss = attention_diversity(l_fea_for_diversity, _x1, num_heads=8)
        diversity_loss = torch.tensor(0.0, device=device)   # DIVERSITY = 0

        # ------------------- Hierarchical merge (subclass → parent) -------------------
        merge_method = getattr(cfg.train, 'merge_train', 'mean') if cfg and hasattr(cfg.train, 'merge_train') else 'mean'

        # cam1 = add_background_cam(cam1)
        # cam2 = add_background_cam(cam2)
        # cam3 = add_background_cam(cam3)
        # cam4 = add_background_cam(cam4)
    

        # === ĐẢM BẢO CAM LUÔN LÀ 4 KÊNH (TUM, STR, LYM, NEC) ===
        def force_4_channels(tensor):
            if tensor is None:
                return torch.zeros((x.size(0), 4, 224, 224), device=x.device)
            if tensor.size(1) > 4:
                return tensor[:, :4]
            elif tensor.size(1) < 4:
                # Nếu thiếu kênh → pad bằng zero
                pad = torch.zeros((tensor.size(0), 4 - tensor.size(1), tensor.size(2), tensor.size(3)), device=tensor.device)
                return torch.cat([tensor, pad], dim=1)
            return tensor

        cam1 = force_4_channels(cam1)
        cam2 = force_4_channels(cam2)
        cam3 = force_4_channels(cam3)
        cam4 = force_4_channels(cam4)

        # Merge (an toàn)
        cam1_merge = merge_subclass_cams_to_parent(cam1, self.k_list, method=merge_method)
        cam2_merge = merge_subclass_cams_to_parent(cam2, self.k_list, method=merge_method)
        cam3_merge = merge_subclass_cams_to_parent(cam3, self.k_list, method=merge_method)
        cam4_merge = merge_subclass_cams_to_parent(cam4, self.k_list, method=merge_method)

        # Ép lại 4 kênh lần cuối
        cam1_merge = force_4_channels(cam1_merge)
        cam2_merge = force_4_channels(cam2_merge)
        cam3_merge = force_4_channels(cam3_merge)
        cam4_merge = force_4_channels(cam4_merge)

        # Tương tự với cls
        cls1_merge = merge_to_parent_predictions(cls1, self.k_list, method=merge_method)
        cls2_merge = merge_to_parent_predictions(cls2, self.k_list, method=merge_method)
        cls3_merge = merge_to_parent_predictions(cls3, self.k_list, method=merge_method)
        cls4_merge = merge_to_parent_predictions(cls4, self.k_list, method=merge_method)

        # Ép cls về 4 class
        def force_4_cls(tensor):
            if tensor is None or tensor.numel() == 0:
                return torch.zeros((x.size(0), 4), device=x.device)
            if tensor.size(1) > 4:
                return tensor[:, :4]
            elif tensor.size(1) < 4:
                pad = torch.zeros((tensor.size(0), 4 - tensor.size(1)), device=tensor.device)
                return torch.cat([tensor, pad], dim=1)
            return tensor

        cls1_merge = force_4_cls(cls1_merge)
        cls2_merge = force_4_cls(cls2_merge)
        cls3_merge = force_4_cls(cls3_merge)
        cls4_merge = force_4_cls(cls4_merge)
        
        
        
        # cam1_merge = merge_subclass_cams_to_parent(cam1, self.k_list, method=merge_method)
        # cam2_merge = merge_subclass_cams_to_parent(cam2, self.k_list, method=merge_method)
        # cam3_merge = merge_subclass_cams_to_parent(cam3, self.k_list, method=merge_method)
        # cam4_merge = merge_subclass_cams_to_parent(cam4, self.k_list, method=merge_method)
        
        # # Merge cũng chỉ trả về 4 kênh
        # cam1_merge = merge_subclass_cams_to_parent(cam1, self.k_list, method=merge_method)[:, :4]
        # cam2_merge = merge_subclass_cams_to_parent(cam2, self.k_list, method=merge_method)[:, :4]
        # cam3_merge = merge_subclass_cams_to_parent(cam3, self.k_list, method=merge_method)[:, :4]
        # cam4_merge = merge_subclass_cams_to_parent(cam4, self.k_list, method=merge_method)[:, :4]
        
        # # Luôn ép về 4 kênh
        # cam1_merge = cam1_merge[:, :4] if cam1_merge.size(1) >= 4 else cam1_merge
        # cam2_merge = cam2_merge[:, :4] if cam2_merge.size(1) >= 4 else cam2_merge
        # cam3_merge = cam3_merge[:, :4] if cam3_merge.size(1) >= 4 else cam3_merge
        # cam4_merge = cam4_merge[:, :4] if cam4_merge.size(1) >= 4 else cam4_merge
        
        
        # cls1_merge = merge_to_parent_predictions(cls1, self.k_list, method=merge_method)[:, :4]
        # cls2_merge = merge_to_parent_predictions(cls2, self.k_list, method=merge_method)[:, :4]
        # cls3_merge = merge_to_parent_predictions(cls3, self.k_list, method=merge_method)[:, :4]
        # cls4_merge = merge_to_parent_predictions(cls4, self.k_list, method=merge_method)[:, :4]
        

        # ------------------- Final Loss -------------------
        loss = torch.tensor(0.0, device=device)
        if labels is not None:
            loss = self.loss_bce(cls4_merge, labels.float())
        # loss = loss + 0.25 * diversity_loss

        return (cls1_merge, cam1_merge,
        cls2_merge, cam2_merge,
        cls3_merge, cam3_merge,
        cls4_merge, cam4_merge,
        l_fea_final, loss, sim_i2t, self.k_list, _x1)
                

