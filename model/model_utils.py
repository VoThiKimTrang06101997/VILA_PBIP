import logging
from typing import Optional, Tuple
import nmslib
from torch.nn import Module, Linear
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
from torch import Tensor
from torch.nn.functional import linear, softmax, dropout, pad
import math
import numpy as np
import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class Attn_Net_Gated(nn.Module):
    def __init__(self, in_channels=64, D=256, dropout=False, n_classes=4, num_patches=3136):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(in_channels, D),
            nn.Tanh()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(in_channels, D),
            nn.Sigmoid()
        )
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_c = nn.Linear(D, n_classes)
        self.num_patches = num_patches

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        # Ensure A shape is [batch_size, num_patches, n_classes]
        if A.shape[1] != self.num_patches:
            logger.warning(f"Attention weights num_patches ({A.shape[1]}) does not match expected {self.num_patches}. Padding or truncating.")
            if A.shape[1] < self.num_patches:
                padding = torch.zeros(A.shape[0], self.num_patches - A.shape[1], A.shape[2], device=A.device)
                A = torch.cat([A, padding], dim=1)
            else:
                A = A[:, :self.num_patches, :]
        return A, x

def init_max_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            if m.bias is not None:
                m.bias.data.zero_()

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    need_raw: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    with torch.amp.autocast('cuda'):
        if query.size(0) < query.size(1):
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len = query.size(0)
        bsz = query.size(1)
        src_len = key.size(0)

        assert key.size(1) == bsz and value.size(1) == bsz, f"Batch size mismatch: query batch {bsz}, key batch {key.size(1)}, value batch {value.size(1)}"

        if not use_separate_proj_weight:
            expected_rows = 3 * embed_dim_to_check
            expected_cols = embed_dim_to_check
            if in_proj_weight.size(0) != expected_rows or in_proj_weight.size(1) != expected_cols:
                raise ValueError(f"Expected in_proj_weight shape [{expected_rows}, {expected_cols}], got {in_proj_weight.shape}")

            if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
                qkv = linear(query, in_proj_weight, in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)
            else:
                q = linear(query, in_proj_weight[:embed_dim_to_check, :], in_proj_bias[:embed_dim_to_check] if in_proj_bias is not None else None)
                k = linear(key, in_proj_weight[embed_dim_to_check:2 * embed_dim_to_check, :], 
                        in_proj_bias[embed_dim_to_check:2 * embed_dim_to_check] if in_proj_bias is not None else None)
                v = linear(value, in_proj_weight[2 * embed_dim_to_check:, :], 
                        in_proj_bias[2 * embed_dim_to_check:] if in_proj_bias is not None else None)
        else:
            q = linear(query, q_proj_weight, in_proj_bias[:embed_dim_to_check] if in_proj_bias is not None else None)
            k = linear(key, k_proj_weight, in_proj_bias[embed_dim_to_check:2 * embed_dim_to_check] if in_proj_bias is not None else None)
            v = linear(value, v_proj_weight, in_proj_bias[2 * embed_dim_to_check:] if in_proj_bias is not None else None)

        q_embed_dim = q.size(-1)
        head_dim = q_embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            k = static_k
        if static_v is not None:
            v = static_v

        if bias_k is not None and bias_v is not None:
            k = torch.cat([k, bias_k.repeat(1, 1, 1)], dim=1)
            v = torch.cat([v, bias_v.repeat(1, 1, 1)], dim=1)
            src_len += 1

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights_raw = attn_output_weights.clone() if need_raw else None
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            if need_raw:
                attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
                return attn_output, attn_output_weights_raw
            else:
                attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
                return attn_output, attn_output_weights.sum(dim=1) / num_heads
        return attn_output, None

class MultiheadAttention(Module):
    bias_k: Optional[Tensor]
    bias_v: Optional[Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        if 'batch_first' not in state:
            state['batch_first'] = True
        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, need_raw=True, attn_mask=None):
        with torch.amp.autocast('cuda'):
            if self.batch_first:
                query = query.transpose(0, 1)
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)
            else:
                raise ValueError("batch_first must be True for this implementation")

            if not self._qkv_same_embed_dim:
                attn_output, attn_output_weights = multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    None, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                    attn_mask=attn_mask, use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight)
            else:
                attn_output, attn_output_weights = multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                    attn_mask=attn_mask)

            if self.batch_first:
                attn_output = attn_output.transpose(0, 1)
                if attn_output_weights is not None:
                    attn_output_weights = attn_output_weights.transpose(1, 2)

            return attn_output, attn_output_weights

class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params or {'M': 16, 'post': 0, 'efConstruction': 400}
        query_params = self.query_params or {'ef': 90}

        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

def pt2graph(coords, features, threshold=5000, radius=9):
    from torch_geometric.data import Data as geomData
    from itertools import chain
    coords, features = np.array(coords.cpu().detach()), np.array(features.cpu().detach())
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]

    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius - 1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_spatial = torch.tensor(np.stack([a, b])).type(torch.long)

    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius - 1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_latent = torch.tensor(np.stack([a, b])).type(torch.long)

    start_point = edge_spatial[0, :]
    end_point = edge_spatial[1, :]
    start_coord = coords[start_point]
    end_coord = coords[end_point]
    tmp = start_coord - end_coord
    edge_distance = [math.hypot(tmp[i][0], tmp[i][1]) for i in range(tmp.shape[0])]

    filter_edge_spatial = edge_spatial[:, np.array(edge_distance) <= threshold]

    G = geomData(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=filter_edge_spatial,
        edge_latent=edge_latent,
        centroid=torch.tensor(coords, dtype=torch.float)
    )
    return G

def attention_diversity(prototypes: Tensor, features: Tensor, num_heads: int = 8) -> Tensor:
    batch_size, num_patches, D = features.shape
    num_prototypes = prototypes.shape[0]

    prototypes = prototypes.squeeze(1).to(features.device)
    query = prototypes.unsqueeze(0).repeat(batch_size, 1, 1)
    key = features
    value = features

    attn = MultiheadAttention(embed_dim=D, num_heads=num_heads, batch_first=True).to(features.device)
    attn_output, attn_weights = attn(query, key, value)

    attn_weights = attn_weights.mean(dim=1)  # Average over heads
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
    diversity_loss = -torch.mean(entropy)

    return diversity_loss

def pairwise_distances(x: Tensor) -> Tensor:
    bn = x.size(0)
    x = x.reshape(bn, -1)
    dist = torch.cdist(x, x, p=2.0)
    return dist

def calculate_gram_mat(x: Tensor, sigma: float) -> Tensor:
    dist = pairwise_distances(x)
    return torch.exp(-dist / sigma)

def reyi_entropy(x: Tensor, sigma: float) -> Tensor:
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.linalg.eigvalsh(k, UPLO='L'))
    return -torch.sum(eigv * torch.log(eigv + 1e-20))

def joint_entropy(x: Tensor, y: Tensor, s_x: float, s_y: float) -> Tensor:
    alpha = 1.01
    x_gram = calculate_gram_mat(x, s_x)
    y_gram = calculate_gram_mat(y, s_y)
    k = x_gram * y_gram
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy

def calculate_mi(x: Tensor, y: Tensor, s_x: float = 1.0, s_y: float = 1.0) -> Tensor:
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    k = calculate_gram_mat(x, sigma=s_x) * calculate_gram_mat(y, sigma=s_y)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.linalg.eigvalsh(k, UPLO='L'))
    Hxy = -torch.sum(eigv * torch.log(eigv + 1e-20))
    return Hx + Hy - Hxy
