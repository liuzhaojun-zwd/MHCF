


import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import copy
import numpy as np
# ------------------------------ Mona 模块 -----------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

class SequentialCrossModalHeatConductionOperator(nn.Module):
    def __init__(self, feature_dim: int, spatial_height: int, spatial_width: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width

        self.alpha_cls = nn.Parameter(torch.tensor(0.3))
        self.alpha_patch = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.2))

        self.conductivity_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Softplus()
        )

        self.cls_patch_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        self.patch_spatial_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

        self.register_buffer('spatial_adjacency', self._init_spatial_adjacency())

    def _init_spatial_adjacency(self):
        H, W = self.spatial_height, self.spatial_width
        N = H * W
        adjacency = torch.zeros(N, N)

        for i in range(H):
            for j in range(W):
                center_idx = i * W + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            neighbor_idx = ni * W + nj
                            distance = (di ** 2 + dj ** 2) ** 0.5
                            adjacency[center_idx, neighbor_idx] = 1.0 / distance

        row_sums = adjacency.sum(dim=1, keepdim=True)
        adjacency = adjacency / (row_sums + 1e-8)
        return adjacency

    def spatial_diffusion(self, patches, gate_weights, diffusion_steps=3):
        B, N, C = patches.shape
        current_patches = patches

        adj = self.spatial_adjacency.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        for step in range(diffusion_steps):
            neighbor_avg = torch.bmm(adj, current_patches)  # [B, N, C]
            diffused = gate_weights * neighbor_avg + (1 - gate_weights) * current_patches
            current_patches = 0.7 * current_patches + 0.3 * diffused

        return current_patches

    def cls_to_patches_conduction(self, cls_token, patches, conductivity):
        B, N, C = patches.shape
        cls_broadcast = cls_token.expand(-1, N, -1)
        similarity = F.cosine_similarity(cls_broadcast, patches, dim=-1)
        similarity = torch.softmax(similarity, dim=-1).unsqueeze(-1)  # [B, N, 1]

        conductivity = conductivity.view(B, 1, 1)
        cls_influence = cls_broadcast * similarity * conductivity * self.alpha_cls
        return patches + cls_influence

    def patches_to_cls_conduction(self, patches, cls_token, interaction_gate):
        B, N, C = patches.shape
        patch_importance = torch.norm(patches, dim=-1, keepdim=True) + 1e-6
        patch_importance = torch.softmax(patch_importance.squeeze(-1), dim=-1).unsqueeze(-1)  # [B, N, 1]
        interaction_gate = interaction_gate.expand(-1, N, -1)
        weighted_patches = patches * patch_importance * interaction_gate
        patch_summary = weighted_patches.sum(dim=1, keepdim=True)  # [B, 1, C]
        return cls_token + self.alpha_cls * patch_summary

    def cross_modal_conduction(self, source_cls, source_patches, target_patches):
        B, N, C = target_patches.shape

        source_global = F.adaptive_avg_pool1d(source_patches.transpose(1, 2), 1).squeeze(-1)
        target_global = F.adaptive_avg_pool1d(target_patches.transpose(1, 2), 1).squeeze(-1)
        combined_feat = torch.cat([source_global, target_global], dim=1)
        conductivity = self.conductivity_predictor(combined_feat)

        cls_patch_combined = torch.cat([source_cls.squeeze(1), target_global], dim=1)
        interaction_gate = self.cls_patch_gate(cls_patch_combined).unsqueeze(1)

        spatial_gate = self.patch_spatial_gate(target_patches)

        target_patches_cls_conducted = self.cls_to_patches_conduction(
            source_cls, target_patches, conductivity
        )

        cross_similarity = torch.bmm(target_patches, source_patches.transpose(1, 2))
        cross_similarity = torch.softmax(cross_similarity, dim=-1)
        cross_patch_info = torch.bmm(cross_similarity, source_patches)

        interaction_gate_exp = interaction_gate.expand(-1, N, -1)
        target_patches_cross_conducted = target_patches_cls_conducted + \
            self.beta * interaction_gate_exp * cross_patch_info

        target_patches_final = self.spatial_diffusion(target_patches_cross_conducted, spatial_gate)

        return target_patches_final, conductivity, interaction_gate

    def forward(self, source_cls, source_patches, target_cls, target_patches):
        enhanced_target_patches, conductivity, interaction_gate = self.cross_modal_conduction(
            source_cls, source_patches, target_patches
        )

        enhanced_target_cls = self.patches_to_cls_conduction(
            enhanced_target_patches, target_cls, interaction_gate
        )

        stats = {
            'conductivity': conductivity,
            'interaction_gate': interaction_gate.mean(),
            'cls_change': torch.norm(enhanced_target_cls - target_cls, dim=-1).mean(),
            'patch_change': torch.norm(enhanced_target_patches - target_patches, dim=-1).mean()
        }

        return enhanced_target_cls, enhanced_target_patches, stats


class MultiModalHeatCycleFusion(nn.Module):
    def __init__(self, patches_per_modal=128, feature_dim=768,
                 spatial_height=16, spatial_width=8, num_cycles=2):
        super().__init__()
        self.patches_per_modal = patches_per_modal
        self.feature_dim = feature_dim
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.num_cycles = num_cycles

        self.heat_conductor = SequentialCrossModalHeatConductionOperator(
            feature_dim, spatial_height, spatial_width
        )

   

    def sequential_conduction_cycle(self, rgb_cls, rgb_patches, nir_cls, nir_patches, tir_cls, tir_patches, cycle_idx):
        stats = {}

        nir_cls, nir_patches, stats['rgb_to_nir'] = self.heat_conductor(
            rgb_cls, rgb_patches, nir_cls, nir_patches
        )
        tir_cls, tir_patches, stats['nir_to_tir'] = self.heat_conductor(
            nir_cls, nir_patches, tir_cls, tir_patches
        )
        rgb_cls, rgb_patches, stats['tir_to_rgb'] = self.heat_conductor(
            tir_cls, tir_patches, rgb_cls, rgb_patches
        )

        return rgb_cls, rgb_patches, nir_cls, nir_patches, tir_cls, tir_patches, stats

    def forward(self, rgb_features, nir_features, tir_features):
        device = rgb_features.device


        rgb_cls, rgb_patches = rgb_features[:, 0:1, :], rgb_features[:, 1:, :]
        nir_cls, nir_patches = nir_features[:, 0:1, :], nir_features[:, 1:, :]
        tir_cls, tir_patches = tir_features[:, 0:1, :], tir_features[:, 1:, :]

        all_cycle_stats = []
        for cycle in range(self.num_cycles):
            rgb_cls, rgb_patches, nir_cls, nir_patches, tir_cls, tir_patches, stats = \
                self.sequential_conduction_cycle(rgb_cls, rgb_patches, nir_cls, nir_patches, tir_cls, tir_patches, cycle)
            all_cycle_stats.append(stats)


        cls_features = torch.cat([
           rgb_cls.squeeze(1),nir_cls.squeeze(1), tir_cls.squeeze(1)
        ], dim=1)

        rgb_global = F.adaptive_avg_pool1d(rgb_patches.transpose(1, 2), 1).squeeze(-1)
        nir_global = F.adaptive_avg_pool1d(nir_patches.transpose(1, 2), 1).squeeze(-1)
        tir_global = F.adaptive_avg_pool1d(tir_patches.transpose(1, 2), 1).squeeze(-1)

        global_features = torch.cat([rgb_global, nir_global, tir_global], dim=1)

       

        return {
            'cls': cls_features,
            'pat': global_features,
        }

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class MHCF(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(MHCF, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.tta_pth = cfg.TEST.WEIGHT
 
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        self.miss = cfg.TEST.MISS

        self.cfg = cfg
        
        
     
     

        
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.heat =  MultiModalHeatCycleFusion(
            patches_per_modal=128,
            feature_dim=768,
            spatial_height=self.h_resolution,
            spatial_width=self.w_resolution,
            num_cycles=3
        )
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
    
        
        if self.cfg.MODEL.DIRECT:
            self.classifier = nn.Linear(self.in_planes*3, self.num_classes, bias=False)  # 768*3 Class_num
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(self.in_planes*3)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)

            self.fc_heat_pat = nn.Linear(self.in_planes*3, self.num_classes, bias=False)
            self.fc_heat_pat.apply(weights_init_classifier)
            self.bn_heat_pat = nn.BatchNorm1d(self.in_planes*3)
            self.bn_heat_pat.bias.requires_grad_(False)
            self.bn_heat_pat.apply(weights_init_kaiming)

            self.fc_heat_cls = nn.Linear(self.in_planes*3, self.num_classes, bias=False)
            self.fc_heat_cls.apply(weights_init_classifier)
            self.bn_heat_cls = nn.BatchNorm1d(self.in_planes*3)
            self.bn_heat_cls.bias.requires_grad_(False)
            self.bn_heat_cls.apply(weights_init_kaiming)

         

        else:
            self.classifier = nn.Linear(self.in_planes*3, self.num_classes, bias=False)  # 768*3 Class_num
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(self.in_planes*3)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
            self.fc_r = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.fc_r.apply(weights_init_classifier)
    
            self.fc_n = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.fc_n.apply(weights_init_classifier)
    
            self.fc_t = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.fc_t.apply(weights_init_classifier)
    
            self.bn_r = nn.BatchNorm1d(self.in_planes)
            self.bn_r.bias.requires_grad_(False)
            self.bn_r.apply(weights_init_kaiming)
    
            self.bn_n = nn.BatchNorm1d(self.in_planes)
            self.bn_n.bias.requires_grad_(False)
            self.bn_n.apply(weights_init_kaiming)
    
            self.bn_t = nn.BatchNorm1d(self.in_planes)
            self.bn_t.bias.requires_grad_(False)
            self.bn_t.apply(weights_init_kaiming)
       
            self.fc_heat_pat = nn.Linear(self.in_planes*3, self.num_classes, bias=False)
            self.fc_heat_pat.apply(weights_init_classifier)
            self.bn_heat_pat = nn.BatchNorm1d(self.in_planes*3)
            self.bn_heat_pat.bias.requires_grad_(False)
            self.bn_heat_pat.apply(weights_init_kaiming)

            self.fc_heat_cls = nn.Linear(self.in_planes*3, self.num_classes, bias=False)
            self.fc_heat_cls.apply(weights_init_classifier)
            self.bn_heat_cls = nn.BatchNorm1d(self.in_planes*3)
            self.bn_heat_cls.bias.requires_grad_(False)
            self.bn_heat_cls.apply(weights_init_kaiming)
        
        

    
     
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

  
    def forward(self, x, label=None, cam_label= None, view_label=None) :

            layer_r, r_x12, _ = self.image_encoder(x['RGB'], None) 
            layer_n, n_x12, _ = self.image_encoder(x['NI'], None) 
            layer_t, t_x12, _ = self.image_encoder(x['TI'], None) 

            ori_f = torch.cat([r_x12[:,0], n_x12[:,0], t_x12[:,0]], dim=-1)
            
            rgb, ni, ti = [r_x12[:,0], n_x12[:,0], t_x12[:,0]]
            out = self.heat(r_x12, n_x12, t_x12)
        
            h_cls = out['cls']
            h_pat = out['pat']

            if self.training: 
                if self.cfg.MODEL.DIRECT:
                    ori_id = self.classifier(self.bottleneck(ori_f) )
                    h_cls_id = self.fc_heat_cls(self.bn_heat_cls(h_cls))
                    h_pat_id = self.fc_heat_pat(self.bn_heat_pat(h_pat))
    
                    return [ori_id, h_cls_id, h_pat_id], [ori_f, h_cls, h_pat], 0
           
                else:
                    ori_id = self.classifier(self.bottleneck(ori_f) )
                    rgb_id = self.fc_r(self.bn_r(rgb))
                    ni_id = self.fc_n(self.bn_n(ni))
                    ti_id = self.fc_t(self.bn_t(ti))
                 
                    h_cls_id = self.fc_heat_cls(self.bn_heat_cls(h_cls))
                    h_pat_id = self.fc_heat_pat(self.bn_heat_pat(h_pat))
            else:
                return torch.cat([ori_f,h_cls, h_pat],dim=-1)

    
    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)
        
    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = MHCF(num_class, camera_num, view_num, cfg)
    return model

from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model



