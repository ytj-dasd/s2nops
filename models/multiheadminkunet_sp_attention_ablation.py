import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.minkunet import MinkUNet34C

class SPGlobalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead=4, bias_hidden=64, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.bias_mlp = nn.Sequential(
            nn.Linear(3, bias_hidden), nn.ReLU(inplace=True), nn.Linear(bias_hidden, 1)
        )

    def forward(self, T, C, R):
        S, d = T.shape
        Q = self.Wq(T).view(S, self.nhead, self.d_head).transpose(0, 1)  
        K = self.Wk(T).view(S, self.nhead, self.d_head).transpose(0, 1) 
        V = self.Wv(T).view(S, self.nhead, self.d_head).transpose(0, 1)  

        # 位置偏置 
        # eps = 1e-6
        # delta = C.unsqueeze(1) - C.unsqueeze(0)             
        # r_norm = 0.5 * (R.unsqueeze(1) + R.unsqueeze(0))    
        # hat_delta = delta / (r_norm.clamp_min(eps).unsqueeze(-1))
        # hat_delta = hat_delta.tanh()
        # B = self.bias_mlp(hat_delta).squeeze(-1)            
        # B = B.unsqueeze(0)                                  

        # # 注意力
        # logits = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)  
        # logits = logits + B                                
        
        # 去除位置编码
        B = 0.0
        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head) 
                  
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)

        O = attn @ V                             
        O = O.transpose(0, 1).contiguous().view(S, d)
        O = self.Wo(O)
        return T + O


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()

        self.prototypes = ME.MinkowskiConvolution(
            output_dim,
            num_prototypes,
            kernel_size=1,
            bias=False,
            dimension=D)

    def forward(self, x):
        return self.prototypes(x).F


class MultiHead(nn.Module):
    def __init__(
        self, input_dim, num_prototypes, num_heads
    ):
        super().__init__()
        self.num_heads = num_heads

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(input_dim, num_prototypes) for _ in range(num_heads)]
        )

    def forward_head(self, head_idx, feats):
        return self.prototypes[head_idx](feats), feats.F

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadMinkUnet(nn.Module):
    def __init__(
        self,
        num_labeled,
        num_unlabeled,
        overcluster_factor=None,
        num_heads=1,
        clip_feat_dim=None
    ):
        super().__init__()

        print("------ SuperPoint Attention wo position encoding------")

        # backbone -> pretrained model + identity as final
        # self.encoder = MinkUNet34C(111, num_labeled) # intensity & rgb (1 + 3) & geo_feature(7)
        self.encoder = MinkUNet34C(102, num_labeled) # rgb (3)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()
        
        self.sp_attn = SPGlobalSelfAttention(d_model=self.feat_dim, nhead=4, dropout=0.0)
        # self.sp_attn = SPGlobalSelfAttention(d_model=self.feat_dim, nhead=1, dropout=0.0)

        self.head_lab = Prototypes(output_dim=self.feat_dim,
                                    num_prototypes=num_labeled)
        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads
            )

        if overcluster_factor is not None:
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads
            )
        
        if clip_feat_dim is not None:
            self.clip_projector = torch.nn.Sequential(
                ME.MinkowskiConvolution(
                    self.feat_dim,
                    clip_feat_dim//2,
                    kernel_size=1,
                    bias=True,
                    dimension=3),
                ME.MinkowskiBatchNorm(clip_feat_dim//2),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(
                    clip_feat_dim//2,
                    clip_feat_dim,
                    kernel_size=1,
                    bias=True,
                    dimension=3)
            )

    def _sp_self_attn_injection(self, feats: ME.SparseTensor, superpoints: torch.Tensor):
        
        F = feats.F                              
        C_all = feats.C                           
        b = C_all[:, 0].long()                    
        xyz = C_all[:, 1:].float()             
        superpoints = superpoints.to(F.device)   
        sp_coarse = superpoints[:, 2].long()      

        pair = torch.stack([b, sp_coarse], dim=1)
        uniq, g = torch.unique(pair, dim=0, return_inverse=True)
        G = uniq.size(0)
        d = F.size(1)
        device = F.device
        dtype = F.dtype

        counts = torch.bincount(g, minlength=G).clamp_min(1).to(F.dtype)  

        # SP token: 特征均值
        T = torch.zeros(G, d, device=device, dtype=dtype)
        T.index_add_(0, g, F)
        T = T / counts.unsqueeze(1)

        # SP 质心、半径
        C = torch.zeros(G, 3, device=device, dtype=xyz.dtype)
        C.index_add_(0, g, xyz)
        C = C / counts.unsqueeze(1)

        dist = (xyz - C[g]).pow(2).sum(dim=1).sqrt()     
        R = torch.zeros(G, device=device, dtype=xyz.dtype)
        R.index_add_(0, g, dist)
        R = (R / counts).clamp_min(1e-6)                 

        # 按 batch 分开做自注意力
        Z = torch.zeros_like(T)                         
        batch_ids = uniq[:, 0].unique(sorted=True)
        for bid in batch_ids:
            mask = (uniq[:, 0] == bid)
            idx = mask.nonzero(as_tuple=False).squeeze(1)     
            if idx.numel() == 0:
                continue
            T_b = T[idx]                                       
            C_b = C[idx]                                       
            R_b = R[idx]                                       
            Z_b = self.sp_attn(T_b, C_b, R_b)                 
            Z[idx] = Z_b

        # 广播回点
        inj = Z[g]                                             
        return inj

    # 平均池化
    # def _sp_self_attn_injection(self, feats: ME.SparseTensor, superpoints: torch.Tensor):
    #     F = feats.F
    #     C_all = feats.C
    #     b = C_all[:, 0].long()
    #     xyz = C_all[:, 1:].float()
    #     superpoints = superpoints.to(F.device)
    #     sp_coarse = superpoints[:, 2].long()

    #     pair = torch.stack([b, sp_coarse], dim=1)
    #     uniq, g = torch.unique(pair, dim=0, return_inverse=True)
    #     G = uniq.size(0)
    #     d = F.size(1)
    #     device = F.device
    #     dtype = F.dtype

    #     counts = torch.bincount(g, minlength=G).clamp_min(1).to(F.dtype)

    #     # === 平均池化 token ===
    #     T = torch.zeros(G, d, device=device, dtype=dtype)
    #     T.index_add_(0, g, F)
    #     T = T / counts.unsqueeze(1)  # [G, d]

    #     inj = T[g]  # [N, d]
    #     return inj



    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(feats)}
        if hasattr(self, "head_unlab"):
            logits_unlab, _ = self.head_unlab(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    # "proj_feats_unlab": proj_feats_unlab,
                }
            )
        if hasattr(self, "head_unlab_over"):
            logits_unlab_over, _ = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab_over": logits_unlab_over,
                    # "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, input, superpoints: torch.Tensor = None):
        feats = self.encoder(input)
        
        if superpoints is not None:
            # print("------ Has ------")
            inj = self._sp_self_attn_injection(feats, superpoints)
            feats = ME.SparseTensor(
                features=feats.F + inj,
                coordinate_map_key=feats.coordinate_map_key,
                coordinate_manager=feats.coordinate_manager,
            )
        
        out = self.forward_heads(feats)
        out['feats'] = feats.F
        if hasattr(self, "clip_projector"):
            out['clip_feats'] = self.clip_projector(feats).F
        return out