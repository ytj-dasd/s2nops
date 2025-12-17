import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.minkunet import MinkUNet34C

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

        # backbone -> pretrained model + identity as final
        print('·········hello lwy·········')
        self.encoder = MinkUNet34C(105, num_labeled) # intensity & rgb (1 + 3)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()
        self.img_adapter = torch.nn.Sequential(
            ME.MinkowskiConvolution(101, self.feat_dim, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(self.feat_dim), ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(self.feat_dim, self.feat_dim, kernel_size=1, dimension=3),
        )

        self.fuse = ME.MinkowskiConvolution(2 * self.feat_dim, self.feat_dim, kernel_size=1, dimension=3)

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

    def forward(self, input):
        cm, k = input.coordinate_manager, input.coordinate_map_key
        feats_all = input.F

        sp_rgbi = ME.SparseTensor(features=feats_all,  coordinate_map_key=k, coordinate_manager=cm)
        sp_img  = ME.SparseTensor(features=feats_all[:, 4:],  coordinate_map_key=k, coordinate_manager=cm)
        
        F_enc = self.encoder(sp_rgbi)  # 1) rgbi 编码 (N, 96)

        L = self.img_adapter(sp_img)   # 2) 图像 logits 编码 (N, 96) 
        
        with torch.no_grad():  # 3) 熵门控
            p = sp_img.F.clamp_min(1e-12)            
            H = -(p * p.log()).sum(dim=1, keepdim=True) 
            Hmax = math.log(101)        
            w = 1.0 - (H / Hmax)
        Lw = ME.SparseTensor(
            L.F * w, coordinate_map_key=L.coordinate_map_key, coordinate_manager=L.coordinate_manager
        )

        F = self.fuse(ME.cat(F_enc, Lw)) # 4） 融合
        
        out = self.forward_heads(F)
        out['feats'] = F.F
        if hasattr(self, "clip_projector"):
            out['clip_feats'] = self.clip_projector(F).F
        return out