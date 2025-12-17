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
        
        print('·········Input Encoder·········')

        # backbone -> pretrained model + identity as final
        # self.encoder = MinkUNet34C(112, num_labeled) # intensity & rgb (1 + 3) & geo_feature(7)
        # self.encoder = MinkUNet34C(105, num_labeled) # intensity & rgb (1 + 3) & logit
        self.encoder = MinkUNet34C(64, num_labeled) # rgbi(32) & logit(32)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()

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
            
        self.rgb_stem = torch.nn.Sequential(
            ME.MinkowskiConvolution(4, 32, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),
        )
        self.img_stem = torch.nn.Sequential(
            ME.MinkowskiConvolution(101, 32, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),
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
        feats_all = input.F  
        rgbi = feats_all[:, :4]
        p    = feats_all[:, 4:]       
        
        with torch.no_grad():
            p_safe = p.clamp_min(1e-12)
            H = -(p_safe * p_safe.log()).sum(dim=1, keepdim=True)     
            Hmax = math.log(p.shape[1]) if p.shape[1] > 0 else 1.0
            w = 1.0 - (H / Hmax)
        
        p_gate = p * w  
        if self.training:
            p_gate = F.dropout(p_gate, p=0.3, inplace=False)
        
        # feats_gated = torch.cat([rgbi, p_gate], dim=1)                
        # sp = ME.SparseTensor(
        #     features=feats_gated,
        #     coordinate_map_key=input.coordinate_map_key,
        #     coordinate_manager=input.coordinate_manager,
        # )
        # feats_enc = self.encoder(sp)  
        
        # sp_rgbi = ME.SparseTensor(rgbi, coordinates=input.C)
        # sp_img  = ME.SparseTensor(p_gate, coordinates=input.C)
        sp_rgbi = ME.SparseTensor(
            rgbi,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager
        )
        sp_img = ME.SparseTensor(
            p_gate,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager
        )
        rgb32   = self.rgb_stem(sp_rgbi)      
        img32   = self.img_stem(sp_img)        
        feats64 = ME.cat(rgb32, img32)         
        feats_enc = self.encoder(feats64)                                 

        out = self.forward_heads(feats_enc)
        out["feats"] = feats_enc.F
        if hasattr(self, "clip_projector"):
            out["clip_feats"] = self.clip_projector(feats_enc.F)
        return out