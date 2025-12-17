import os
import sys
from itertools import chain as chain_iterators
import numpy as np

import MinkowskiEngine as ME
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.functional import jaccard_index
from tqdm import tqdm
from models.minkunet import MinkUNet18A

# from models.multiheadminkunet_sp_attention_ablation import MultiHeadMinkUnet
from models.multiheadminkunet_sp_attention import MultiHeadMinkUnet
# from models.multiheadminkunet_feat_fusion import MultiHeadMinkUnet
from utils.collation import (
    collation_fn_restricted_dataset,
    collation_fn_restricted_dataset_two_samples,
)
from utils.dataset_sp import dataset_wrapper, get_dataset
from utils.scheduler import LinearWarmupCosineAnnealingLR
from utils.sinkhorn_knopp import SinkhornKnopp


class Discoverer(pl.LightningModule):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, save_predictions=False, output_dir="predictions",  **kwargs):

        super().__init__()
        self.save_hyperparameters(
            {k: v for (k, v) in kwargs.items() if not callable(v)}
        )

        print("------ SuperPoint Module3 Weighted Loss------")
        
        self.save_predictions = save_predictions
        self.output_dir = output_dir
        
        if self.save_predictions:
            self.val_probs = {}  # {point_key: prob_array}
            self.val_counts = {}  # {point_key: count}
            self.test_smooth = 0.98  # smooth
        
        # DRW 起始 epoch：后 50 个 epoch 开启
        self.cb_drw_start_epoch = 100
        unk = getattr(self.hparams, "unknown_labels", [])
        if isinstance(unk, int):
            unk = [unk]
        self.register_buffer("unknown_labels", torch.as_tensor(unk, dtype=torch.long))
        # 新类权重范围与比例上限
        self.cb_weight_min = 0.5
        self.cb_weight_max = 2.5
        self.cb_ratio_cap = 80.0
        self.cb_ratio_threshold = 30.0

        # 类别占比向量
        self.register_buffer(
            "cb_priors_full",
            torch.tensor([
                0.0,
                4.1047e-01,
                3.2170e-02,
                1.2526e-01,
                2.4986e-04,
                3.4417e-01,
                9.8642e-03,
                1.1918e-02,
                6.5899e-02
            ], dtype=torch.float)
        )
    
        # 超点损失权重 Default:0.3  Toronto3D_2:0.15
        self.w_sp_loss = 0.15 
        
        if self.hparams.clip_path is not None:
            clip_feat_dim = 768

            self.model = MultiHeadMinkUnet(
                num_labeled=self.hparams.num_labeled_classes,
                num_unlabeled=self.hparams.num_unlabeled_classes,
                overcluster_factor=self.hparams.overcluster_factor,
                num_heads=self.hparams.num_heads,
                clip_feat_dim=clip_feat_dim,
            )

            self.clip_model = MinkUNet18A(3, clip_feat_dim)
            clip_state_dict = torch.load(self.hparams.clip_path)["state_dict"]
            if "nuscenes" in self.hparams.clip_path:
                clip_state_dict = {
                    key.replace("module.net3d.", ""): value
                    for key, value in clip_state_dict.items()
                }
            elif (
                "scannet" in self.hparams.clip_path
                or "matterport" in self.hparams.clip_path
            ):
                clip_state_dict = {
                    key.replace("net3d.", ""): value
                    for key, value in clip_state_dict.items()
                }
            self.clip_model.load_state_dict(clip_state_dict)

        else:

            self.model = MultiHeadMinkUnet(
                num_labeled=self.hparams.num_labeled_classes,
                num_unlabeled=self.hparams.num_unlabeled_classes,
                overcluster_factor=self.hparams.overcluster_factor,
                num_heads=self.hparams.num_heads,
            )

        self.label_mapping = label_mapping
        self.label_mapping_inv = label_mapping_inv
        self.unknown_label = unknown_label

        if self.hparams.pretrained is not None:
            state_dict = torch.load(self.hparams.pretrained)
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )
            print(f"Missing: {missing_keys}", f"Unexpected: {unexpected_keys}")

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(
            num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.initial_epsilon_sk
        )

        self.sk_queue = None
        self.sk_indices = []

        self.loss_per_head = torch.zeros(self.hparams.num_heads, device=self.device)

        # wCE as loss
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        weights = torch.ones(len(self.label_mapping)) / len(self.label_mapping)
        self.criterion.weight = weights

        self.valid_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        weights = torch.ones(len(self.label_mapping)) / len(self.label_mapping)
        self.valid_criterion.weight = weights

        self.clip_align_criterion = torch.nn.CosineSimilarity()

        # Mapping numeric_label -> word_label
        dataset_config_file = self.hparams.dataset_config
        with open(dataset_config_file, "r") as f:
            dataset_config = yaml.safe_load(f)
        map_inv = dataset_config["learning_map_inv"]
        lab_dict = dataset_config["labels"]
        label_dict = {}
        for new_label, old_label in map_inv.items():
            label_dict[new_label] = lab_dict[old_label]
        self.label_dict = label_dict

        return

    def configure_optimizers(self):
        if self.hparams.pretrained is not None:
            encoder_params = self.model.encoder.parameters()
            rest_params = chain_iterators(
                self.model.head_lab.parameters(), self.model.head_unlab.parameters()
            )
            if hasattr(self.model, "head_unlab_over"):
                rest_params = chain_iterators(
                    rest_params, self.model.head_unlab_over.parameters()
                )
            optimizer = optim.SGD(
                [
                    {"params": rest_params, "lr": self.hparams.train_lr},
                    {"params": encoder_params},
                ],
                lr=self.hparams.finetune_lr,
                momentum=self.hparams.momentum_for_optim,
                weight_decay=self.hparams.weight_decay_for_optim,
            )
        else:
            optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.hparams.train_lr,
                momentum=self.hparams.momentum_for_optim,
                weight_decay=self.hparams.weight_decay_for_optim,
            )

        if self.hparams.use_scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_epochs,
                max_epochs=self.hparams.epochs,
                warmup_start_lr=self.hparams.min_lr,
                eta_min=self.hparams.min_lr,
            )

            return [optimizer], [scheduler]

        return optimizer

    def on_train_start(self):
        # Compute/load weights for weighted CE loss
        if not os.path.exists("weights.pt"):
            dataset = get_dataset(self.hparams.dataset)(
                config_file=self.hparams.dataset_config,
                split="train",
                voxel_size=self.hparams.voxel_size,
                downsampling=self.hparams.downsampling,
                augment=True,
                label_mapping=self.label_mapping,
            )

            weights = torch.zeros((self.hparams.num_classes), device=self.device)

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                shuffle=False,
            )

            # Split each unknown point across the 5 (or 4) unknown classes
            unk_labels_num = self.hparams.num_unlabeled_classes
            with tqdm(
                total=len(dataloader),
                desc="Evaluating weights for wCE",
                file=sys.stdout,
            ) as pbar:
                for data in dataloader:
                    _, _, _, _, labels, _, _ = data
                    for label in set(self.label_mapping.values()):
                        n_points = (labels == label).nonzero().numel()
                        if label != self.unknown_label:
                            weights[label] += n_points
                        else:
                            weights[-unk_labels_num:] += n_points / unk_labels_num
                    pbar.update()

            weights += 1
            weights = 1 / weights
            weights = weights / torch.sum(weights)
            self.criterion.weight = weights
            torch.save(weights, "weights.pt")
        else:
            print("\nLoading weights.pt ...", flush=True)
            weights = torch.load("weights.pt").to(self.device)
            self.criterion.weight = weights

    def train_dataloader(self):
        dataset = get_dataset(self.hparams.dataset)(
            config_file=self.hparams.dataset_config,
            split="train",
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
        )

        dataset = dataset_wrapper(dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset_two_samples,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader

    def val_dataloader(self):

        dataset = get_dataset(self.hparams.dataset)(
            config_file=self.hparams.dataset_config,
            split="valid",
            voxel_size=self.hparams.voxel_size,
            label_mapping=self.label_mapping,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
        )

        return dataloader

    def on_train_epoch_start(self):
        # Reset best_head tracker
        self.loss_per_head = torch.zeros_like(self.loss_per_head, device=self.device)

        # Compute the actual epsilon for Sinkhorn-Knopp
        if self.hparams.adapting_epsilon_sk and self.hparams.epochs > 1:
            eps_0 = self.hparams.initial_epsilon_sk
            eps_n = self.hparams.final_epsilon_sk
            n_ep = self.hparams.epochs
            act_ep = self.current_epoch
            self.sk.epsilon = eps_0 + act_ep * (eps_n - eps_0) / (n_ep - 1)

    def training_step(self, data, _):
        def get_uncertainty_mask(preds: torch.Tensor, p=0.5):
            """
            returns a boolean mask selecting the p-th percentile of the predictions with highest confidence for each class

            :param preds: Tensor of predicted logits (N x Nc)
            :param p: float describing the percentile to use in the selection
            """

            self.log(f"utils/tot_p", preds.shape[0])

            # init mask
            uncertainty_mask = torch.zeros(
                preds.shape[0], dtype=torch.bool, device=self.device
            )

            # get hard predictions
            hard_preds = preds.argmax(dim=-1)

            # generate indexes for consistent mapping
            indexes = torch.arange(preds.shape[0], device=self.device)

            # for each novel class
            for un_tmp in range(self.hparams.num_unlabeled_classes):
                # select points with given novel class
                un_idx_tmp = hard_preds == un_tmp

                if (un_idx_tmp.sum() * p).int() > 0:
                    # select confident novel pts
                    if self.hparams.dataset == "S3DIS":
                        un_conf = preds[un_idx_tmp].softmax(-1)[:, un_tmp]
                    else:
                        un_conf = preds[un_idx_tmp][:, un_tmp]
                    un_sel_tmp = indexes[un_idx_tmp]

                    # sort them
                    sorted_conf_tmp, sorted_idx_tmp = torch.sort(un_conf)
                    un_conf = un_conf[sorted_idx_tmp]
                    un_sel_tmp = un_sel_tmp[sorted_idx_tmp]

                    # get percentile idx
                    perc_tmp = (un_idx_tmp.sum() * p).int()

                    # update th
                    un_th_tmp = un_conf[perc_tmp]

                    # find valid pts
                    mask_tmp = un_conf > un_th_tmp

                    self.log(f"utils/thr_{un_tmp}", un_th_tmp)
                    self.log(
                        f"utils/perc_{un_tmp}", mask_tmp.sum() / un_sel_tmp.shape[0]
                    )
                    self.log(f"utils/tot_p_{un_tmp}", un_sel_tmp.shape[0])

                    uncertainty_mask[un_sel_tmp[mask_tmp]] = 1

            return uncertainty_mask

        nlc = self.hparams.num_labeled_classes

        (
            coords,
            feats,
            _,
            selected_idx,
            mapped_labels,
            superpoints,
            coords1,
            feats1,
            _,
            selected_idx1,
            mapped_labels1,
            superpoints1,
            pcd_indexes,
        ) = data

        # Forward
        coords = coords.int()
        coords1 = coords1.int()

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords)
        sp_tensor1 = ME.SparseTensor(features=feats1.float(), coordinates=coords1)

        # Clear cache at regular interval
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor, superpoints=superpoints)
        out1 = self.model(sp_tensor1, superpoints=superpoints1)

        coords_bck = torch.clone(coords)
        coords_bck1 = torch.clone(coords1)

        pcd_masks = []
        pcd_masks1 = []
        for i in range(pcd_indexes.shape[0]):
            pcd_masks.append(coords[:, 0] == i)
            pcd_masks1.append(coords1[:, 0] == i)

        # Gather outputs
        out["logits_lab"] = (
            out["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1)
        )
        out1["logits_lab"] = (
            out1["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1)
        )
        logits = torch.cat([out["logits_lab"], out["logits_unlab"]], dim=-1)
        logits1 = torch.cat([out1["logits_lab"], out1["logits_unlab"]], dim=-1)
        if self.hparams.overcluster_factor is not None:
            logits_over = torch.cat(
                [out["logits_lab"], out["logits_unlab_over"]], dim=-1
            )
            logits_over1 = torch.cat(
                [out1["logits_lab"], out1["logits_unlab_over"]], dim=-1
            )


        mask_lab = mapped_labels != self.unknown_label
        mask_lab1 = mapped_labels1 != self.unknown_label

        # Generate one-hot targets for the base points
        targets_lab = (
            F.one_hot(
                mapped_labels[mask_lab].to(torch.long),
                num_classes=self.hparams.num_labeled_classes,
            )
            .float()
            .to(self.device)
        )
        targets_lab1 = (
            F.one_hot(
                mapped_labels1[mask_lab1].to(torch.long),
                num_classes=self.hparams.num_labeled_classes,
            )
            .float()
            .to(self.device)
        )

        # Generate empty targets for all the points
        targets = torch.zeros_like(logits)
        targets1 = torch.zeros_like(logits1)
        if self.hparams.overcluster_factor is not None:
            targets_over = torch.zeros_like(logits_over)
            targets_over1 = torch.zeros_like(logits_over1)

        # Generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        act_queue = (
            None
            if self.current_epoch < self.hparams.queue_start_epoch
            else self.sk_queue
        )
        for h in range(self.hparams.num_heads):
            # Insert the one-hot labels
            targets[h, mask_lab, :nlc] = targets_lab.type_as(targets)
            targets1[h, mask_lab1, :nlc] = targets_lab1.type_as(targets1)

            if self.hparams.use_uncertainty_queue or self.hparams.use_uncertainty_loss:
                # Get masks for certain points
                unc_mask = get_uncertainty_mask(
                    out["logits_unlab"][h][~mask_lab].detach(),
                    p=self.hparams.uncertainty_percentile,
                )
                unc_mask1 = get_uncertainty_mask(
                    out1["logits_unlab"][h][~mask_lab1].detach(),
                    p=self.hparams.uncertainty_percentile,
                )
                if h == 0:
                    unc_mask_overall = unc_mask
                    unc_mask_overall1 = unc_mask1
                else:
                    unc_mask_overall = torch.logical_and(unc_mask_overall, unc_mask)
                    unc_mask_overall1 = torch.logical_and(unc_mask_overall1, unc_mask1)

            if self.hparams.use_uncertainty_loss:
                # Get predictions from Sinkhorn only for high-confidence points
                pred_sk = self.sk(
                    out["feats"][~mask_lab][unc_mask],
                    self.model.head_unlab.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets)
                pred_sk1 = self.sk(
                    out1["feats"][~mask_lab1][unc_mask1],
                    self.model.head_unlab.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets)

                new_mask_unlab = ~mask_lab.clone()
                new_mask_unlab[new_mask_unlab == True] = unc_mask
                new_mask_unlab1 = ~mask_lab1.clone()
                new_mask_unlab1[new_mask_unlab1 == True] = unc_mask1
                # Use sinkhorn labels only with the confident points (unconfident ones remain zero_labelled)
                targets[h, new_mask_unlab, nlc:] = pred_sk
                targets1[h, new_mask_unlab1, nlc:] = pred_sk1
            else:
                # Insert sinkhorn labels
                targets[h, ~mask_lab, nlc:] = self.sk(
                    out["feats"][~mask_lab],
                    self.model.head_unlab.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets)
                targets1[h, ~mask_lab1, nlc:] = self.sk(
                    out1["feats"][~mask_lab1],
                    self.model.head_unlab.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets)

            if self.hparams.overcluster_factor is not None:
                # Manage also overclustering heads
                targets_over[h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets_over[h, ~mask_lab, nlc:] = self.sk(
                    out["feats"][~mask_lab],
                    self.model.head_unlab_over.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets)
                targets_over1[h, mask_lab1, :nlc] = targets_lab1.type_as(targets1)
                targets_over1[h, ~mask_lab1, nlc:] = self.sk(
                    out1["feats"][~mask_lab1],
                    self.model.head_unlab_over.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets1)


        # Evaluate loss
        loss_cluster = self.loss(
            logits, targets1, selected_idx, selected_idx1, pcd_masks, pcd_masks1
        )
        loss_cluster += self.loss(
            logits1, targets, selected_idx1, selected_idx, pcd_masks1, pcd_masks
        )

        if self.hparams.overcluster_factor is not None:
            loss_overcluster = self.loss(
                logits_over,
                targets_over1,
                selected_idx,
                selected_idx1,
                pcd_masks,
                pcd_masks1,
            )
            loss_overcluster += self.loss(
                logits_over1,
                targets_over,
                selected_idx1,
                selected_idx,
                pcd_masks1,
                pcd_masks,
            )
        else:
            loss_overcluster = loss_cluster

        # Keep track of the loss for each head
        self.loss_per_head += loss_cluster.clone().detach()

        loss_cluster = loss_cluster.mean()
        loss_overcluster = loss_overcluster.mean()
        loss = (loss_cluster + loss_overcluster) / 2

        # SP consistency: JS + detach(mean) + entropy gate (scheme B) + |S| filter
        def _sp_js_loss(_logits, _coords, _spids_col, _alpha):
            eps = 1e-6
            H, N, C = _logits.shape
            _b = _coords[:, 0].to(torch.long)
            _pair = torch.stack([_b, _spids_col.to(torch.long)], dim=1)
            _, _g = torch.unique(_pair, return_inverse=True, dim=0)
            G = int(_g.max().item()) + 1
            _counts = torch.bincount(_g, minlength=G).clamp_min(1)

            _logC = torch.log(torch.tensor(float(C), device=self.device, dtype=_logits.dtype))
            _h = _alpha * _logC

            loss_sum = _logits.new_tensor(0.0)
            total_w  = _logits.new_tensor(0.0)
            eff_groups = _logits.new_tensor(0.0)

            for h in range(H):
                p = torch.softmax(_logits[h], dim=-1)  
                bar = torch.zeros(G, C, device=p.device, dtype=p.dtype)
                bar.index_add_(0, _g, p)
                bar = bar / _counts.unsqueeze(1)
                bar_det = bar.detach()

                H_bar = -(bar_det * (bar_det + eps).log()).sum(dim=1)
                w_ent = ((_h - H_bar) / _h).clamp(0.0, 1.0)             
                size_mask = (_counts >= 2).float()                       
                w_group = w_ent * size_mask                              

                bar_g = bar_det[_g]                                      
                m = 0.5 * (p + bar_g)
                kl_p_m = (p * ((p + eps).log() - (m + eps).log())).sum(dim=1)
                kl_b_m = (bar_g * ((bar_g + eps).log() - (m + eps).log())).sum(dim=1)
                js = 0.5 * (kl_p_m + kl_b_m)                             

                w_point = (w_group[_g] / _counts[_g].to(p.dtype))        
                loss_sum = loss_sum + (w_point * js).sum()
                total_w  = total_w  + w_group.sum()
                # eff_groups = eff_groups + (w_group > 0).float().sum()

            return loss_sum / total_w.clamp_min(1e-6) 
            # return loss_sum / eff_groups.clamp_min(1.0)
        
        # Toronto_1 Toronto_3
        if self.current_epoch < 10:
            _levels  = [2]                    
            _level_w = {2: 1.0}
        elif self.current_epoch < 20:
            _levels  = [2, 1]                
            _level_w = {2: 0.6, 1: 0.6}
        else:
            _levels  = [2, 1, 0]              
            _level_w = {2: 0.4, 1: 0.6, 0: 0.3}

        _level_alpha = {2: 0.45, 1: 0.60, 0: 0.50}
        
        # Toronto_0
        # if self.current_epoch < 10:
        #     _levels  = [2]                          # 仅粗层
        #     _level_w = {2: 1.0}
        # elif self.current_epoch < 20:
        #     _levels  = [2, 1]                       # 加中层
        #     _level_w = {2: 0.50, 1: 0.80}           # 粗降权，中升权
        # else:
        #     _levels  = [2, 1, 0]                    # 再加细层
        #     _level_w = {2: 0.25, 1: 0.75, 0: 0.50} 
        # _level_alpha = {2: 0.45, 1: 0.60, 0: 0.50}
            
        _sp_logs = {}
        loss_sp_js = logits.new_tensor(0.0)
        for _L in _levels:
            _spids  = superpoints[:,  _L].to(self.device)
            _spids1 = superpoints1[:, _L].to(self.device)
            _l = 0.5 * (
                _sp_js_loss(logits,  coords,  _spids,  _level_alpha[_L]) +
                _sp_js_loss(logits1, coords1, _spids1, _level_alpha[_L])
            )
            loss_sp_js = loss_sp_js + _level_w[_L] * _l
            _sp_logs[f"train/loss_sp_js_L{_L}"] = _l.detach()
        _sp_logs["train/sp_levels_active"] = float(len(_levels))
        
        loss = loss + self.w_sp_loss * loss_sp_js

        if self.hparams.clip_path is not None:
            # sp_tensor_clip = ME.SparseTensor(
            #     features=feats.float().repeat(1, 3), coordinates=coords_bck
            # )
            # sp_tensor_clip1 = ME.SparseTensor(
            #     features=feats1.float().repeat(1, 3), coordinates=coords_bck1
            # )
            
            # intensity & rgb
            clip_feats_input = feats[:, 1:4]
            sp_tensor_clip = ME.SparseTensor(
                features=clip_feats_input.float(), coordinates=coords_bck
            )
            clip_feats_input1 = feats1[:, 1:4]
            sp_tensor_clip1 = ME.SparseTensor(
                features=clip_feats_input1.float(), coordinates=coords_bck1
            )


            self.clip_model.eval()

            with torch.no_grad():
                clip_feats = self.clip_model(sp_tensor_clip).F
                clip_feats1 = self.clip_model(sp_tensor_clip1).F

            clip_align_loss = self.clip_align_loss(
                out["clip_feats"],
                out1["clip_feats"],
                clip_feats,
                clip_feats1,
                mask=~mask_lab,
                mask1=~mask_lab1,
            )

            if not torch.isnan(clip_align_loss):
                loss += self.hparams.clip_weight * clip_align_loss

        # logging
        results = {
            "train/loss": loss.detach(),
            "train/loss_cluster": loss_cluster.detach(),
        }

        if self.hparams.overcluster_factor is not None:
            results["train/loss_overcluster"] = loss_overcluster.detach()

        if self.hparams.clip_path is not None:
            results["train/loss_clip_align"] = clip_align_loss.detach()

        results["train/loss_sp_js"] = loss_sp_js.detach()
        results.update(_sp_logs)    

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        if self.hparams.queue_start_epoch != -1:
            if self.hparams.use_uncertainty_queue:
                self.update_queue(
                    torch.cat(
                        (
                            out["feats"][~mask_lab][unc_mask_overall],
                            out1["feats"][~mask_lab1][unc_mask_overall1],
                        )
                    )
                )
            else:
                self.update_queue(
                    torch.cat((out["feats"][~mask_lab], out1["feats"][~mask_lab1]))
                )

        return loss

    def update_queue(self, feats: torch.Tensor):
        """
        Updates self.queue with the features of the novel points in the current batch

        :param feats: the features for the novel points in the current batch
        """
        feats = feats.detach()
        if not self.hparams.use_uncertainty_queue:
            n_feats_to_retain = int(feats.shape[0] * self.hparams.queue_percentage)
            mask = torch.randperm(feats.shape[0])[:n_feats_to_retain]
        else:
            n_feats_to_retain = feats.shape[0]
            mask = torch.ones(n_feats_to_retain, device=feats.device, dtype=torch.bool)
        if self.sk_queue is None:
            self.sk_queue = feats[mask]
            self.sk_indices.append(n_feats_to_retain)
            return

        if len(self.sk_indices) < self.hparams.queue_batches:
            self.sk_queue = torch.vstack((feats[mask], self.sk_queue))
            self.sk_indices.insert(0, n_feats_to_retain)
        else:
            self.sk_queue = torch.vstack(
                (feats[mask], self.sk_queue[: -self.sk_indices[-1]])
            )
            self.sk_indices.insert(0, n_feats_to_retain)
            del self.sk_indices[-1]

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        idx_logits: torch.Tensor,
        idx_targets: torch.Tensor,
        pcd_mask_logits: torch.Tensor,
        pcd_mask_targets: torch.Tensor,
    ):
        """
        Evaluates the loss function of the predicted logits w.r.t. the targets
        """

        if self.criterion.weight.shape[0] != targets.shape[2]:
            weight_bck = self.criterion.weight.clone()
            weight_new = torch.zeros(targets.shape[2], device=self.device)
            weight_new[: self.hparams.num_labeled_classes] = weight_bck[
                : self.hparams.num_labeled_classes
            ]
            new_weight_tmp = weight_bck[-1] / self.hparams.overcluster_factor
            weight_new[
                -self.hparams.num_unlabeled_classes * self.hparams.overcluster_factor :
            ] = new_weight_tmp
            self.criterion.weight = weight_new
        else:
            weight_bck = None

        # apply_cb = self.current_epoch >= self.cb_drw_start_epoch
        # priors_full    = self.cb_priors_full          
        # unknown_labels = self.unknown_labels         
        # cb_weight_min  = self.cb_weight_min
        # cb_weight_max  = self.cb_weight_max
        # cb_ratio_cap   = self.cb_ratio_cap
        # cb_ratio_threshold = self.cb_ratio_threshold

        heads_loss = None

        for head in range(self.hparams.num_heads):
            head_loss = None
            for pcd in range(len(pcd_mask_logits)):
                pcd_logits = logits[head][pcd_mask_logits[pcd]]
                pcd_targets = targets[head][pcd_mask_targets[pcd]]
                ####
                logit_shape = pcd_logits.shape[0]
                target_shape = pcd_targets.shape[0]
                ####
                mask_logits = torch.isin(
                    idx_logits[pcd_mask_logits[pcd]], idx_targets[pcd_mask_targets[pcd]]
                )
                mask_targets = torch.isin(
                    idx_targets[pcd_mask_targets[pcd]], idx_logits[pcd_mask_logits[pcd]]
                )
                pcd_logits = pcd_logits[mask_logits]
                pcd_targets = pcd_targets[mask_targets]
                ####
                perc_to_log = (
                    pcd_logits.shape[0] / logit_shape
                    + pcd_targets.shape[0] / target_shape
                ) / 2
                self.log("utils/points_in_common", perc_to_log)
                ####

                # ---------- 分组加权：构造 w_cls_full ----------
                # if apply_cb:
                #     Ctot = pcd_logits.shape[-1]
                #     w_cls_full = torch.ones(Ctot, device=self.device)

                #     priors = priors_full
                #     if priors.numel() < Ctot:
                #         priors = torch.cat([priors, torch.zeros(Ctot - priors.numel(), device=self.device)])
                #     else:
                #         priors = priors[:Ctot]

                #     # 基类 = 全部类别剔除“新类集合 unknown_labels”
                #     base_mask = torch.ones(Ctot, dtype=torch.bool, device=self.device)
                #     unk_idx = unknown_labels.to(self.device)
                #     unk_idx = unk_idx[(unk_idx >= 0) & (unk_idx < Ctot)]
                #     if unk_idx.numel() > 0:
                #         base_mask[unk_idx] = False

                #     base_sum = priors[base_mask].sum()
                #     new_sum  = priors[unk_idx].sum() if unk_idx.numel() > 0 else torch.tensor(0.0, device=self.device)

                #     eps = torch.tensor(1e-6, device=self.device)
                #     ratio = base_sum / new_sum.clamp_min(eps)

                #     if ratio.item() > cb_ratio_threshold:
                #         cap = torch.tensor(cb_ratio_cap, device=self.device)
                #         log_r = torch.log(ratio.clamp(1.0 / cap, cap))

                #         w_new = 1.0 + (log_r / torch.log(cap)) * (cb_weight_max - 1.0)  # r=cap→w_max, r=1→1
                #         w_new = torch.clamp(w_new, cb_weight_min, cb_weight_max)

                #         if unk_idx.numel() > 0:
                #             w_cls_full[unk_idx] = w_new
                #     else:
                #         w_cls_full = None
                # else:
                #     w_cls_full = None
                # # ---------- 分组加权构造结束 ----------

                loss = self.criterion(pcd_logits, pcd_targets)   # [N] per-point CE（你原有的 criterion）

                # 应用新类分组加权：软目标用期望权重；若是硬标签可改为 w_cls_full[labels]
                # if apply_cb and (w_cls_full is not None):
                #     w_point = (pcd_targets * w_cls_full.unsqueeze(0)).sum(dim=1)  # [N]
                #     loss = loss * w_point

                if self.hparams.use_uncertainty_loss:
                    loss = loss[loss > 0]

                multiplier = 1 / ((self.criterion.weight * pcd_targets).sum(1)).sum(0)
                loss *= multiplier
                loss = loss.sum()

                if head_loss is None:
                    head_loss = loss
                else:
                    head_loss = torch.hstack((head_loss, loss))

            if heads_loss is None:
                heads_loss = head_loss.mean()
            else:
                heads_loss = torch.hstack((heads_loss, head_loss.mean()))

        if weight_bck is not None:
            self.criterion.weight = weight_bck

        heads_loss[heads_loss.isnan()] = (
            heads_loss[~heads_loss.isnan()].mean()
            if not torch.isnan(heads_loss[~heads_loss.isnan()].mean())
            else torch.tensor(0.0, device=heads_loss.device)
        )
        return heads_loss


    def clip_align_loss(self, pred, pred1, gt, gt1, mask=None, mask1=None):
        if mask is not None:
            loss = (1 - self.clip_align_criterion(pred[mask], gt[mask])).mean()
        else:
            loss = (1 - self.clip_align_criterion(pred, gt)).mean()

        if mask1 is not None:
            loss += (1 - self.clip_align_criterion(pred1[mask1], gt1[mask1])).mean()
        else:
            loss += (1 - self.clip_align_criterion(pred1, gt1)).mean()

        return loss

    def on_validation_epoch_start(self):
        # Run the hungarian algorithm to map each novel class to the related semantic class
        if (
            self.hparams.hungarian_at_each_step
            or len(self.label_mapping_inv) < self.hparams.num_classes
        ):
            cost_matrix = torch.zeros(
                (
                    self.hparams.num_unlabeled_classes,
                    self.hparams.num_unlabeled_classes,
                ),
                device=self.device,
            )

            dataset = get_dataset(self.hparams.dataset)(
                config_file=self.hparams.dataset_config,
                split="valid",
                voxel_size=self.hparams.voxel_size,
                label_mapping=self.label_mapping,
            )

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
            )

            real_labels_to_be_matched = [
                label
                for label in self.label_mapping
                if self.label_mapping[label] == self.unknown_label
            ]

            with tqdm(
                total=len(dataloader), desc="Cost matrix build-up", file=sys.stdout
            ) as pbar:
                for step, data in enumerate(dataloader):
                    coords, feats, real_labels, _, mapped_labels, superpoints, _ = data

                    # Forward
                    coords = coords.int().to(self.device)
                    feats = feats.to(self.device)
                    real_labels = real_labels.to(self.device)

                    sp_tensor = ME.SparseTensor(
                        features=feats.float(), coordinates=coords
                    )

                    # Must clear cache at regular interval
                    if self.global_step % self.hparams.clear_cache_int == 0:
                        torch.cuda.empty_cache()

                    out = self.model(sp_tensor, superpoints=superpoints)

                    best_head = torch.argmin(self.loss_per_head)

                    mask_unknown = mapped_labels == self.unknown_label

                    preds = out["logits_unlab"][best_head]

                    preds = torch.argmax(preds[mask_unknown].softmax(1), dim=1)

                    for pseudo_label in range(self.hparams.num_unlabeled_classes):
                        mask_pseudo = preds == pseudo_label
                        for j, real_label in enumerate(real_labels_to_be_matched):
                            mask_real = real_labels[mask_unknown] == real_label
                            cost_matrix[pseudo_label, j] += torch.logical_and(
                                mask_pseudo, mask_real
                            ).sum()

                    pbar.update()

            cost_matrix = cost_matrix / (
                torch.negative(cost_matrix)
                + torch.sum(cost_matrix, dim=0)
                + torch.sum(cost_matrix, dim=1).unsqueeze(1)
                + 1e-5
            )

            # Hungarian
            cost_matrix = cost_matrix.cpu()
            row_ind, col_ind = linear_sum_assignment(
                cost_matrix=cost_matrix, maximize=True
            )
            label_mapping = {
                row_ind[i] + self.unknown_label: real_labels_to_be_matched[col_ind[i]]
                for i in range(len(row_ind))
            }
            self.label_mapping_inv.update(label_mapping)

        # Reorder weights for validation loss
        weights = self.criterion.weight.clone()
        sorted_label_mapping_inv = dict(
            sorted(self.label_mapping_inv.items(), key=lambda item: item[1])
        )
        sorter = list(sorted_label_mapping_inv.keys())
        self.valid_criterion.weight = weights[sorter]

        return

    def validation_step(self, data, batch_idx):
        coords, feats, real_labels, _, _, superpoints, scene_info = data

        # Forward
        coords = coords.int()
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords)

        # Must clear cache at regular interval
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor, superpoints=superpoints)
        best_head = torch.argmin(self.loss_per_head)

        preds = torch.cat([out["logits_lab"], out["logits_unlab"][best_head]], dim=-1)

        sorted_label_mapping_inv = dict(
            sorted(self.label_mapping_inv.items(), key=lambda item: item[1])
        )
        sorter = list(sorted_label_mapping_inv.keys())
        preds = preds[:, sorter]

        loss = self.valid_criterion(preds, real_labels.long())

        gt_labels = real_labels
        avail_labels = torch.unique(gt_labels).long()
        _, pred_labels = torch.max(torch.softmax(preds.detach(), dim=1), dim=1)

        IoU = jaccard_index(gt_labels, pred_labels, reduction="none")
        IoU = IoU[avail_labels]

        self.log("valid/loss", loss, on_epoch=True, sync_dist=True, rank_zero_only=True)
        IoU_to_log = {
            f"valid/IoU/{self.label_dict[int(avail_labels[i])]}": label_IoU
            for i, label_IoU in enumerate(IoU)
        }
        for label, value in IoU_to_log.items():
            self.log(label, value, on_epoch=True, sync_dist=True, rank_zero_only=True)

        # 保存场景级别的预测结果
        if self.save_predictions:
            self.accumulate_scene_predictions(data, preds, batch_idx)

        return loss

    # def save_point_cloud_predictions(self, batch, pred_logits, batch_idx):

    #     coords, _, _, _, _, _ = batch                
    #     xyz = coords[:, 1:].float().cpu().numpy()      

    #     pred_labels = torch.argmax(pred_logits, dim=1).cpu().numpy()
    
    #     out = np.hstack((xyz, pred_labels[:, None]))   # (N, 4)

        
    #     os.makedirs(self.output_dir, exist_ok=True)
    #     fname = os.path.join(self.output_dir, f"prediction_{batch_idx:04d}.txt")
    #     np.savetxt(fname, out, fmt="%.6f %.6f %.6f %d")

    def accumulate_scene_predictions(self, batch, pred_logits, batch_idx):
        """
        累积场景级别的预测结果，使用坐标作为唯一键进行投票融合
        """
        
        coords, feats, real_labels, _, _, _, scene_info = batch
        
        coords_np = coords[:, 1:].float().cpu().numpy()  
        probs = torch.softmax(pred_logits, dim=1).cpu().numpy()
        num_classes = probs.shape[1]
        
        # 对每个点进行投票融合
        for i, point in enumerate(coords_np):
            point_key = tuple(point) 
            
            if point_key not in self.val_probs:
                self.val_probs[point_key] = np.zeros(num_classes, dtype=np.float32)
                self.val_counts[point_key] = 0
            
            # 平滑更新概率
            self.val_probs[point_key] = (
                self.test_smooth * self.val_probs[point_key] + 
                (1 - self.test_smooth) * probs[i]
            )
            self.val_counts[point_key] += 1

    def save_scene_ply(self):
        from plyfile import PlyData, PlyElement
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not self.val_probs:
            print("No predictions to save")
            return
        
        color_map = {
            0: [255, 255, 255],   # unlabeled     
            1: [140, 140, 140],   # Ground
            2: [  7, 255, 235],   # Road marking   
            3: [  3, 200,   4],   # Natural
            4: [120, 120, 180],   # Building
            5: [230, 230,   6],   # Utility line
            6: [255,   0,  51],   # Pole
            7: [200, 102,   0],   # Car
            8: [  6, 184, 255]    # Fence
        }
        
        coords_list = []
        probs_list = []
        counts_list = []
        
        for point_key, probs in self.val_probs.items():
            coords_list.append(point_key)  # (x, y, z)
            probs_list.append(probs)
            counts_list.append(self.val_counts[point_key])
        
        coords = np.array(coords_list)
        probs = np.array(probs_list)
        counts = np.array(counts_list)
        
        pred_labels = np.argmax(probs, axis=1)
        
        colors = np.array([color_map.get(label, [128, 128, 128]) for label in pred_labels])
        
        vertex_data = np.array([
            (coords[i, 0], coords[i, 1], coords[i, 2], 
            colors[i, 0], colors[i, 1], colors[i, 2], pred_labels[i])
            for i in range(len(coords))
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('pred_label', 'u1')])
        
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        
        filename = "scene.ply"
        filepath = os.path.join(self.output_dir, filename)
        PlyData([vertex_element]).write(filepath)
        
        print(f"Saved scene predictions to {filepath}")
        print(f"Total points: {len(coords)}, avg votes: {np.mean(counts):.2f}")

    def save_scene_predictions(self):
        """
        保存融合后的预测结果
        """
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not self.val_probs:
            print("No predictions to save")
            return
        
        coords_list = []
        probs_list = []
        counts_list = []
        
        for point_key, probs in self.val_probs.items():
            coords_list.append(point_key)  # (x, y, z)
            probs_list.append(probs)
            counts_list.append(self.val_counts[point_key])
        
        coords = np.array(coords_list)
        probs = np.array(probs_list)
        counts = np.array(counts_list)
        
        # 获取最终预测标签
        pred_labels = np.argmax(probs, axis=1)
        
        result = np.column_stack([
            coords,  # x, y, z
            pred_labels,  # predicted label
            counts  # number of votes
        ])
        
        filename = "scene_predictions.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        np.savetxt(filepath, result, fmt='%.6f %.6f %.6f %d %d', 
                  header='x y z pred_label vote_count')
        
        print(f"Saved scene predictions to {filepath}")
        print(f"Total points: {len(coords)}, avg votes: {np.mean(counts):.2f}")

    def on_validation_epoch_end(self):
        """
        验证结束后保存所有场景的预测结果
        """
        if self.save_predictions and self.val_probs:
            # self.save_scene_predictions()
            self.save_scene_ply()
            self.val_probs.clear()
            self.val_counts.clear()

    def on_save_checkpoint(self, checkpoint):
        # Maintain info about best head when saving checkpoints
        checkpoint["loss_per_head"] = self.loss_per_head
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        self.loss_per_head = checkpoint.get(
            "loss_per_head",
            torch.zeros(
                checkpoint["hyper_parameters"]["num_heads"], device=self.device
            ),
        )
        return super().on_load_checkpoint(checkpoint)


    
