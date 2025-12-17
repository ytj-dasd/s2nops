import torch
import torch.nn as nn
import torch.nn.functional as F

class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, features, head, queue=None):
        if queue is None or queue.shape[0] == 0:
            queue = None
        if queue is not None:
            features = torch.vstack((features, queue))

        features = torch.nn.functional.normalize(features, dim=1, p=2)
        head = torch.nn.functional.normalize(head, dim=1, p=2)

        logits = features@head

        logits = logits.to(torch.float64)
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        to_ret = Q.t() if queue is None else Q.t()[:-queue.shape[0]]

        return to_ret

class SemiSinkhornKnopp(nn.Module):
    def __init__(self, epsilon=0.1, gamma=1.0, stoperr=1e-6, numItermax=1000):
        super().__init__()
        self.epsilon = float(epsilon)      # 熵正则温度
        self.gamma = float(gamma)          # 半松弛强度
        self.stoperr = float(stoperr)      # 停止阈值（||b_t - b_{t-1}||）
        self.numItermax = int(numItermax)  # 最大迭代步数
        self.w = None                      # 记录列边际（监控用）

    @torch.no_grad()
    def forward(self, P: torch.Tensor):
        dev = P.device
        N, K = P.shape
        eps = max(self.epsilon, 1e-8)

        # Stabilize kernel: use log_softmax-style costs above, but still guard underflow
        Q = torch.exp(-P / eps).clamp_min(1e-12)  # [N, K]

        # 均匀先验
        Pa = torch.full((N, 1), 1.0 / N, device=dev)
        Pb = torch.full((K, 1), 1.0 / K, device=dev)

        # 半松弛幂次：fi=gamma/(gamma+epsilon)；fi=1→等量，fi=0→完全不等量
        fi = float(self.gamma) / (float(self.gamma) + eps)

        # 迭代标度
        b = torch.full((K, 1), 1.0 / K, device=dev)
        last_b = b
        err = torch.tensor(1.0, device=dev)
        iters = 0
        while float(err) > self.stoperr and iters < self.numItermax:
            denom_ab = (Q @ b).clamp_min(1e-12)
            a = Pa / denom_ab                      # 行标度（保证列归一/每样本分布）
            denom_b = (Q.t() @ a).clamp_min(1e-12)
            b = torch.pow(Pb / denom_b, fi)        # 列标度（半松弛）
            err = torch.norm(b - last_b)
            last_b = b
            iters += 1

        # 运输计划
        OT_plan = N * a * Q * b.t()                      # [N, K]

        # OT 期望代价
        ot_loss = torch.mean(torch.sum(OT_plan * P, dim=1))

        # 列边际与 KL
        w = OT_plan.mean(dim=0, keepdim=True)            # [1, K]
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        self.w = w
        reg = F.kl_div(torch.log(w + 1e-7), Pb.reshape(1, -1), reduction="batchmean")

        return OT_plan, ot_loss, reg
