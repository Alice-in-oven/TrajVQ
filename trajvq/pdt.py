import math

import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from faiss.contrib.inspect_tools import get_pq_centroids


class Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)


class NonLinear(nn.Module):
    def __init__(self, d, d_hidden, heads=1):
        super().__init__()
        self.non_linear = nn.ModuleList()
        for _ in range(heads):
            self.non_linear.append(nn.Sequential(
                nn.Linear(d, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d),
                Normalize(),
            ))
        self.to_score = nn.Linear(d, 1)

    def forward(self, x):
        xs = torch.stack([layer(x) for layer in self.non_linear], dim=1)
        score = self.to_score(xs).softmax(1)
        return (xs * score).sum(dim=1)


class MultiStepNonLinear(nn.Module):
    def __init__(self, d, d_hidden, heads=1, steps=1):
        super().__init__()
        self.transform = nn.ModuleList([NonLinear(d, d_hidden, heads) for _ in range(steps)])

    def forward(self, x):
        for step in self.transform:
            x = x + step(x)
        return x


class MultiStepDistributionTrans(nn.Module):
    def __init__(self, d, d_hidden, steps=1, heads=1):
        super().__init__()
        self.encoder = MultiStepNonLinear(d, d_hidden, heads, steps)
        self.decoder = MultiStepNonLinear(d, d_hidden, heads, steps)

    def encode(self, x):
        if self.training:
            x = x + 0.001 * torch.randn(x.shape).to(x.device)
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


def _one_hot(y, depth):
    y_flat = y.to(torch.int64).view(-1, 1)
    y_one_hot = torch.zeros(y_flat.size()[0], depth, device=y.device).scatter_(1, y_flat, 1)
    return y_one_hot.view(*(tuple(y.shape) + (-1,)))


def gumbel_softmax(dist, noise=1.0, eps=1e-20):
    if noise != 0:
        u = torch.rand(*dist.shape, device=dist.device, dtype=dist.dtype)
        dist = dist + noise * -torch.log(-torch.log(u + eps) + eps)
    probs = torch.softmax(dist, dim=-1)
    _, argmax = torch.max(probs, dim=-1)
    return (_one_hot(argmax, depth=dist.shape[-1]) - probs).detach() + probs


class Quantizer(nn.Module):
    def __init__(self, d, groups, centroids):
        super().__init__()
        assert d % groups == 0
        self.groups = groups
        self.codebook = nn.Parameter(torch.randn(groups, centroids, d // groups), requires_grad=True)
        self.log_temperatures = nn.Parameter(data=torch.zeros(groups), requires_grad=True)

    def compute_score(self, x):
        x = rearrange(x, 'n (g d) -> n g d', g=self.groups)
        norm_x = torch.sum(x ** 2, dim=-1, keepdim=True)
        norm_c = torch.sum(self.codebook ** 2, dim=-1).unsqueeze(0)
        dot = torch.matmul(x.permute(1, 0, 2), self.codebook.permute(0, 2, 1))
        score = - norm_x + 2 * dot.permute(1, 0, 2) - norm_c
        score *= torch.exp(-self.log_temperatures[:, None])
        return score

    def get_codes(self, x):
        return self.compute_score(x).argmax(dim=-1)

    def reconstruction(self, codes):
        x_recon = torch.stack([self.codebook[g, codes[:, g]] for g in range(self.groups)])
        return rearrange(x_recon, 'g n d -> n (g d)')

    def forward(self, x):
        score = gumbel_softmax(self.compute_score(x), noise=1.0 if self.training else 0.0)
        x_recon = torch.einsum('ngk,gkd->ngd', score, self.codebook)
        return rearrange(x_recon, 'n g d -> n (g d)'), score


class DPQ(nn.Module):
    def __init__(self, d, M, K):
        super().__init__()
        self.M, self.K = M, K
        self.quantizer = Quantizer(d, M, K)

    def init_codebook(self, x):
        pq = faiss.ProductQuantizer(x.shape[-1], self.M, int(math.log2(self.K)))
        pq.train(x.data.cpu().numpy())
        self.quantizer.codebook.copy_(torch.as_tensor(get_pq_centroids(pq)).to(x.device))

    def get_codes(self, x):
        return self.quantizer.get_codes(x)

    def reconstruction(self, codes):
        return self.quantizer.reconstruction(codes)

    def forward(self, x):
        return self.quantizer(x)


def distance_loss(x1, x2):
    dist1 = torch.cdist(x1, x1, 2)
    dist2 = torch.cdist(x2, x2, 2)
    return torch.sqrt(torch.pow(dist1 - dist2, 2).clamp(min=1e-10)).mean(-1)


def uniform_loss(codebook):
    dots = torch.matmul(codebook, codebook.transpose(-1, -2))
    m, k, _ = codebook.shape
    dots.view(m, -1)[:, ::(k + 1)].fill_(-1)
    _, nearest = torch.topk(dots, 1, -1)
    neighbours = torch.gather(codebook, 1, nearest.repeat(1, 1, codebook.shape[-1]))
    return - torch.log(codebook.shape[-1] * F.pairwise_distance(codebook, neighbours)).mean()


class PDT(nn.Module):
    def __init__(self, d, M, K, steps, heads, d_hidden=256):
        super().__init__()
        self.trans = MultiStepDistributionTrans(d, d_hidden, steps, heads)
        self.vq = DPQ(d, M, K)
        self.loss_schedule = {0: (0, 10), 20: (10, 10), 40: (10, 1), 60: (5, 0.1), 80: (1, 0.01)}
        self.rank_weight, self.dist_weight = 1.0, 1.0

    def set_epoch(self, epoch):
        if epoch in self.loss_schedule:
            self.rank_weight, self.dist_weight = self.loss_schedule[epoch]

    def init_codebook(self, x):
        self.vq.init_codebook(self.encode(x))

    def encode(self, x):
        return self.trans.encode(x)

    def decode(self, x):
        return self.trans.decode(x)

    def get_codes(self, x):
        return self.vq.get_codes(x)

    def reconstruction(self, codes):
        return self.vq.reconstruction(codes)

    def soft_quantize(self, x):
        x_enc = self.encode(x)
        x_recon_vq, soft_codes = self.vq(x_enc)

        codebook_loss = F.mse_loss(x_recon_vq, x_enc.detach())
        commitment_loss = F.mse_loss(x_recon_vq.detach(), x_enc)
        avg_probs = soft_codes.mean(dim=0).clamp_min(1e-8)
        entropy = -(avg_probs * avg_probs.log()).sum(dim=-1).mean()
        return {
            "commitment": codebook_loss + 0.25 * commitment_loss,
            "entropy": x_enc.new_tensor(math.log(float(soft_codes.shape[-1]))) - entropy,
            "uniform": uniform_loss(self.vq.quantizer.codebook),
        }

    def soft_quantize_latent(self, x):
        x_enc = self.encode(x)
        x_recon_vq, _ = self.vq(x_enc)
        return x_enc, x_recon_vq

    def _reconstruct(self, x):
        x_enc = self.encode(x)
        x_recon_vq, _ = self.vq(x_enc)
        x_recon = self.decode(x_recon_vq)
        return x_recon, torch.norm(x_recon - x, 2, dim=-1) + distance_loss(x_recon, x)

    @staticmethod
    def _normalize_dist(tensor):
        return tensor / (tensor.mean() + 1e-8)

    def forward(self, anchor, pos, neg, pn_dist, ap_dist, an_dist):
        anc_emb, anc_loss = self._reconstruct(anchor)
        pos_emb, pos_loss = self._reconstruct(pos)
        neg_emb, neg_loss = self._reconstruct(neg)
        d_ap = self._normalize_dist(torch.norm(anc_emb - pos_emb, p=2, dim=1))
        d_an = self._normalize_dist(torch.norm(anc_emb - neg_emb, p=2, dim=1))
        d_pn = self._normalize_dist(torch.norm(pos_emb - neg_emb, p=2, dim=1))
        gt_ap = self._normalize_dist(ap_dist)
        gt_an = self._normalize_dist(an_dist)
        gt_pn = self._normalize_dist(pn_dist)

        threshold = gt_an - gt_ap
        rank_loss = torch.nn.functional.relu(d_ap - d_an + threshold)
        mse_loss = (d_ap - gt_ap) ** 2 + (d_an - gt_an) ** 2 + (d_pn - gt_pn) ** 2
        recon_loss = anc_loss + pos_loss + neg_loss
        return recon_loss + self.rank_weight * rank_loss + self.dist_weight * torch.sqrt(mse_loss + 1e-12)
