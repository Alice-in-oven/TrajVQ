import torch
import torch.nn.functional as F


class TripletLoss:
    def __init__(self, epochs, target_scale=None):
        step = epochs // 5
        self.schedule = {step * 0: (0, 10), step * 1: (10, 10), step * 2: (10, 1), step * 3: (5, 0.1), step * 4: (1, 0.01)}
        self.rank_weight, self.dist_weight = 1, 1
        self.target_scale = target_scale

    def __call__(self, epoch, d_ap, gt_ap, d_an, gt_an, d_pn, gt_pn):
        if epoch in self.schedule:
            self.rank_weight, self.dist_weight = self.schedule[epoch]
        if self.target_scale is not None:
            gt_ap, gt_an, gt_pn = gt_ap / self.target_scale, gt_an / self.target_scale, gt_pn / self.target_scale
        threshold = gt_an - gt_ap
        rank_loss = F.relu(d_ap - d_an + threshold)
        mse_loss = (d_ap - gt_ap) ** 2 + (d_an - gt_an) ** 2 + (d_pn - gt_pn) ** 2
        return torch.mean(self.rank_weight * rank_loss + self.dist_weight * torch.sqrt(mse_loss))
