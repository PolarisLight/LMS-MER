import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mer_weighted(
    feat_norm: torch.Tensor,
    logits: torch.Tensor,
    k: int = 8,
    sigma: float = 0.5,
    T: float = 1.0,
    use_js: bool = False,
    detach_p_hat: bool = True,
    detach_g: bool = True,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Support-weighted MER.

    This term measures local semantic inconsistency and scales it by
    geometric support to emphasize structurally fragile samples.
    """
    device = feat_norm.device
    bsz, dim = feat_norm.shape

    if bsz <= 1:
        if reduction == "none":
            return torch.zeros(bsz, device=device)
        return torch.zeros((), device=device)

    # Build local neighborhoods in feature space.
    feat_for_graph = feat_norm.detach() if detach_g else feat_norm
    dist2 = torch.cdist(feat_for_graph, feat_for_graph, p=2) ** 2
    eye = torch.eye(bsz, device=device, dtype=torch.bool)
    dist2 = dist2.masked_fill(eye, float("inf"))

    k_eff = min(k, bsz - 1)
    knn_dist, knn_idx = torch.topk(dist2, k_eff, dim=1, largest=False)

    w = torch.exp(-knn_dist / (sigma ** 2 + eps))
    w = w / (w.sum(dim=1, keepdim=True) + eps)

    # Geometric support g_i: smaller values imply local collapse.
    g = (w * knn_dist.sqrt()).sum(dim=1) / math.sqrt(dim)
    alpha = 1.0 / (g + eps)
    alpha = alpha / (alpha.mean() + eps)
    alpha = alpha.clamp(max=5.0)

    # Semantic divergence between each sample and its neighborhood prediction.
    p = F.softmax(logits / T, dim=1)
    neigh_p = p[knn_idx]
    p_hat = (w.unsqueeze(-1) * neigh_p).sum(dim=1)

    if detach_p_hat:
        p_hat = p_hat.detach()

    p_safe = p.clamp_min(eps)
    p_hat_safe = p_hat.clamp_min(eps)

    if not use_js:
        m = (p_safe * (p_safe.log() - p_hat_safe.log())).sum(dim=1)
    else:
        m_mid = 0.5 * (p_safe + p_hat_safe)
        kl1 = (p_safe * (p_safe.log() - m_mid.log())).sum(dim=1)
        kl2 = (p_hat_safe * (p_hat_safe.log() - m_mid.log())).sum(dim=1)
        m = 0.5 * (kl1 + kl2)

    # Final MER value: support-weighted local semantic mismatch.
    loss = alpha * m

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


class MERLoss(nn.Module):
    def __init__(
        self,
        k: int = 8,
        sigma: float = 0.5,
        T: float = 1.0,
        use_js: bool = False,
        detach_p_hat: bool = True,
        detach_g: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.T = T
        self.use_js = use_js
        self.detach_p_hat = detach_p_hat
        self.detach_g = detach_g
        self.reduction = reduction

    def forward(self, feat_norm: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return compute_mer_weighted(
            feat_norm,
            logits,
            k=self.k,
            sigma=self.sigma,
            T=self.T,
            use_js=self.use_js,
            detach_p_hat=self.detach_p_hat,
            detach_g=self.detach_g,
            reduction=self.reduction,
        )


def compute_feature_expansion_loss(
    feat_norm: torch.Tensor,
    k: int = 8,
    sigma: float = 0.5,
    margin: float = 0.12,
    detach_g_weight: bool = True,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Feature-space expansion regularization for LA compatibility."""
    device = feat_norm.device
    bsz, dim = feat_norm.shape

    if bsz <= 1:
        if reduction == "none":
            return torch.zeros(bsz, device=device)
        return torch.zeros((), device=device)

    dist2 = torch.cdist(feat_norm, feat_norm, p=2) ** 2
    eye = torch.eye(bsz, device=device, dtype=torch.bool)
    dist2 = dist2.masked_fill(eye, float("inf"))

    k_eff = min(k, bsz - 1)
    knn_dist2, _ = torch.topk(dist2, k_eff, dim=1, largest=False)

    with torch.no_grad():
        w = torch.exp(-knn_dist2.detach() / (sigma ** 2 + eps))
        w = w / (w.sum(dim=1, keepdim=True) + eps)

    knn_dist = knn_dist2.sqrt()
    # Penalize overly compact neighborhoods to avoid representation collapse.
    g = (w * knn_dist).sum(dim=1) / math.sqrt(dim)

    g_for_weight = g.detach() if detach_g_weight else g
    alpha = 1.0 / (g_for_weight + eps)
    alpha = alpha / (alpha.mean() + eps)
    alpha = alpha.clamp(max=5.0)

    loss_i = alpha * torch.relu(margin - g)

    if reduction == "none":
        return loss_i
    if reduction == "sum":
        return loss_i.sum()
    return loss_i.mean()
