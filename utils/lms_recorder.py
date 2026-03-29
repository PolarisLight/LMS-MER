import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class LMSRecorder:
    """LMS recorder for per-sample semantic-gradient statistics."""

    def __init__(
        self,
        num_samples,
        num_classes,
        cls_num_list,
        save_dir=None,
        k=8,
        sigma=0.5,
        T=1.0,
        use_js=False,
        normalize_lms=True,
    ):
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.cls_num_list = np.asarray(cls_num_list, dtype=np.float64)

        self.k = int(k)
        self.sigma = float(sigma)
        self.T = float(T)
        self.use_js = bool(use_js)
        self.normalize_lms = bool(normalize_lms)

        self.sum_lms = np.zeros(num_samples, dtype=np.float64)
        self.cnt_lms = np.zeros(num_samples, dtype=np.int64)

        self.storage = {
            "features": None,
            "logits": None,
            "kl": None,
            "dist": None,
        }
        self.recorded_mask = np.zeros(num_samples, dtype=bool)

        self.labels = None
        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_ids = []
        self.epoch_t_c = []
        self.epoch_pear_f = []
        self.epoch_spear_f = []
        self.epoch_pear_inv = []
        self.epoch_spear_inv = []
        self.epoch_pear_log = []
        self.epoch_spear_log = []

    def set_labels(self, labels):
        self.labels = np.asarray(labels)

    def reset_epoch(self):
        self.sum_lms.fill(0.0)
        self.cnt_lms.fill(0)

    @torch.no_grad()
    def update(self, batch_idx, feat, logits):
        device = feat.device
        bsz = feat.size(0)
        if bsz <= 1:
            return

        feat_norm = F.normalize(feat, p=2, dim=1)
        _, dim = feat_norm.size()

        if self.storage["features"] is None:
            cls_cnt = logits.size(1)
            self.storage["features"] = np.zeros((self.num_samples, dim), dtype=np.float32)
            self.storage["logits"] = np.zeros((self.num_samples, cls_cnt), dtype=np.float32)
            self.storage["kl"] = np.zeros(self.num_samples, dtype=np.float32)
            self.storage["dist"] = np.zeros(self.num_samples, dtype=np.float32)

        dist2 = torch.cdist(feat_norm, feat_norm, p=2) ** 2
        eye = torch.eye(bsz, device=device, dtype=torch.bool)
        dist2 = dist2.masked_fill(eye, 1e9)

        k_eff = min(self.k, bsz - 1)
        knn_dist2, knn_idx = torch.topk(dist2, k_eff, dim=1, largest=False)
        knn_dist = torch.sqrt(knn_dist2 + 1e-12)

        w = torch.exp(-knn_dist2 / (self.sigma ** 2 + 1e-12))
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        avg_dist = (w * knn_dist).sum(dim=1)

        p = F.softmax(logits / self.T, dim=1)
        neigh_p = p[knn_idx]
        p_hat = (w.unsqueeze(-1) * neigh_p).sum(dim=1)

        p_safe = p.clamp_min(1e-12)
        p_hat_safe = p_hat.clamp_min(1e-12)

        if not self.use_js:
            kl_div = (p_safe * (p_safe.log() - p_hat_safe.log())).sum(dim=1)
        else:
            m_mid = 0.5 * (p_safe + p_hat_safe)
            kl1 = (p_safe * (p_safe.log() - m_mid.log())).sum(dim=1)
            kl2 = (p_hat_safe * (p_hat_safe.log() - m_mid.log())).sum(dim=1)
            kl_div = 0.5 * (kl1 + kl2)

        scaling = 10.0 / (math.sqrt(dim) + 1e-12)
        denom = avg_dist * scaling
        lms_gradient = kl_div / (denom + 1e-6)

        if self.normalize_lms:
            val = torch.log1p(lms_gradient)
            v_min, v_max = val.min(), val.max()
            lms_processed = (val - v_min) / (v_max - v_min + 1e-12)
        else:
            lms_processed = lms_gradient

        lms_np = lms_processed.detach().cpu().numpy()
        idx_np = batch_idx.detach().cpu().numpy()

        for bi, li in zip(idx_np, lms_np):
            self.sum_lms[bi] += float(li)
            self.cnt_lms[bi] += 1

        self.storage["features"][idx_np] = feat_norm.detach().cpu().numpy()
        self.storage["logits"][idx_np] = logits.detach().cpu().numpy()
        self.storage["kl"][idx_np] = kl_div.detach().cpu().numpy()
        self.storage["dist"][idx_np] = avg_dist.detach().cpu().numpy()
        self.recorded_mask[idx_np] = True

    @torch.no_grad()
    def compute_lms_epoch(self, epoch: int):
        assert self.labels is not None, "Call set_labels() before compute_lms_epoch()."

        sample_values = np.zeros(self.num_samples, dtype=np.float64)
        valid = self.cnt_lms > 0
        sample_values[valid] = self.sum_lms[valid] / self.cnt_lms[valid]

        t_c = np.zeros(self.num_classes, dtype=np.float64)
        for c in range(self.num_classes):
            mask = self.labels == c
            if mask.any():
                t_c[c] = sample_values[mask].mean()

        freq = self.cls_num_list.copy()
        inv_freq = 1.0 / (freq + 1e-12)
        neg_log = -np.log(freq + 1.0)

        def pearson(a, b):
            a = a - a.mean()
            b = b - b.mean()
            num = (a * b).sum()
            den = math.sqrt((a * a).sum()) * math.sqrt((b * b).sum()) + 1e-12
            return float(num / den)

        def spearman(a, b):
            ra = np.argsort(np.argsort(a))
            rb = np.argsort(np.argsort(b))
            return pearson(ra, rb)

        pear_f = pearson(t_c, freq)
        spear_f = spearman(t_c, freq)
        pear_inv = pearson(t_c, inv_freq)
        spear_inv = spearman(t_c, inv_freq)
        pear_log = pearson(t_c, neg_log)
        spear_log = spearman(t_c, neg_log)

        print(f"\n[Epoch {epoch}] LMS(Gradient)-Freq Correlation:")
        print(f"  Pearson( t_c , f_c )      = {pear_f:.4f}")
        print(f"  Spearman( t_c , f_c )     = {spear_f:.4f}")
        print(f"  Pearson( t_c , 1/f_c )    = {pear_inv:.4f}")
        print(f"  Spearman( t_c , 1/f_c )   = {spear_inv:.4f}")

        self.epoch_ids.append(epoch)
        self.epoch_t_c.append(t_c)
        self.epoch_pear_f.append(pear_f)
        self.epoch_spear_f.append(spear_f)
        self.epoch_pear_inv.append(pear_inv)
        self.epoch_spear_inv.append(spear_inv)
        self.epoch_pear_log.append(pear_log)
        self.epoch_spear_log.append(spear_log)

        return {"t_c": t_c, "pearson_inv": pear_inv, "spearman_inv": spear_inv}

    def finalize(self):
        if self.save_dir is None or not self.epoch_ids:
            return

        epoch_arr = np.array(self.epoch_ids, dtype=np.int32)
        pear_f = np.array(self.epoch_pear_f, dtype=np.float32)
        spear_f = np.array(self.epoch_spear_f, dtype=np.float32)
        pear_inv = np.array(self.epoch_pear_inv, dtype=np.float32)
        spear_inv = np.array(self.epoch_spear_inv, dtype=np.float32)
        pear_log = np.array(self.epoch_pear_log, dtype=np.float32)
        spear_log = np.array(self.epoch_spear_log, dtype=np.float32)
        t_c_mat = np.stack(self.epoch_t_c, axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epoch_arr, pear_f, label="Pearson(t_c, f_c)")
        ax.plot(epoch_arr, pear_inv, label="Pearson(t_c, 1/f_c)")
        ax.plot(epoch_arr, pear_log, label="Pearson(t_c, -log f_c)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson correlation")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        plt.tight_layout()
        pear_fig_path = self.save_dir / "lms_pearson_vs_epoch.png"
        plt.savefig(pear_fig_path, dpi=300)
        plt.close()
        print(f"[LMS] Pearson-vs-epoch curve saved to: {pear_fig_path}")

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(epoch_arr, spear_f, label="Spearman(t_c, f_c)")
        ax2.plot(epoch_arr, spear_inv, label="Spearman(t_c, 1/f_c)")
        ax2.plot(epoch_arr, spear_log, label="Spearman(t_c, -log f_c)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Spearman correlation")
        ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax2.legend()
        plt.tight_layout()
        spear_fig_path = self.save_dir / "lms_spearman_vs_epoch.png"
        plt.savefig(spear_fig_path, dpi=300)
        plt.close()
        print(f"[LMS] Spearman-vs-epoch curve saved to: {spear_fig_path}")

        t_c_last = t_c_mat[-1]
        x = np.arange(self.num_classes)

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(x, t_c_last, marker="o", linewidth=1.5)
        ax3.set_xlabel("Class index")
        ax3.set_ylabel("LMS (last epoch)")
        ax3.set_title("Per-class LMS")
        ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        per_class_path = self.save_dir / "lms_per_class.png"
        plt.savefig(per_class_path, dpi=300)
        plt.close()
        print(f"[LMS] Per-class LMS plot saved to: {per_class_path}")

        freq = self.cls_num_list
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(x, t_c_last, marker="o", linewidth=1.5, label="LMS per class")
        ax4.set_xlabel("Class index")
        ax4.set_ylabel("LMS (last epoch)")
        ax4.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        ax5 = ax4.twinx()
        ax5.plot(x, freq, marker="x", linestyle="--", label="Class frequency")
        ax5.set_ylabel("Class frequency")

        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax5.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout()
        per_class_dual_path = self.save_dir / "lms_per_class_with_freq.png"
        plt.savefig(per_class_dual_path, dpi=300)
        plt.close()
        print(f"[LMS] Per-class LMS + freq plot saved to: {per_class_dual_path}")

        if self.storage["features"] is not None:
            snapshot_path = self.save_dir / "lms_full_snapshot.npz"
            print(f"[LMS] Saving full diagnostic snapshot to {snapshot_path}...")
            mask = self.recorded_mask
            np.savez_compressed(
                snapshot_path,
                features=self.storage["features"][mask],
                logits=self.storage["logits"][mask],
                kl=self.storage["kl"][mask],
                dist=self.storage["dist"][mask],
                labels=self.labels[mask] if self.labels is not None else None,
                freq=self.cls_num_list,
            )
            print("[LMS] Snapshot saved.")


class LMSMemoryRecorder:
    """
    LMS Recorder (Queue-Augmented) -- Single NPZ Compatible
    ----------------------------------------------------------------------
    Goal:
      - Save ALL epoch-wise info into ONE npz (like the previous recorder)
    - Save class-wise LMS as `lms` [E,C], with `fz` [C], `log_fz` [C], `epochs` [E]
      - NEW: also save `S` [E,C] and `G` [E,C]
      - Optionally also save a full aligned snapshot for the LAST epoch:
          features_last [N,D], logits_last [N,C],
          kl_last [N]   (== S, numerator),
          dist_last [N] (== avg_dist, before /sqrt(D)),
          G_last [N]    (== denom, after /sqrt(D) + eps),
          recorded_mask_last [N], indices_last [N]
    """

    def __init__(self,
                 num_samples,
                 num_classes,
                 cls_num_list,
                 save_dir=None,
                 k=8,
                 sigma=0.5,
                 T=1.0,
                 use_js=False,
                 normalize_lms=True,
                 queue_size=8192):

        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.cls_num_list = np.asarray(cls_num_list, dtype=np.float64)

        self.k = int(k)
        self.sigma = float(sigma)
        self.T = float(T)
        self.use_js = bool(use_js)
        self.normalize_lms = bool(normalize_lms)
        self.queue_size = int(queue_size)

        # --- Memory Queue (Lazy Init) ---
        self.queue_feat = None
        self.queue_logits = None
        self.queue_ptr = 0
        self.queue_full = False

        # --- epoch accumulators (per-sample index) ---
        self.sum_lms = np.zeros(self.num_samples, dtype=np.float64)
        self.cnt_lms = np.zeros(self.num_samples, dtype=np.int64)

        # Epoch accumulators for numerator S and denominator G.
        self.sum_S = np.zeros(self.num_samples, dtype=np.float64)   # semantic divergence numerator
        self.sum_G = np.zeros(self.num_samples, dtype=np.float64)   # geometric support denom

        # --- Full aligned snapshot buffers (LAST epoch) ---
        self.storage = {
            "features": None,  # [N, D] float32
            "logits": None,    # [N, C] float32
            "kl": None,        # [N] float32  (kept for backward compatibility; equals S)
            "dist": None,      # [N] float32  (kept; equals avg_dist BEFORE /sqrt(D))
            "G": None,         # [N] float32  (NEW; equals denom AFTER /sqrt(D) + eps)
        }
        self.recorded_mask = np.zeros(self.num_samples, dtype=bool)

        self.labels = None

        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- epoch history ---
        self.epoch_ids = []
        self.epoch_t_c = []   # LMS per class, [C]
        self.epoch_S_c = []
        self.epoch_G_c = []

        self.epoch_pear_f = []
        self.epoch_spear_f = []
        self.epoch_pear_inv = []
        self.epoch_spear_inv = []
        self.epoch_pear_log = []
        self.epoch_spear_log = []

        # --- accuracy history ---
        self.epoch_acc_overall = []
        self.epoch_acc_head = []
        self.epoch_acc_mid = []
        self.epoch_acc_tail = []

    def set_labels(self, labels):
        self.labels = np.asarray(labels, dtype=np.int64)

    def reset_epoch(self):
        # reset accumulators for this epoch
        self.sum_lms.fill(0.0)
        self.cnt_lms.fill(0)

        # NEW
        self.sum_S.fill(0.0)
        self.sum_G.fill(0.0)

        # reset epoch snapshot mask
        self.recorded_mask.fill(False)
        if self.storage["kl"] is not None:
            self.storage["kl"].fill(np.nan)
            self.storage["dist"].fill(np.nan)
            self.storage["G"].fill(np.nan)

    def record_acc(self, acc_overall, acc_head, acc_mid, acc_tail):
        self.epoch_acc_overall.append(float(acc_overall))
        self.epoch_acc_head.append(float(acc_head))
        self.epoch_acc_mid.append(float(acc_mid))
        self.epoch_acc_tail.append(float(acc_tail))

    @torch.no_grad()
    def _update_queue(self, batch_feat_norm, batch_logits):
        batch_size = batch_feat_norm.shape[0]
        dim = batch_feat_norm.shape[1]
        num_classes = batch_logits.shape[1]

        if self.queue_feat is None:
            device = batch_feat_norm.device
            self.queue_feat = torch.zeros(self.queue_size, dim, device=device)
            self.queue_logits = torch.zeros(self.queue_size, num_classes, device=device)
            self.queue_ptr = 0
            self.queue_full = False

        assert self.queue_size >= batch_size, "Queue size must be larger than batch size"

        ptr = self.queue_ptr
        space_left = self.queue_size - ptr
        if space_left >= batch_size:
            self.queue_feat[ptr: ptr + batch_size] = batch_feat_norm
            self.queue_logits[ptr: ptr + batch_size] = batch_logits
            self.queue_ptr = (ptr + batch_size) % self.queue_size
        else:
            part1 = space_left
            part2 = batch_size - part1
            self.queue_feat[ptr:] = batch_feat_norm[:part1]
            self.queue_logits[ptr:] = batch_logits[:part1]
            self.queue_feat[:part2] = batch_feat_norm[part1:]
            self.queue_logits[:part2] = batch_logits[part1:]
            self.queue_ptr = part2

        if self.queue_ptr < ptr:
            self.queue_full = True

    @torch.no_grad()
    def update(self, batch_idx, feat, logits):
        """
        Update with Queue-Augmented LMS.
        batch_idx: dataset indices for this batch (LongTensor [B])
        feat:  [B,D]
        logits:[B,C]
        """
        device = feat.device
        B = feat.size(0)
        if B <= 1:
            return

        # 1) normalize feature
        feat_norm = F.normalize(feat, p=2, dim=1)
        _, D = feat_norm.size()
        C = logits.size(1)

        # lazy init storage buffers
        if self.storage["features"] is None:
            self.storage["features"] = np.zeros((self.num_samples, D), dtype=np.float32)
            self.storage["logits"] = np.zeros((self.num_samples, C), dtype=np.float32)
            self.storage["kl"] = np.full((self.num_samples,), np.nan, dtype=np.float32)
            self.storage["dist"] = np.full((self.num_samples,), np.nan, dtype=np.float32)
            self.storage["G"] = np.full((self.num_samples,), np.nan, dtype=np.float32)

        # 2) Context = current batch + memory queue.
        if self.queue_feat is not None:
            context_feat = torch.cat([feat_norm, self.queue_feat], dim=0)
            context_logits = torch.cat([logits, self.queue_logits], dim=0)
        else:
            context_feat = feat_norm
            context_logits = logits

        # 3) Cosine similarity mapped to squared distance on the hypersphere.
        sim_matrix = torch.matmul(feat_norm, context_feat.T)          # [B, N_ctx]
        dist2 = (2.0 - 2.0 * sim_matrix).clamp_min(0.0)              # [B, N_ctx]

        # 4) mask self in first B positions
        N_ctx = dist2.size(1)
        mask_self = torch.eye(B, N_ctx, device=device, dtype=torch.bool)
        dist2 = dist2.masked_fill(mask_self, 1e9)

        # 5) knn
        k = min(self.k, N_ctx - 1)
        knn_dist2, knn_idx = torch.topk(dist2, k, dim=1, largest=False)
        knn_dist = torch.sqrt(knn_dist2 + 1e-12)

        # 6) Neighbor weights and geometric denominator G.
        w = torch.exp(-knn_dist2 / (self.sigma ** 2 + 1e-12))
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

        avg_dist = (w * knn_dist).sum(dim=1)                       # [B] (raw)
        scaling = 1.0 / (math.sqrt(D) + 1e-12)
        denom = avg_dist * scaling + 1e-9                          # [B] (this is G)

        # 7) Numerator S as KL/JS semantic divergence.
        p_query = F.softmax(logits / self.T, dim=1)
        neigh_logits = context_logits[knn_idx]                     # [B,K,C]
        neigh_p = F.softmax(neigh_logits / self.T, dim=2)
        p_hat = (w.unsqueeze(-1) * neigh_p).sum(dim=1)

        p_safe = p_query.clamp_min(1e-12)
        p_hat_safe = p_hat.clamp_min(1e-12)

        if not self.use_js:
            numerator = (p_safe * (p_safe.log() - p_hat_safe.log())).sum(dim=1)  # [B] (this is S)
        else:
            m_mid = 0.5 * (p_safe + p_hat_safe)
            kl1 = (p_safe * (p_safe.log() - m_mid.log())).sum(dim=1)
            kl2 = (p_hat_safe * (p_hat_safe.log() - m_mid.log())).sum(dim=1)
            numerator = 0.5 * (kl1 + kl2)

        lms = numerator / denom

        # 8) Normalize LMS for stable visualization.
        if self.normalize_lms:
            val = torch.log1p(lms)
            m_min, m_max = val.min(), val.max()
            m_proc = (val - m_min) / (m_max - m_min + 1e-12)
        else:
            m_proc = lms

        # 9) update per-sample accumulators
        idx_np = batch_idx.detach().cpu().numpy().astype(np.int64)
        lms_np = m_proc.detach().cpu().numpy().astype(np.float64)

        S_np = numerator.detach().cpu().numpy().astype(np.float64)
        G_np = denom.detach().cpu().numpy().astype(np.float64)

        for bi, li, si, gi in zip(idx_np, lms_np, S_np, G_np):
            self.sum_lms[bi] += float(li)
            self.sum_S[bi] += float(si)
            self.sum_G[bi] += float(gi)
            self.cnt_lms[bi] += 1

        # 10) store LAST epoch aligned snapshot buffers
        feat_np = feat_norm.detach().cpu().numpy().astype(np.float32)
        logits_np = logits.detach().cpu().numpy().astype(np.float32)

        self.storage["features"][idx_np] = feat_np
        self.storage["logits"][idx_np] = logits_np
        self.storage["kl"][idx_np] = numerator.detach().cpu().numpy().astype(np.float32)   # S (compat)
        self.storage["dist"][idx_np] = avg_dist.detach().cpu().numpy().astype(np.float32) # raw avg_dist (compat)
        self.storage["G"][idx_np] = denom.detach().cpu().numpy().astype(np.float32)        # NEW: G

        self.recorded_mask[idx_np] = True

        # 11) update queue
        self._update_queue(feat_norm.detach(), logits.detach())

    @torch.no_grad()
    def compute_lms_epoch(self, epoch: int):
        """
                Compute per-class means:
                    - t_c : mean LMS
          - S_c : mean numerator
          - G_c : mean denom
        and correlations w.r.t. frequency.
        """
        assert self.labels is not None, "Call set_labels() before compute_lms_epoch()."

        valid = self.cnt_lms > 0

        # per-sample averages for this epoch
        v_lms = np.zeros(self.num_samples, dtype=np.float64)
        v_S = np.zeros(self.num_samples, dtype=np.float64)
        v_G = np.zeros(self.num_samples, dtype=np.float64)

        v_lms[valid] = self.sum_lms[valid] / (self.cnt_lms[valid] + 1e-12)
        v_S[valid] = self.sum_S[valid] / (self.cnt_lms[valid] + 1e-12)
        v_G[valid] = self.sum_G[valid] / (self.cnt_lms[valid] + 1e-12)

        # per-class mean
        t_c = np.zeros(self.num_classes, dtype=np.float64)
        S_c = np.zeros(self.num_classes, dtype=np.float64)
        G_c = np.zeros(self.num_classes, dtype=np.float64)

        for c in range(self.num_classes):
            m = (self.labels == c)
            if m.any():
                t_c[c] = float(v_lms[m].mean())
                S_c[c] = float(v_S[m].mean())
                G_c[c] = float(v_G[m].mean())

        freq = self.cls_num_list.copy().astype(np.float64)
        inv_freq = 1.0 / (freq + 1e-12)
        neg_log = -np.log(freq + 1.0)

        def pearson(a, b):
            a = a - a.mean()
            b = b - b.mean()
            num = float((a * b).sum())
            den = float(math.sqrt((a * a).sum()) * math.sqrt((b * b).sum()) + 1e-12)
            return num / den

        def spearman(a, b):
            ra = np.argsort(np.argsort(a))
            rb = np.argsort(np.argsort(b))
            return pearson(ra.astype(np.float64), rb.astype(np.float64))

        pear_f = pearson(t_c, freq)
        spear_f = spearman(t_c, freq)
        pear_inv = pearson(t_c, inv_freq)
        spear_inv = spearman(t_c, inv_freq)
        pear_log = pearson(t_c, neg_log)
        spear_log = spearman(t_c, neg_log)

        print(f"\n[Epoch {epoch}] LMS(Queue-Augmented)-Freq Correlation:")
        print(f"  Pearson( t_c , f_c )      = {pear_f:.4f}")
        print(f"  Spearman( t_c , f_c )     = {spear_f:.4f}")
        print(f"  Pearson( t_c , 1/f_c )    = {pear_inv:.4f}")
        print(f"  Spearman( t_c , 1/f_c )   = {spear_inv:.4f}")
        print(f"  Pearson( t_c , -log f_c ) = {pear_log:.4f}")
        print(f"  Spearman( t_c , -log f_c )= {spear_log:.4f}")

        # Record histories.
        self.epoch_ids.append(int(epoch))
        self.epoch_t_c.append(t_c.astype(np.float64))
        self.epoch_S_c.append(S_c.astype(np.float64))
        self.epoch_G_c.append(G_c.astype(np.float64))

        self.epoch_pear_f.append(float(pear_f))
        self.epoch_spear_f.append(float(spear_f))
        self.epoch_pear_inv.append(float(pear_inv))
        self.epoch_spear_inv.append(float(spear_inv))
        self.epoch_pear_log.append(float(pear_log))
        self.epoch_spear_log.append(float(spear_log))

        return dict(t_c=t_c, S_c=S_c, G_c=G_c, pearson_inv=pear_inv, spearman_inv=spear_inv)

    def finalize(self):
        """
        Save ONE npz:
          - keys: epochs, lms, fz, log_fz
          - extra curves: pearson/spearman...
          - NEW: S [E,C], G [E,C]
          - optional last snapshot: features_last/logits_last/kl_last/dist_last/G_last/labels/mask/indices
        """
        if self.save_dir is None:
            return
        if len(self.epoch_ids) == 0:
            print("[LMS] finalize(): no epoch records collected, skip saving.")
            return

        epoch_arr = np.asarray(self.epoch_ids, dtype=np.int32)                 # [E]
        t_c_mat = np.stack(self.epoch_t_c, axis=0).astype(np.float32)          # [E,C]

        # NEW
        S_mat = np.stack(self.epoch_S_c, axis=0).astype(np.float32)            # [E,C]
        G_mat = np.stack(self.epoch_G_c, axis=0).astype(np.float32)            # [E,C]

        pear_f = np.asarray(self.epoch_pear_f, dtype=np.float32)
        spear_f = np.asarray(self.epoch_spear_f, dtype=np.float32)
        pear_inv = np.asarray(self.epoch_pear_inv, dtype=np.float32)
        spear_inv = np.asarray(self.epoch_spear_inv, dtype=np.float32)
        pear_log = np.asarray(self.epoch_pear_log, dtype=np.float32)
        spear_log = np.asarray(self.epoch_spear_log, dtype=np.float32)

        acc_ov = np.asarray(self.epoch_acc_overall, dtype=np.float32)
        acc_hd = np.asarray(self.epoch_acc_head, dtype=np.float32)
        acc_md = np.asarray(self.epoch_acc_mid, dtype=np.float32)
        acc_tl = np.asarray(self.epoch_acc_tail, dtype=np.float32)

        fz = self.cls_num_list.astype(np.float32)
        log_fz = np.log(np.clip(fz, 1e-12, None)).astype(np.float32)

        snapshot_path = self.save_dir / "lms_full_snapshot.npz"
        print(f"[LMS] Saving ONE npz to: {snapshot_path}")

        # If never updated, still save the epoch-level curves.
        if self.storage["features"] is None:
            np.savez_compressed(
                snapshot_path,
                epochs=epoch_arr,
                lms=t_c_mat,
                S=S_mat,
                G=G_mat,
                fz=fz,
                log_fz=log_fz,
                pearson_f=pear_f,
                spearman_f=spear_f,
                pearson_inv=pear_inv,
                spearman_inv=spear_inv,
                pearson_neglog=pear_log,
                spearman_neglog=spear_log,
                acc_overall=acc_ov,
                acc_head=acc_hd,
                acc_mid=acc_md,
                acc_tail=acc_tl,
            )
            print("[LMS] Saved (no per-sample snapshot).")
            return

        mask_last = self.recorded_mask.astype(np.bool_)
        indices_last = np.arange(self.num_samples, dtype=np.int64)

        np.savez_compressed(
            snapshot_path,
            # --- compatible ---
            epochs=epoch_arr,
            lms=t_c_mat,
            fz=fz,
            log_fz=log_fz,

            # --- NEW per-epoch components ---
            S=S_mat,
            G=G_mat,

            # --- correlations ---
            pearson_f=pear_f,
            spearman_f=spear_f,
            pearson_inv=pear_inv,
            spearman_inv=spear_inv,
            pearson_neglog=pear_log,
            spearman_neglog=spear_log,

            # --- accuracy ---
            acc_overall=acc_ov,
            acc_head=acc_hd,
            acc_mid=acc_md,
            acc_tail=acc_tl,

            # --- last-epoch snapshot (optional but useful) ---
            features_last=self.storage["features"],
            logits_last=self.storage["logits"],
            kl_last=self.storage["kl"],      # == S (numerator), kept name for compatibility
            dist_last=self.storage["dist"],  # == avg_dist (raw)
            G_last=self.storage["G"],        # == denom (scaled + eps)
            labels=self.labels if self.labels is not None else None,
            recorded_mask_last=mask_last,
            indices_last=indices_last,
        )
        print("[LMS] Saved.")

    def memory_bank_report(self, verbose=True):
        """
        Report memory usage of the memory-bank tensors (GPU queue only).
        Returns:
            dict:
                - queue_feat_mb
                - queue_logits_mb
                - total_bank_mb
        """
        def tensor_mb(x):
            if x is None:
                return 0.0
            return x.numel() * x.element_size() / 1024 / 1024

        queue_feat_mb = tensor_mb(self.queue_feat)
        queue_logits_mb = tensor_mb(self.queue_logits)
        total_bank_mb = queue_feat_mb + queue_logits_mb

        report = {
            "queue_feat_mb": queue_feat_mb,
            "queue_logits_mb": queue_logits_mb,
            "total_bank_mb": total_bank_mb,
        }

        if verbose:
            print("\n===== Memory Bank Report =====")
            print(f"queue_feat:   {queue_feat_mb:.3f} MB")
            print(f"queue_logits: {queue_logits_mb:.3f} MB")
            print(f"total_bank:   {total_bank_mb:.3f} MB")

        return report

    # Inference helper for test-time LMS estimation.
    @torch.no_grad()
    def calculate_test_lms(self, test_feats, test_probs, k=None, sigma=None):
        k = k if k is not None else self.k
        sigma = sigma if sigma is not None else self.sigma

        if self.queue_feat is None:
            return torch.zeros(test_feats.size(0), device=test_feats.device)

        if self.queue_full:
            bank_feats = self.queue_feat
            bank_logits = self.queue_logits
        else:
            if self.queue_ptr == 0:
                return torch.zeros(test_feats.size(0), device=test_feats.device)
            bank_feats = self.queue_feat[:self.queue_ptr]
            bank_logits = self.queue_logits[:self.queue_ptr]

        dists = torch.cdist(test_feats, bank_feats) ** 2
        curr_k = min(k, bank_feats.size(0))
        knn_dist2, knn_idx = torch.topk(dists, k=curr_k, dim=1, largest=False)

        w = torch.exp(-knn_dist2 / (sigma ** 2 + 1e-12))
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

        neigh_logits = bank_logits[knn_idx]
        neigh_p = F.softmax(neigh_logits / self.T, dim=2)
        p_hat = (w.unsqueeze(-1) * neigh_p).sum(dim=1)

        p_safe = test_probs.clamp_min(1e-12)
        p_hat_safe = p_hat.clamp_min(1e-12)

        if not self.use_js:
            numerator = (p_safe * (p_safe.log() - p_hat_safe.log())).sum(dim=1)
        else:
            m_mid = 0.5 * (p_safe + p_hat_safe)
            kl1 = (p_safe * (p_safe.log() - m_mid.log())).sum(dim=1)
            kl2 = (p_hat_safe * (p_hat_safe.log() - m_mid.log())).sum(dim=1)
            numerator = 0.5 * (kl1 + kl2)

        avg_dist = (w * torch.sqrt(knn_dist2 + 1e-12)).sum(dim=1)
        scaling = 1.0 / (math.sqrt(test_feats.shape[1]) + 1e-12)
        denom = avg_dist * scaling + 1e-9

        return numerator / denom
