import os
import math
import random
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import nni
from loss.compared_loss import FocalLoss, LDAMLoss
from loss.merloss import MERLoss, compute_feature_expansion_loss
from utils.utils import set_seed


# ===============================================================
# ResNet-32
# ===============================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.short = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.short = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet32(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 5, 1)
        self.layer2 = self._make_layer(32, 5, 2)
        self.layer3 = self._make_layer(64, 5, 2)

        self.linear = nn.Linear(64, num_classes)

        # A separate classifier head for CRT fine-tuning stage.
        self.linear_rt = nn.Linear(64, num_classes)


    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers=[]
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False,rt=False):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.avg_pool2d(out, 8)
        feat = out.view(out.size(0), -1)     # [B, 64]
        feat_norm = F.normalize(feat, dim=1)
        if rt:
            logits = self.linear_rt(feat)
        else:
            logits = self.linear(feat)
        if return_feat:
            return logits, feat_norm
        return logits


# ===============================================================
# Long-Tailed CIFAR
# ===============================================================

def build_long_tailed_indices(targets, num_classes=100, imb_factor=100, oversample_tail=1):
    # Generate class-imbalanced indices with optional tail oversampling.
    targets = torch.tensor(targets, dtype=torch.long)
    cls_num = []
    max_num = (targets == 0).sum().item()

    for c in range(num_classes):
        n = int(max_num * (imb_factor ** (-(c / (num_classes - 1)))))
        n = max(n, 1)
        cls_num.append(n)

    indices = []
    tail_start = int(num_classes * 2 / 3)

    for c in range(num_classes):
        cls_idx = (targets == c).nonzero(as_tuple=True)[0].tolist()
        random.shuffle(cls_idx)

        repeat = oversample_tail if c >= tail_start else 1
        indices.extend(cls_idx[:cls_num[c]] * repeat)

    return indices, cls_num


class LTTrainDataset(Dataset):
    def __init__(self, base_dataset, lt_indices):
        super().__init__()
        self.base = base_dataset
        self.indices = lt_indices
        self.labels = [self.base.targets[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img, label = self.base[base_idx]
        return img, label, idx


# ===============================================================
# Evaluate
# ===============================================================


@torch.no_grad()
def evaluate(model,
             loader,
             device,
             num_classes=100,
             method="ce",
             log_prior=None,
             la_tau=1.0,
             cls_head=None,
             cls_mid=None,
             cls_tail=None,
             rt=False,
             clip_text_emb=None,
             conf_tau=0.7,
             ret_temp=10.0):
    model.eval()
    total, correct = 0, 0
    per_cls_correct = torch.zeros(num_classes, device=device)
    per_cls_total   = torch.zeros(num_classes, device=device)

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if method == "crt":
            logits = model(x, rt=rt)  # [B,C]
        else:
            logits = model(x)  # [B,C]

        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total   += y.numel()

        for cidx in range(num_classes):
            m = (y == cidx)
            n = m.sum()
            if n.item() > 0:
                per_cls_correct[cidx] += (pred[m] == y[m]).sum()
                per_cls_total[cidx]   += n

    acc = correct / total
    per_cls_acc = (per_cls_correct / (per_cls_total + 1e-12))

    if cls_head is None or cls_mid is None or cls_tail is None:
        head = mid = tail = float(acc)
    else:
        head = float(per_cls_acc[cls_head].mean().item())
        mid  = float(per_cls_acc[cls_mid].mean().item())
        tail = float(per_cls_acc[cls_tail].mean().item())

    model.train()
    return float(acc), head, mid, tail


# ===============================================================
# SupCon Loss (for SCL)
# ===============================================================

class SupConLoss(nn.Module):
    """
    Simplified supervised contrastive loss.
    - features: [B, D] (L2-normalized features are recommended)
    - labels: [B]
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B = features.size(0)
        if B <= 1:
            return features.new_zeros(())

        # Pairwise similarity matrix [B, B].
        sim = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        # Improve numerical stability.
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)  # [B,1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [B,B]

        # Exclude self-comparisons from positive pairs.
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        # log-softmax
        exp_sim = torch.exp(sim) * logits_mask  # [B,B]
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # Average log-probabilities over positive pairs per sample.
        pos_sum = (mask * log_prob).sum(dim=1)
        pos_count = mask.sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = pos_sum / pos_count

        # Loss is the negative mean positive log-probability.
        loss = -mean_log_prob_pos.mean()
        return loss





# ===============================================================
# Mixup helper
# ===============================================================

def mixup_data(x, y, alpha=0.2):
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def update_best(best_d, val, epoch, name="Overall"):
    if val > best_d["value"]:
        best_d["value"] = val
        best_d["epoch"] = epoch
        print(f"==> New best {name}: {val*100:.2f}% @ epoch {epoch}")
    return best_d
def split_head_mid_tail(cls_num_list):
    """
    Sort classes by frequency and split into head/mid/tail thirds.
    """
    cls_num = torch.tensor(cls_num_list, dtype=torch.float32)
    num_classes = cls_num.numel()
    sorted_idx = torch.argsort(cls_num, descending=True)
    n_head = num_classes // 3
    n_mid  = num_classes // 3
    head = sorted_idx[:n_head]
    mid  = sorted_idx[n_head:n_head + n_mid]
    tail = sorted_idx[n_head + n_mid:]
    return head, mid, tail

# ===============================================================
# Training
# ===============================================================

def run(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 100 if args.dataset=="cifar100" else 10

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    method_dir = save_root / (args.method + f"_mer") if args.mer else save_root / args.method
    method_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)

    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    tf_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    root = args.data_root
    if args.dataset=="cifar100":
        full_train = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=tf_train)
        test_set   = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=tf_test)
    else:
        full_train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tf_train)
        test_set   = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf_test)

    lt_indices, cls_num_list = build_long_tailed_indices(
        full_train.targets,
        num_classes=num_classes,
        imb_factor=args.imb_factor,
        oversample_tail=args.oversample_tail
    )
    train_set = LTTrainDataset(full_train, lt_indices)

    # Precompute class prior and reweighting terms once per run.
    cls_num_arr = np.asarray(cls_num_list, dtype=np.float64)
    prior = cls_num_arr / cls_num_arr.sum()
    log_prior = torch.from_numpy(np.log(prior + 1e-12)).float().to(device)

    # RW/DRW class weights: inverse-frequency with mean normalization.
    rw_weight_np = (cls_num_arr + 1e-12) ** (-args.rw_beta)
    rw_weight_np = rw_weight_np / rw_weight_np.mean()
    class_weight_rw = torch.from_numpy(rw_weight_np).float().to(device)

    # Build dataloader; resample mode uses a weighted sampler.
    if args.method == "resample":
        labels_tensor = torch.tensor(train_set.labels, dtype=torch.long)
        class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()
        inv_counts = 1.0 / (class_counts + 1e-12)
        sample_weights = inv_counts[labels_tensor]  # [N]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        )

    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    from models.MyResnet import MyNet
    if args.backbone == "resnet32": 
        model = ResNet32(num_classes=num_classes).to(device)
    else:
        model = MyNet(backbone=args.backbone,num_classes=num_classes).to(device)
    
    if args.method == "crt":
            
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.rt_start)
        optimizer_rt = optim.SGD(
        model.linear_rt.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
        scheduler_rt = optim.lr_scheduler.CosineAnnealingLR(optimizer_rt, T_max=args.epochs-args.rt_start)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

   

    # Instantiate losses and regularizers based on the selected training method.
    criterion = None
    scl_criterion = None

    if args.method == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma)
    elif args.method == "ldam":
        criterion = LDAMLoss(cls_num_list=cls_num_list,
                             max_m=args.ldam_max_m,
                             s=args.ldam_s)
    elif args.method == "scl":
        scl_criterion = SupConLoss(temperature=args.scl_temp)
    
    mer = MERLoss(k=args.mer_k).to(device)

    if criterion is not None:
        criterion = criterion.to(device)
    if scl_criterion is not None:
        scl_criterion = scl_criterion.to(device)
    from utils.lms_recorder import LMSMemoryRecorder
    # LMSMemoryRecorder tracks per-sample LMS statistics and queue-based neighbors.
    recorder = LMSMemoryRecorder(
        num_samples=len(train_set),
        num_classes=num_classes,
        cls_num_list=cls_num_list,
        save_dir=str(method_dir / f"{args.dataset}-{int(args.imb_factor)}_lms"),
        k=args.mer_k,
        sigma=0.5,
        T=1.0,
        use_js=False,
        normalize_lms=False,
        queue_size=128
    )
    recorder.set_labels(train_set.labels)

    best_overall = {"value": 0.0, "epoch": 0}
    best_head    = {"value": 0.0, "epoch": 0}
    best_mid     = {"value": 0.0, "epoch": 0}
    best_tail    = {"value": 0.0, "epoch": 0}

    # head/mid/tail
    cls_head, cls_mid, cls_tail = split_head_mid_tail(cls_num_list)
    for epoch in range(1, args.epochs+1):
        model.train()
        recorder.reset_epoch()

        loop = tqdm(train_loader, desc=f"[{args.method}] Epoch {epoch}/{args.epochs}")

        for x, y, idx in loop:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Compute method-specific objective.
            if args.method == "ce" or args.method == "resample":
                logits, feat_norm = model(x, return_feat=True)
                loss = F.cross_entropy(logits, y)
                mer_logits = logits

            elif args.method == "mixup":
                mixed_x, y_a, y_b, lam, index_perm = mixup_data(x, y, alpha=args.mixup_alpha)
                logits, feat_norm = model(mixed_x, return_feat=True)
                loss = lam * F.cross_entropy(logits, y_a) + \
                       (1.0 - lam) * F.cross_entropy(logits, y_b)
                mer_logits = logits

            elif args.method == "la":
                logits_raw, feat_norm = model(x, return_feat=True)
                logits_adj = logits_raw + args.la_tau * log_prior.view(1, -1)
                loss = F.cross_entropy(logits_adj, y)
                loss_mer = compute_feature_expansion_loss(feat_norm=feat_norm,
                                                          k=10,
                                                          margin=args.mer_margin,
                                                          sigma=args.mer_sigma if hasattr(args, "mer_sigma") else 0.5,
                                                          )
                mer_logits = logits_raw

            elif args.method == "focal":
                logits, feat_norm = model(x, return_feat=True)
                loss = criterion(logits, y)
                mer_logits = logits

            elif args.method == "ldam":
                logits, feat_norm = model(x, return_feat=True)
                loss = criterion(logits, y)
                mer_logits = logits

            elif args.method == "rw":
                logits, feat_norm = model(x, return_feat=True)
                loss = F.cross_entropy(logits, y, weight=class_weight_rw)
                mer_logits = logits

            elif args.method == "drw":
                logits, feat_norm = model(x, return_feat=True)
                if epoch >= args.drw_start_epoch:
                    loss = F.cross_entropy(logits, y, weight=class_weight_rw)
                else:
                    loss = F.cross_entropy(logits, y)
                mer_logits = logits

            elif args.method == "scl":
                logits, feat_norm = model(x, return_feat=True)
                ce_loss = F.cross_entropy(logits, y)
                scl_loss = scl_criterion(feat_norm, y)
                loss = ce_loss + args.scl_lambda * scl_loss
                mer_logits = logits
            elif args.method == "mer":
                logits, feat_norm = model(x, return_feat=True)
                ce_loss = F.cross_entropy(logits, y)
                mer_loss = mer(feat_norm, logits)
                loss = ce_loss + 0.5 * mer_loss
                mer_logits = logits
            elif args.method == "crt":
                if epoch < args.rt_start:
                    logits, feat_norm = model(x, return_feat=True, rt=False)
                    loss = F.cross_entropy(logits, y)
                    mer_logits = logits
                else:
                    logits_raw, feat_norm = model(x, return_feat=True, rt=True)
                    # CRT stage: optionally apply LA shift then MER regularization.
                    logits_adj = logits_raw + args.la_tau * log_prior.view(1, -1)
                    loss = F.cross_entropy(logits_adj, y)
                    loss -= args.lambda_mer * mer(feat_norm, logits_raw)
                    mer_logits = logits_raw
            else:
                raise ValueError(f"Unknown method: {args.method}")
            if args.mer:
                loss -= args.lambda_mer * mer(feat_norm, mer_logits)
            if args.method == "crt" and epoch >= args.rt_start:
                optimizer_rt.zero_grad()
                loss.backward()
                optimizer_rt.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            recorder.update(idx, feat_norm.detach(), mer_logits.detach())

            loop.set_postfix(loss=float(loss.item()))
        if args.method == "crt" and epoch >= args.rt_start:
            scheduler_rt.step()
        else:
            scheduler.step()

        acc, head, mid, tail = evaluate(
            model,
            test_loader,
            device,
            method=args.method,
            cls_head=cls_head,
            cls_mid=cls_mid,
            cls_tail=cls_tail,
            log_prior=log_prior,
            la_tau=args.la_tau,
            rt=(epoch >= args.rt_start)
        )
        nni.report_intermediate_result({'default': acc})
        recorder.record_acc(acc, head, mid, tail)
        print(f"[Epoch {epoch}] Overall={acc*100:.2f}% | Head={head*100:.2f}% "
              f"Mid={mid*100:.2f}% Tail={tail*100:.2f}%")

        best_overall = update_best(best_overall, acc, epoch, "Overall")
        best_head    = update_best(best_head, head, epoch, "Head")
        best_mid     = update_best(best_mid, mid, epoch, "Mid")
        best_tail    = update_best(best_tail, tail, epoch, "Tail")

        recorder.compute_lms_epoch(epoch)
    nni.report_final_result({'default': best_overall['value']})
    print("\n===== Training Finished =====")
    print(f"Best Overall: {best_overall['value']*100:.2f}% @ {best_overall['epoch']}")
    print(f"Best Head:    {best_head['value']*100:.2f}% @ {best_head['epoch']}")
    print(f"Best Mid:     {best_mid['value']*100:.2f}% @ {best_mid['epoch']}")
    print(f"Best Tail:    {best_tail['value']*100:.2f}% @ {best_tail['epoch']}")
    recorder.finalize()


# ===============================================================
# CLI
# ===============================================================

def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-LT | MER loss with LMS recorder under different LTVR methods")
    p.add_argument("--data_root", type=str, default="data/")
    p.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cifar10"])
    p.add_argument("--imb_factor", type=float, default=100.0)
    p.add_argument("--oversample_tail", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--rt_start", type=int, default=180)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="results_ce_bs/")
    p.add_argument("--lambda_mer", type=float, default=0.6)
    p.add_argument("--method", type=str, default="la",
                   choices=["ce", "mixup", "la", "focal", "ldam",
                            "rw", "drw", "resample", "scl", "mer", "crt"])
    p.add_argument("--mer", action="store_true", help="Whether to use MER loss as a regularization term.")
    p.add_argument("--backbone", type=str, default="resnet32",
                   choices=["resnet32","resnet18","resnet50",
                            "densenet121",
                            "mobilenet_v2",
                            "vgg16"
                            ])

    p.add_argument("--mixup_alpha", type=float, default=0.4)

    p.add_argument("--la_tau", type=float, default=1.0)

    p.add_argument("--focal_gamma", type=float, default=1.0)

    p.add_argument("--ldam_max_m", type=float, default=0.5)
    p.add_argument("--ldam_s", type=float, default=1.0)

    p.add_argument("--rw_beta", type=float, default=0.999)
    p.add_argument("--drw_start_epoch", type=int, default=160)

    # SCL
    p.add_argument("--scl_lambda", type=float, default=0.1)
    p.add_argument("--scl_temp", type=float, default=0.07)
    p.add_argument("--mer_k", type=int, default=5)
    p.add_argument("--mer_margin", type=float, default=0.1)

    return p.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    tuner_params = nni.get_next_parameter()
    if 'lambda_mer' in tuner_params:
        args.lambda_mer = tuner_params['lambda_mer']
    if 'bs' in tuner_params:
        args.batch_size = tuner_params['bs']
    if 'mer_k' in tuner_params:
        args.mer_k = tuner_params['mer_k']
    if 'mer_margin' in tuner_params:
        args.mer_margin = tuner_params['mer_margin']
    args.save_dir = args.save_dir + f"k_{args.mer_k}/"
    run(args)
