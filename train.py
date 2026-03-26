"""
Baseline training loop for cadresearch.

This file is intentionally weak. It is the file the autonomous agent should edit.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET,
    evaluate_model,
    get_dataset_spec,
    make_dataloader,
    move_batch_to_device,
)


@dataclass
class TrainConfig:
    batch_size: int = 4
    eval_batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    global_dim: int = 256
    param_loss_weight: float = 0.2
    boundary_loss_weight: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 25
    k_neighbors: int = 8


def knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    with torch.no_grad():
        cpu_points = points.detach().cpu()
        dist = torch.cdist(cpu_points, cpu_points)
        _, indices = dist.topk(k + 1, dim=-1, largest=False)
    return indices[:, :, 1:].to(device=points.device)


def gather_neighbors(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    bsz, num_points, channels = x.shape
    k = indices.size(-1)
    batch_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1).expand(-1, num_points, k)
    return x[batch_idx, indices]


class LocalPointModel(nn.Module):
    """
    Stronger but still compact:
    - encode xyz + normals together
    - aggregate simple k-NN local geometry
    - combine local and mixed global context for per-point predictions
    """

    def __init__(self, num_classes: int, param_dim: int, hidden_dim: int, global_dim: int, k_neighbors: int):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.point_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.local_mlp = nn.Sequential(
            nn.Linear((2 * hidden_dim) + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, global_dim),
            nn.ReLU(),
        )
        self.global_pool_proj = nn.Sequential(
            nn.Linear(2 * global_dim, global_dim),
            nn.ReLU(),
        )
        fused_dim = hidden_dim + global_dim + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.param_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.cat([points, normals], dim=-1)
        point_feat = self.point_encoder(inputs)
        neighbor_idx = knn_indices(points, self.k_neighbors)
        neighbor_feat = gather_neighbors(point_feat, neighbor_idx)
        neighbor_points = gather_neighbors(points, neighbor_idx)
        neighbor_normals = gather_neighbors(normals, neighbor_idx)
        center_feat = point_feat.unsqueeze(2).expand_as(neighbor_feat)
        center_points = points.unsqueeze(2).expand_as(neighbor_points)
        center_normals = normals.unsqueeze(2).expand_as(neighbor_normals)
        rel_points = neighbor_points - center_points
        rel_normals = neighbor_normals - center_normals
        edge_feat = torch.cat([center_feat, neighbor_feat - center_feat, rel_points, rel_normals], dim=-1)
        local_feat = self.local_mlp(edge_feat).max(dim=2).values
        context_feat = self.context_proj(torch.cat([point_feat, local_feat], dim=-1))
        global_max = context_feat.max(dim=1, keepdim=True).values
        global_mean = context_feat.mean(dim=1, keepdim=True)
        global_feat = self.global_pool_proj(torch.cat([global_max, global_mean], dim=-1))
        global_feat = global_feat.expand(-1, points.size(1), -1)
        fused = torch.cat([point_feat, local_feat, global_feat], dim=-1)
        logits = self.classifier(fused)
        param_pred = self.param_head(fused)
        boundary_logits = self.boundary_head(fused).squeeze(-1)
        return logits, param_pred, boundary_logits


def compute_loss(
    logits: torch.Tensor,
    param_pred: torch.Tensor,
    boundary_logits: torch.Tensor,
    labels: torch.Tensor,
    target_params: torch.Tensor,
    param_mask: torch.Tensor,
    boundary_target: torch.Tensor,
    param_scale: torch.Tensor,
    param_loss_weight: float,
    boundary_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    scale = param_scale.view(1, 1, -1)
    param_error = F.smooth_l1_loss(param_pred / scale, target_params / scale, reduction="none")
    param_loss = (param_error * param_mask).sum() / param_mask.sum().clamp(min=1.0)
    boundary_pos = boundary_target.mean().clamp(min=1e-3, max=1.0 - 1e-3)
    pos_weight = ((1.0 - boundary_pos) / boundary_pos).detach()
    boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target, pos_weight=pos_weight)
    loss = cls_loss + param_loss_weight * param_loss + boundary_loss_weight * boundary_loss
    return loss, {
        "cls_loss": float(cls_loss.detach().item()),
        "param_loss": float(param_loss.detach().item()),
        "boundary_loss": float(boundary_loss.detach().item()),
        "loss": float(loss.detach().item()),
    }


def get_device() -> torch.device:
    requested = os.environ.get("CADRESEARCH_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_time_budget() -> float:
    override = os.environ.get("CADRESEARCH_TIME_BUDGET")
    if override:
        return float(override)
    return float(TIME_BUDGET)


def maybe_peak_vram_mb(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    return 0


def main() -> None:
    torch.manual_seed(1337)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    cfg = TrainConfig()
    spec = get_dataset_spec()
    device = get_device()
    time_budget = get_time_budget()
    num_classes = len(spec.class_names)
    param_scale = torch.tensor(spec.param_scale, dtype=torch.float32, device=device)

    train_loader = make_dataloader("train", batch_size=cfg.batch_size, drop_last=True)
    val_loader = make_dataloader("val", batch_size=cfg.eval_batch_size, shuffle=False)

    model = LocalPointModel(
        num_classes=num_classes,
        param_dim=spec.param_dim,
        hidden_dim=cfg.hidden_dim,
        global_dim=cfg.global_dim,
        k_neighbors=cfg.k_neighbors,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step = 0
    last_stats = {"loss": math.nan, "cls_loss": math.nan, "param_loss": math.nan, "boundary_loss": math.nan}
    train_start = None
    budget_deadline = None
    train_iter = iter(train_loader)

    while True:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if train_start is None:
            train_start = time.time()
            budget_deadline = train_start + time_budget
        elif time.time() >= budget_deadline:
            break

        batch = move_batch_to_device(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, param_pred, boundary_logits = model(batch["points"], batch["normals"])
        loss, last_stats = compute_loss(
            logits,
            param_pred,
            boundary_logits,
            batch["labels"],
            batch["params"],
            batch["param_mask"],
            batch["boundary"],
            param_scale,
            cfg.param_loss_weight,
            cfg.boundary_loss_weight,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        step += 1

        if step % cfg.log_interval == 0:
            elapsed = time.time() - train_start
            print(
                f"step={step:04d} elapsed={elapsed:7.1f}s "
                f"loss={last_stats['loss']:.4f} "
                f"cls={last_stats['cls_loss']:.4f} "
                f"param={last_stats['param_loss']:.4f} "
                f"boundary={last_stats['boundary_loss']:.4f}"
            )

    training_seconds = 0.0 if train_start is None else (time.time() - train_start)
    metrics = evaluate_model(
        model,
        val_loader,
        device,
        param_scale=param_scale,
        num_classes=num_classes,
    )
    peak_vram_mb = maybe_peak_vram_mb(device)

    print("---")
    print(f"val_score:        {metrics['val_score']:.6f}")
    print(f"macro_iou:        {metrics['macro_iou']:.6f}")
    print(f"param_rmse_norm:  {metrics['param_rmse_norm']:.6f}")
    print(f"param_score:      {metrics['param_score']:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"steps:            {step}")
    print(f"peak_vram_mb:     {peak_vram_mb}")
    print(f"device:           {device.type}")
    print(f"last_loss:        {last_stats['loss']:.6f}")
    print("---")


if __name__ == "__main__":
    main()
