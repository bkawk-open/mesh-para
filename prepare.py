"""
Read-only infrastructure for cadresearch.

This file owns:
- dataset locations and fixed constants
- loading packed .npz shards
- evaluation

The autonomous loop should treat this file as read-only.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

TIME_BUDGET = int(os.environ.get("CADRESEARCH_TIME_BUDGET", "300"))
CACHE_DIR = os.environ.get(
    "CADRESEARCH_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "cadresearch"),
)
DATA_DIR = os.path.join(CACHE_DIR, "shards")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")


def load_metadata(data_dir: str = DATA_DIR) -> dict[str, object]:
    path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `uv run build_dataset.py synthetic` or pack a real dataset first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_metadata_cached() -> dict[str, object]:
    return load_metadata(DATA_DIR)


@dataclass(frozen=True)
class DatasetSpec:
    class_names: tuple[str, ...]
    num_points: int
    param_dim: int
    param_scale: tuple[float, ...]


def get_dataset_spec(data_dir: str = DATA_DIR) -> DatasetSpec:
    metadata = load_metadata(data_dir)
    return DatasetSpec(
        class_names=tuple(metadata["class_names"]),
        num_points=int(metadata["num_points"]),
        param_dim=int(metadata["param_dim"]),
        param_scale=tuple(float(x) for x in metadata["param_scale"]),
    )


class PackedShardDataset(Dataset):
    def __init__(self, split: str, data_dir: str = DATA_DIR):
        metadata = load_metadata(data_dir)
        if split not in metadata["splits"]:
            raise ValueError(f"Unknown split {split!r}")
        self.data_dir = data_dir
        self.split = split
        self.entries = list(metadata["splits"][split])
        if not self.entries:
            raise ValueError(f"Split {split!r} is empty")
        self.num_points = int(metadata["num_points"])
        self.param_dim = int(metadata["param_dim"])
        self.cumulative = []
        total = 0
        for entry in self.entries:
            total += int(entry["num_samples"])
            self.cumulative.append(total)

    def __len__(self) -> int:
        return self.cumulative[-1]

    @lru_cache(maxsize=4)
    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        path = os.path.join(self.data_dir, self.entries[shard_index]["path"])
        with np.load(path) as data:
            return {
                "points": data["points"].astype(np.float32),
                "normals": data["normals"].astype(np.float32),
                "labels": data["labels"].astype(np.int64),
                "params": data["params"].astype(np.float32),
                "param_mask": data["param_mask"].astype(np.float32),
                "boundary": data["boundary"].astype(np.float32) if "boundary" in data else np.zeros(data["labels"].shape, dtype=np.float32),
            }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = int(np.searchsorted(self.cumulative, index, side="right"))
        prev_total = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        local_index = index - prev_total
        shard = self._load_shard(shard_index)
        return {
            "points": torch.from_numpy(shard["points"][local_index]),
            "normals": torch.from_numpy(shard["normals"][local_index]),
            "labels": torch.from_numpy(shard["labels"][local_index]),
            "params": torch.from_numpy(shard["params"][local_index]),
            "param_mask": torch.from_numpy(shard["param_mask"][local_index]),
            "boundary": torch.from_numpy(shard["boundary"][local_index]),
        }


def make_dataloader(
    split: str,
    batch_size: int,
    *,
    shuffle: bool | None = None,
    num_workers: int = 0,
    drop_last: bool = False,
    data_dir: str = DATA_DIR,
) -> DataLoader:
    dataset = PackedShardDataset(split, data_dir=data_dir)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if value.dtype.is_floating_point:
            out[key] = value.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            out[key] = value.to(device=device, non_blocking=True)
    return out


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    flat = (target.reshape(-1) * num_classes + pred.reshape(-1)).detach().cpu().numpy().astype(np.int64, copy=False)
    bins = np.bincount(flat, minlength=num_classes * num_classes)
    return torch.from_numpy(bins.reshape(num_classes, num_classes))


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    param_scale: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    accum_dtype = torch.float32
    total_confusion = torch.zeros((num_classes, num_classes), dtype=accum_dtype, device=device)
    total_sq_error = torch.zeros(1, dtype=accum_dtype, device=device)
    total_param_count = torch.zeros(1, dtype=accum_dtype, device=device)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["points"], batch["normals"])
        logits, param_pred = outputs[:2] if isinstance(outputs, tuple) else outputs
        pred_labels = logits.argmax(dim=-1)
        total_confusion += confusion_matrix(pred_labels, batch["labels"], num_classes).to(device=device, dtype=accum_dtype)

        scale = param_scale.view(1, 1, -1)
        diff = (param_pred - batch["params"]) / scale
        masked_sq = diff.square() * batch["param_mask"]
        total_sq_error += masked_sq.sum(dtype=accum_dtype)
        total_param_count += batch["param_mask"].sum(dtype=accum_dtype)

    intersection = total_confusion.diag()
    union = total_confusion.sum(dim=1) + total_confusion.sum(dim=0) - intersection
    valid_classes = union > 0
    per_class_iou = torch.zeros_like(intersection)
    per_class_iou[valid_classes] = intersection[valid_classes] / union[valid_classes]
    macro_iou = float(per_class_iou[valid_classes].mean().item()) if valid_classes.any() else 0.0

    denom = torch.clamp(total_param_count, min=1.0)
    param_rmse_norm = float(torch.sqrt(total_sq_error / denom).item())
    param_score = 1.0 / (1.0 + param_rmse_norm)
    val_score = 0.7 * macro_iou + 0.3 * param_score

    return {
        "val_score": float(val_score),
        "macro_iou": float(macro_iou),
        "param_rmse_norm": float(param_rmse_norm),
        "param_score": float(param_score),
    }


def summarize_dataset(data_dir: str = DATA_DIR) -> str:
    metadata = load_metadata(data_dir)
    lines = [
        f"data_dir:     {data_dir}",
        f"description:  {metadata.get('description', 'n/a')}",
        f"num_points:   {metadata['num_points']}",
        f"param_dim:    {metadata['param_dim']}",
        f"class_names:  {', '.join(metadata['class_names'])}",
        f"class_counts: {metadata['class_counts']}",
    ]
    for split in ("train", "val"):
        total = sum(int(entry["num_samples"]) for entry in metadata["splits"][split])
        lines.append(f"{split}_samples: {total}")
        lines.append(f"{split}_shards:  {len(metadata['splits'][split])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the fixed cadresearch dataset contract")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    print(summarize_dataset())
    loader = make_dataloader("train", batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))
    print("batch points: ", tuple(batch["points"].shape))
    print("batch normals:", tuple(batch["normals"].shape))
    print("batch labels: ", tuple(batch["labels"].shape))
    print("batch params: ", tuple(batch["params"].shape))
    print("batch boundary:", tuple(batch["boundary"].shape))


if __name__ == "__main__":
    main()
