"""
Dataset builder for cadresearch.

Two modes are supported:

1. synthetic: generate a procedural bootstrap dataset of primitive patches
2. pack: pack preprocessed per-object .npz samples into fixed shards
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

CLASS_NAMES = ["plane", "cylinder", "cone", "sphere", "other"]
PARAM_DIM = 8
DEFAULT_NUM_POINTS = 2048
DEFAULT_CACHE_DIR = os.environ.get(
    "CADRESEARCH_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "cadresearch"),
)
DEFAULT_DATA_DIR = os.path.join(DEFAULT_CACHE_DIR, "shards")


@dataclass(frozen=True)
class Sample:
    points: np.ndarray
    normals: np.ndarray
    labels: np.ndarray
    params: np.ndarray
    param_mask: np.ndarray
    boundary: np.ndarray


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    norm = np.linalg.norm(v)
    while norm < 1e-8:
        v = rng.normal(size=3)
        norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32)


def orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(axis, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    v = v / np.linalg.norm(v)
    return u.astype(np.float32), v.astype(np.float32)


def normalize_vectors(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return (x / norms).astype(np.float32)


def make_param_rows(count: int, values: list[float], mask: list[float]) -> tuple[np.ndarray, np.ndarray]:
    params = np.tile(np.array(values, dtype=np.float32), (count, 1))
    param_mask = np.tile(np.array(mask, dtype=np.float32), (count, 1))
    return params, param_mask


def sample_plane_patch(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = rng.uniform(-0.7, 0.7, size=3).astype(np.float32)
    normal = random_unit_vector(rng)
    u, v = orthonormal_basis(normal)
    half_width = float(rng.uniform(0.15, 0.6))
    half_height = float(rng.uniform(0.15, 0.6))
    a = rng.uniform(-half_width, half_width, size=count).astype(np.float32)
    b = rng.uniform(-half_height, half_height, size=count).astype(np.float32)
    points = center + a[:, None] * u + b[:, None] * v
    normals = np.repeat(normal[None, :], count, axis=0)
    params, mask = make_param_rows(
        count,
        [*center.tolist(), *normal.tolist(), half_width, half_height],
        [1, 1, 1, 1, 1, 1, 1, 1],
    )
    return points, normals, params, mask


def sample_cylinder_patch(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = rng.uniform(-0.5, 0.5, size=3).astype(np.float32)
    axis = random_unit_vector(rng)
    u, v = orthonormal_basis(axis)
    radius = float(rng.uniform(0.08, 0.35))
    half_height = float(rng.uniform(0.2, 0.7))
    theta_span = float(rng.uniform(math.pi / 3.0, 2.0 * math.pi))
    theta0 = float(rng.uniform(-math.pi, math.pi))
    theta = theta0 + rng.uniform(0.0, theta_span, size=count).astype(np.float32)
    h = rng.uniform(-half_height, half_height, size=count).astype(np.float32)
    radial = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    points = center + h[:, None] * axis + radius * radial
    normals = normalize_vectors(radial)
    params, mask = make_param_rows(
        count,
        [*center.tolist(), *axis.tolist(), radius, half_height],
        [1, 1, 1, 1, 1, 1, 1, 1],
    )
    return points.astype(np.float32), normals.astype(np.float32), params, mask


def sample_cone_patch(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    apex = rng.uniform(-0.5, 0.5, size=3).astype(np.float32)
    axis = random_unit_vector(rng)
    u, v = orthonormal_basis(axis)
    half_angle = float(rng.uniform(math.radians(8.0), math.radians(35.0)))
    height = float(rng.uniform(0.25, 0.8))
    theta_span = float(rng.uniform(math.pi / 3.0, 2.0 * math.pi))
    theta0 = float(rng.uniform(-math.pi, math.pi))
    theta = theta0 + rng.uniform(0.0, theta_span, size=count).astype(np.float32)
    h = rng.uniform(0.08, height, size=count).astype(np.float32)
    radial_dir = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    radial_mag = np.tan(half_angle) * h
    points = apex + h[:, None] * axis + radial_mag[:, None] * radial_dir
    dp_dtheta = radial_mag[:, None] * (-np.sin(theta)[:, None] * u + np.cos(theta)[:, None] * v)
    dp_dh = axis + np.tan(half_angle) * radial_dir
    normals = normalize_vectors(np.cross(dp_dtheta, dp_dh))
    params, mask = make_param_rows(
        count,
        [*apex.tolist(), *axis.tolist(), half_angle, height],
        [1, 1, 1, 1, 1, 1, 1, 1],
    )
    return points.astype(np.float32), normals.astype(np.float32), params, mask


def sample_sphere_patch(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = rng.uniform(-0.5, 0.5, size=3).astype(np.float32)
    pole = random_unit_vector(rng)
    u, v = orthonormal_basis(pole)
    radius = float(rng.uniform(0.12, 0.4))
    gamma_max = float(rng.uniform(math.pi / 10.0, math.pi / 2.5))
    theta = rng.uniform(-math.pi, math.pi, size=count).astype(np.float32)
    gamma = rng.uniform(0.0, gamma_max, size=count).astype(np.float32)
    dirs = (
        np.cos(gamma)[:, None] * pole
        + np.sin(gamma)[:, None] * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    )
    dirs = normalize_vectors(dirs)
    points = center + radius * dirs
    normals = dirs
    params, mask = make_param_rows(
        count,
        [*center.tolist(), 0.0, 0.0, 0.0, radius, gamma_max],
        [1, 1, 1, 0, 0, 0, 1, 1],
    )
    return points.astype(np.float32), normals.astype(np.float32), params, mask


def sample_other_patch(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = rng.uniform(-0.6, 0.6, size=3).astype(np.float32)
    w = random_unit_vector(rng)
    u, v = orthonormal_basis(w)
    half_width = float(rng.uniform(0.18, 0.55))
    half_height = float(rng.uniform(0.18, 0.55))
    a = float(rng.uniform(-0.6, 0.6))
    b = float(rng.uniform(-0.6, 0.6))
    c = float(rng.uniform(-0.3, 0.3))
    x = rng.uniform(-half_width, half_width, size=count).astype(np.float32)
    y = rng.uniform(-half_height, half_height, size=count).astype(np.float32)
    z = a * x * x + b * y * y + c * x * y
    points = center + x[:, None] * u + y[:, None] * v + z[:, None] * w
    dp_dx = u + (2.0 * a * x + c * y)[:, None] * w
    dp_dy = v + (2.0 * b * y + c * x)[:, None] * w
    normals = normalize_vectors(np.cross(dp_dx, dp_dy))
    params, mask = make_param_rows(count, [0.0] * PARAM_DIM, [0.0] * PARAM_DIM)
    return points.astype(np.float32), normals.astype(np.float32), params, mask


PATCH_GENERATORS = {
    0: sample_plane_patch,
    1: sample_cylinder_patch,
    2: sample_cone_patch,
    3: sample_sphere_patch,
    4: sample_other_patch,
}


def allocate_patch_counts(num_points: int, num_patches: int, rng: np.random.Generator) -> np.ndarray:
    minimum = max(8, num_points // (num_patches * 6))
    minimum = min(minimum, max(1, num_points // num_patches))
    counts = np.full(num_patches, minimum, dtype=np.int64)
    remaining = num_points - int(counts.sum())
    if remaining < 0:
        counts = np.ones(num_patches, dtype=np.int64)
        remaining = num_points - int(counts.sum())
        if remaining < 0:
            raise ValueError("num_points too small for the requested number of patches")
    if remaining > 0:
        probs = rng.dirichlet(np.ones(num_patches, dtype=np.float32))
        counts += rng.multinomial(remaining, probs)
    return counts


def add_measurement_noise(points: np.ndarray, normals: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    point_noise = rng.normal(0.0, 0.004, size=points.shape).astype(np.float32)
    normal_noise = rng.normal(0.0, 0.02, size=normals.shape).astype(np.float32)
    points = points + point_noise
    normals = normalize_vectors(normals + normal_noise)
    return points.astype(np.float32), normals.astype(np.float32)


def generate_synthetic_sample(num_points: int, rng: np.random.Generator) -> Sample:
    num_patches = int(rng.integers(3, 8))
    patch_types = rng.choice(
        len(CLASS_NAMES),
        size=num_patches,
        p=np.array([0.40, 0.25, 0.12, 0.08, 0.15], dtype=np.float32),
    )
    counts = allocate_patch_counts(num_points, num_patches, rng)
    points_all = []
    normals_all = []
    labels_all = []
    params_all = []
    masks_all = []
    for patch_type, count in zip(patch_types, counts):
        points, normals, params, param_mask = PATCH_GENERATORS[int(patch_type)](rng, int(count))
        points_all.append(points)
        normals_all.append(normals)
        labels_all.append(np.full(int(count), int(patch_type), dtype=np.int64))
        params_all.append(params)
        masks_all.append(param_mask)

    points = np.concatenate(points_all, axis=0)
    normals = np.concatenate(normals_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    params = np.concatenate(params_all, axis=0)
    param_mask = np.concatenate(masks_all, axis=0)
    points, normals = add_measurement_noise(points, normals, rng)

    shuffle = rng.permutation(points.shape[0])
    return Sample(
        points=points[shuffle].astype(np.float32),
        normals=normals[shuffle].astype(np.float32),
        labels=labels[shuffle].astype(np.int64),
        params=params[shuffle].astype(np.float32),
        param_mask=param_mask[shuffle].astype(np.float32),
        boundary=np.zeros(num_points, dtype=np.float32),
    )


def validate_sample(sample: Sample, num_points: int) -> None:
    if sample.points.shape != (num_points, 3):
        raise ValueError(f"points must have shape {(num_points, 3)}, got {sample.points.shape}")
    if sample.normals.shape != (num_points, 3):
        raise ValueError(f"normals must have shape {(num_points, 3)}, got {sample.normals.shape}")
    if sample.labels.shape != (num_points,):
        raise ValueError(f"labels must have shape {(num_points,)}, got {sample.labels.shape}")
    if sample.params.shape != (num_points, PARAM_DIM):
        raise ValueError(f"params must have shape {(num_points, PARAM_DIM)}, got {sample.params.shape}")
    if sample.param_mask.shape != (num_points, PARAM_DIM):
        raise ValueError(f"param_mask must have shape {(num_points, PARAM_DIM)}, got {sample.param_mask.shape}")
    if sample.boundary.shape != (num_points,):
        raise ValueError(f"boundary must have shape {(num_points,)}, got {sample.boundary.shape}")
    if sample.labels.min() < 0 or sample.labels.max() >= len(CLASS_NAMES):
        raise ValueError("labels contain out-of-range class indices")


def resample_points(sample: Sample, num_points: int, rng: np.random.Generator) -> Sample:
    current = sample.points.shape[0]
    if current == num_points:
        return sample
    if current > num_points:
        indices = rng.choice(current, size=num_points, replace=False)
    else:
        extra = rng.choice(current, size=num_points - current, replace=True)
        indices = np.concatenate([np.arange(current), extra], axis=0)
    return Sample(
        points=sample.points[indices].astype(np.float32),
        normals=sample.normals[indices].astype(np.float32),
        labels=sample.labels[indices].astype(np.int64),
        params=sample.params[indices].astype(np.float32),
        param_mask=sample.param_mask[indices].astype(np.float32),
        boundary=sample.boundary[indices].astype(np.float32),
    )


def stack_samples(samples: list[Sample]) -> dict[str, np.ndarray]:
    return {
        "points": np.stack([s.points for s in samples], axis=0).astype(np.float32),
        "normals": np.stack([s.normals for s in samples], axis=0).astype(np.float32),
        "labels": np.stack([s.labels for s in samples], axis=0).astype(np.int64),
        "params": np.stack([s.params for s in samples], axis=0).astype(np.float32),
        "param_mask": np.stack([s.param_mask for s in samples], axis=0).astype(np.float32),
        "boundary": np.stack([s.boundary for s in samples], axis=0).astype(np.float32),
    }


def save_shards(
    split: str,
    samples: Iterable[Sample],
    data_dir: str,
    shard_size: int,
) -> list[dict[str, object]]:
    shard_entries = []
    buffer: list[Sample] = []
    shard_index = 0
    for sample in samples:
        buffer.append(sample)
        if len(buffer) >= shard_size:
            filename = f"{split}_{shard_index:04d}.npz"
            path = os.path.join(data_dir, filename)
            np.savez_compressed(path, **stack_samples(buffer))
            shard_entries.append({"path": filename, "num_samples": len(buffer)})
            buffer.clear()
            shard_index += 1
    if buffer:
        filename = f"{split}_{shard_index:04d}.npz"
        path = os.path.join(data_dir, filename)
        np.savez_compressed(path, **stack_samples(buffer))
        shard_entries.append({"path": filename, "num_samples": len(buffer)})
    return shard_entries


def compute_metadata(
    data_dir: str,
    num_points: int,
    description: str,
) -> dict[str, object]:
    split_entries = {}
    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    param_sum_sq = np.zeros(PARAM_DIM, dtype=np.float64)
    param_count = np.zeros(PARAM_DIM, dtype=np.float64)

    for split in ("train", "val"):
        entries = []
        for filename in sorted(f for f in os.listdir(data_dir) if f.startswith(f"{split}_") and f.endswith(".npz")):
            path = os.path.join(data_dir, filename)
            with np.load(path) as shard:
                labels = shard["labels"]
                params = shard["params"]
                param_mask = shard["param_mask"]
                bincount = np.bincount(labels.reshape(-1), minlength=len(CLASS_NAMES))
                class_counts += bincount.astype(np.int64)
                if split == "train":
                    param_sum_sq += np.square(params * param_mask).sum(axis=(0, 1))
                    param_count += param_mask.sum(axis=(0, 1))
                entries.append({"path": filename, "num_samples": int(labels.shape[0])})
        split_entries[split] = entries

    param_scale = np.sqrt(param_sum_sq / np.clip(param_count, 1.0, None))
    param_scale = np.clip(param_scale, 1e-2, None)
    return {
        "version": 1,
        "description": description,
        "num_points": int(num_points),
        "param_dim": PARAM_DIM,
        "class_names": CLASS_NAMES,
        "splits": split_entries,
        "class_counts": class_counts.tolist(),
        "param_scale": param_scale.astype(np.float32).tolist(),
    }


def write_metadata(data_dir: str, metadata: dict[str, object]) -> None:
    path = os.path.join(data_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def build_synthetic_dataset(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    for filename in os.listdir(args.output_dir):
        if filename.endswith(".npz") or filename == "metadata.json":
            os.remove(os.path.join(args.output_dir, filename))

    rng = np.random.default_rng(args.seed)
    train_samples = [generate_synthetic_sample(args.num_points, rng) for _ in range(args.train_samples)]
    val_samples = [generate_synthetic_sample(args.num_points, rng) for _ in range(args.val_samples)]
    for sample in train_samples + val_samples:
        validate_sample(sample, args.num_points)

    save_shards("train", train_samples, args.output_dir, args.shard_size)
    save_shards("val", val_samples, args.output_dir, args.shard_size)
    metadata = compute_metadata(
        args.output_dir,
        args.num_points,
        "Synthetic primitive patches for cadresearch smoke tests",
    )
    write_metadata(args.output_dir, metadata)
    print(f"Wrote synthetic dataset to {args.output_dir}")
    print(f"train samples: {args.train_samples}")
    print(f"val samples:   {args.val_samples}")
    print(f"num points:    {args.num_points}")


def iter_input_samples(input_dir: str) -> tuple[list[str], list[str]]:
    train_dir = os.path.join(input_dir, "train")
    val_dir = os.path.join(input_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise ValueError("pack mode expects input_dir/train and input_dir/val")
    train_files = sorted(
        os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".npz")
    )
    val_files = sorted(
        os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".npz")
    )
    if not train_files or not val_files:
        raise ValueError("pack mode requires at least one .npz file in both train and val")
    return train_files, val_files


def load_external_sample(path: str, num_points: int, rng: np.random.Generator) -> Sample:
    with np.load(path) as data:
        sample = Sample(
            points=data["points"].astype(np.float32),
            normals=data["normals"].astype(np.float32),
            labels=data["labels"].astype(np.int64),
            params=data["params"].astype(np.float32),
            param_mask=data["param_mask"].astype(np.float32),
            boundary=data["boundary"].astype(np.float32) if "boundary" in data else np.zeros(data["labels"].shape, dtype=np.float32),
        )
    sample = resample_points(sample, num_points, rng)
    validate_sample(sample, num_points)
    return sample


def build_packed_dataset(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    for filename in os.listdir(args.output_dir):
        if filename.endswith(".npz") or filename == "metadata.json":
            os.remove(os.path.join(args.output_dir, filename))

    train_files, val_files = iter_input_samples(args.input_dir)
    rng = np.random.default_rng(args.seed)
    train_samples = [load_external_sample(path, args.num_points, rng) for path in train_files]
    val_samples = [load_external_sample(path, args.num_points, rng) for path in val_files]
    save_shards("train", train_samples, args.output_dir, args.shard_size)
    save_shards("val", val_samples, args.output_dir, args.shard_size)
    metadata = compute_metadata(
        args.output_dir,
        args.num_points,
        "Packed external mesh-to-CAD training shards",
    )
    write_metadata(args.output_dir, metadata)
    print(f"Wrote packed dataset to {args.output_dir}")
    print(f"train samples: {len(train_samples)}")
    print(f"val samples:   {len(val_samples)}")
    print(f"num points:    {args.num_points}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed shards for cadresearch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="generate a synthetic bootstrap dataset")
    synthetic.add_argument("--output-dir", default=DEFAULT_DATA_DIR)
    synthetic.add_argument("--train-samples", type=int, default=1024)
    synthetic.add_argument("--val-samples", type=int, default=128)
    synthetic.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    synthetic.add_argument("--shard-size", type=int, default=128)
    synthetic.add_argument("--seed", type=int, default=1337)
    synthetic.set_defaults(func=build_synthetic_dataset)

    pack = subparsers.add_parser("pack", help="pack preprocessed per-object .npz files")
    pack.add_argument("--input-dir", required=True)
    pack.add_argument("--output-dir", default=DEFAULT_DATA_DIR)
    pack.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    pack.add_argument("--shard-size", type=int, default=128)
    pack.add_argument("--seed", type=int, default=1337)
    pack.set_defaults(func=build_packed_dataset)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
