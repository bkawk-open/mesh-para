"""
Convert extracted ABC raw samples into per-object .npz files for cadresearch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

CLASS_NAMES = ["plane", "cylinder", "cone", "sphere", "other"]
PARAM_DIM = 8
TYPE_TO_LABEL = {
    "Plane": 0,
    "Cylinder": 1,
    "Cone": 2,
    "Sphere": 3,
}


def load_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header = f.read(80)
        if len(header) != 80:
            raise ValueError(f"Invalid STL header in {path}")
        tri_count = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        data = np.frombuffer(f.read(), dtype=np.uint8)
    expected = tri_count * 50
    if data.size != expected:
        raise ValueError(f"Unexpected STL payload size in {path}: {data.size} != {expected}")
    tri = data.reshape(tri_count, 50)
    normals = tri[:, :12].view("<f4").reshape(tri_count, 3).astype(np.float32)
    vertices = tri[:, 12:48].view("<f4").reshape(tri_count, 3, 3).astype(np.float32)
    return vertices, normals


def normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.clip(denom, 1e-8, None)
    return (x / denom).astype(np.float32)


def triangle_areas(vertices: np.ndarray) -> np.ndarray:
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    return (0.5 * np.linalg.norm(cross, axis=1)).astype(np.float32)


def parse_surface(surface: dict) -> tuple[int, np.ndarray, np.ndarray]:
    label = TYPE_TO_LABEL.get(surface.get("type"), 4)
    params = np.zeros(PARAM_DIM, dtype=np.float32)
    mask = np.zeros(PARAM_DIM, dtype=np.float32)
    location = np.array(surface.get("location", [0.0, 0.0, 0.0]), dtype=np.float32)

    if label == 0:
        z_axis = np.array(surface.get("z_axis", [0.0, 0.0, 1.0]), dtype=np.float32)
        params[:3] = location
        params[3:6] = normalize(z_axis[None, :])[0]
        mask[:6] = 1.0
    elif label == 1:
        z_axis = np.array(surface.get("z_axis", [0.0, 0.0, 1.0]), dtype=np.float32)
        params[:3] = location
        params[3:6] = normalize(z_axis[None, :])[0]
        params[6] = float(surface.get("radius", 0.0))
        mask[:7] = 1.0
    elif label == 2:
        z_axis = np.array(surface.get("z_axis", [0.0, 0.0, 1.0]), dtype=np.float32)
        radius = float(surface.get("radius", 0.0))
        apex_angle = float(surface.get("angle", 0.0))
        if apex_angle == 0.0 and radius > 0.0:
            apex_angle = math.atan(radius / max(1e-6, np.linalg.norm(location)))
        params[:3] = location
        params[3:6] = normalize(z_axis[None, :])[0]
        params[6] = apex_angle
        params[7] = radius
        mask[:] = 1.0
    elif label == 3:
        params[:3] = location
        params[6] = float(surface.get("radius", 0.0))
        mask[0:3] = 1.0
        mask[6] = 1.0
    return label, params, mask


def sample_points(
    triangles: np.ndarray,
    face_normals: np.ndarray,
    face_labels: np.ndarray,
    face_params: np.ndarray,
    face_masks: np.ndarray,
    face_boundary: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    areas = triangle_areas(triangles)
    probs = areas / np.clip(areas.sum(), 1e-8, None)
    face_ids = rng.choice(len(triangles), size=num_points, replace=True, p=probs)
    chosen = triangles[face_ids]
    r1 = np.sqrt(rng.random(num_points, dtype=np.float32))
    r2 = rng.random(num_points, dtype=np.float32)
    points = (
        (1.0 - r1)[:, None] * chosen[:, 0]
        + (r1 * (1.0 - r2))[:, None] * chosen[:, 1]
        + (r1 * r2)[:, None] * chosen[:, 2]
    ).astype(np.float32)
    normals = face_normals[face_ids].astype(np.float32)
    labels = face_labels[face_ids].astype(np.int64)
    params = face_params[face_ids].astype(np.float32)
    masks = face_masks[face_ids].astype(np.float32)
    boundary = face_boundary[face_ids].astype(np.float32)
    return {
        "points": points,
        "normals": normals,
        "labels": labels,
        "params": params,
        "param_mask": masks,
        "boundary": boundary,
    }


def vertex_key(vertex: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(x) for x in np.round(vertex.astype(np.float64), decimals=6))


def compute_face_boundary(triangles: np.ndarray, face_labels: np.ndarray) -> np.ndarray:
    edge_to_faces: dict[tuple[tuple[float, float, float], tuple[float, float, float]], list[int]] = {}
    for face_idx, tri in enumerate(triangles):
        verts = [vertex_key(tri[i]) for i in range(3)]
        edges = [
            tuple(sorted((verts[0], verts[1]))),
            tuple(sorted((verts[1], verts[2]))),
            tuple(sorted((verts[2], verts[0]))),
        ]
        for edge in edges:
            edge_to_faces.setdefault(edge, []).append(face_idx)

    face_boundary = np.zeros(len(triangles), dtype=np.float32)
    for adjacent_faces in edge_to_faces.values():
        if len(adjacent_faces) < 2:
            continue
        labels = {int(face_labels[idx]) for idx in adjacent_faces}
        if len(labels) > 1:
            for idx in adjacent_faces:
                face_boundary[idx] = 1.0
    return face_boundary


def convert_sample(sample_dir: Path, output_path: Path, num_points: int, rng: np.random.Generator) -> None:
    triangles, stl_normals = load_binary_stl(sample_dir / "mesh.stl")
    tri_normals = normalize(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]))
    valid = np.linalg.norm(stl_normals, axis=1) > 1e-6
    face_normals = np.where(valid[:, None], normalize(stl_normals), tri_normals).astype(np.float32)

    with (sample_dir / "features.yml").open("r", encoding="utf-8") as f:
        features = yaml.safe_load(f)
    surfaces = features.get("surfaces", [])

    face_labels = np.full(len(triangles), 4, dtype=np.int64)
    face_params = np.zeros((len(triangles), PARAM_DIM), dtype=np.float32)
    face_masks = np.zeros((len(triangles), PARAM_DIM), dtype=np.float32)
    for surface in surfaces:
        label, params, mask = parse_surface(surface)
        for face_idx in surface.get("face_indices", []):
            if 0 <= face_idx < len(triangles):
                face_labels[face_idx] = label
                face_params[face_idx] = params
                face_masks[face_idx] = mask

    face_boundary = compute_face_boundary(triangles, face_labels)

    sample = sample_points(
        triangles,
        face_normals,
        face_labels,
        face_params,
        face_masks,
        face_boundary,
        num_points,
        rng,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **sample)


def process_task(task: tuple[str, str, int, int]) -> None:
    sample_dir_str, output_path_str, num_points, seed = task
    rng = np.random.default_rng(seed)
    convert_sample(Path(sample_dir_str), Path(output_path_str), num_points, rng)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess raw ABC subset into per-object .npz files")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--val-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    raw_dir = input_dir / "raw"
    sample_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not sample_dirs:
        raise SystemExit(f"No raw sample directories found under {raw_dir}")

    train_dirs = sample_dirs[:-args.val_count] if args.val_count < len(sample_dirs) else sample_dirs[:-1]
    val_dirs = sample_dirs[len(train_dirs):]
    tasks = []
    seed_base = args.seed
    for split, dirs in [("train", train_dirs), ("val", val_dirs)]:
        for i, sample_dir in enumerate(dirs):
            tasks.append(
                (
                    str(sample_dir),
                    str(output_dir / split / f"{sample_dir.name}.npz"),
                    args.num_points,
                    seed_base + len(tasks) + i,
                )
            )

    if args.workers <= 1:
        for task in tasks:
            process_task(task)
    else:
        with Pool(processes=args.workers) as pool:
            pool.map(process_task, tasks)

    with (output_dir / "preprocess_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "num_points": args.num_points,
                "class_names": CLASS_NAMES,
                "train_count": len(train_dirs),
                "val_count": len(val_dirs),
            },
            f,
            indent=2,
        )
    print(output_dir)
    print(f"train={len(train_dirs)} val={len(val_dirs)}")


if __name__ == "__main__":
    main()
