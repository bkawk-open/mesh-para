"""Extract a small matched subset from ABC archives for cadresearch."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def list_archive_members(archive: Path) -> list[str]:
    result = subprocess.run(
        ["bsdtar", "-tf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.endswith("/")]


def shared_key(member: str) -> str:
    name = Path(member).name
    if "_trimesh_" in name:
        base = name.split("_trimesh_", 1)[0]
    elif "_step_" in name:
        base = name.split("_step_", 1)[0]
    elif "_features_" in name:
        base = name.split("_features_", 1)[0]
    else:
        raise ValueError(f"Unrecognized ABC member name: {member}")
    return f"{Path(member).parent.as_posix()}/{base}"


def build_index(members: list[str]) -> dict[str, str]:
    index = {}
    for member in members:
        index[shared_key(member)] = member
    return index


def extract_members_bulk(archive: Path, members: list[str], destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["bsdtar", "-xf", str(archive), "-C", str(destination_root), *members],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract matched ABC subset for cadresearch")
    parser.add_argument("--stl-archive", required=True)
    parser.add_argument("--step-archive", required=True)
    parser.add_argument("--feat-archive", required=True)
    parser.add_argument(
        "--output-dir",
        default="/Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/abc_subset",
    )
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stl_archive = Path(args.stl_archive)
    step_archive = Path(args.step_archive)
    feat_archive = Path(args.feat_archive)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    stl_index = build_index(list_archive_members(stl_archive))
    step_index = build_index(list_archive_members(step_archive))
    feat_index = build_index(list_archive_members(feat_archive))
    shared = sorted(set(stl_index) & set(step_index) & set(feat_index))
    if not shared:
        raise SystemExit("No matched STL/STEP/features entries found across the provided archives.")

    selected = shared[args.offset : args.offset + args.limit]
    if not selected:
        raise SystemExit("Requested slice produced an empty selection.")

    stl_members = [stl_index[key] for key in selected]
    step_members = [step_index[key] for key in selected]
    feat_members = [feat_index[key] for key in selected]

    manifest = []
    with tempfile.TemporaryDirectory(prefix="cadresearch_abc_extract_") as temp_dir:
        temp_root = Path(temp_dir)
        temp_stl = temp_root / "stl"
        temp_step = temp_root / "step"
        temp_feat = temp_root / "feat"
        extract_members_bulk(stl_archive, stl_members, temp_stl)
        extract_members_bulk(step_archive, step_members, temp_step)
        extract_members_bulk(feat_archive, feat_members, temp_feat)

        for i, key in enumerate(selected):
            sample_id = f"{i + args.offset:06d}"
            sample_dir = raw_dir / sample_id
            stl_member = stl_index[key]
            step_member = step_index[key]
            feat_member = feat_index[key]
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "mesh.stl").write_bytes((temp_stl / stl_member).read_bytes())
            (sample_dir / "model.step").write_bytes((temp_step / step_member).read_bytes())
            (sample_dir / "features.yml").write_bytes((temp_feat / feat_member).read_bytes())
            manifest.append(
                {
                    "sample_id": sample_id,
                    "shared_key": key,
                    "source_files": {
                        "mesh": Path(stl_member).name,
                        "step": Path(step_member).name,
                        "features": Path(feat_member).name,
                    },
                    "paths": {
                        "mesh": str((sample_dir / "mesh.stl").relative_to(output_dir)),
                        "step": str((sample_dir / "model.step").relative_to(output_dir)),
                        "features": str((sample_dir / "features.yml").relative_to(output_dir)),
                    },
                }
            )

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(manifest),
                "offset": args.offset,
                "archives": {
                    "stl": str(stl_archive),
                    "step": str(step_archive),
                    "features": str(feat_archive),
                },
                "samples": manifest,
            },
            f,
            indent=2,
        )

    print(f"Matched entries available: {len(shared)}")
    print(f"Extracted subset size:     {len(manifest)}")
    print(f"Output directory:          {output_dir}")


if __name__ == "__main__":
    main()
