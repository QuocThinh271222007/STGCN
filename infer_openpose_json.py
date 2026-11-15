#!/usr/bin/env python
"""
Run the original ST-GCN Kinetics checkpoint on OpenPose COCO-18 JSON files.

Run this file from the root of the yysijie/st-gcn repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from net.st_gcn import Model


NUM_JOINTS = 18
CHANNELS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer an action from OpenPose COCO-18 JSON using ST-GCN."
    )
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/st_gcn.kinetics.pt"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("resource/kinetics_skeleton/label_name.txt"),
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-persons", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Uniformly sample long sequences to at most this many frames; 0 keeps all.",
    )
    parser.add_argument(
        "--save-npy",
        type=Path,
        default=None,
        help="Optional path for saving the produced C,T,V,M skeleton tensor.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path.resolve()}")


def video_size(video_path: Path) -> tuple[int, int]:
    require_file(video_path, "Video")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path.resolve()}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions: {width}x{height}")
    return width, height


def choose_frame_files(files: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(files) <= max_frames:
        return files

    indices = np.linspace(0, len(files) - 1, max_frames).round().astype(int)
    # linspace+round can theoretically duplicate an index; keep order and uniqueness.
    selected: list[Path] = []
    seen: set[int] = set()
    for index in indices.tolist():
        if index not in seen:
            selected.append(files[index])
            seen.add(index)
    return selected


def load_people(json_path: Path, width: int, height: int) -> list[np.ndarray]:
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    people: list[np.ndarray] = []
    for person in payload.get("people", []):
        values = person.get("pose_keypoints_2d", [])
        if len(values) != NUM_JOINTS * CHANNELS:
            continue

        pose = np.asarray(values, dtype=np.float32).reshape(NUM_JOINTS, CHANNELS)
        confidence = pose[:, 2]
        missing = confidence <= 0

        # Match the normalization in processor/demo_offline.py:
        # x = x / W - 0.5, y = y / H - 0.5.
        pose[:, 0] = pose[:, 0] / float(width) - 0.5
        pose[:, 1] = pose[:, 1] / float(height) - 0.5
        pose[missing, 0:2] = 0.0
        people.append(pose)

    people.sort(key=lambda item: float(item[:, 2].sum()), reverse=True)
    return people


def pose_distance(previous: np.ndarray, current: np.ndarray) -> float:
    visible = (previous[:, 2] > 0) & (current[:, 2] > 0)
    if not np.any(visible):
        return float("inf")
    delta = previous[visible, 0:2] - current[visible, 0:2]
    return float(np.linalg.norm(delta, axis=1).mean())


def assign_people(
    candidates: list[np.ndarray],
    previous_slots: list[np.ndarray | None],
    max_persons: int,
) -> list[np.ndarray | None]:
    """
    Greedy pose tracking for one or two people.

    The original repository calls its tracker "naive_pose_tracker" and likewise
    uses nearest-pose distance. This adapter keeps the same intent while reading
    saved JSON rather than pyopenpose objects.
    """
    slots: list[np.ndarray | None] = [None] * max_persons
    unused = set(range(len(candidates)))

    # Preserve existing identities where a previous pose is available.
    for slot_index, previous in enumerate(previous_slots):
        if previous is None or not unused:
            continue
        best = min(unused, key=lambda i: pose_distance(previous, candidates[i]))
        distance = pose_distance(previous, candidates[best])
        if np.isfinite(distance):
            slots[slot_index] = candidates[best]
            unused.remove(best)

    # Fill new/empty slots by pose confidence.
    for slot_index in range(max_persons):
        if slots[slot_index] is not None or not unused:
            continue
        best = max(unused, key=lambda i: float(candidates[i][:, 2].sum()))
        slots[slot_index] = candidates[best]
        unused.remove(best)

    return slots


def build_skeleton(
    json_files: Iterable[Path],
    width: int,
    height: int,
    max_persons: int,
) -> np.ndarray:
    files = list(json_files)
    data = np.zeros(
        (CHANNELS, len(files), NUM_JOINTS, max_persons),
        dtype=np.float32,
    )
    previous_slots: list[np.ndarray | None] = [None] * max_persons
    detected_frames = 0

    for frame_index, json_path in enumerate(files):
        candidates = load_people(json_path, width, height)
        if not candidates:
            # Keep zero padding for a frame with no detected person.
            continue

        slots = assign_people(candidates, previous_slots, max_persons)
        found_person = False
        for person_index, pose in enumerate(slots):
            if pose is None:
                continue
            data[:, frame_index, :, person_index] = pose.T
            found_person = True

        if found_person:
            detected_frames += 1
            previous_slots = [
                pose.copy() if pose is not None else previous_slots[index]
                for index, pose in enumerate(slots)
            ]

    if detected_frames == 0:
        raise RuntimeError(
            "OpenPose JSON files were found, but no valid COCO-18 body pose was detected."
        )

    print(f"Frames containing at least one pose: {detected_frames}/{len(files)}")
    return data


def load_labels(path: Path) -> list[str]:
    require_file(path, "Kinetics label file")
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) != 400:
        raise RuntimeError(f"Expected 400 Kinetics labels, found {len(labels)}")
    return labels


def load_checkpoint(model: Model, path: Path) -> None:
    require_file(path, "ST-GCN Kinetics checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint object: {type(checkpoint).__name__}"
        )

    state_dict = OrderedDict(
        (key.removeprefix("module."), value)
        for key, value in checkpoint.items()
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not exactly match the ST-GCN Kinetics model.\n"
            f"Missing keys: {missing}\nUnexpected keys: {unexpected}"
        )


def main() -> int:
    args = parse_args()

    if not args.json_dir.is_dir():
        raise NotADirectoryError(
            f"OpenPose JSON directory not found: {args.json_dir.resolve()}"
        )

    json_files = sorted(args.json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files were found in: {args.json_dir.resolve()}"
        )

    json_files = choose_frame_files(json_files, args.max_frames)
    width, height = video_size(args.video)

    print(f"Video dimensions: {width}x{height}")
    print(f"JSON frames used: {len(json_files)}")

    data = build_skeleton(
        json_files=json_files,
        width=width,
        height=height,
        max_persons=args.max_persons,
    )
    print(f"Skeleton shape (C,T,V,M): {tuple(data.shape)}")

    if args.save_npy is not None:
        args.save_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_npy, data)
        print(f"Saved skeleton tensor: {args.save_npy.resolve()}")

    labels = load_labels(args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(
        in_channels=3,
        num_class=400,
        graph_args={"layout": "openpose", "strategy": "spatial"},
        edge_importance_weighting=True,
    )
    load_checkpoint(model, args.weights)
    model = model.to(device)
    model.eval()

    tensor = torch.from_numpy(data).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(
            probabilities,
            k=min(args.topk, probabilities.numel()),
        )

    print(f"Device: {device}")
    print("\nTop predictions:")
    for rank, (value, index) in enumerate(
        zip(values.tolist(), indices.tolist()),
        start=1,
    ):
        print(f"{rank:>2}. {labels[index]:<45} {value * 100:7.3f}%")

    print(
        "\nNote: low probabilities or semantically odd labels can be caused by "
        "pose-estimation errors, short/ambiguous clips, or the mismatch between "
        "your video and the Kinetics-Skeleton training distribution."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
