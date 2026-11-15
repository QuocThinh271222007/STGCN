from collections import OrderedDict
from pathlib import Path

import torch

from net.st_gcn import Model


CHECKPOINT_PATH = Path("models/st_gcn.kinetics.pt")


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {CHECKPOINT_PATH.resolve()}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Kinetics-Skeleton:
    # 3 channels: x, y, confidence
    # 18 OpenPose COCO joints
    # 400 action classes
    model = Model(
        in_channels=3,
        num_class=400,
        graph_args={
            "layout": "openpose",
            "strategy": "spatial",
        },
        edge_importance_weighting=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    clean_state_dict = OrderedDict()

    for key, value in checkpoint.items():
        clean_key = key.split("module.")[-1]
        clean_state_dict[clean_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        clean_state_dict,
        strict=False,
    )

    print(f"Missing keys    : {missing_keys}")
    print(f"Unexpected keys : {unexpected_keys}")

    model = model.to(device)
    model.eval()

    # N, C, T, V, M
    skeleton = torch.randn(
        1,      # một video
        3,      # x, y, confidence
        128,    # 128 frame
        18,     # 18 khớp OpenPose COCO
        2,      # tối đa hai người
        device=device,
    )

    with torch.inference_mode():
        output = model(skeleton)

    print(f"Device          : {device}")
    print(f"Input shape     : {tuple(skeleton.shape)}")
    print(f"Output shape    : {tuple(output.shape)}")
    print(f"Predicted index : {output.argmax(dim=1).item()}")
    print("Kinetics checkpoint: SUCCESS")


if __name__ == "__main__":
    main()
