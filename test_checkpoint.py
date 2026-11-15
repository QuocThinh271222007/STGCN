from collections import OrderedDict
from pathlib import Path

import torch

from net.st_gcn import Model


CHECKPOINT_PATH = Path("models/st_gcn.ntu-xsub.pt")


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {CHECKPOINT_PATH.resolve()}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(
        in_channels=3,
        num_class=60,
        graph_args={
            "layout": "ntu-rgb+d",
            "strategy": "spatial",
        },
        edge_importance_weighting=True,
        dropout=0.5,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    # Một số checkpoint có thể bọc state_dict trong dictionary.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Xóa tiền tố "module." nếu checkpoint được lưu bằng DataParallel.
    clean_state_dict = OrderedDict()

    for key, value in checkpoint.items():
        clean_key = key.removeprefix("module.")
        clean_state_dict[clean_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        clean_state_dict,
        strict=False,
    )

    print(f"Missing keys    : {missing_keys}")
    print(f"Unexpected keys : {unexpected_keys}")

    model = model.to(device)
    model.eval()

    skeleton = torch.randn(
        1,
        3,
        64,
        25,
        2,
        device=device,
    )

    with torch.inference_mode():
        output = model(skeleton)

    print(f"Device          : {device}")
    print(f"Output shape    : {tuple(output.shape)}")
    print(f"Predicted index : {output.argmax(dim=1).item()}")
    print("Checkpoint load : SUCCESS")


if __name__ == "__main__":
    main()
