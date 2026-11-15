import sys
import traceback

import torch

from net.st_gcn import Model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("ST-GCN SMOKE TEST")
    print("=" * 60)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"Selected device : {device}")

    if torch.cuda.is_available():
        print(f"GPU name        : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU VRAM        : {total_vram / 1024**3:.2f} GB")

    # Mô hình dùng cấu trúc NTU RGB+D:
    # 3 channel tọa độ x, y, z
    # 25 khớp
    # 60 lớp hành động
    model = Model(
        in_channels=3,
        num_class=60,
        graph_args={
            "layout": "ntu-rgb+d",
            "strategy": "spatial",
        },
        edge_importance_weighting=True,
        dropout=0.5,
    ).to(device)

    model.eval()

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"Parameters      : {parameter_count:,}")

    # Tensor skeleton giả:
    # N = 1 mẫu
    # C = 3 tọa độ x, y, z
    # T = 64 frame
    # V = 25 khớp
    # M = tối đa 2 người
    skeleton = torch.randn(
        1,
        3,
        64,
        25,
        2,
        device=device,
    )

    print(f"Input shape     : {tuple(skeleton.shape)}")

    with torch.inference_mode():
        output = model(skeleton)

    print(f"Output shape    : {tuple(output.shape)}")
    print(f"Predicted index : {output.argmax(dim=1).item()}")
    print("ST-GCN forward pass: SUCCESS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ST-GCN forward pass: FAILED")
        traceback.print_exc()
        sys.exit(1)
