import os
import json
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
from pretrain_pos import ConvNeXtPosition


def test():
    parser = argparse.ArgumentParser(
        description="Test ConvNeXtPosition on a dataset — reports 3D position error"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained_convnext_pos.pth",
        help="Path to .pth weights file (default: pretrained_convnext_pos.pth)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Dataset directory containing labels.json and images (default: dataset)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of random samples to evaluate (default: 20)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading ConvNeXtPosition weights from: {args.weights}")
    model = ConvNeXtPosition().to(device)
    if not os.path.exists(args.weights):
        print(f"Error: weights file not found: {args.weights}")
        return
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # ── Load labels ───────────────────────────────────────────────────────────
    label_path = os.path.join(args.dataset, "labels.json")
    if not os.path.exists(label_path):
        print(f"Error: {label_path} not found. Did you run collect_pretrain_data.py?")
        return
    with open(label_path) as f:
        labels = json.load(f)

    n = min(args.n, len(labels))
    indices = np.random.choice(len(labels), n, replace=False)
    print(f"Evaluating on {n} random samples from {args.dataset}...\n")

    out_dir = "test_results_pos"
    os.makedirs(out_dir, exist_ok=True)

    errors_cm = []

    for i, idx in enumerate(indices):
        item   = labels[idx]
        img_pil = Image.open(os.path.join(args.dataset, item["rgb"])).convert("RGB")

        # Prepare tensor
        img_arr     = np.array(img_pil).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            pred_pos = model(input_tensor).squeeze(0).cpu().numpy()   # (3,)

        gt_pos  = np.array(item["pose_6d"][:3], dtype=np.float32)
        err_cm  = np.linalg.norm(pred_pos - gt_pos) * 100.0
        errors_cm.append(err_cm)

        per_axis_cm = np.abs(pred_pos - gt_pos) * 100.0

        print(
            f"Sample {i:02d} | "
            f"GT=({gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f}) m | "
            f"Pred=({pred_pos[0]:.3f}, {pred_pos[1]:.3f}, {pred_pos[2]:.3f}) m | "
            f"Err={err_cm:.1f} cm  (dx={per_axis_cm[0]:.1f} dy={per_axis_cm[1]:.1f} dz={per_axis_cm[2]:.1f})"
        )

        # Annotate image
        draw = ImageDraw.Draw(img_pil)
        draw.text((4,  4), f"GT  : ({gt_pos[0]:.2f}, {gt_pos[1]:.2f}, {gt_pos[2]:.2f}) m",  fill=(0, 255, 0))
        draw.text((4, 16), f"Pred: ({pred_pos[0]:.2f}, {pred_pos[1]:.2f}, {pred_pos[2]:.2f}) m", fill=(255, 80, 80))
        draw.text((4, 28), f"Err : {err_cm:.1f} cm",                                          fill=(255, 255, 255))
        img_pil.save(os.path.join(out_dir, f"test_{i:02d}.png"))

    errors_cm = np.array(errors_cm)
    print(f"\n── Summary ({n} samples) ──────────────────────────────")
    print(f"  Mean error : {errors_cm.mean():.2f} cm")
    print(f"  Median     : {np.median(errors_cm):.2f} cm")
    print(f"  Std        : {errors_cm.std():.2f} cm")
    print(f"  Max        : {errors_cm.max():.2f} cm")
    print(f"  < 2 cm     : {(errors_cm < 2).mean()*100:.1f}%")
    print(f"  < 5 cm     : {(errors_cm < 5).mean()*100:.1f}%")
    print(f"\nAnnotated images saved to {out_dir}/")


if __name__ == "__main__":
    test()
