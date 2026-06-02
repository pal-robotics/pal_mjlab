import os
import json
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
from pretrain_kp_pos import ConvNeXtKPPos, PolicyCNNKPPos


def test():
    parser = argparse.ArgumentParser(
        description="Test Dual-Head Model (4 Keypoints + 3D Position) on Dataset"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["cnn", "convnext"],
        default="convnext",
        help="Visual backbone architecture type (default: convnext)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to .pth weights file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Dataset directory to load validation labels and images from"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of random samples to evaluate"
    )
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ── Resolve default weights path ──
    if args.weights is None:
        args.weights = f"pretrained_{args.backbone}_kp_pos.pth"

    print(f"Loading {args.backbone.upper()} model weights from: {args.weights}")
    if args.backbone == "cnn":
        model = PolicyCNNKPPos(num_keypoints=4).to(device)
    else:
        model = ConvNeXtKPPos(num_keypoints=4).to(device)
        
    if not os.path.exists(args.weights):
        print(f"Error: Weights file {args.weights} not found.")
        return
        
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    
    # ── Load labels ──
    label_path = os.path.join(args.dataset, "labels.json")
    if not os.path.exists(label_path):
        print(f"Error: {label_path} not found. Did you run collect_pretrain_data.py?")
        return

    with open(label_path, "r") as f:
        labels = json.load(f)
        
    output_dir = f"test_results_{args.backbone}_kp_pos"
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = min(args.n, len(labels))
    indices = np.random.choice(len(labels), n_samples, replace=False)
    print(f"Running inference on {n_samples} random samples...")
    
    total_pos_err = 0.0
    total_kp_err_pixels = 0.0
    
    for i, idx in enumerate(indices):
        item = labels[idx]
        rgb_path = os.path.join(args.dataset, item["rgb"])
        img_pil = Image.open(rgb_path).convert("RGB")
        
        # Prepare input tensor
        img_arr = np.array(img_pil).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor) # Shape: (1, 11)
            
        pred_kps_norm = output[0, :8]
        pred_pos = output[0, 8:].cpu().numpy()
        
        # Denormalize output keypoints [-1, 1] -> [0, 128] pixels
        # Flip back coordinates from [v, u] (row, col) back to [u, v] (col, row) for visualization
        preds = pred_kps_norm.cpu().numpy().reshape(-1, 2)
        preds = (preds + 1.0) * 64.0
        preds = preds[:, [1, 0]] # Flip back to [u, v]
        
        gt_kps = np.array(item["keypoints"][:4])
        gt_pos = np.array(item["pose_6d"][:3])
        
        # Calculate Position Error (meters -> cm)
        pos_err = np.linalg.norm(pred_pos - gt_pos) * 100.0 # in cm
        total_pos_err += pos_err
        
        # Calculate Keypoints Error (pixels)
        kp_err = np.mean(np.linalg.norm(preds - gt_kps, axis=1))
        total_kp_err_pixels += kp_err
        
        per_axis_cm = np.abs(pred_pos - gt_pos) * 100.0
        
        print(
            f"Sample {i:02d} | "
            f"Pos Err: {pos_err:.2f} cm (dx={per_axis_cm[0]:.1f} dy={per_axis_cm[1]:.1f} dz={per_axis_cm[2]:.1f}) | "
            f"KP Err: {kp_err:.2f} px"
        )
        
        # Draw on image
        draw = ImageDraw.Draw(img_pil)
        
        # Draw GT keypoints (Green circles)
        for u, v in gt_kps:
            r = 1.5
            draw.ellipse([u-r, v-r, u+r, v+r], fill=(0, 255, 0))
            
        # Draw Predicted keypoints (Red crosses)
        for u, v in preds:
            r = 2.5
            draw.line([u-r, v-r, u+r, v+r], fill=(255, 80, 80), width=1)
            draw.line([u-r, v+r, u+r, v-r], fill=(255, 80, 80), width=1)
            
        # Add labels
        draw.text((4, 4), f"Pos Err: {pos_err:.1f} cm", fill=(255, 255, 255))
        draw.text((4, 16), f"KP Err : {kp_err:.1f} px", fill=(255, 255, 255))
            
        img_pil.save(os.path.join(output_dir, f"test_{i:02d}.png"))
        
    mean_pos_err = total_pos_err / n_samples
    mean_kp_err = total_kp_err_pixels / n_samples
    print(f"\nEvaluation Summary ({args.backbone.upper()}):")
    print(f"  Mean Translation Error: {mean_pos_err:.2f} cm")
    print(f"  Mean Keypoint Error:    {mean_kp_err:.2f} px")
    print(f"Saved visualization results to {output_dir}/")


if __name__ == "__main__":
    test()
