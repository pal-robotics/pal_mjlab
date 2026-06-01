import os
import json
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
from pretrain import PolicyCNNBackbone, ConvNeXtBackbone

def test():
    parser = argparse.ArgumentParser(description="Test trained dual-head backbone on RGB dataset")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["cnn", "convnext"],
        default="cnn",
        help="Visual backbone architecture type (default: cnn)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Dataset directory to load validation labels and images from"
    )
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading trained model ({args.backbone.upper()})...")
    if args.backbone == "cnn":
        model = PolicyCNNBackbone(num_keypoints=6).to(device)
        model_path = "pretrained_backbone.pth"
    else:
        model = ConvNeXtBackbone(num_keypoints=6).to(device)
        model_path = "pretrained_convnext.pth"
        
    if not os.path.exists(model_path):
        print(f"Error: Trained model file {model_path} not found. Please train the backbone first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Load Dataset labels
    label_path = os.path.join(args.dataset, "labels.json")
    if not os.path.exists(label_path):
        print(f"Error: {label_path} not found. Did you run collect_pretrain_data.py?")
        return

    with open(label_path, "r") as f:
        labels = json.load(f)
        
    output_dir = f"test_results_{args.backbone}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Test on 20 random samples
    indices = np.random.choice(len(labels), min(20, len(labels)), replace=False)
    print(f"Running inference on {len(indices)} samples...")
    
    total_pos_err = 0.0
    total_rot_err = 0.0
    
    for i, idx in enumerate(indices):
        item = labels[idx]
        rgb_path = os.path.join(args.dataset, item["rgb"])
        img_pil = Image.open(rgb_path)
        
        # Prepare input tensor
        img_arr = np.array(img_pil).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor) # Output shape: (1, 18)
            
        pred_kps_norm = output[0, :12]
        pred_pose = output[0, 12:].cpu().numpy()
        
        # Denormalize output [-1, 1] -> [0, 128] pixels
        # Flip back coordinates from [v, u] (row, col) back to [u, v] (col, row) for visualization
        preds = pred_kps_norm.cpu().numpy().reshape(-1, 2)
        preds = (preds + 1.0) * 64.0
        preds = preds[:, [1, 0]] # Flip back to [u, v]
        
        gt_kps = np.array(item["keypoints"])
        gt_pose = np.array(item["pose_6d"])
        
        # Calculate Errors
        pos_err = np.linalg.norm(pred_pose[:3] - gt_pose[:3]) # in meters
        # Mean absolute angle error in degrees
        rot_err_rad = np.abs(pred_pose[3:] - gt_pose[3:])
        # Handle wrap-around for euler angles
        rot_err_rad = np.minimum(rot_err_rad, 2 * np.pi - rot_err_rad)
        rot_err = np.mean(rot_err_rad) * 180.0 / np.pi
        
        total_pos_err += pos_err
        total_rot_err += rot_err
        
        print(f"Sample {i:02d} | Position Error: {pos_err*100:.2f} cm | Orientation Error: {rot_err:.2f}°")
        
        # Draw on RGB Image
        draw = ImageDraw.Draw(img_pil)
        
        # Draw Ground Truth keypoints (Green dots)
        for u, v in gt_kps:
            r = 1.5
            draw.ellipse([u-r, v-r, u+r, v+r], fill=(0, 255, 0))
            
        # Draw Predictions (Red crosses)
        for u, v in preds:
            r = 2.5
            draw.line([u-r, v-r, u+r, v+r], fill=(255, 0, 0), width=1)
            draw.line([u-r, v+r, u+r, v-r], fill=(255, 0, 0), width=1)
            
        # Add labels
        draw.text((5, 5), f"Pos Err: {pos_err*100:.1f}cm", fill=(255, 255, 255))
        draw.text((5, 15), f"Ori Err: {rot_err:.1f} deg", fill=(255, 255, 255))
            
        img_pil.save(os.path.join(output_dir, f"test_{i:02d}.png"))
        
    mean_pos_err = (total_pos_err / len(indices)) * 100.0
    mean_rot_err = total_rot_err / len(indices)
    print(f"\nEvaluation Summary ({args.backbone.upper()}):")
    print(f"  Mean Translation Error: {mean_pos_err:.2f} cm")
    print(f"  Mean Rotation Error:    {mean_rot_err:.2f}°")
    print(f"Saved visualization results to {output_dir}/")

if __name__ == "__main__":
    test()
