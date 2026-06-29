import os
import json
import numpy as np
import pandas as pd

def main():
    scratch_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(scratch_dir, "occlusion_joint_data.csv")
    out_json = os.path.join(scratch_dir, "critical_configurations.json")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Run the collection script first.")
        return
        
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter only occluded configurations
    occluded_df = df[df["occluded"] == 1]
    total_occluded = len(occluded_df)
    
    if total_occluded == 0:
        print("Error: No occluded steps found in the data.")
        return
        
    print(f"Found {total_occluded} occluded steps.")
    
    # Focus only on the policy actuated joints: right arm joints and gripper_right_finger_joint
    active_joints = [
        "arm_right_1_joint",
        "arm_right_2_joint",
        "arm_right_3_joint",
        "arm_right_4_joint",
        "arm_right_5_joint",
        "arm_right_6_joint",
        "arm_right_7_joint",
        # "gripper_right_finger_joint"
    ]
    print(f"Clustering based on {len(active_joints)} active joints:")
    for j in active_joints:
        print(f"  - {j}")
        
    X = occluded_df[active_joints].values
    
    # Choose K = 50 (or max possible if total samples < 50)
    k = min(50, total_occluded)
    print(f"Running K-Means clustering with K = {k}...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # K-Means implementation
    # 1. Random initialization
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    # 2. Main loop
    max_iters = 150
    for iteration in range(max_iters):
        # Compute Euclidean distance from X to all centroids
        # X: (N, D), centroids: (K, D)
        # We can use broadcast subtraction: (N, 1, D) - (1, K, D) -> (N, K, D)
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2) # Shape: (N, K)
        labels = np.argmin(dists, axis=1)
        
        # Recompute centroids
        new_centroids = []
        for i in range(k):
            members = X[labels == i]
            if len(members) > 0:
                new_centroids.append(members.mean(axis=0))
            else:
                # Re-initialize to a random point if cluster becomes empty
                new_centroids.append(X[np.random.choice(X.shape[0])])
        
        new_centroids = np.array(new_centroids)
        
        # Check convergence
        if np.allclose(centroids, new_centroids, atol=1e-5):
            print(f"Converged at iteration {iteration}.")
            break
        centroids = new_centroids
    else:
        print("Reached max iterations without complete convergence.")
        
    # Recalculate final cluster sizes and labels
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    
    # Calculate sizes
    cluster_sizes = np.bincount(labels, minlength=k)
    
    # Sort cluster centroids by size in descending order (highest size = rank 1)
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    
    critical_configs = []
    print("\nTop 10 Critical Configurations (Ranked by Cluster Size):")
    print(f"{'Rank':<6} | {'Cluster Size':<12} | {'Percentage':<10} | {'Representative Joint Angles (first 4 joints)':<60}")
    print("-" * 100)
    
    for rank, idx in enumerate(sorted_indices, 1):
        size = int(cluster_sizes[idx])
        percentage = (size / total_occluded) * 100
        joint_values = centroids[idx].tolist()
        
        critical_configs.append({
            "rank": rank,
            "cluster_size": size,
            "percentage": percentage,
            "joints": joint_values
        })
        
        if rank <= 10:
            formatted_joints = ", ".join([f"{val:.3f}" for val in joint_values[:4]]) + " ..."
            print(f"{rank:<6} | {size:<12} | {percentage:<9.2f}% | {formatted_joints:<60}")
            
    # Save output to JSON
    output_data = {
        "active_joints": active_joints,
        "critical_configurations": critical_configs
    }
    
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nSuccessfully saved {len(critical_configs)} ranked critical configurations to {out_json}")

    # Also save to the package directory to keep it in sync for HPC runs
    pkg_json = os.path.abspath(os.path.join(scratch_dir, "..", "src", "pal_mjlab", "tasks", "manipulation", "mdp", "critical_configurations.json"))
    try:
        with open(pkg_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Also saved in-sync configurations to Python package directory: {pkg_json}")
    except Exception as e:
        print(f"Warning: could not save to package directory: {e}")

if __name__ == "__main__":
    main()
