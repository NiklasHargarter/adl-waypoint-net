import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from external.transfuser.team_code_transfuser.config import GlobalConfig
from external.transfuser.team_code_transfuser.data import CARLA_Data


def visualize_sample(sample, idx):
    """Visualize a single sample from the dataset."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Dataset Sample {idx}", fontsize=16)

        # RGB Image
    rgb_img = sample['rgb'].squeeze(0)  # Remove batch dimension (1, C, H, W) -> (C, H, W)
    rgb_img = rgb_img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    rgb_img = rgb_img / 255.0  # Normalize to [0, 1] for matplotlib
    rgb_img = np.clip(rgb_img, 0, 1)
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')

        # BEV (Bird's Eye View)
    bev_img = sample['bev'].squeeze(0).numpy()
    axes[0, 1].imshow(bev_img, cmap='tab10')
    axes[0, 1].set_title('BEV Map')
    axes[0, 1].axis('off')

    # Depth
    depth_img = sample['depth'].squeeze(0)
    axes[0, 2].imshow(depth_img, cmap='plasma')
    axes[0, 2].set_title('Depth Map')
    axes[0, 2].axis('off')

    # Semantic Segmentation
    semantic_img = sample['semantic'].squeeze(0)
    axes[0, 3].imshow(semantic_img, cmap='tab20')
    axes[0, 3].set_title('Semantic Segmentation')
    axes[0, 3].axis('off')

    # LiDAR Histogram (above ground)
    lidar_above = sample['lidar'].squeeze(0)[0].numpy()  # Above ground channel
    axes[1, 0].imshow(lidar_above, cmap='viridis')
    axes[1, 0].set_title('LiDAR Above Ground')
    axes[1, 0].axis('off')

    # LiDAR Histogram (below ground)
    lidar_below = sample['lidar'].squeeze(0)[1].numpy()  # Below ground channel
    axes[1, 1].imshow(lidar_below, cmap='viridis')
    axes[1, 1].set_title('LiDAR Below Ground')
    axes[1, 1].axis('off')

    # Target Point
    target_point_img = sample['target_point_image'].squeeze(0).squeeze()
    axes[1, 2].imshow(target_point_img, cmap='gray')
    axes[1, 2].set_title('Target Point')
    axes[1, 2].axis('off')

    # Ego Waypoints
    waypoints = sample['ego_waypoint'].squeeze(0).numpy()
    axes[1, 3].plot(waypoints[:, 0], waypoints[:, 1], "r-o", markersize=4)
    axes[1, 3].axhline(0, color="k", linewidth=0.5)
    axes[1, 3].axvline(0, color="k", linewidth=0.5)

    # Add padding to prevent cramped plot
    all_x = waypoints[:, 0]
    all_y = waypoints[:, 1]
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    margin = max(0.5, 0.1 * max(x_range, y_range))  # at least 0.5m, or 10% of range
    axes[1, 3].set_xlim(all_x.min() - margin, all_x.max() + margin)
    axes[1, 3].set_ylim(all_y.min() - margin, all_y.max() + margin)

    axes[1, 3].set_aspect("equal")
    axes[1, 3].set_title("Ego Waypoints")
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].set_xlabel("x (m)")
    axes[1, 3].set_ylabel("y (m)")

    plt.tight_layout()
    return fig


def main():
    # Configuration - use the same root_dir as training script
    root_dir = "data/small"
    cfg = GlobalConfig(root_dir=root_dir)

    # Use data/small/train
    data_root = "data/small/train"

    # Create dataset with correct format (list of paths)
    dataset = CARLA_Data(root=[data_root], config=cfg)

    print(f"Dataset size: {len(dataset)}")

    # Create data loader with small batch size (no workers for visualization)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)    # Create output directory
    output_dir = "viz_dataset"
    os.makedirs(output_dir, exist_ok=True)

    # Visualize first few samples
    num_samples_to_visualize = 5

    for i, sample in enumerate(dataloader):
        if i >= num_samples_to_visualize:
            break

        fig = visualize_sample(sample, i)
        plt.savefig(
            os.path.join(output_dir, f"dataset_sample_{i}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"Saved visualization for sample {i}")

    print("Dataset visualization complete!")


if __name__ == "__main__":
    main()
