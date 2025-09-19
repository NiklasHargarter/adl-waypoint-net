import os
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import tomllib
import tomli_w
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from external.transfuser.team_code_transfuser.config import GlobalConfig
from external.transfuser.team_code_transfuser.data import CARLA_Data
from src.waypoint_net import MultimodalWaypointNet
from src.dataset_wrapper import ImagenetNormWrapper

# Module-level constants for data paths
ROOT_DIR = "data/small/"
TEST_DIR = os.path.join(ROOT_DIR, "test")


def get_short_model_name(model_name, info_dict):
    """Convert full model name to short display name."""
    variant = info_dict.get("hyperparameters", {}).get("variant", "unknown")
    if variant == "fullretrain":
        return "Full Retrain"
    elif variant == "convadapter":
        return "ConvAdapter"
    elif variant == "lora":
        return "LoRA"
    elif variant == "dora":
        return "DoRA"
    else:
        return variant.upper()


def create_model_from_path(model_path, device):
    """Create the appropriate model based on the model path and info file."""
    base_name = os.path.basename(model_path).replace(".pth", "")
    info_path = os.path.join(os.path.dirname(model_path), f"{base_name}_info.toml")

    # Load existing info if available
    info_dict = {}
    if os.path.exists(info_path):
        try:
            with open(info_path, "rb") as f:
                info_dict = tomllib.load(f)
        except Exception as e:
            print(f"Warning: Could not load info file {info_path}: {e}")

    cfg = GlobalConfig(root_dir=ROOT_DIR)

    # Determine model type from name or info
    model_name = info_dict.get("metadata", {}).get(
        "model_name", base_name.split("_")[0]
    )
    variant = info_dict.get("hyperparameters", {}).get("variant", "normal")
    if variant == "fullretrain":
        variant = "normal"

    print(f"Creating ResNetGRUWaypointNet model (variant: {variant}) for {base_name}")

    model = MultimodalWaypointNet(
        config=cfg,
        backbone_variant=variant,
        adapter_dim=64,
        adapter_stages="late",
    ).to(device)

    return model, info_dict, info_path


def compute_ade_fde(pred, target):
    # pred/target: [B, T, 2]
    diff = pred - target
    dists = torch.linalg.norm(diff, dim=-1)  # [B, T]
    ade = dists.mean(dim=1)  # [B]
    fde = dists[:, -1]  # [B]
    return ade, fde


def save_rgb_image(rgb_tensor, out_path):
    """Save RGB image from tensor [3, H, W] or [H, W, 3]"""
    # rgb_tensor: [3, H, W] normalized with ImageNet mean/std by ImagenetNormWrapper
    if rgb_tensor.shape[0] == 3:
        # Convert from [3, H, W] to [H, W, 3]
        rgb_img = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        rgb_img = rgb_tensor.cpu().numpy()

    # Denormalize from ImageNet normalization to [0, 1] range
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    rgb_img = (rgb_img * imagenet_std) + imagenet_mean

    # Clip to [0, 1] range
    rgb_img = np.clip(rgb_img, 0, 1)

    # Scale to [0, 255] and convert to uint8
    rgb_img = (rgb_img * 255.0).astype(np.uint8)

    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_img)
    plt.axis("off")
    plt.title("RGB Input Image")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_ego_plot(p0, t0, out_path, short_name):
    # p0, t0: [T, 2] in meters (ego frame)
    plt.figure(figsize=(8, 6))
    plt.plot(t0[:, 0], t0[:, 1], "g-o", label="GT", linewidth=2, markersize=6)
    plt.plot(p0[:, 0], p0[:, 1], "r-o", label="Pred", linewidth=2, markersize=6)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)

    # Add padding to prevent squished/overlapping plots
    all_x = np.concatenate([t0[:, 0], p0[:, 0]])
    all_y = np.concatenate([t0[:, 1], p0[:, 1]])
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    margin = max(0.5, 0.1 * max(x_range, y_range))  # at least 0.5m, or 10% of range
    plt.xlim(all_x.min() - margin, all_x.max() + margin)
    plt.ylim(all_y.min() - margin, all_y.max() + margin)

    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.title(f"Ego-frame waypoints - {short_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_combined_ego_plot(all_predictions, target, rgb_tensor, out_path, short_names):
    """Save a combined plot showing RGB image and predictions from all models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left subplot: RGB image
    rgb_img = rgb_tensor.permute(1, 2, 0).cpu().numpy()

    # Denormalize from ImageNet normalization to [0, 1] range
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    rgb_img = (rgb_img * imagenet_std) + imagenet_mean

    # Clip to [0, 1] range
    rgb_img = np.clip(rgb_img, 0, 1)

    # Scale to [0, 255] and convert to uint8
    rgb_img = (rgb_img * 255.0).astype(np.uint8)

    ax1.imshow(rgb_img)
    ax1.axis("off")
    ax1.set_title("RGB Input Image")

    # Right subplot: Ego trajectory plot
    # Plot ground truth
    ax2.plot(target[:, 0], target[:, 1], "k-o", label="GT", linewidth=3, markersize=8)

    # Define colors for different models
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]

    for i, (pred, short_name) in enumerate(zip(all_predictions, short_names)):
        color = colors[i % len(colors)]
        ax2.plot(
            pred[:, 0],
            pred[:, 1],
            color=color,
            marker="o",
            label=f"{short_name}",
            linewidth=2,
            markersize=6,
        )

    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.axvline(0, color="k", linewidth=0.5)

    # Add padding to prevent squished/overlapping plots
    all_x = [target[:, 0]]
    all_y = [target[:, 1]]
    for pred in all_predictions:
        all_x.append(pred[:, 0])
        all_y.append(pred[:, 1])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    margin = max(0.5, 0.1 * max(x_range, y_range))  # at least 0.5m, or 10% of range
    ax2.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax2.set_ylim(all_y.min() - margin, all_y.max() + margin)

    ax2.set_aspect("equal", "box")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.set_title("Model Comparison - Ego-frame Waypoints")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = GlobalConfig(root_dir=TEST_DIR)

    # Data
    test_set = CARLA_Data(root=[TEST_DIR], config=cfg)

    # Wrap with ImageNet normalization
    test_set = ImagenetNormWrapper(test_set)
    test_loader = DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )

    # Get all model checkpoints
    models_dir = "models"
    pth_files = glob.glob(os.path.join(models_dir, "*.pth"))

    # Load all models first
    models_info = []
    for ckpt_path in pth_files:
        base_name = os.path.basename(ckpt_path).replace(".pth", "")

        print(f"Loading model: {base_name}")

        # Load model using proper configuration from info file
        model, info_dict, info_path = create_model_from_path(ckpt_path, device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model = model.to(memory_format=torch.channels_last)
        model.eval()

        models_info.append(
            {
                "model": model,
                "name": base_name,
                "short_name": get_short_model_name(base_name, info_dict),
                "info_dict": info_dict,
                "info_path": info_path,
            }
        )

    # Create directories
    viz_combined_dir = "viz_combined"
    viz_rgb_dir = "viz_rgb"
    os.makedirs(viz_combined_dir, exist_ok=True)
    os.makedirs(viz_rgb_dir, exist_ok=True)

    # Create individual model directories
    for model_info in models_info:
        viz_ego_dir = f"viz_ego_{model_info['short_name'].replace(' ', '_').lower()}"
        os.makedirs(viz_ego_dir, exist_ok=True)

    # Collect metrics for each model across all batches
    model_metrics = {
        model_info["name"]: {"ade": [], "fde": []} for model_info in models_info
    }

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            rgb = (
                batch["rgb"]
                .to(device, non_blocking=True)
                .to(memory_format=torch.channels_last)
                .float()
            )
            lidar = (
                batch["lidar"]
                .to(device, non_blocking=True)
                .to(memory_format=torch.channels_last)
                .float()
            )
            target = (
                batch["ego_waypoint"].to(device, non_blocking=True).float()
            )  # [B, T, 2]

            # Load target point (required for ResNetGRUWaypointNet)
            target_point = (
                batch["target_point"].to(device, non_blocking=True).float()
            )  # [B, 2]

            # Run inference on all models
            all_predictions = []

            for j, model_info in enumerate(models_info):
                model = model_info["model"]

                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=(device.type == "cuda"),
                ):
                    pred = model(rgb, lidar, target_point)  # [B, T, 2]

                ade, fde = compute_ade_fde(pred, target)
                model_metrics[model_info["name"]]["ade"].append(ade.cpu().numpy())
                model_metrics[model_info["name"]]["fde"].append(fde.cpu().numpy())

                # Store prediction for first sample in batch
                p0 = pred[0].detach().cpu().numpy()  # [T, 2]
                all_predictions.append(p0)

                # Save individual model plot
                t0 = target[0].detach().cpu().numpy()  # [T, 2]
                viz_ego_dir = (
                    f"viz_ego_{model_info['short_name'].replace(' ', '_').lower()}"
                )
                save_ego_plot(
                    p0,
                    t0,
                    os.path.join(viz_ego_dir, f"ego_{i:06d}.png"),
                    model_info["short_name"],
                )

            # Save combined plot
            t0 = target[0].detach().cpu().numpy()  # [T, 2]
            rgb_img = batch["rgb"][0].detach().cpu()  # [3, H, W]
            short_names = [model_info["short_name"] for model_info in models_info]
            save_combined_ego_plot(
                all_predictions,
                t0,
                rgb_img,
                os.path.join(viz_combined_dir, f"combined_{i:06d}.png"),
                short_names,
            )

            # Save RGB image
            rgb_img = batch["rgb"][0].detach().cpu()  # [3, H, W]
            save_rgb_image(rgb_img, os.path.join(viz_rgb_dir, f"rgb_{i:06d}.png"))

    # Calculate final metrics for each model
    results = {}
    for i, model_info in enumerate(models_info):
        model_name = model_info["name"]
        metrics = model_metrics[model_name]
        all_ade = np.concatenate(metrics["ade"])
        all_fde = np.concatenate(metrics["fde"])
        ade_mean = float(all_ade.mean())
        fde_mean = float(all_fde.mean())
        results[model_name] = {"ADE": ade_mean, "FDE": fde_mean}
        short_name = get_short_model_name(model_name, model_info["info_dict"])
        print(f"{short_name} | ADE: {ade_mean:.4f} | FDE: {fde_mean:.4f}")

        # Save test results to info file
        info_dict = model_info["info_dict"]
        info_path = model_info["info_path"]
        if "test_results" not in info_dict:
            info_dict["test_results"] = {}
        info_dict["test_results"]["ade"] = ade_mean
        info_dict["test_results"]["fde"] = fde_mean

        with open(info_path, "wb") as f:
            tomli_w.dump(info_dict, f)
        print(f"   Test results saved to {info_path}")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'ADE':<10} {'FDE':<10}")
    print("-" * 60)
    for model_name, metrics in results.items():
        short_name = get_short_model_name(
            model_name,
            next(
                model_info["info_dict"]
                for model_info in models_info
                if model_info["name"] == model_name
            ),
        )
        print(f"{short_name:<15} {metrics['ADE']:<10.4f} {metrics['FDE']:<10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
