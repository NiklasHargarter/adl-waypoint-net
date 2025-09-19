# train_resnet_variants.py
import datetime
import math
import os
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from external.transfuser.team_code_transfuser.config import GlobalConfig
from external.transfuser.team_code_transfuser.data import CARLA_Data
from src.dataset_wrapper import ImagenetNormWrapper
from src.waypoint_net import MultimodalWaypointNet

import tomli_w


# =====================
# Shared/default variables (used by all variants)
# =====================
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 16
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_ROOT_DIR = "data/full/"
DEFAULT_NUM_EPOCHS = 20
DEFAULT_TIME_LIMIT_SEC = 2 * 60 * 60  # 2 hours
HEAD_LR = 2e-4  # Used by all variants

BACKBONE_LR = 1e-4  # Used by fullretrain
DEFAULT_WEIGHT_DECAY = 0.01  # Used by fullretrain and as fallback

# =====================
# LoRA & DoRA specific
# =====================
ADAPTER_LR_LORA_DORA = 5e-4
ADAPTER_WEIGHT_DECAY = 0.0

# =====================
# ConvAdapter specific
# =====================
ADAPTER_LR_CONV = 1e-3
CONV_ADAPTER_WEIGHT_DECAY = 1e-4
CONV_ADAPTER_MAX_NORM = 1.0
CONV_BASE_LR = 1e-3
CONV_FINAL_LR = 1e-5


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
    warmup_steps=-1,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def setup_data_loaders(
    config,
    root_dir=DEFAULT_ROOT_DIR,
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
    prefetch_factor=DEFAULT_PREFETCH_FACTOR,
):
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")

    # Create base datasets
    train_set = CARLA_Data(root=[train_dir], config=config)
    val_set = CARLA_Data(root=[val_dir], config=config)

    # Wrap with ImageNet normalization (uses standard ImageNet values internally)
    train_set = ImagenetNormWrapper(train_set)
    val_set = ImagenetNormWrapper(val_set)

    print(
        f"There are {len(train_set)} samples in training set, {len(val_set)} in validation set"
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )
    return train_loader, val_loader


def build_model(
    variant: str, config: GlobalConfig, device: torch.device
) -> MultimodalWaypointNet:
    """
    Build model for the specified variant.
    """
    # Model configuration per variant
    variant_configs = {
        "fullretrain": {
            "backbone_variant": "normal",
        },
        "lora": {
            "backbone_variant": "lora",
        },
        "dora": {
            "backbone_variant": "dora",
        },
        "convadapter": {
            "backbone_variant": "convadapter",
            "adapter_dim": 64,
            "adapter_stages": "late",
        },
    }

    if variant not in variant_configs:
        raise ValueError(
            f"Unknown variant: {variant}. Supported: {list(variant_configs.keys())}"
        )

    model = MultimodalWaypointNet(config=config, **variant_configs[variant])
    return model.to(device).to(memory_format=torch.channels_last)


def _get_parameter_groups(
    named_params: list[tuple[str, torch.nn.Parameter]],
    head_names: set,
    adapter_names: set = None,
) -> tuple[list, list, list]:
    """Extract parameter groups: backbone/adapters, heads, and other parameters."""
    backbone_or_adapter_params = []
    head_params = []
    other_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue

        if name in head_names:
            head_params.append(param)
        elif adapter_names and name in adapter_names:
            backbone_or_adapter_params.append(param)
        elif adapter_names is None and name.startswith("rgb_features."):
            backbone_or_adapter_params.append(param)
        else:
            other_params.append(param)

    return backbone_or_adapter_params, head_params, other_params


def param_groups_for_variant(model: torch.nn.Module, variant: str) -> list[dict]:
    """
    Build optimizer parameter groups with sensible defaults per variant.
    """
    named = list(model.named_parameters())

    def match(names: tuple[str, ...], require_grad=True):
        return [
            n
            for n, p in named
            if p.requires_grad == require_grad and any(k in n for k in names)
        ]

    # Common heads outside the backbone
    head_names = set(match(("fuse.", "gru.", "head.", "lidar_enc.")))

    # Variant-specific configurations
    variant_configs = {
        "fullretrain": {
            "adapter_names": None,
            "backbone_lr": BACKBONE_LR,
            "head_lr": HEAD_LR,
            "backbone_wd": DEFAULT_WEIGHT_DECAY,
            "head_wd": DEFAULT_WEIGHT_DECAY,
        },
        "lora": {
            "adapter_names": set(match(("rgb_features.",))),
            "backbone_lr": ADAPTER_LR_LORA_DORA,
            "head_lr": HEAD_LR,
            "backbone_wd": ADAPTER_WEIGHT_DECAY,
            "head_wd": DEFAULT_WEIGHT_DECAY,
        },
        "dora": {
            "adapter_names": set(match(("rgb_features.",))),
            "backbone_lr": ADAPTER_LR_LORA_DORA,
            "head_lr": HEAD_LR,
            "backbone_wd": ADAPTER_WEIGHT_DECAY,
            "head_wd": DEFAULT_WEIGHT_DECAY,
        },
        "convadapter": {
            "adapter_names": set(
                [n for n, p in named if p.requires_grad and "tuning_module" in n]
            ),
            "backbone_lr": ADAPTER_LR_CONV,
            "head_lr": HEAD_LR,
            "backbone_wd": CONV_ADAPTER_WEIGHT_DECAY,
            "head_wd": CONV_ADAPTER_WEIGHT_DECAY,
        },
    }

    if variant not in variant_configs:
        # Fallback: single group
        return [
            {
                "params": [p for _, p in named if p.requires_grad],
                "lr": BACKBONE_LR,
                "weight_decay": DEFAULT_WEIGHT_DECAY,
            }
        ]

    config = variant_configs[variant]
    backbone_params, head_params, other_params = _get_parameter_groups(
        named, head_names, config["adapter_names"]
    )

    groups = []
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": config["backbone_lr"],
                "weight_decay": config["backbone_wd"],
            }
        )
    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": config["head_lr"],
                "weight_decay": config["head_wd"],
            }
        )
    if other_params:
        groups.append(
            {
                "params": other_params,
                "lr": BACKBONE_LR,
                "weight_decay": DEFAULT_WEIGHT_DECAY,
            }
        )

    return groups


def _process_batch(batch, device):
    """Process a batch of data for training/validation."""
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
    target = batch["ego_waypoint"].to(device, non_blocking=True).float()
    target_point = batch["target_point"].to(device, non_blocking=True).float()
    return rgb, lidar, target, target_point


def train_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    device,
    epoch,
    num_epochs,
    variant="normal",
    max_norm=None,
):
    model.train()
    losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")

    for batch in pbar:
        rgb, lidar, target, target_point = _process_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(rgb, lidar, target_point)
            loss = F.mse_loss(pred, target)
        scaler.scale(loss).backward()

        # Gradient clipping for ConvAdapter (reference-compliant)
        if variant == "convadapter" and max_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.5f}"})

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate_epoch(model, val_loader, device, epoch, num_epochs):
    model.eval()
    losses = []
    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]")

    for batch in pbar:
        rgb, lidar, target, target_point = _process_batch(batch, device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(rgb, lidar, target_point)
            loss = F.mse_loss(pred, target)
        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.5f}"})

    return float(np.mean(losses)) if losses else 0.0


def save_model_and_info(
    model: torch.nn.Module,
    variant: str,
    start_timestamp: str,
    total_time: float,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
    num_epochs: int,
    root_dir: str,
    optimizer: torch.optim.Optimizer,
    epoch_done: int,
    avg_train_loss: float,
    avg_val_loss: float,
    epochs_hist: list[int],
    train_hist: list[float],
    val_hist: list[float],
):
    model_name = f"ResNetGRUWaypointNet_{variant}"
    os.makedirs("models", exist_ok=True)

    # Calculate parameters before potential adapter merging
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optionally merge LoRA/DoRA adapters for inference
    if variant in ("lora", "dora") and hasattr(model.rgb_features, "merge_and_unload"):
        try:
            model.rgb_features.merge_and_unload()
            print("[Info] Merged adapters into backbone weights for inference.")
        except Exception as e:
            print(f"[Warn] merge_and_unload failed: {e}")

    save_path = f"models/{model_name}_{start_timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    trainable_pct = 100.0 * trainable_params / max(1, total_params)

    info = {
        "metadata": {
            "model_name": model_name,
            "timestamp": start_timestamp,
            "total_training_time_seconds": total_time,
            "total_training_time_hours": total_time / 3600.0,
            "model_save_path": save_path,
        },
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rates": [g.get("lr", 0.0) for g in optimizer.param_groups],
            "weight_decays": [
                g.get("weight_decay", 0.0) for g in optimizer.param_groups
            ],
            "optimizer": type(optimizer).__name__,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "device": str(device),
            "num_epochs": num_epochs,
            "dataset_root": root_dir,
            "variant": variant,
            "cosine_lr_scheduling": variant == "convadapter",
            "gradient_clipping": variant == "convadapter",
        },
        "model_stats": {
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "trainable_percentage": float(trainable_pct),
        },
        "training_stats": {
            "epochs_completed": int(epoch_done),
            "final_train_loss": float(avg_train_loss),
            "final_val_loss": float(avg_val_loss),
            "epochs": [int(e) for e in epochs_hist],
            "train_losses": [float(x) for x in train_hist],
            "val_losses": [float(x) for x in val_hist],
        },
    }

    info_path = f"models/{model_name}_{start_timestamp}_info.toml"
    with open(info_path, "wb") as f:
        tomli_w.dump(info, f)
    print(f"Training info saved to {info_path}")


def train_one_variant(
    variant: str,
    config: GlobalConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    root_dir: str,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    time_limit_sec: int = DEFAULT_TIME_LIMIT_SEC,
):
    print(f"\n{'=' * 60}\nTraining variant: {variant.upper()}\n{'=' * 60}")
    model = build_model(variant, config, device)
    param_groups = param_groups_for_variant(model, variant)
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler(device=device)

    # ConvAdapter-specific settings (reference-compliant)
    use_cosine_lr = variant == "convadapter"
    lr_schedule_values = None
    max_norm = None

    if use_cosine_lr:
        # Reference-inspired: cosine LR with warmup, adapted for 20 epochs
        num_training_steps_per_epoch = len(train_loader)
        lr_schedule_values = cosine_scheduler(
            base_value=CONV_BASE_LR,  # Reference LR for adapters
            final_value=CONV_FINAL_LR,  # Slightly higher final LR for fewer epochs
            epochs=num_epochs,
            niter_per_ep=num_training_steps_per_epoch,
            warmup_epochs=0,  # Reference uses 0 warmup
        )
        max_norm = CONV_ADAPTER_MAX_NORM  # Reference uses optional gradient clipping
        print(
            f"Using cosine LR scheduler: {lr_schedule_values[0]:.6f} -> {lr_schedule_values[-1]:.6f}"
        )
        print(f"Using gradient clipping with max_norm={max_norm}")

    start = time.time()
    start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    epochs_hist, train_hist, val_hist = [], [], []
    for epoch in range(1, num_epochs + 1):
        # Update LR for cosine scheduling
        if use_cosine_lr and lr_schedule_values is not None:
            step = (epoch - 1) * len(train_loader)
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0:  # Adapter group gets cosine LR
                    param_group["lr"] = lr_schedule_values[
                        min(step, len(lr_schedule_values) - 1)
                    ]

        avg_train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            num_epochs,
            variant=variant,
            max_norm=max_norm,
        )
        avg_val_loss = validate_epoch(model, val_loader, device, epoch, num_epochs)
        print(
            f"Epoch {epoch}/{num_epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}"
        )

        epochs_hist.append(epoch)
        train_hist.append(avg_train_loss)
        val_hist.append(avg_val_loss)

        if time.time() - start > time_limit_sec:
            print("Time limit reached, stopping early.")
            break

    total_time = time.time() - start
    save_model_and_info(
        model=model,
        variant=variant,
        start_timestamp=start_timestamp,
        total_time=total_time,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        device=device,
        num_epochs=num_epochs,
        root_dir=root_dir,
        optimizer=optimizer,
        epoch_done=epochs_hist[-1] if epochs_hist else 0,
        avg_train_loss=train_hist[-1] if train_hist else 0.0,
        avg_val_loss=val_hist[-1] if val_hist else 0.0,
        epochs_hist=epochs_hist,
        train_hist=train_hist,
        val_hist=val_hist,
    )
    print(f"{'=' * 60}\n")


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)

    # Shared configuration
    root_dir = DEFAULT_ROOT_DIR
    config = GlobalConfig(root_dir=root_dir)
    batch_size = DEFAULT_BATCH_SIZE
    num_workers = DEFAULT_NUM_WORKERS
    prefetch_factor = DEFAULT_PREFETCH_FACTOR

    # Shared dataloaders with ImageNet-normalized RGB produced by wrapper
    train_loader, val_loader = setup_data_loaders(
        config=config,
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # Train all variants
    variants = ["fullretrain", "lora", "dora", "convadapter"]
    for variant in variants:
        train_one_variant(
            variant=variant,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            root_dir=root_dir,
            num_epochs=DEFAULT_NUM_EPOCHS,
            time_limit_sec=DEFAULT_TIME_LIMIT_SEC,
        )


if __name__ == "__main__":
    main()
