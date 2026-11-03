
import os
import gc
import sys
import random
import subprocess
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "segmentation-models-pytorch"
    ])
    import segmentation_models_pytorch as smp

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

classes = ["defect_a", "defect_b", "defect_c", "defect_d"]
height, width = 128, 800
output = Path("/kaggle/working")
output.mkdir(exist_ok=True)

print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("\nQ1")

bottleneck_h = height // 16
bottleneck_w = width // 16
odd_width = 799

widths = []
for _ in range(4):
    odd_width = odd_width // 2
    widths.append(odd_width)

mismatch = widths[-2] - widths[-1] * 2
downsample_factor = (height * width) // (
    bottleneck_h * bottleneck_w
)

print(
    f"{bottleneck_h}, {bottleneck_w}; "
    f"{mismatch}; {downsample_factor}"
)

print("\nQ2")

model_a = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=4
)

model_b = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights=None,
    in_channels=1,
    classes=4
)

params_a = sum(p.numel() for p in model_a.parameters())
params_b = sum(p.numel() for p in model_b.parameters())

for p in model_a.encoder.parameters():
    p.requires_grad = False

trainable_params = sum(
    p.numel() for p in model_a.parameters() if p.requires_grad
)

trainable_pct = trainable_params / params_a * 100
param_diff = abs(params_a - params_b)

efficient_model = (
    "efficientnet-b0" if params_b < params_a else "resnet34"
)

print("ResNet34:", f"{params_a:,}")
print("EfficientNet-B0:", f"{params_b:,}")
print(
    f"{efficient_model}, {param_diff}; "
    f"{trainable_pct:.1f}"
)

for p in model_a.parameters():
    p.requires_grad = True

del model_b
gc.collect()

print("\nQ3")
print("sum, 3x")

print("\nQ4")
print("13, 25")
print(
    "ASPP uses different dilation rates to capture both small and large "
    "defects without repeatedly downsampling the feature maps. "
    "This helps retain thin crack boundaries while still detecting "
    "larger defect regions."
)

print("\nQ5")

def soft_dice_loss_value(pred, target, smooth=1.0):
    intersection = np.sum(pred * target)
    score = (
        2 * intersection + smooth
    ) / (
        np.sum(pred) + np.sum(target) + smooth
    )
    return 1 - score

target = np.zeros(10000, dtype=np.float32)
target[:100] = 1

pred1 = np.full(10000, 0.01, dtype=np.float32)
pred2 = np.zeros(10000, dtype=np.float32)
pred2[:100] = 0.90

loss1 = soft_dice_loss_value(pred1, target)
loss2 = soft_dice_loss_value(pred2, target)

print(f"{loss1:.4f}, {loss2:.4f}")

print("\nQ6")
print(
    "Yes; Softmax allows only one class per pixel, so overlapping "
    "defect labels cannot be represented and one class will suppress "
    "the other."
)

print("\nQ7")
print(
    "Stray positive predictions increase the Dice loss and produce "
    "gradients that push those predictions down. With a completely "
    "empty prediction, the Dice loss is zero."
)
print(
    "BCE provides direct pixel-level supervision and penalises false "
    "positives on clean images, while Dice focuses on defect overlap. "
    "Combining both gives more balanced training on highly imbalanced data."
)

csv_files = list(Path("/kaggle/input").rglob("train.csv"))
csv_path = None

for file in csv_files:
    try:
        sample = pd.read_csv(file, nrows=5)
        if {"image_id", "defect_type", "mask_rle"}.issubset(sample.columns):
            csv_path = file
            break
    except Exception:
        pass

if csv_path is None:
    raise FileNotFoundError("Could not find the competition train.csv")

df = pd.read_csv(csv_path)

df["image_id"] = df["image_id"].astype(str)
df["mask_rle"] = df["mask_rle"].fillna("").astype(str).str.strip()
df.loc[
    df["mask_rle"].str.lower().isin(["nan", "none", "null"]),
    "mask_rle"
] = ""

wide = df.pivot(
    index="image_id",
    columns="defect_type",
    values="mask_rle"
)

wide = wide.reindex(columns=classes).fillna("").sort_index()

ids = wide.index.to_numpy()
rles = wide.to_numpy()
labels = (wide != "").astype(np.int8)

image_paths = {}

for path in Path("/kaggle/input").rglob("*"):
    if path.is_file() and path.suffix.lower() in [
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    ]:
        image_paths[path.name.lower()] = path
        image_paths[path.stem.lower()] = path

def decode_rle(rle):
    mask = np.zeros(height * width, dtype=np.uint8)

    if not rle:
        return mask.reshape(height, width, order="F")

    values = np.fromstring(str(rle), sep=" ", dtype=np.int64)

    if len(values) % 2:
        raise ValueError("Invalid RLE")

    for start, length in values.reshape(-1, 2):
        start -= 1
        end = start + length

        if start < 0 or end > height * width:
            raise ValueError("RLE outside image boundaries")

        mask[start:end] = 1

    return mask.reshape(height, width, order="F")

def get_image(image_id):
    key = str(image_id).lower()
    path = image_paths.get(key)

    if path is None:
        path = image_paths.get(Path(key).stem)

    if path is None:
        raise FileNotFoundError(f"Image not found: {image_id}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_id}")

    if image.shape != (height, width):
        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

    image = image.astype(np.float32) / 255.0
    image = (image - 0.449) / 0.226

    return image

class SteelDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        image_id = ids[row]

        image = get_image(image_id)[None, :, :]

        mask = np.stack([
            decode_rle(rles[row, i])
            for i in range(4)
        ]).astype(np.float32)

        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(mask))
        )

strata = labels.astype(str).agg("".join, axis=1)

stratify = strata if strata.value_counts().min() >= 2 else None

train_idx, val_idx = train_test_split(
    np.arange(len(ids)),
    test_size=0.2,
    random_state=seed,
    stratify=stratify
)

train_dataset = SteelDataset(train_idx)
val_dataset = SteelDataset(val_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=use_amp,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=use_amp,
    persistent_workers=True
)

print("\nDataset")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

x, y = next(iter(train_loader))

print("Input shape:", tuple(x.shape))
print("Mask shape:", tuple(y.shape))

def compound_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)

    prob = torch.sigmoid(logits)

    intersection = (prob * target).sum(dim=(2, 3))
    denominator = (
        prob.sum(dim=(2, 3)) +
        target.sum(dim=(2, 3))
    )

    dice = (
        2 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    dice_loss = 1 - dice.mean()

    return 0.5 * bce + 0.5 * dice_loss

def calculate_metrics(model):
    model.eval()

    foreground_sum = 0.0
    foreground_count = 0
    empty_correct = 0
    empty_count = 0
    false_pixels = 0
    background_pixels = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)

            predictions = torch.sigmoid(logits) >= 0.5
            targets = masks >= 0.5

            intersection = (predictions & targets).sum(
                dim=(2, 3)
            ).float()

            pred_sum = predictions.sum(dim=(2, 3)).float()
            target_sum = targets.sum(dim=(2, 3)).float()

            dice = (
                2 * intersection
            ) / (
                pred_sum + target_sum
            ).clamp_min(1)

            positive = targets.flatten(2).any(dim=2)
            empty = ~positive

            foreground_sum += dice[positive].sum().item()
            foreground_count += positive.sum().item()

            pred_any = predictions.flatten(2).any(dim=2)

            empty_correct += (
                (~pred_any) & empty
            ).sum().item()

            empty_count += empty.sum().item()

            false_pixels += (
                predictions & ~targets
            ).sum().item()

            background_pixels += (~targets).sum().item()

    foreground_dice = (
        foreground_sum / max(foreground_count, 1)
    )

    empty_accuracy = (
        empty_correct / max(empty_count, 1)
    )

    false_positive_fraction = (
        false_pixels / max(background_pixels, 1)
    )

    false_positive_score = 1 - false_positive_fraction

    official_score = (
        0.70 * foreground_dice +
        0.20 * false_positive_score +
        0.10 * empty_accuracy
    )

    return {
        "foreground_dice": foreground_dice,
        "empty_accuracy": empty_accuracy,
        "false_positive_score": false_positive_score,
        "official_score": official_score
    }

def train_model(name, model):
    model = model.to(device)

    if use_amp:
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []

    for epoch in range(1, 6):
        model.train()
        train_total = 0.0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if use_amp:
                images = images.contiguous(
                    memory_format=torch.channels_last
                )

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = compound_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_total += loss.item() * images.size(0)

        train_loss = train_total / len(train_dataset)

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                if use_amp:
                    images = images.contiguous(
                        memory_format=torch.channels_last
                    )

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(images)
                    loss = compound_loss(logits, masks)

                val_total += loss.item() * images.size(0)

        val_loss = val_total / len(val_dataset)
        metrics = calculate_metrics(model)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics
        })

        print(
            f"{name} | Epoch {epoch}/5 | "
            f"Train: {train_loss:.5f} | "
            f"Val: {val_loss:.5f} | "
            f"Foreground Dice: {metrics['foreground_dice']:.5f} | "
            f"Official Score: {metrics['official_score']:.5f}"
        )

    torch.save(
        model.cpu().state_dict(),
        output / f"{name.lower()}_milestone.pt"
    )

    result = history[-1].copy()
    result["epoch1_foreground_dice"] = history[0]["foreground_dice"]
    result["epoch1_val_loss"] = history[0]["val_loss"]

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, pd.DataFrame(history)

print("\nLoading ImageNet-pretrained models")

scratch_model = model_a

try:
    pretrained_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=4
    )

    efficientnet_model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=1,
        classes=4
    )
except Exception as e:
    raise RuntimeError(
        "Could not load ImageNet weights. Enable Internet in Kaggle "
        "settings and run again."
    ) from e

print("\nQ8: Scratch U-Net")
scratch_result, scratch_history = train_model(
    "scratch_resnet34",
    scratch_model
)

print("\nQ8: Pretrained ResNet34 U-Net")
pretrained_result, pretrained_history = train_model(
    "pretrained_resnet34",
    pretrained_model
)

print("\nQ9: Pretrained EfficientNet-B0 U-Net")
efficient_result, efficient_history = train_model(
    "efficientnet_b0",
    efficientnet_model
)

print("\nQ8 RESULTS")

print(
    "Scratch epoch 5:",
    scratch_result["train_loss"],
    scratch_result["val_loss"],
    scratch_result["official_score"]
)

print(
    "Pretrained epoch 5:",
    pretrained_result["train_loss"],
    pretrained_result["val_loss"],
    pretrained_result["official_score"]
)

if (
    pretrained_result["val_loss"] < scratch_result["val_loss"]
    and pretrained_result["official_score"] > scratch_result["official_score"]
):
    winner = "pretrained"
elif (
    scratch_result["val_loss"] < pretrained_result["val_loss"]
    and scratch_result["official_score"] > pretrained_result["official_score"]
):
    winner = "scratch"
else:
    winner = (
        "pretrained"
        if pretrained_result["official_score"] > scratch_result["official_score"]
        else "scratch"
    )

faster = (
    "pretrained"
    if pretrained_result["epoch1_foreground_dice"] >
    scratch_result["epoch1_foreground_dice"]
    else "scratch"
)

scratch_gap = abs(
    scratch_result["val_loss"] - scratch_result["train_loss"]
)

pretrained_gap = abs(
    pretrained_result["val_loss"] - pretrained_result["train_loss"]
)

tighter_gap = "Yes" if pretrained_gap <= scratch_gap else "No"

score_gain = (
    pretrained_result["official_score"] -
    scratch_result["official_score"]
)

if score_gain < 0.02:
    improvement_range = "< +0.02"
elif score_gain <= 0.08:
    improvement_range = "+0.02 to +0.08"
else:
    improvement_range = "> +0.10"

print(
    "Q8:",
    f"{winner}; {faster}; {tighter_gap}; {improvement_range}"
)

print("\nQ9 RESULTS")

ratio = round(params_a / params_b)

efficient_score = efficient_result["official_score"]
resnet_score = pretrained_result["official_score"]

competitive = (
    "Yes" if efficient_score >= resnet_score - 0.02 else "No"
)

print("ResNet34 parameters:", params_a)
print("EfficientNet-B0 parameters:", params_b)
print("Parameter ratio:", ratio)
print("ResNet34 official score:", round(resnet_score, 5))
print("EfficientNet-B0 official score:", round(efficient_score, 5))

print(
    "Q9:",
    f"{ratio}; efficientnet-b0; {competitive}; "
    "depthwise separable convolutions (MBConv)"
)

all_results = pd.DataFrame([
    {"model": "scratch_resnet34", **scratch_result},
    {"model": "pretrained_resnet34", **pretrained_result},
    {"model": "efficientnet_b0", **efficient_result}
])

all_results.to_csv(
    output / "milestone_3_results.csv",
    index=False
)

scratch_history.to_csv(
    output / "scratch_training.csv",
    index=False
)

pretrained_history.to_csv(
    output / "resnet34_training.csv",
    index=False
)

efficient_history.to_csv(
    output / "efficientnet_training.csv",
    index=False
)

print("\nFinal results saved in /kaggle/working")
print(all_results.to_string(index=False))
