
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

cv2.setNumThreads(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
outdir = Path("/kaggle/working")
outdir.mkdir(parents=True, exist_ok=True)

csv_files = list(Path("/kaggle/input").rglob("train.csv"))
if not csv_files:
    raise FileNotFoundError("Add the competition dataset containing train.csv")

csv_path = next((p for p in csv_files if p.parent.name == "public"), csv_files[0])
root = csv_path.parent
df = pd.read_csv(csv_path)

classes = ["defect_a", "defect_b", "defect_c", "defect_d"]
h, w = 128, 800
canvas = h * w

df["image_id"] = df["image_id"].astype(str)
df["mask_rle"] = df["mask_rle"].fillna("").astype(str).str.strip()
df.loc[df["mask_rle"].str.lower().isin(["nan", "none", "null"]), "mask_rle"] = ""

assert set(df["defect_type"].unique()) == set(classes)
assert not df.duplicated(["image_id", "defect_type"]).any()

wide = df.pivot(index="image_id", columns="defect_type", values="mask_rle")
wide = wide.reindex(columns=classes).fillna("")
wide = wide.sort_index()

ids = wide.index.to_numpy()
rles = wide.to_numpy()
labels = (wide != "").astype(np.uint8)
cardinality = labels.sum(axis=1)

assert len(ids) == 11393
assert len(df) == 45572
assert labels.shape == (11393, 4)

def decode(rle):
    flat = np.zeros(canvas, dtype=np.uint8)
    if not rle:
        return flat.reshape(h, w, order="F")

    nums = np.fromstring(rle, sep=" ", dtype=np.int64)
    if len(nums) % 2:
        raise ValueError("Invalid RLE token count")

    for start, length in nums.reshape(-1, 2):
        begin = max(0, int(start) - 1)
        end = min(canvas, int(start + length - 1))
        if begin < end:
            flat[begin:end] = 1

    return flat.reshape(h, w, order="F")

counts = labels.sum()
clean = int((cardinality == 0).sum())

q1 = int(counts.max()) + clean
q2 = int((cardinality > 1).sum())

print("Class counts:", counts.to_dict())
print("Clean images:", clean)
print("Q1:", q1)
print("Q2:", q2)

overlap = 0

for i in np.flatnonzero(cardinality.to_numpy() > 1):
    merged = np.zeros((h, w), dtype=np.uint8)
    for rle in rles[i]:
        if rle:
            merged += decode(rle)
    overlap += int((merged >= 2).sum())

print("Q3:", overlap)

bad_rows = 0
max_index = 0

for rle in df["mask_rle"]:
    if not rle:
        continue

    nums = np.fromstring(rle, sep=" ", dtype=np.int64)
    runs = nums.reshape(-1, 2)
    ends = runs[:, 0] + runs[:, 1] - 1

    bad_rows += int(np.any(ends > canvas))
    max_index = max(max_index, int(ends.max()))

q4 = f"{bad_rows} rows, max index {max_index}"
print("Q4:", q4)

sx = 256 / 800
sy = 256 / 128
q5 = f"{sx}, {sy}, {sy / sx}"
print("Q5:", q5)

image_paths = {}

for folder in [root / "train", root / "train_images", root / "images", root]:
    if folder.exists():
        for p in folder.rglob("*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                image_paths.setdefault(p.name, p)
                image_paths.setdefault(p.stem, p)

def get_path(image_id):
    p = image_paths.get(str(image_id))
    if p is None:
        raise FileNotFoundError(f"Missing training image: {image_id}")
    return p

def read_image(image_id):
    img = cv2.imread(str(get_path(image_id)), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_id}")
    if img.shape != (h, w):
        raise ValueError(f"Unexpected image shape for {image_id}: {img.shape}")
    return img

def pooled_cohen(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan

    mean1 = np.mean(a, dtype=np.float64)
    mean2 = np.mean(b, dtype=np.float64)
    var1 = np.var(a, ddof=1, dtype=np.float64)
    var2 = np.var(b, ddof=1, dtype=np.float64)

    pooled = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    )

    if pooled == 0:
        return np.nan

    return (mean1 - mean2) / pooled

moment = {
    c: np.zeros((2, 3), dtype=np.float64)
    for c in classes
}

areas = {c: [] for c in classes}
pixel_totals = Counter()

for i, image_id in enumerate(ids):
    if cardinality.iloc[i] == 0:
        continue

    gray = read_image(image_id).astype(np.float64)

    for k, c in enumerate(classes):
        if not rles[i, k]:
            continue

        mask = decode(rles[i, k])
        fg = mask.astype(bool)
        bg = ~fg

        for j, selected in enumerate((fg, bg)):
            values = gray[selected]
            moment[c][j, 0] += values.size
            moment[c][j, 1] += values.sum()
            moment[c][j, 2] += np.square(values).sum()

        pixel_totals[c] += int(mask.sum())

        n, component_map, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        if n > 1:
            areas[c].extend(
                stats[1:, cv2.CC_STAT_AREA].astype(int).tolist()
            )

effects = {}

for c in classes:
    a, b = moment[c]
    n1, sum1, ss1 = a
    n2, sum2, ss2 = b

    m1 = sum1 / n1
    m2 = sum2 / n2

    v1 = max(0.0, (ss1 - sum1 * sum1 / n1) / (n1 - 1))
    v2 = max(0.0, (ss2 - sum2 * sum2 / n2) / (n2 - 1))

    pooled = np.sqrt(
        ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    )

    effects[c] = (m1 - m2) / pooled if pooled > 0 else np.nan

print("Cohen d:", effects)

closest = min(classes, key=lambda c: abs(effects[c]))
q6 = f"{closest}, texture/frequency"

print("Q6:", q6)
print("Q7: 1.0, 0.0")

fg_toy = (6 * 0.8 + 4 * 0.4) / 10
fp_toy = 1 - ((4 * 1024) / (40 * canvas))
ea_toy = 36 / 40
score_toy = 0.7 * fg_toy + 0.2 * fp_toy + 0.1 * ea_toy

q8 = f"{fg_toy:.2f}, {fp_toy:.3f}, {ea_toy:.1f}, {score_toy:.4f}"
print("Q8:", q8)

empty_pairs = int((labels.to_numpy() == 0).sum())
naive_empty = empty_pairs / labels.size

q9 = f"{naive_empty:.4f}, 0.3000"

print("Empty image-class pairs:", empty_pairs)
print("Q9:", q9)

b_areas = np.asarray(areas["defect_b"])
a_areas = np.asarray(areas["defect_a"])

min_b = int(b_areas.min())
lost_b = 100 * b_areas[b_areas < 50].sum() / pixel_totals["defect_b"]
lost_a = 100 * a_areas[a_areas < 200].sum() / pixel_totals["defect_a"]

q10 = f"{min_b}, {lost_b:.4f}, {lost_a:.4f}"
print("Q10:", q10)

combinations = labels.apply(
    lambda row: "+".join(
        c for c in classes if row[c]
    ) or "clean",
    axis=1
)

n_combos = combinations.nunique()
expected_b = round(0.2 * int(counts["defect_b"]))
q11 = f"{n_combos}, {expected_b}"

print("Combination distribution:")
print(combinations.value_counts().to_string())
print("Q11:", q11)

strata_counts = combinations.value_counts()
if strata_counts.min() < 2:
    raise ValueError("Singleton combination stratum: define a documented split policy")

train_idx, val_idx = train_test_split(
    np.arange(len(ids)),
    test_size=0.2,
    random_state=seed,
    stratify=combinations
)

pd.DataFrame({
    "image_id": ids,
    "combination": combinations.to_numpy(),
    "partition": np.where(
        np.isin(np.arange(len(ids)), val_idx),
        "validation",
        "train"
    )
}).to_csv(outdir / "holdout.csv", index=False)

print("Train images:", len(train_idx))
print("Validation images:", len(val_idx))
print("Validation class counts:", labels.iloc[val_idx].sum().to_dict())

class SteelDataset(Dataset):
    def __init__(self, indices, augment=False):
        self.indices = np.asarray(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = int(self.indices[item])
        img = read_image(ids[i]).astype(np.float32) / 255.0
        masks = np.stack(
            [decode(rle) for rle in rles[i]],
            axis=0
        ).astype(np.float32)

        if self.augment and random.random() < 0.5:
            img = np.flip(img, axis=1)
            masks = np.flip(masks, axis=2)

        x = torch.from_numpy(np.ascontiguousarray(img[None]))
        y = torch.from_numpy(np.ascontiguousarray(masks))

        return x, y

def conv_block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1, bias=False),
        nn.BatchNorm2d(b),
        nn.ReLU(inplace=True),
        nn.Conv2d(b, b, 3, padding=1, bias=False),
        nn.BatchNorm2d(b),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = conv_block(1, 16)
        self.e2 = conv_block(16, 32)
        self.e3 = conv_block(32, 64)
        self.e4 = conv_block(64, 128)
        self.mid = conv_block(128, 256)

        self.d4 = conv_block(384, 128)
        self.d3 = conv_block(192, 64)
        self.d2 = conv_block(96, 32)
        self.d1 = conv_block(48, 16)

        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(16, 4, 1)

    def up(self, x, skip, layer):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        return layer(torch.cat([x, skip], dim=1))

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.e4(self.pool(c))
        x = self.mid(self.pool(d))

        x = self.up(x, d, self.d4)
        x = self.up(x, c, self.d3)
        x = self.up(x, b, self.d2)
        x = self.up(x, a, self.d1)

        return self.head(x)

def loss_fn(logits, targets):
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    intersection = (probs * targets).sum((2, 3))
    denominator = probs.sum((2, 3)) + targets.sum((2, 3))

    dice = 1 - (2 * intersection + 1) / (denominator + 1)

    positive = targets.sum((2, 3)) > 0
    fg_loss = dice[positive].mean() if positive.any() else bce * 0

    return bce + fg_loss

batch_size = 8
epochs = 5

train_loader = DataLoader(
    SteelDataset(train_idx, augment=True),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=device.type == "cuda"
)

val_loader = DataLoader(
    SteelDataset(val_idx),
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=device.type == "cuda"
)

model = UNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

history = []
best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    train_total = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        train_total += loss.item() * len(x)

    model.eval()
    val_total = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            loss = loss_fn(model(x), y)
            val_total += loss.item() * len(x)

    train_loss = train_total / len(train_idx)
    val_loss = val_total / len(val_idx)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"train={train_loss:.5f} val={val_loss:.5f}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), outdir / "best.pt")

pd.DataFrame(history).to_csv(
    outdir / "training_history.csv",
    index=False
)

model.load_state_dict(
    torch.load(outdir / "best.pt", map_location=device, weights_only=True)
)
model.eval()

thresholds = [0.3, 0.5, 0.7, 0.9]

results = {
    t: {
        "naive": 0.0,
        "foreground": 0.0,
        "fp_pixels": 0,
        "empty_correct": 0,
        "empty_false": 0
    }
    for t in thresholds
}

positive_pairs = 0
negative_pairs = 0

with torch.no_grad():
    for x, y in val_loader:
        probs = torch.sigmoid(model(x.to(device))).cpu()
        truth = y.bool()

        truth_size = truth.sum((2, 3))
        positive = truth_size > 0
        negative = ~positive

        positive_pairs += int(positive.sum().item())
        negative_pairs += int(negative.sum().item())

        for threshold in thresholds:
            pred = probs >= threshold
            pred_size = pred.sum((2, 3))
            intersection = (pred & truth).sum((2, 3))

            denominator = pred_size + truth_size

            dice = torch.where(
                denominator > 0,
                2 * intersection.float() / denominator.clamp(min=1),
                torch.ones_like(denominator, dtype=torch.float32)
            )

            r = results[threshold]
            r["naive"] += dice.sum().item()
            r["foreground"] += dice[positive].sum().item()
            r["fp_pixels"] += int(pred_size[negative].sum().item())
            r["empty_correct"] += int(
                (pred_size[negative] == 0).sum().item()
            )
            r["empty_false"] += int(
                (pred_size[negative] > 0).sum().item()
            )

rows = []

for threshold, r in results.items():
    naive = r["naive"] / (positive_pairs + negative_pairs)
    foreground = r["foreground"] / positive_pairs
    fp_fraction = r["fp_pixels"] / (negative_pairs * canvas)
    fp_score = 1 - fp_fraction
    empty_accuracy = r["empty_correct"] / negative_pairs
    pair_fp_rate = r["empty_false"] / negative_pairs

    official = (
        0.7 * foreground
        + 0.2 * fp_score
        + 0.1 * empty_accuracy
    )

    rows.append({
        "threshold": threshold,
        "naive_dice": naive,
        "foreground_dice": foreground,
        "fp_pixel_fraction": fp_fraction,
        "fp_pair_rate": pair_fp_rate,
        "false_positive_score": fp_score,
        "empty_accuracy": empty_accuracy,
        "official_score": official
    })

sweep = pd.DataFrame(rows)
sweep.to_csv(outdir / "threshold_sweep.csv", index=False)

best_naive = sweep.loc[sweep["naive_dice"].idxmax()]
best_official = sweep.loc[sweep["official_score"].idxmax()]

print("\nQ12")
print(sweep.round(5).to_string(index=False))

print("\nBest naive threshold:", best_naive["threshold"])
print("Best official threshold:", best_official["threshold"])

answers = {
    "Q1": str(q1),
    "Q2": str(q2),
    "Q3": str(overlap),
    "Q4": q4,
    "Q5": q5,
    "Q6": q6,
    "Q7": "1.0, 0.0",
    "Q8": q8,
    "Q9": q9,
    "Q10": q10,
    "Q11": q11
}

pd.DataFrame(
    list(answers.items()),
    columns=["Question", "Answer"]
).to_csv(outdir / "milestone_answers.csv", index=False)

print("\nFINAL GOOGLE FORM ANSWERS\n")

for question, answer in answers.items():
    print(question + ":", answer)

print("\nQ12 SUMMARY")
print("Validation empty fraction:", round(negative_pairs / (positive_pairs + negative_pairs), 5))
print("Best naive threshold:", best_naive["threshold"])
print("Best naive Dice:", round(best_naive["naive_dice"], 5))
print("Best official threshold:", best_official["threshold"])
print("Best official score:", round(best_official["official_score"], 5))
print("\nSaved to:", outdir)
