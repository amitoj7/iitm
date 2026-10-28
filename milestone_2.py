
import random
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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cv2.setNumThreads(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["defect_a", "defect_b", "defect_c", "defect_d"]

files = list(Path("/kaggle/input").rglob("train.csv"))
if not files:
    raise FileNotFoundError("Add the competition dataset first")

csv_path = next((p for p in files if p.parent.name == "public"), files[0])
df = pd.read_csv(csv_path)

df["image_id"] = df["image_id"].astype(str)
df["mask_rle"] = df["mask_rle"].fillna("").astype(str).str.strip()
df.loc[df["mask_rle"].str.lower().isin(["nan", "none", "null"]), "mask_rle"] = ""

data = df.pivot(index="image_id", columns="defect_type", values="mask_rle")
data = data.reindex(columns=classes).fillna("").sort_index()

ids = data.index.to_numpy()
labels = (data != "").astype(int)

def decode(rle):
    mask = np.zeros(128 * 800, dtype=np.uint8)

    if not rle:
        return mask.reshape(128, 800, order="F")

    numbers = np.fromstring(rle, sep=" ", dtype=int)

    for start, length in numbers.reshape(-1, 2):
        start -= 1
        mask[start:start + length] = 1

    return mask.reshape(128, 800, order="F")

def encode(mask):
    pixels = mask.flatten(order="F")
    pixels = np.r_[0, pixels, 0]
    changes = np.where(pixels[1:] != pixels[:-1])[0]

    runs = []
    for i in range(0, len(changes), 2):
        runs.extend([
            int(changes[i] + 1),
            int(changes[i + 1] - changes[i])
        ])

    return " ".join(map(str, runs))

active_pixels = []

if active_pixels:
    grid = np.zeros((8, 8), dtype=np.uint8)

    for row, col in active_pixels:
        grid[row - 1, col - 1] = 1

    print("Q1:", encode(grid))
else:
    print("Q1: Enter the highlighted grid coordinates in active_pixels")

image_id = "0ZUU8LH57Y"
masks = [decode(data.loc[image_id, c]) for c in classes]
q2 = sum(int(mask.sum()) for mask in masks)

print("Q2:", q2)

counts = labels.sum()
most = counts.idxmax()
least = counts.idxmin()

print("Q3:", f"{most} ({counts[most]}), {least} ({counts[least]})")

matrix = labels.to_numpy()
cooc = matrix.T @ matrix

pairs = []

for i, j in combinations(range(4), 2):
    pairs.append((cooc[i, j], i, j))

count, i, j = max(pairs)
pair_name = f"{classes[i]}+{classes[j]}"

print("Q4:", f"{pair_name}, {count}")

corr = labels.corr()
value = corr.iloc[i, j]

print("Q5:", f"{value:.2f}, {'Yes' if value >= 0.3 else 'No'}")

max_overlap = 0

for image_id in ids:
    present = [c for c in classes if data.loc[image_id, c]]

    if len(present) < 2:
        continue

    image_masks = {c: decode(data.loc[image_id, c]) for c in present}

    for a, b in combinations(present, 2):
        overlap = np.logical_and(image_masks[a], image_masks[b]).sum()
        max_overlap = max(max_overlap, int(overlap))

print("Q6:", f"{max_overlap}, {'Yes' if max_overlap == 0 else 'No'}")

image_files = {}

for folder in [
    csv_path.parent / "train",
    csv_path.parent / "train_images",
    csv_path.parent / "images",
    csv_path.parent
]:
    if folder.exists():
        for file in folder.rglob("*"):
            if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                image_files.setdefault(file.name, file)
                image_files.setdefault(file.stem, file)

def get_image(image_id):
    path = image_files.get(str(image_id))

    if path is None:
        raise FileNotFoundError(image_id)

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(image_id)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class SteelDataset(Dataset):
    def __init__(self, image_ids):
        self.image_ids = list(image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image = get_image(image_id).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)).copy()

        masks = np.stack([
            decode(data.loc[image_id, c]) for c in classes
        ]).astype(np.float32)

        return (
            torch.from_numpy(image),
            torch.from_numpy(masks.copy())
        )

strata = labels.astype(str).agg("".join, axis=1)

train_ids, val_ids = train_test_split(
    ids,
    test_size=0.2,
    random_state=42,
    stratify=strata
)

train_data = SteelDataset(train_ids)
val_data = SteelDataset(val_ids)

train_loader = DataLoader(
    train_data,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    pin_memory=device == "cuda"
)

val_loader = DataLoader(
    val_data,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=device == "cuda"
)

images, masks = next(iter(train_loader))

print(
    "Q7:",
    f"{tuple(images.shape)}, {tuple(masks.shape)}, "
    f"{masks.dtype}, "
    "Masks need to be float32 because BCEWithLogitsLoss "
    "works with floating-point targets, not integer or boolean masks."
)

def block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(b, b, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = block(3, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.middle = block(64, 128)

        self.dec3 = block(192, 64)
        self.dec2 = block(96, 32)
        self.dec1 = block(48, 16)

        self.pool = nn.MaxPool2d(2)
        self.out = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        x = self.middle(self.pool(c))

        x = F.interpolate(x, size=c.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, c], dim=1))

        x = F.interpolate(x, size=b.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, b], dim=1))

        x = F.interpolate(x, size=a.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, a], dim=1))

        return self.out(x)

model = UNet().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)

history = []
best_loss = float("inf")
best_epoch = 0

for epoch in range(5):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(x)

    train_loss /= len(train_data)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            val_loss += loss.item() * len(x)

    val_loss /= len(val_data)

    history.append((train_loss, val_loss))

    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss = {train_loss:.5f}, "
        f"Val Loss = {val_loss:.5f}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "/kaggle/working/milestone_2.pt")

first_train, first_val = history[0]
last_train, last_val = history[-1]

if best_epoch < 5 and last_val > best_loss * 1.05:
    status = "The validation loss started increasing while the training loss kept improving, suggesting some overfitting."
elif last_train > 0.4 and last_val > 0.4:
    status = "Both losses are still fairly high, so the model may need more training."
else:
    status = "Training and validation losses decreased together without a clear sign of overfitting."

print("\nQ8:")
print(
    f"epoch1_({first_train:.5f}, {first_val:.5f}), "
    f"epoch5_({last_train:.5f}, {last_val:.5f}), "
    f"{best_epoch}, {status}"
)

pd.DataFrame(
    history,
    columns=["train_loss", "val_loss"]
).to_csv("/kaggle/working/milestone_2_losses.csv", index=False)
