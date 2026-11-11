import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch', 'wandb'])

import os, glob, random, math, warnings, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd, cv2
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import wandb
from kaggle_secrets import UserSecretsClient
warnings.filterwarnings("ignore")

H, W = 128, 800
N_CLASSES, BATCH_SIZE, THRESHOLD = 4, 16, 0.5
SEED, SUBSET_SIZE = 42, 1200
LR, WEIGHT_DECAY, EPOCHS, WARMUP_STEPS = 3e-4, 1e-2, 6, 60
MAX_WEIGHT_RATIO = 5.0
ENCODER_WEIGHTS = None
WANDB_PROJECT = "milestone4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

USE_WANDB = True
try:
    wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
except Exception as e:
    print("W&B login failed, logging disabled:", e)
    USE_WANDB = False

csv_path = None
for f in sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True)):
    cols = pd.read_csv(f, nrows=2).columns
    if {"image_id", "defect_type", "mask_rle"} <= set(cols):
        csv_path = f
        break
print("CSV:", csv_path)
sub = pd.read_csv(csv_path)
sub["image_id"] = sub["image_id"].astype(str)
sub["mask_rle"] = sub["mask_rle"].fillna("").astype(str).str.strip()

class_names = ["defect_a", "defect_b", "defect_c", "defect_d"]
class_to_idx = {c: i for i, c in enumerate(class_names)}
print("Defect types in CSV:", sorted(sub["defect_type"].unique()))

image_ids = sorted(sub["image_id"].unique())[:SUBSET_SIZE]
assert len(image_ids) == SUBSET_SIZE

path_lookup = {}
for p in Path("/kaggle/input").rglob("*"):
    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        path_lookup[p.name.lower()] = str(p)
        path_lookup[p.stem.lower()] = str(p)

def find_path(image_id):
    k = image_id.lower()
    if k in path_lookup:
        return path_lookup[k]
    return path_lookup[Path(k).stem]

image_paths = {x: find_path(x) for x in image_ids}

def decode_rle(rle):
    m = np.zeros(H * W, dtype=np.uint8)
    if rle:
        v = np.asarray(rle.split(), dtype=np.int64)
        for s, l in zip(v[0::2] - 1, v[1::2]):
            e = min(s + l, H * W)
            if 0 <= s < H * W and e > s:
                m[s:e] = 1
    return m.reshape((H, W), order="F")

rle_lookup = {(r.image_id, r.defect_type): r.mask_rle for r in sub.itertuples(index=False)}

masks, combination = {}, {}
for image_id in image_ids:
    target = np.zeros((N_CLASSES, H, W), dtype=np.uint8)
    present = []
    for cls in class_names:
        m = decode_rle(rle_lookup.get((image_id, cls), ""))
        target[class_to_idx[cls]] = m
        if m.any():
            present.append(cls)
    masks[image_id] = target
    combination[image_id] = "+".join(present) if present else "clean"

combo_counts = Counter(combination.values())
merged_combo = {i: (c if combo_counts[c] >= 3 else "rare_other") for i, c in combination.items()}

train_ids, val_ids = train_test_split(
    image_ids, test_size=0.20, random_state=42, stratify=[merged_combo[x] for x in image_ids])
print("Subset images:", len(image_ids), "| pairs:", len(image_ids) * 4)
print("Train:", len(train_ids), "| Val:", len(val_ids))
print("Combination groups before merging:", len(combo_counts))
print(combo_counts)
print("Groups after merging:", Counter(merged_combo.values()))

class DefectDataset(Dataset):
    def __init__(self, ids, augment=False):
        self.ids, self.augment = list(ids), augment
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        image_id = self.ids[i]
        img = cv2.imread(image_paths[image_id], cv2.IMREAD_GRAYSCALE)
        if img.shape != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        mask = masks[image_id].transpose(1, 2, 0)
        if self.augment and random.random() < 0.5:
            img = np.flip(img, axis=1).copy(); mask = np.flip(mask, axis=1).copy()
        img = (img.astype(np.float32) / 255.0 - 0.449) / 0.226
        img = torch.from_numpy(img[None].copy())
        mask = torch.from_numpy(mask.transpose(2, 0, 1).copy()).float()
        return img, mask

def sampler_image_weights(ids):
    n = Counter(combination[x] for x in ids)
    n_clean = n.get("clean", max(n.values()))
    gw = {g: min(math.sqrt(n_clean / c), MAX_WEIGHT_RATIO) for g, c in n.items()}
    return {x: gw[combination[x]] for x in ids}

def make_loaders(train_ids, val_ids, use_sampler=False, seed=42):
    g = torch.Generator(); g.manual_seed(seed)
    if use_sampler:
        w = sampler_image_weights(train_ids)
        sampler = WeightedRandomSampler(torch.as_tensor([w[x] for x in train_ids], dtype=torch.double),
                                        num_samples=len(train_ids), replacement=True, generator=g)
        tl = DataLoader(DefectDataset(train_ids, True), batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True)
    else:
        tl = DataLoader(DefectDataset(train_ids, True), batch_size=BATCH_SIZE, shuffle=True, generator=g,
                        num_workers=2, pin_memory=True)
    vl = DataLoader(DefectDataset(val_ids, False), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=2, pin_memory=True)
    return tl, vl

clean_count = sum(combination[x] == "clean" for x in image_ids)
class_positive = {c: int(sum(masks[x][class_to_idx[c]].any() for x in image_ids)) for c in class_names}
print("Q1  clean images:", clean_count)
print("Q1  positive images per class:", class_positive)
print("Q1  unique combination strings:", len(combo_counts),
      "| largest group:", max(combo_counts.values()), "| smallest:", min(combo_counts.values()))

b_idx = class_to_idx[class_names[1]]
b_area = np.array([masks[x][b_idx].sum() for x in image_ids], dtype=np.int64)
pos_b = b_area > 0
pos_pix, total_pix = int(b_area.sum()), len(image_ids) * H * W
print("\nQ2  positive defect_b pixel %:", round(100 * pos_pix / total_pix, 4))
print("Q2  mean defect_b area (positive images):", round(b_area[pos_b].mean(), 1) if pos_b.any() else 0)
print("Q2  BCE pos_weight (neg/pos):", round((total_pix - pos_pix) / pos_pix) if pos_pix else 0)

draw_w = sampler_image_weights(image_ids)
total_w = sum(draw_w.values())
clean_w = sum(draw_w[x] for x in image_ids if combination[x] == "clean")
b_w = sum(draw_w[x] for x, p in zip(image_ids, pos_b) if p)
exp_b = len(image_ids) * b_w / total_w
print("\nQ3  expected clean fraction:", round(clean_w / total_w, 4), "| natural:", round(clean_count / len(image_ids), 4))
print("Q3  expected defect_b draws:", round(exp_b, 2), "| natural:", int(pos_b.sum()),
      "| sampling factor:", round(exp_b / max(1, pos_b.sum()), 2))

recalls = {"1-100": 0.30, "101-500": 0.60, "501-1000": 0.80, "1001+": 0.90}
areas = np.array([masks[x][c].sum() for x in image_ids for c in range(N_CLASSES)])
areas = areas[areas > 0]
bins = {"1-100": int(((areas >= 1) & (areas <= 100)).sum()),
        "101-500": int(((areas >= 101) & (areas <= 500)).sum()),
        "501-1000": int(((areas >= 501) & (areas <= 1000)).sum()),
        "1001+": int((areas >= 1001).sum())}
missed = {b: bins[b] * (1 - recalls[b]) for b in bins}
print("\nQ9  positive pairs per bin:", bins)
print("Q9  expected missed pairs:", round(sum(missed.values()), 2), "| worst bin:", max(missed, key=missed.get))
print("Q9  missed by bin:", missed)
print("Q9  % of positive pixels in 501+ bin:", round(100 * areas[areas >= 501].sum() / areas.sum(), 2))
n_clean_all = combo_counts["clean"]
print("Q3  groups hitting the cap of 5:", sum(math.sqrt(n_clean_all / c) >= MAX_WEIGHT_RATIO for c in combo_counts.values()))

print("Q4: warmup + cosine")
base_lr = 3e-4
def q4_mult(step):
    if step < 60: return step / 60
    return 0.5 * (1 + math.cos(math.pi * (step - 60) / 420))
for s in [15, 270, 480]:
    print(f"  LR at step {s}: {base_lr * q4_mult(s):.8e}")
print(f"  LR after 8 epoch-level scheduler.step() calls: {base_lr * q4_mult(8):.8e}")
print("  Q4(c): stepping once per epoch instead of per optimizer step makes warmup/cosine advance far too slowly.")

print("\nQ5: early stopping trace")
val_trace = [0.512, 0.641, 0.618, 0.661, 0.664, 0.665, 0.671, 0.670, 0.672, 0.669, 0.673, 0.665]
def early_stop_trace(values, min_delta=0.002, patience=3):
    best, best_epoch, bad, stopped = -float("inf"), None, 0, None
    for e, s in enumerate(values, 1):
        if s > best + min_delta:
            best, best_epoch, bad = s, e, 0
        else:
            bad += 1
            if bad >= patience:
                stopped = e; break
    return best_epoch, best, stopped
be, bs, se = early_stop_trace(val_trace)
print(f"  best epoch (restored): {be} (score {bs}) | stops at epoch: {se}")

print("\nQ6: mixed precision and accumulation")
state_mb = 25_000_000 * 4 * 4 / 1e6
fp32_16 = state_mb + 16 * 150
amp_16 = state_mb + 16 * 75
print("  (a) weights+grads+AdamW MB:", state_mb)
print("  (b) batch 16 FP32 / AMP MB:", fp32_16, amp_16)
print("  (c) max batch FP32 / AMP in 3400 MB:", int((3400 - state_mb) // 150), int((3400 - state_mb) // 75))
micro_per_epoch = len(train_ids) // 8
print("  (d) effective batch:", 8 * 4)
print("  (e) micro-batches, optimizer steps per epoch:", micro_per_epoch, micro_per_epoch // 4)
print("  (f) loss divisor:", 4)
print("  (g) warmup steps for 2 epochs:", 2 * (micro_per_epoch // 4))

print("\nQ7: official score conventions")
all_pairs = len(image_ids) * N_CLASSES
empty_pairs = sum(not masks[x][c].any() for x in image_ids for c in range(N_CLASSES))
base_naive = empty_pairs / all_pairs
base_official = 0.70 * base_naive + 0.20 * 1.0 + 0.10 * 1.0
print("  (a) empty GT pairs in 1,200-image subset:", empty_pairs, "of", all_pairs)
print("  (b) all-empty baseline naive Dice:", round(base_naive, 4))
print("  (c) all-empty baseline official score:", round(base_official, 4))
m_naive = (60 * 0.50 + 306 * 1.0 + 34 * 0.0) / 400
m_fp = 1 - (34 / 340) * (1024 / 102400)
m_empty_acc = 306 / 340
m_official = 0.70 * 0.50 + 0.20 * m_fp + 0.10 * m_empty_acc
toy_base_naive = 340 / 400
toy_base_official = 0.70 * 0.0 + 0.20 * 1.0 + 0.10 * 1.0
print("  (d) model official:", round(m_official, 4), "| model naive Dice:", round(m_naive, 4))
print("  (e) toy baseline naive Dice:", round(toy_base_naive, 4), "| toy baseline official:", round(toy_base_official, 4),
      "| naive ranks model below baseline:", m_naive < toy_base_naive, "| official ranks model below baseline:", m_official < toy_base_official)

def build_model():
    return smp.Unet(encoder_name="resnet34", encoder_weights=ENCODER_WEIGHTS,
                    in_channels=1, classes=N_CLASSES, activation=None)

def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    prob = torch.sigmoid(logits)
    inter = (prob * targets).sum(dim=(2, 3))
    den = prob.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + 1.0) / (den + 1.0)
    return 0.5 * bce + 0.5 * (1 - dice.mean())

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses, pair_dice, empty_fp_frac, empty_total, empty_correct = [], [], [], 0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            losses.append(combined_loss(logits.float(), targets).item())
        preds = torch.sigmoid(logits.float()) >= THRESHOLD
        truth = targets >= 0.5
        inter = (preds & truth).sum((2, 3)).float()
        ps, ts = preds.sum((2, 3)).float(), truth.sum((2, 3)).float()
        dice = torch.where((ps == 0) & (ts == 0), torch.ones_like(inter), 2 * inter / (ps + ts).clamp(min=1))
        pair_dice.append(dice.cpu())
        empty = ts == 0
        fp = (preds & ~truth).sum((2, 3)).float()
        empty_fp_frac.append((fp[empty] / (H * W)).cpu())
        empty_total += int(empty.sum()); empty_correct += int((empty & (ps == 0)).sum())
    pd_ = torch.cat(pair_dice)
    fpf = torch.cat(empty_fp_frac)
    mean_dice = pd_.mean().item()
    fp_score = 1 - (fpf.mean().item() if len(fpf) else 0.0)
    empty_acc = empty_correct / max(1, empty_total)
    return {"loss": float(np.mean(losses)), "mean_dice": mean_dice, "fp_score": fp_score,
            "empty_acc": empty_acc, "official": 0.70 * mean_dice + 0.20 * fp_score + 0.10 * empty_acc,
            "class_dice": pd_.mean(0).tolist()}

results = []

def run_experiment(run_name, seed, use_scheduler, use_sampler, epochs=EPOCHS):
    set_seed(seed)
    train_loader, val_loader = make_loaders(train_ids, val_ids, use_sampler, seed)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * len(train_loader)
    sched = None
    if use_scheduler:
        def lam(step):
            if step < WARMUP_STEPS: return step / WARMUP_STEPS
            prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
            return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        sched = LambdaLR(opt, lam)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    cfg = dict(seed=seed, scheduler=use_scheduler, sampler=use_sampler, epochs=epochs, lr=LR,
               batch_size=BATCH_SIZE, threshold=THRESHOLD, encoder="resnet34", weights=str(ENCODER_WEIGHTS))
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=run_name, config=cfg, reinit=True)
    best, best_epoch, ckpt = None, None, f"/kaggle/working/{run_name}_best.pt"
    for epoch in range(1, epochs + 1):
        model.train(); tl, t0 = [], time.time()
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(images)
            loss = combined_loss(logits.float(), targets)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched: sched.step()
            tl.append(loss.item())
        m = evaluate(model, val_loader)
        print(f"[{run_name}] ep{epoch} train_loss={np.mean(tl):.4f} val_loss={m['loss']:.4f} "
              f"dice={m['mean_dice']:.4f} official={m['official']:.4f} ({time.time()-t0:.0f}s)")
        if USE_WANDB:
            wandb.log({"epoch": epoch, "train/loss": float(np.mean(tl)), "val/loss": m["loss"],
                       "val/mean_dice": m["mean_dice"], "val/fp_score": m["fp_score"],
                       "val/empty_acc": m["empty_acc"], "val/official": m["official"],
                       "lr": opt.param_groups[0]["lr"],
                       **{f"val/dice_{class_names[i]}": d for i, d in enumerate(m["class_dice"])}})
        if best is None or m["official"] > best["official"]:
            best, best_epoch = m, epoch
            torch.save(model.state_dict(), ckpt)
    if USE_WANDB:
        wandb.summary["best_epoch"] = best_epoch; wandb.summary["best_official"] = best["official"]
        wandb.finish()
    res = {"name": run_name, "seed": seed, "scheduler": use_scheduler, "sampler": use_sampler,
           "best_epoch": best_epoch, "checkpoint": ckpt, **best}
    results.append(res)
    del model, opt; torch.cuda.empty_cache()
    return res

R0  = run_experiment("M4_R0_baseline",  seed=42, use_scheduler=False, use_sampler=False)
R0b = run_experiment("M4_R0b_seed7",    seed=7,  use_scheduler=False, use_sampler=False)
R1  = run_experiment("M4_R1_scheduler", seed=42, use_scheduler=True,  use_sampler=False)
R2  = run_experiment("M4_R2_sampler",   seed=42, use_scheduler=False, use_sampler=True)

rows = []
for r in results:
    row = {"run": r["name"], "seed": r["seed"], "scheduler": r["scheduler"], "sampler": r["sampler"],
           "best_epoch": r["best_epoch"], "official": r["official"], "mean_dice": r["mean_dice"],
           "fp_score": r["fp_score"], "empty_acc": r["empty_acc"], "val_loss": r["loss"]}
    row.update({f"dice_{class_names[i]}": d for i, d in enumerate(r["class_dice"])})
    rows.append(row)
comparison = pd.DataFrame(rows)
base = comparison.loc[comparison.run == "M4_R0_baseline", "official"].iloc[0]
comparison["delta_vs_R0"] = comparison["official"] - base
seed_noise = abs(R0["official"] - R0b["official"])
comparison["beyond_seed_noise"] = comparison["delta_vs_R0"].abs() > seed_noise
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("Seed noise |R0 - R0b| (official score):", round(seed_noise, 5))
print(comparison.round(4).to_string(index=False))
comparison.to_csv("/kaggle/working/milestone4_results.csv", index=False)
for r in results: print(r["name"], "->", r["checkpoint"])
print("Saved: /kaggle/working/milestone4_results.csv")

def fa_rate(r):
    n_empty = sum(not masks[x][c].any() for x in val_ids for c in range(N_CLASSES))
    return round(100 * (1 - r["empty_acc"]), 1), round(n_empty * (1 - r["empty_acc"]))

print("\n========== FINAL ANSWERS ==========")
for r in (R0, R1):
    print(r["name"], "best_epoch:", r["best_epoch"], "best official:", round(r["official"], 3), "val_loss@best:", round(r["loss"], 4))
print("Q10 a:", ", ".join(f"{r['official']:.3f}" for r in (R0, R0b, R1, R2)))
noise = abs(R0["official"] - R0b["official"])
g1, g2 = R1["official"] - R0["official"], R2["official"] - R0["official"]
print("Q10 b:", f"{noise:.3f}, {g1:.3f}, {g2:.3f}")
print("Q10 c:", "yes" if g1 > 2 * noise else "no", ",", "yes" if g2 > 2 * noise else "no")
print("Q10 d (false-alarm % of empty val pairs, count) R0:", fa_rate(R0), "R2:", fa_rate(R2))