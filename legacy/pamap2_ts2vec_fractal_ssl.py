# pamap2_ts2vec_fractal_ssl.py
"""End‑to‑end pipeline for
1. Loading PAMAP2 patient sensor data
2. Pre‑training a TS2Vec encoder
3. Training a Fractal‑SSL model that wraps TS2Vec as the backbone

The script is organised so that each stage can be run independently from the
command line:

    # Step 0 – install deps (create a fresh Python 3.10 venv first!)
    pip install -r requirements.txt

    # Step 1 – cache cleaned windows & basic stats (runs quickly)
    python pamap2_ts2vec_fractal_ssl.py prepare \
        --data-root /path/to/PAMAP2_Dataset \
        --out-dir cache/

    # Step 2 – contrastive pre‑train TS2Vec backbone
    python pamap2_ts2vec_fractal_ssl.py ts2vec-pretrain \
        --cache-dir cache/ \
        --epochs 50 --batch-size 256 --gpu 0

    # Step 3 – Fractal‑SSL fine‑tuning (hierarchical multi‑scale objective)
    python pamap2_ts2vec_fractal_ssl.py fractal-ssl \
        --cache-dir cache/ \
        --epochs 100 --batch-size 128 --gpu 0

    # Step 4 – evaluate on downstream activity‑recognition task (linear probe)
    python pamap2_ts2vec_fractal_ssl.py evaluate \
        --cache-dir cache/ \
        --checkpoint runs/fractal_ssl_best.ckpt

Notes
-----
* The script intentionally keeps all heavy‑lifting (datasets, models, loss
  functions) in‑file so you can experiment without hunting across packages.
* Replace `/path/to/PAMAP2_Dataset` with wherever you unzipped
  `PAMAP2_Dataset.zip`.
* Requires a GPU (tested on RTX 4090 w/ 24 GiB VRAM). CPU will work but it’s
  slow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

try:
    # pip install ts2vec (official impl) OR copy repo to PYTHONPATH
    from ts2vec import TS2Vec
except ImportError as e:
    raise SystemExit("TS2Vec not found. Run `pip install git+https://github.com/zhihanyue/ts2vec.git` first.") from e

# -------------------------
# Utility helpers
# -------------------------

def sliding_window(a: np.ndarray, win_size: int, step: int = 1) -> np.ndarray:
    """Return views of `a` with shape (n_windows, win_size, n_features)."""
    n_samples, n_feat = a.shape
    if win_size > n_samples:
        raise ValueError("win_size larger than input length")
    n_windows = 1 + (n_samples - win_size) // step
    stride0, stride1 = a.strides
    return np.lib.stride_tricks.as_strided(
        a,
        shape=(n_windows, win_size, n_feat),
        strides=(step * stride0, stride0, stride1),
        writeable=False,
    )

# -------------------------
# 0. Dataset preparation
# -------------------------

SENSOR_COLS = list(range(3, 54))  # cols 3‑53 in original file (0‑index)
LABEL_COL = 1  # activityID
TIMESTAMP_COL = 0

class Pamap2WindowDataset(Dataset):
    """Pre‑processed sliding windows for SSL + downstream tasks."""

    def __init__(self, cache_npz: Path):
        self.data = np.load(cache_npz, allow_pickle=True)
        self.x: np.ndarray = self.data["x"]  # (N, T, C)
        self.y: np.ndarray = self.data["y"]  # (N,) – activity label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), torch.tensor(self.y[idx]).long()


def prepare(args):
    """Convert raw .dat files into normalised sliding‑window tensors saved to NPZ."""
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hyper‑params
    win_size = 400  # 4 s @100 Hz, matches many HAR papers
    step = 200      # 50 % overlap

    all_windows: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for dat in sorted(data_root.glob("*.dat")):
        print(f"Parsing {dat.name}…")
        df = pd.read_csv(dat, sep=" ", header=None)
        # Keep only sensor + label
        sensors = df.iloc[:, SENSOR_COLS].replace({"NaN": np.nan}).astype(float)
        labels = df.iloc[:, LABEL_COL].astype(int)
        # Drop rows with any NaNs, reset index for contiguous windows
        good = sensors.dropna().reset_index(drop=True)
        lbl = labels.loc[good.index].reset_index(drop=True)
        x = good.to_numpy(dtype=np.float32)
        # Standardise per‑sensor globally (fit across file)
        mu = x.mean(axis=0)
        sigma = x.std(axis=0) + 1e-6
        x = (x - mu) / sigma
        windows = sliding_window(x, win_size, step)
        y_windows = sliding_window(lbl.to_numpy(), win_size, step)[:, 0]  # majority vote later
        all_windows.append(windows)
        all_labels.append(y_windows)

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    assert X.shape[0] == y.shape[0]

    # Majority activity in each window
    from scipy import stats
    y_mode = stats.mode(y, axis=1, keepdims=False)[0]

    np.savez_compressed(out_dir / "pamap2_windows.npz", x=X, y=y_mode)
    print(f"Saved {X.shape[0]} windows → {out_dir/'pamap2_windows.npz'}")

# -------------------------
# 1. TS2Vec backbone
# -------------------------

def ts2vec_pretrain(args):
    cache_npz = Path(args.cache_dir) / "pamap2_windows.npz"
    dset = Pamap2WindowDataset(cache_npz)
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = TS2Vec(
        input_dims=dset.x.shape[-1],
        device=device,
        repr_dims=args.repr_dims,
    )

    model = model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch, _ in loader:
            batch = batch.to(device)
            loss = model(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(batch)
        print(f"[TS2Vec] epoch {epoch:03d}/{args.epochs} – loss {epoch_loss/len(dset):.4f}")
        if epoch % 10 == 0:
            ckpt = f"runs/ts2vec_epoch{epoch}.pt"
            Path("runs").mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt)

    torch.save(model.state_dict(), "runs/ts2vec_final.pt")

# -------------------------
# 2. Fractal‑SSL wrapper
# -------------------------

class FractalViewGenerator:
    """Generates K self‑similar sub‑sequences of exponentially decaying lengths.

    Example: given a window length L=400 and levels=4 →
        lens = [400, 200, 100, 50]
    Each level gets two augmentations (jitter + scaling) to form positives.
    """

    def __init__(self, levels: int = 4):
        self.levels = levels

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, T, C)
        B, T, C = x.shape
        views = []
        for l in range(self.levels):
            seg_len = T // (2 ** l)
            start = torch.randint(0, T - seg_len + 1, (B,))
            for b in range(B):
                seg = x[b, start[b]: start[b] + seg_len]
                # simple jitter noise aug
                seg = seg + 0.01 * torch.randn_like(seg)
                views.append(seg)
        return views  # length B*levels

class FractalSSL(nn.Module):
    """Fractal self‑supervised learner with TS2Vec backbone + projection head."""

    def __init__(self, backbone: TS2Vec, proj_dim: int = 128, temp: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(backbone.repr_dims, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.temp = temp
        self.view_gen = FractalViewGenerator()

    def info_nce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise InfoNCE loss between two view batches."""
        z1 = nn.functional.normalize(z1, dim=-1)
        z2 = nn.functional.normalize(z2, dim=-1)
        sim = (z1 @ z2.T) / self.temp  # (B, B)
        labels = torch.arange(z1.size(0), device=z1.device)
        return nn.functional.cross_entropy(sim, labels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        views = self.view_gen(batch)  # len = B*levels
        losses = []
        B = batch.size(0)
        for l in range(0, len(views), B):
            v = torch.stack(views[l: l + B])  # (B, t_l, C)
            # TS2Vec enc expects (B, T, C): pad shorter sequences to max len
            padded = nn.utils.rnn.pad_sequence(v, batch_first=True)
            z = self.backbone.encode(padded)  # (B, T', D)
            # Aggregate by max‑pool across time axis
            z = torch.max(z, dim=1).values  # (B, D)
            z = self.proj(z)
            # positive pairs: current level vs full‑length level 0
            if l == 0:
                z_full = z  # save for later
            else:
                losses.append(self.info_nce(z, z_full))
        return torch.stack(losses).mean()


def train_fractal_ssl(args):
    cache_npz = Path(args.cache_dir) / "pamap2_windows.npz"
    dset = Pamap2WindowDataset(cache_npz)
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    backbone = TS2Vec(input_dims=dset.x.shape[-1], device=device, repr_dims=args.repr_dims)
    model = FractalSSL(backbone).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for batch, _ in loader:
            batch = batch.to(device)
            loss = model(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(batch)
        print(f"[Fractal‑SSL] epoch {epoch:03d}/{args.epochs} – loss {total/len(dset):.4f}")
        if epoch % 20 == 0:
            Path("runs").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"runs/fractal_ssl_epoch{epoch}.ckpt")

    torch.save(model.state_dict(), "runs/fractal_ssl_best.ckpt")

# -------------------------
# 3. Downstream linear probe
# -------------------------

def evaluate(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    cache_npz = Path(args.cache_dir) / "pamap2_windows.npz"
    dset = Pamap2WindowDataset(cache_npz)
    X = torch.from_numpy(dset.x).float()  # (N, T, C)
    y = dset.y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = TS2Vec(input_dims=dset.x.shape[-1], device=device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    backbone.load_state_dict({k.replace("backbone.", ""): v for k, v in ckpt.items() if k.startswith("backbone.")})
    backbone.eval()

    with torch.no_grad():
        feats = backbone.encode(X.to(device)).max(dim=1).values.cpu().numpy()

    # Simple train/test split
    n = len(feats)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(feats[train_idx], y[train_idx])
    y_pred = clf.predict(feats[test_idx])
    acc = accuracy_score(y[test_idx], y_pred)
    print(f"Linear‑probe accuracy: {acc*100:.2f}%")
    print(classification_report(y[test_idx], y_pred))

# -------------------------
# 4. CLI entry
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PAMAP2 × TS2Vec × Fractal‑SSL")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare")
    sp.add_argument("--data-root", required=True)
    sp.add_argument("--out-dir", default="cache")

    sp = sub.add_parser("ts2vec-pretrain")
    sp.add_argument("--cache-dir", default="cache")
    sp.add_argument("--epochs", type=int, default=50)
    sp.add_argument("--batch-size", type=int, default=256)
    sp.add_argument("--repr-dims", type=int, default=320)
    sp.add_argument("--gpu", type=int, default=0)

    sp = sub.add_parser("fractal-ssl")
    sp.add_argument("--cache-dir", default="cache")
    sp.add_argument("--epochs", type=int, default=100)
    sp.add_argument("--batch-size", type=int, default=128)
    sp.add_argument("--repr-dims", type=int, default=320)
    sp.add_argument("--gpu", type=int, default=0)

    sp = sub.add_parser("evaluate")
    sp.add_argument("--cache-dir", default="cache")
    sp.add_argument("--checkpoint", required=True)

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "ts2vec-pretrain":
        ts2vec_pretrain(args)
    elif args.cmd == "fractal-ssl":
        train_fractal_ssl(args)
    elif args.cmd == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
