#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a TSMixer PoC model with advanced sampling and weighting options.

Features
--------
- Supports balanced sampling over labels/view_id/symbol
- Optional per-symbol loss weighting
- Optional hard-negative boosting from a JSON list of indices
- AMP training with cosine LR schedule and early stopping
- Saves stats/meta consistent with runtime loader requirements
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **_: x  # type: ignore

from src.models.tsmixer_poc import TSMixerPoC  # noqa: E402

LOG = logging.getLogger("train_tsmixer_poc")


def str2bool(value: Optional[str]) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean flag value: {value}")


class NumpyDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        symbol_ids: Optional[np.ndarray] = None,
        view_ids: Optional[np.ndarray] = None,
    ) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.symbol_ids = np.asarray(symbol_ids, dtype=np.int64) if symbol_ids is not None else None
        self.view_ids = np.asarray(view_ids, dtype=np.int64) if view_ids is not None else None
        self.indices = np.arange(len(self.X), dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        symbol = torch.tensor(self.symbol_ids[idx]) if self.symbol_ids is not None else torch.tensor(-1, dtype=torch.int64)
        view = torch.tensor(self.view_ids[idx]) if self.view_ids is not None else torch.tensor(-1, dtype=torch.int64)
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(self.indices[idx]),
            symbol,
            view,
        )


def load_arrays(dataset_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(dataset_dir / f"X.{split}.npy")
    y = np.load(dataset_dir / f"y.{split}.npy")
    return X, y


def load_optional(dataset_dir: Path, name: str) -> Optional[np.ndarray]:
    path = dataset_dir / name
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    return arr


def load_stats(dataset_dir: Path) -> Dict[str, object]:
    stats_path = dataset_dir / "stats.json"
    with stats_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_balanced_tokens(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    tokens = []
    for item in raw.split(","):
        part = item.strip().lower()
        if part:
            tokens.append(part)
    valid = {"label", "symbol", "view_id"}
    filtered = [tok for tok in tokens if tok in valid]
    for tok in tokens:
        if tok not in valid:
            LOG.warning("Ignoring unsupported balanced sampler token '%s'", tok)
    return filtered


def parse_symbol_weights(raw: Optional[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if not raw:
        return weights
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        sym, val = item.split(":", 1)
        try:
            weights[sym.strip().upper()] = float(val.strip())
        except ValueError:
            LOG.warning("Invalid symbol weight entry '%s'", item)
    return weights


def build_symbol_id_map(symbols: np.ndarray) -> Dict[str, int]:
    unique = sorted(set(str(sym).upper() for sym in symbols))
    return {sym: idx for idx, sym in enumerate(unique)}


def compute_balanced_weights(
    tokens: List[str],
    y: np.ndarray,
    symbol_ids: Optional[np.ndarray],
    view_ids: Optional[np.ndarray],
) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float32)
    n = len(y)
    if "label" in tokens:
        classes, counts = np.unique(y.astype(int), return_counts=True)
        cls_weights = {cls: n / (len(classes) * count) for cls, count in zip(classes, counts) if count > 0}
        weights *= np.array([cls_weights.get(int(lbl), 1.0) for lbl in y], dtype=np.float32)
    if "symbol" in tokens and symbol_ids is not None:
        classes, counts = np.unique(symbol_ids, return_counts=True)
        sym_weights = {cls: n / (len(classes) * count) for cls, count in zip(classes, counts) if count > 0}
        weights *= np.array([sym_weights.get(int(sid), 1.0) for sid in symbol_ids], dtype=np.float32)
    if "view_id" in tokens and view_ids is not None:
        classes, counts = np.unique(view_ids, return_counts=True)
        view_weights = {cls: n / (len(classes) * count) for cls, count in zip(classes, counts) if count > 0}
        weights *= np.array([view_weights.get(int(vid), 1.0) for vid in view_ids], dtype=np.float32)
    weights[weights <= 0] = 1.0
    return weights


def load_hard_negative_lookup(path: Optional[str], weight: float) -> Dict[int, float]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        LOG.warning("Hard-negative file not found: %s", file_path)
        return {}
    with file_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        indices = payload.get("indices") or payload.get("hard_indices") or []
    else:
        indices = payload
    lookup: Dict[int, float] = {}
    for item in indices:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if idx >= 0:
            lookup[idx] = float(weight)
    LOG.info("Loaded %d hard-negative indices from %s", len(lookup), file_path)
    return lookup


def build_model(cfg: argparse.Namespace, stats: Dict[str, object]) -> nn.Module:
    seq_len = int(stats.get("seq_len", cfg.seq_len))
    features = stats.get("features")
    if not features:
        raise ValueError("stats.json missing 'features' entry")
    model = TSMixerPoC(
        seq_len=seq_len,
        n_features=len(features),
        hidden=cfg.d_model,
        n_blocks=cfg.n_layers,
        dropout=cfg.dropout,
        task="classification",
    )
    return model


def configure_optimizer(model: nn.Module, cfg: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def compute_pos_weight(y: np.ndarray) -> float:
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


def parse_pos_weight(raw: str, y: np.ndarray) -> float:
    if raw.lower() == "auto":
        return compute_pos_weight(y)
    return float(raw)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: argparse.Namespace,
    device: torch.device,
    pos_weight: float,
    hard_lookup: Dict[int, float],
    symbol_lookup: Dict[int, float],
) -> Dict[str, float]:
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = configure_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", disable=cfg.quiet):
            X, y, idx_tensor, sym_tensor, _ = batch
            X = X.to(device)
            y = y.to(device)
            idx_tensor = idx_tensor.to(device)
            sym_tensor = sym_tensor.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(X)
                targets = y
                if cfg.label_smoothing > 0:
                    targets = targets * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
                loss = criterion(logits.squeeze(-1), targets)

            weights = torch.ones_like(loss)
            if cfg.sample_weight_map:
                view_col = batch[4].to(device)
                for view_id, value in cfg.sample_weight_map.items():
                    weights = torch.where(view_col == view_id, weights * value, weights)
            if symbol_lookup:
                for sym_id, value in symbol_lookup.items():
                    weights = torch.where(sym_tensor == sym_id, weights * value, weights)
            if hard_lookup:
                for idx, value in hard_lookup.items():
                    weights = torch.where(idx_tensor == idx, weights * value, weights)

            loss = (loss * weights).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * X.size(0)

        scheduler.step()
        train_loss = running / max(len(train_loader.dataset), 1)
        val_loss = evaluate(model, val_loader, criterion, device, cfg)
        LOG.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

        if val_loss + cfg.early_stop_delta < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= cfg.early_stop:
                LOG.info("Early stopping triggered at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_loss": float(best_val), "epoch": best_epoch}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: argparse.Namespace,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            X, y, idx_tensor, sym_tensor, view_tensor = batch
            X = X.to(device)
            y = y.to(device)
            idx_tensor = idx_tensor.to(device)
            sym_tensor = sym_tensor.to(device)
            view_tensor = view_tensor.to(device)

            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                logits = model(X).squeeze(-1)
                targets = y
                if cfg.label_smoothing > 0:
                    targets = targets * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
                loss = criterion(logits, targets)

            weights = torch.ones_like(loss)
            if cfg.sample_weight_map:
                for view_id, value in cfg.sample_weight_map.items():
                    weights = torch.where(view_tensor == view_id, weights * value, weights)
            total += (loss * weights).mean().item() * X.size(0)
    return total / max(len(loader.dataset), 1)


def save_artifacts(
    model: nn.Module,
    cfg: argparse.Namespace,
    stats: Dict[str, object],
    metrics: Dict[str, float],
    device: torch.device,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "seq_len": stats.get("seq_len", cfg.seq_len),
            "n_features": len(stats.get("features", [])),
            "hidden": cfg.d_model,
            "n_layers": cfg.n_layers,
            "dropout": cfg.dropout,
            "task": "classification",
        },
        "metrics": metrics,
    }
    torch.save(payload, cfg.save_pt)

    meta = {
        "trained_at": time.time(),
        "device": str(device),
        "metrics": metrics,
        "hyperparameters": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "label_smoothing": cfg.label_smoothing,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "early_stop": cfg.early_stop,
            "pos_weight": cfg.pos_weight_value,
            "sample_weights": cfg.sample_weights,
            "balanced_sampler": cfg.balanced_sampler,
            "symbol_weights": cfg.symbol_weights,
            "hard_negative_json": cfg.hard_negative_json,
            "hard_negative_weight": cfg.hard_negative_weight,
        },
    }
    Path(cfg.save_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    Path(cfg.save_stats).write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TSMixer PoC model")
    parser.add_argument("--dataset", required=True, help="Dataset directory (output of build_tsmixer_dataset.py)")
    parser.add_argument("--seq-len", type=int, default=600, help="Sequence length fallback when stats.json is missing the value")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--patch-len", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--early-stop-delta", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pos-weight", default="auto")
    parser.add_argument("--sample-weights", default=None, help="Comma separated view weights view:0=1.0,view:1=0.3")
    parser.add_argument("--balanced-sampler", default="", help="Comma separated tokens: label,view_id,symbol")
    parser.add_argument("--symbol-weights", default="", help="Per-symbol weight mapping, e.g. ETHUSDT:2.0")
    parser.add_argument("--hard-negative-json", default=None, help="JSON file listing hard-negative indices")
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--save-pt", required=True)
    parser.add_argument("--save-stats", required=True)
    parser.add_argument("--save-meta", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset_dir = Path(args.dataset).resolve()
    stats = load_stats(dataset_dir)
    features = stats.get("features")
    if not features:
        raise ValueError("stats.json missing 'features'")

    X_train, y_train = load_arrays(dataset_dir, "train")
    X_val, y_val = load_arrays(dataset_dir, "val")
    sym_train = load_optional(dataset_dir, "symbol.train.npy")
    sym_val = load_optional(dataset_dir, "symbol.val.npy")
    view_train = load_optional(dataset_dir, "view.train.npy")
    view_val = load_optional(dataset_dir, "view.val.npy")

    symbol_id_map = build_symbol_id_map(sym_train) if sym_train is not None else {}
    symbol_ids_train = np.array([symbol_id_map.get(str(sym).upper(), -1) for sym in sym_train], dtype=np.int64) if sym_train is not None else None
    symbol_ids_val = np.array([symbol_id_map.get(str(sym).upper(), -1) for sym in sym_val], dtype=np.int64) if sym_val is not None else None

    balanced_tokens = parse_balanced_tokens(args.balanced_sampler)
    sample_weight_map = {}
    if args.sample_weights:
        for entry in args.sample_weights.split(","):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            key, value = entry.split(":", 1)
            try:
                view_id = int(key.split("view:")[-1])
                sample_weight_map[view_id] = float(value)
            except ValueError:
                LOG.warning("Invalid sample weight entry '%s'", entry)

    symbol_weights_raw = parse_symbol_weights(args.symbol_weights)
    symbol_weight_lookup: Dict[int, float] = {}
    for sym, weight in symbol_weights_raw.items():
        if sym in symbol_id_map:
            symbol_weight_lookup[symbol_id_map[sym]] = weight

    train_weights = compute_balanced_weights(balanced_tokens, y_train, symbol_ids_train, view_train)
    if symbol_weight_lookup and symbol_ids_train is not None:
        sym_scale = np.array([symbol_weight_lookup.get(int(sid), 1.0) for sid in symbol_ids_train], dtype=np.float32)
        train_weights *= sym_scale

    hard_lookup = load_hard_negative_lookup(args.hard_negative_json, args.hard_negative_weight)
    if hard_lookup:
        for idx, weight in hard_lookup.items():
            if 0 <= idx < len(train_weights):
                train_weights[idx] *= weight

    train_weights = np.maximum(train_weights, 1e-6)
    if np.any(train_weights != train_weights[0]):
        sampler = WeightedRandomSampler(torch.as_tensor(train_weights, dtype=torch.float32), len(train_weights), replacement=True)
        LOG.info("Using weighted sampler (tokens=%s)", balanced_tokens or list(symbol_weight_lookup.keys()))
    else:
        sampler = None

    train_dataset = NumpyDataset(X_train, y_train, symbol_ids_train, view_train)
    val_dataset = NumpyDataset(X_val, y_val, symbol_ids_val, view_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args, stats).to(device)

    pos_weight = parse_pos_weight(args.pos_weight, y_train)
    args.pos_weight_value = pos_weight
    args.sample_weight_map = sample_weight_map

    metrics = train(
        model,
        train_loader,
        val_loader,
        args,
        device,
        pos_weight,
        hard_lookup,
        symbol_weight_lookup,
    )
    save_artifacts(model, args, stats, metrics, device)
    LOG.info("Training complete. Best epoch=%d val_loss=%.6f", metrics["epoch"], metrics["val_loss"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
