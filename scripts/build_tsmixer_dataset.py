#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct TSMixer training datasets from merged feature dumps.

Highlights
----------
- Splits by gaps so only contiguous segments longer than a per-symbol threshold
  are considered.
- Supports deadband filtering (skip samples where forward return magnitude is
  below a threshold).
- Allows per-symbol OFI winsorisation and tanh squashing.
- Optionally writes pre-standardised fp16/fp32 arrays for fast inference.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

LOG = logging.getLogger("build_tsmixer_dataset")


def parse_symbol_seconds(raw: Optional[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    if not raw:
        return result
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        sym, val = part.split(":", 1)
        try:
            result[sym.strip().upper()] = int(float(val.strip()))
        except ValueError:
            LOG.warning("Invalid min segment entry '%s'", part)
    return result


def parse_symbol_floats(raw: Optional[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not raw:
        return result
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        sym, val = part.split(":", 1)
        try:
            result[sym.strip().upper()] = float(val.strip())
        except ValueError:
            LOG.warning("Invalid per-symbol value '%s'", part)
    return result


def parse_columns(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass
class BuilderConfig:
    resample_sec: int
    seq_len: int
    horizon_sec: int
    cost_bps: float
    deadband_bps: float
    max_gap_sec: int
    drop_cross_gap: bool
    winsorize_ofi: float
    winsorize_ofi_per_symbol: Dict[str, float]
    squash_ofi: str
    nan_short_fill_sec: int
    min_segment_sec: Dict[str, int]
    stale_threshold_sec: int
    label_price_base: str
    label_fallback: Optional[str]
    val_split: float
    time_ordered: bool
    add_missing_mask: bool
    add_view_id: bool
    precompute_standardize: bool
    precompute_dtype: str
    sample_step: int = 1


def load_symbol_frame(root: Path, symbol: str) -> pd.DataFrame:
    path = root / symbol / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset parquet missing for {symbol}: {path}")
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        raise ValueError(f"'ts' column missing in {path}")
    df = df.drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts")


def split_by_gap(df: pd.DataFrame, max_gap_sec: int) -> List[pd.DataFrame]:
    if max_gap_sec <= 0:
        return [df]
    diffs = df.index.to_series().diff().dt.total_seconds().fillna(0.0)
    boundaries = list(np.where(diffs > max_gap_sec)[0])
    segments: List[pd.DataFrame] = []
    start = 0
    for boundary in boundaries:
        seg = df.iloc[start:boundary]
        if not seg.empty:
            segments.append(seg)
        start = boundary
    tail = df.iloc[start:]
    if not tail.empty:
        segments.append(tail)
    return segments


def interpolate_short_nans(seg: pd.DataFrame, cfg: BuilderConfig) -> pd.DataFrame:
    if cfg.nan_short_fill_sec <= 0:
        return seg
    limit = max(1, int(cfg.nan_short_fill_sec / cfg.resample_sec))
    return seg.interpolate(limit=limit, limit_direction="both")


def stale_segment(seg: pd.DataFrame, cfg: BuilderConfig) -> bool:
    if cfg.stale_threshold_sec <= 0:
        return False
    if "mid" not in seg.columns:
        return False
    mid = seg["mid"]
    if mid.isna().all():
        return True
    diff = mid.diff().abs().fillna(0.0)
    zeros = diff <= 1e-12
    if not zeros.any():
        return False
    groups = (~zeros).cumsum()
    run = zeros.groupby(groups).cumcount() + 1
    longest = run.where(zeros, 0).max()
    if longest is None:
        return False
    idle_time = longest * cfg.resample_sec
    return idle_time >= cfg.stale_threshold_sec


def winsorize(series: pd.Series, sigma: float) -> pd.Series:
    if sigma <= 0:
        return series
    std = series.std(skipna=True)
    if std == 0 or np.isnan(std):
        return series
    mean = series.mean(skipna=True)
    limit = sigma * std
    return series.clip(lower=mean - limit, upper=mean + limit)


def squash(series: pd.Series, mode: str) -> pd.Series:
    if mode == "tanh3":
        return np.tanh(series / 3.0)
    return series


def compute_returns(price: pd.Series, horizon_steps: int) -> pd.Series:
    future = price.shift(-horizon_steps)
    ret = (future - price) / price * 10_000.0
    return ret


def generate_sequences(
    seg: pd.DataFrame,
    cfg: BuilderConfig,
    feature_cols: List[str],
    symbol: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def empty_return(n_features: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.empty((0, cfg.seq_len, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
            np.empty((0, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.int16),
        )

    if len(seg) < cfg.seq_len + 1:
        return empty_return(len(feature_cols))

    freq_delta = seg.index.to_series().diff().dropna()
    step_seconds = float(freq_delta.dt.total_seconds().median()) if not freq_delta.empty else cfg.resample_sec
    horizon_steps = int(round(cfg.horizon_sec / step_seconds))
    horizon_steps = max(1, horizon_steps)

    base_price = cfg.label_price_base
    price_series = seg.get(base_price)
    if price_series is None:
        raise ValueError(f"{base_price} column missing for {symbol}")
    if price_series.isna().any() and cfg.label_fallback:
        fallback = seg.get(cfg.label_fallback)
        if fallback is not None:
            price_series = price_series.fillna(fallback)
    price_series = price_series.replace(0.0, np.nan).fillna(method="ffill").fillna(method="bfill")

    returns = compute_returns(price_series, horizon_steps)
    raw_labels = returns > cfg.cost_bps

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    ts_list: List[pd.Timestamp] = []
    snap_list: List[np.ndarray] = []
    view_list: List[int] = []

    matrix = seg[feature_cols].values.astype(np.float32)
    view_series = seg["view_id"].to_numpy() if "view_id" in seg.columns else None
    for k, end_idx in enumerate(range(cfg.seq_len - 1, len(seg) - horizon_steps)):
        if cfg.sample_step > 1 and (k % cfg.sample_step) != 0:
            continue
        start_idx = end_idx - cfg.seq_len + 1
        seq = matrix[start_idx : end_idx + 1]
        if np.isnan(seq).any() or np.isinf(seq).any():
            continue
        forward_ret = returns.iloc[end_idx]
        if cfg.deadband_bps > 0 and abs(forward_ret) <= cfg.deadband_bps:
            continue
        label = float(raw_labels.iloc[end_idx])
        X_list.append(seq)
        y_list.append(label)
        ts_list.append(seg.index[end_idx])
        snap_list.append(matrix[end_idx])
        if view_series is not None:
            view_list.append(int(view_series[end_idx]))
        else:
            view_list.append(-1)

    if not X_list:
        return empty_return(len(feature_cols))

    return (
        np.stack(X_list),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(ts_list, dtype="datetime64[ns]"),
        np.stack(snap_list),
        np.asarray(view_list, dtype=np.int16),
    )


def stream_sequences(
    seg: pd.DataFrame,
    cfg: BuilderConfig,
    feature_cols: List[str],
    symbol: str,
):
    """Yield sequences one by one to avoid large in-memory stacks."""
    if len(seg) < cfg.seq_len + 1:
        return
    freq_delta = seg.index.to_series().diff().dropna()
    step_seconds = float(freq_delta.dt.total_seconds().median()) if not freq_delta.empty else cfg.resample_sec
    horizon_steps = max(1, int(round(cfg.horizon_sec / step_seconds)))

    price_series = seg.get(cfg.label_price_base)
    if price_series is None:
        raise ValueError(f"{cfg.label_price_base} column missing for {symbol}")
    if price_series.isna().any() and cfg.label_fallback:
        fallback = seg.get(cfg.label_fallback)
        if fallback is not None:
            price_series = price_series.fillna(fallback)
    price_series = price_series.replace(0.0, np.nan).fillna(method="ffill").fillna(method="bfill")

    returns = compute_returns(price_series, horizon_steps)
    raw_labels = returns > cfg.cost_bps

    matrix = seg[feature_cols].values.astype(np.float32)
    view_series = seg["view_id"].to_numpy() if "view_id" in seg.columns else None

    for k, end_idx in enumerate(range(cfg.seq_len - 1, len(seg) - horizon_steps)):
        if cfg.sample_step > 1 and (k % cfg.sample_step) != 0:
            continue
        start_idx = end_idx - cfg.seq_len + 1
        seq = matrix[start_idx : end_idx + 1]
        if np.isnan(seq).any() or np.isinf(seq).any():
            continue
        forward_ret = returns.iloc[end_idx]
        if cfg.deadband_bps > 0 and abs(forward_ret) <= cfg.deadband_bps:
            continue
        label = float(raw_labels.iloc[end_idx])
        ts_val = seg.index[end_idx]
        view_val = int(view_series[end_idx]) if view_series is not None else -1
        yield seq, label, ts_val, view_val


def collect_segments(root: Path, symbol: str, cfg: BuilderConfig) -> Tuple[List[pd.DataFrame], List[str]]:
    raw = load_symbol_frame(root, symbol)
    freq = f"{cfg.resample_sec}S"
    segments: List[pd.DataFrame] = []
    feature_cols: List[str] = []

    for seg_raw in split_by_gap(raw, cfg.max_gap_sec):
        if seg_raw.empty:
            continue
        duration = (seg_raw.index[-1] - seg_raw.index[0]).total_seconds()
        min_duration = cfg.min_segment_sec.get(symbol.upper(), 0)
        if duration < max(min_duration, cfg.seq_len * cfg.resample_sec):
            continue
        resampled = seg_raw.resample(freq).last()
        resampled = resampled.asfreq(freq)
        resampled = interpolate_short_nans(resampled, cfg)
        if stale_segment(resampled, cfg):
            continue
        ofi_cols = [col for col in resampled.columns if col.startswith("ofi_")]
        sigma = cfg.winsorize_ofi_per_symbol.get(symbol.upper(), cfg.winsorize_ofi)
        for col in ofi_cols:
            resampled[col] = squash(winsorize(resampled[col], sigma), cfg.squash_ofi)
        seg = resampled.dropna(subset=[cfg.label_price_base])
        if seg.empty:
            continue
        feature_cols = []
        for col in seg.columns:
            if col == "symbol":
                continue
            if col == "view_id":
                if cfg.add_view_id:
                    feature_cols.append(col)
                continue
            if col == cfg.label_price_base:
                continue
            if cfg.label_fallback and col == cfg.label_fallback:
                continue
            feature_cols.append(col)
        segments.append(seg)
    return segments, feature_cols


def compute_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1))
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TSMixer dataset from merged dumps")
    parser.add_argument("--data-root", required=True, help="Patched dataset root (per symbol subdirs)")
    parser.add_argument("--symbols", required=True, help="Comma separated symbols (e.g. BTCUSDT,ETHUSDT)")
    parser.add_argument("--resample-sec", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=600)
    parser.add_argument("--horizon-sec", type=int, default=900)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--deadband-bps", type=float, default=0.0)
    parser.add_argument("--max-gap-sec", type=int, default=0)
    parser.add_argument("--drop-cross-gap", action="store_true")
    parser.add_argument("--winsorize-ofi", type=float, default=5.0)
    parser.add_argument("--winsorize-ofi-per-symbol", default="")
    parser.add_argument("--squash-ofi", choices=["none", "tanh3"], default="tanh3")
    parser.add_argument("--nan-short-fill-sec", type=int, default=30)
    parser.add_argument("--min-segment-sec", default="")
    parser.add_argument("--stale-threshold-sec", type=int, default=0)
    parser.add_argument("--label-price-base", default="mark_price")
    parser.add_argument("--label-fallback", default="mid")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--time-ordered", action="store_true")
    parser.add_argument("--add-missing-mask", action="store_true")
    parser.add_argument("--add-view-id", action="store_true")
    parser.add_argument("--precompute-standardize", action="store_true")
    parser.add_argument("--precompute-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--sample-step", type=int, default=1, help="Keep every nth sequence to reduce memory (default 1)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ------------------------------------------------------------------ #
    # Streaming build (two-pass, low RAM)
    # ------------------------------------------------------------------ #
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cfg = BuilderConfig(
        resample_sec=args.resample_sec,
        seq_len=args.seq_len,
        horizon_sec=args.horizon_sec,
        cost_bps=args.cost_bps,
        deadband_bps=args.deadband_bps,
        max_gap_sec=args.max_gap_sec,
        drop_cross_gap=args.drop_cross_gap,
        winsorize_ofi=args.winsorize_ofi,
        winsorize_ofi_per_symbol=parse_symbol_floats(args.winsorize_ofi_per_symbol),
        squash_ofi=args.squash_ofi,
        nan_short_fill_sec=args.nan_short_fill_sec,
        min_segment_sec=parse_symbol_seconds(args.min_segment_sec),
        stale_threshold_sec=args.stale_threshold_sec,
        label_price_base=args.label_price_base,
        label_fallback=args.label_fallback,
        val_split=args.val_split,
        time_ordered=args.time_ordered,
        add_missing_mask=args.add_missing_mask,
        add_view_id=args.add_view_id,
        precompute_standardize=args.precompute_standardize,
        precompute_dtype=args.precompute_dtype,
        sample_step=max(1, args.sample_step),
    )

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols_global: Optional[List[str]] = None
    # will accumulate tuples (len_so_far, segments_for_symbol)
    seq_meta: List[Tuple[str, List[pd.DataFrame], List[str]]] = []
    total_seq = 0

    # Pass1: gather segments and feature cols per symbol, estimate total sequences
    for symbol in symbols:
        try:
            segments, feature_cols = collect_segments(data_root, symbol, cfg)
        except Exception as exc:
            LOG.warning("Skipping %s due to error: %s", symbol, exc)
            continue
        if not segments:
            LOG.warning("No usable segments for %s", symbol)
            continue
        if feature_cols_global is None:
            feature_cols_global = feature_cols
        elif feature_cols_global != feature_cols:
            LOG.warning("Feature mismatch for %s, aligning to first symbol columns", symbol)
            missing = set(feature_cols_global) - set(feature_cols)
            for seg in segments:
                for col in missing:
                    seg[col] = 0.0
            feature_cols = feature_cols_global

        # estimate counts
        seq_count_sym = 0
        for seg in segments:
            seg = seg.sort_index()
            if cfg.add_view_id and "view_id" not in seg.columns:
                seg["view_id"] = 0
            for _ in stream_sequences(seg, cfg, feature_cols, symbol):
                seq_count_sym += 1
        total_seq += seq_count_sym
        seq_meta.append((symbol, segments, feature_cols))

    total = total_seq
    if total == 0:
        raise RuntimeError("No training samples generated; check input data and filters.")
    # determine split
    split_idx = int(total * (1 - cfg.val_split))
    split_idx = max(1, min(total - 1, split_idx))
    train_count = split_idx
    val_count = total - train_count

    n_feat = len(feature_cols_global)
    X_train_mm = open_memmap(out_dir / "X.train.npy", mode="w+", dtype=np.float32, shape=(train_count, cfg.seq_len, n_feat))
    X_val_mm = open_memmap(out_dir / "X.val.npy", mode="w+", dtype=np.float32, shape=(val_count, cfg.seq_len, n_feat))
    y_train_mm = open_memmap(out_dir / "y.train.npy", mode="w+", dtype=np.float32, shape=(train_count,))
    y_val_mm = open_memmap(out_dir / "y.val.npy", mode="w+", dtype=np.float32, shape=(val_count,))
    sym_train: List[str] = []
    sym_val: List[str] = []
    view_train: List[int] = []
    view_val_list: List[int] = []

    sum_feat = np.zeros(n_feat, dtype=np.float64)
    sumsq_feat = np.zeros(n_feat, dtype=np.float64)
    total_steps_train = train_count * cfg.seq_len

    tr_ptr = 0
    va_ptr = 0
    g_idx = 0
    # Pass2: stream sequences in time order (segments already sorted), fill train then val
    for symbol, segments, feature_cols in seq_meta:
        for seg in segments:
            seg = seg.sort_index()
            if cfg.add_view_id and "view_id" not in seg.columns:
                seg["view_id"] = 0
            for seq, label, ts_val, view_val in stream_sequences(seg, cfg, feature_cols, symbol):
                if g_idx < train_count:
                    X_train_mm[tr_ptr] = seq
                    y_train_mm[tr_ptr] = label
                    sym_train.append(symbol)
                    view_train.append(view_val)
                    sum_feat += seq.sum(axis=0)
                    sumsq_feat += (seq * seq).sum(axis=0)
                    tr_ptr += 1
                else:
                    X_val_mm[va_ptr] = seq
                    y_val_mm[va_ptr] = label
                    sym_val.append(symbol)
                    view_val_list.append(view_val)
                    va_ptr += 1
                g_idx += 1

    mean = sum_feat / float(total_steps_train)
    var = (sumsq_feat / float(total_steps_train)) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))

    for arr in (X_train_mm, X_val_mm):
        arr -= mean.astype(np.float32)
        arr /= std.astype(np.float32)
        arr.flush()
    y_train_mm.flush()
    y_val_mm.flush()

    np.save(out_dir / "symbol.train.npy", np.array(sym_train, dtype=object))
    np.save(out_dir / "symbol.val.npy", np.array(sym_val, dtype=object))
    if cfg.add_view_id:
        np.save(out_dir / "view.train.npy", np.array(view_train, dtype=np.int16))
        np.save(out_dir / "view.val.npy", np.array(view_val_list, dtype=np.int16))

    stats = {
        "scaler": "zscore",
        "features": feature_cols_global,
        "mean": {name: float(val) for name, val in zip(feature_cols_global, mean)},
        "std": {name: float(val) for name, val in zip(feature_cols_global, std)},
        "seq_len": cfg.seq_len,
        "horizon_sec": cfg.horizon_sec,
        "resample_sec": cfg.resample_sec,
        "cost_bps": cfg.cost_bps,
        "deadband_bps": cfg.deadband_bps,
        "precomputed": {"enabled": False, "dtype": None},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    meta = {
        "symbols": symbols,
        "train_samples": int(train_count),
        "val_samples": int(val_count),
        "train_positive_rate": float(float(np.mean(y_train_mm))) if train_count else 0.0,
        "val_positive_rate": float(float(np.mean(y_val_mm))) if val_count else 0.0,
        "train_counts": {sym: int(sum(1 for s in sym_train if s == sym)) for sym in symbols},
        "val_counts": {sym: int(sum(1 for s in sym_val if s == sym)) for sym in symbols},
        "features": feature_cols_global,
        "config": vars(cfg),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    LOG.info(
        "Dataset created at %s (train=%d, val=%d, features=%d)",
        out_dir,
        train_count,
        val_count,
        len(feature_cols_global),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
