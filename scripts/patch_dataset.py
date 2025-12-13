#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to merge websocket feature dumps with REST backfill patches while
producing mask columns suitable for TSMixer training.

Typical usage:

python scripts/patch_dataset.py \
  --src-ws data/tsmixer/datasets/poc \
  --src-rest data/backfill/bybit \
  --out-root data/tsmixer/datasets/merged_v3 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,HYPEUSDT \
  --rest-disallow-micro true \
  --require-complete mid,basis_bps,funding_rate_twap_60m
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger("patch_dataset")

MICRO_FEATURES_DEFAULT = [
    "spread",
    "spread_bps",
    "depth10_bid_usd",
    "depth10_ask_usd",
    "ofi_1s",
    "ofi_5s",
    "ofi_10s",
    "ofi_30s",
]


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


def parse_columns(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_ws_symbol(root: Path, symbol: str) -> pd.DataFrame:
    symbol_dir = root / symbol
    if not symbol_dir.exists():
        LOG.warning("WS root missing for %s: %s", symbol, symbol_dir)
        return pd.DataFrame()
    files = sorted(symbol_dir.glob("*.parquet"))
    if not files:
        LOG.warning("No WS parquet files for %s in %s", symbol, symbol_dir)
        return pd.DataFrame()
    frames = []
    for path in files:
        try:
            frame = pd.read_parquet(path)
            frames.append(frame)
        except Exception as exc:  # pragma: no cover - i/o safety
            LOG.warning("Failed to read %s: %s", path, exc)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "ts" not in df.columns:
        raise ValueError(f"Column 'ts' missing in WS dataset for {symbol}")
    df = df.drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    return df


def load_rest_symbol(root: Path, symbol: str) -> pd.DataFrame:
    path = root / f"{symbol}.parquet"
    if not path.exists():
        LOG.info("REST file missing for %s at %s", symbol, path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        raise ValueError(f"'ts' column missing from REST dataset {path}")
    df = df.drop_duplicates("ts").sort_values("ts")
    # REST backfills store milliseconds since epoch; without the explicit unit
    # pandas interprets them as nanoseconds which explodes the timeline and
    # forces the merger to materialise decades of empty rows.
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    return df


def align_grid(
    ws_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    freq: str,
) -> pd.Index:
    if ws_df.empty and rest_df.empty:
        return pd.Index([], name="ts")
    if ws_df.empty:
        start, end = rest_df.index.min(), rest_df.index.max()
    elif rest_df.empty:
        start, end = ws_df.index.min(), ws_df.index.max()
    else:
        start = min(ws_df.index.min(), rest_df.index.min())
        end = max(ws_df.index.max(), rest_df.index.max())
    return pd.date_range(start=start, end=end, freq=freq, tz="UTC")


def build_symbol_frame(
    ws_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    *,
    freq: str,
    rest_disallow_micro: bool,
    micro_columns: Iterable[str],
    required_cols: List[str],
) -> pd.DataFrame:
    grid = align_grid(ws_df, rest_df, freq)
    if grid.empty:
        return pd.DataFrame()

    ws_aligned = ws_df.reindex(grid)
    rest_aligned = rest_df.reindex(grid)

    combined = ws_aligned.copy()
    if not rest_aligned.empty:
        combined = combined.combine_first(rest_aligned)
    combined["symbol"] = ws_df["symbol"].iloc[0] if "symbol" in ws_df.columns and not ws_df.empty else np.nan

    view_id = pd.Series(0, index=combined.index, dtype=np.int8)
    if not rest_aligned.empty:
        ws_available = ws_aligned.notna().any(axis=1)
        rest_available = rest_aligned.notna().any(axis=1)
        view_id.loc[~ws_available & rest_available] = 1
    combined["view_id"] = view_id

    feature_cols = [col for col in combined.columns if col not in {"symbol", "view_id"}]
    mask = combined[feature_cols].isna().astype(np.int8)

    if rest_disallow_micro:
        rest_rows = view_id == 1
        for col in micro_columns:
            if col in combined.columns:
                combined.loc[rest_rows, col] = 0.0
                mask.loc[rest_rows, col] = 1

    for col in required_cols:
        if col not in combined.columns:
            combined[col] = np.nan
        mask_col = mask[col] if col in mask.columns else combined[col].isna().astype(np.int8)
        valid = mask_col == 0
        combined = combined[valid]
        mask = mask.loc[combined.index]

    combined[feature_cols] = combined[feature_cols].fillna(0.0)
    for col in feature_cols:
        combined[f"feat_missing_{col}"] = mask[col] if col in mask.columns else combined[col].isna().astype(np.int8)
    combined["symbol"] = combined["symbol"].fillna(method="ffill").fillna(method="bfill")
    return combined.dropna(subset=["symbol"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge WS and REST datasets for TSMixer training")
    parser.add_argument("--src-ws", required=True, help="Directory containing WS feature dumps (per symbol subdirs)")
    parser.add_argument("--src-rest", required=True, help="Directory containing REST parquet files per symbol")
    parser.add_argument("--out-root", required=True, help="Output directory for merged parquet files")
    parser.add_argument("--symbols", required=True, help="Comma separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--resample-sec", type=int, default=10, help="Target sampling interval (seconds)")
    parser.add_argument("--rest-disallow-micro", nargs="?", const=True, type=str2bool, default=False, help="Zero out micro features for REST rows")
    parser.add_argument("--micro-columns", default=",".join(MICRO_FEATURES_DEFAULT), help="Comma separated micro feature columns")
    parser.add_argument("--require-complete", default="", help="Columns that must be present (mask==0) in output rows")
    parser.add_argument("--prefer-ws", nargs="?", const=True, type=str2bool, default=True, help="Prefer WS values when overlaps occur")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ws_root = Path(args.src_ws).resolve()
    rest_root = Path(args.src_rest).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    micro_cols = parse_columns(args.micro_columns) or MICRO_FEATURES_DEFAULT
    required_cols = parse_columns(args.require_complete)
    freq = f"{max(1, int(args.resample_sec))}S"
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    for symbol in symbols:
        ws_df = load_ws_symbol(ws_root, symbol)
        rest_df = load_rest_symbol(rest_root, symbol)
        if ws_df.empty and rest_df.empty:
            LOG.warning("No data for symbol %s; skipping", symbol)
            continue
        merged = build_symbol_frame(
            ws_df,
            rest_df,
            freq=freq,
            rest_disallow_micro=bool(args.rest_disallow_micro),
            micro_columns=micro_cols,
            required_cols=required_cols,
        )
        if merged.empty:
            LOG.warning("Merged frame empty for %s; skipping output", symbol)
            continue
        dest_dir = out_root / symbol
        dest_dir.mkdir(parents=True, exist_ok=True)
        merged.index.name = "ts"
        dest_path = dest_dir / f"{symbol}.parquet"
        merged.reset_index().to_parquet(dest_path, index=False)
        LOG.info("Merged %s rows for %s -> %s", len(merged), symbol, dest_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
