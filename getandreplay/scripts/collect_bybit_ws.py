#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect Bybit linear perpetual websocket data and persist it as JSONL archives
for later TS2Vec/TSMixer training or replay.

Example (17 hours, BTC/ETH mainnet):

    python scripts/collect_bybit_ws.py ^
        --symbols BTCUSDT ETHUSDT ^
        --duration 61200 ^
        --out data/raw/bybit_ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.bybit_perp_ws import BybitLinearPerpStream

log = logging.getLogger("collect_bybit_ws")


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Bybit linear perp websocket dumps")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols to subscribe (e.g. BTCUSDT ETHUSDT SOLUSDT)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Total collection duration in seconds (0 = run until stopped; default 0)",
    )
    parser.add_argument(
        "--rotate-sec",
        type=float,
        default=259200.0,
        help="File rotation interval in seconds (default: 259200 = 3 days)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/bybit_ws"),
        help="Output directory root (JSONL files will be written underneath)",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use Bybit testnet endpoints instead of mainnet",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=50,
        help="Orderbook depth to request (default: 50)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress JSONL output via .jsonl.gz",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=60.0,
        help="Progress logging interval in seconds (default: 60)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _writer_path(root: Path, symbol: str, compress: bool, ts: Optional[float] = None) -> Path:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts is not None else datetime.now(timezone.utc)
    day_dir = root / symbol.upper() / dt.strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".jsonl.gz" if compress else ".jsonl"
    filename = f"{dt.strftime('%Y%m%dT%H%M%S')}{suffix}"
    return day_dir / filename


async def _record_symbol(
    symbol: str,
    *,
    duration: float,
    rotate_sec: float,
    out_root: Path,
    testnet: bool,
    depth: int,
    compress: bool,
    log_interval: float,
) -> None:
    start_ts = time.time()
    stop_ts = start_ts + duration if duration > 0 else float("inf")
    rotate_sec = max(1.0, float(rotate_sec))
    opener = (lambda p: open(p, "w", encoding="utf-8")) if not compress else _open_gzip  # type: ignore
    restart_delay = 5.0
    idle_timeout = max(60.0, log_interval * 2.0)

    def _open_writer() -> tuple[any, Path, float]:
        ts_open = time.time()
        path = _writer_path(out_root, symbol, compress, ts_open)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle_local = opener(path)
        rotate_at = ts_open + rotate_sec
        log.info("Recording %s -> %s (rotate every %.0fs)", symbol, path, rotate_sec)
        return handle_local, path, rotate_at

    handle = None
    try:
        handle, current_path, rotate_at = _open_writer()
        next_log = time.time() + log_interval
        count = 0
        stop_run = False
        while not stop_run:
            stream = BybitLinearPerpStream(symbol, testnet=testnet, depth=depth)
            try:
                aiter = stream.__aiter__()
                while True:
                    try:
                        event = await asyncio.wait_for(aiter.__anext__(), timeout=idle_timeout)
                    except asyncio.TimeoutError:
                        raise RuntimeError(f"No events received for {idle_timeout:.0f}s (symbol={symbol})")
                    except StopAsyncIteration:
                        break

                    ts = time.time()
                    payload = {
                        "symbol": symbol,
                        "ts_recorded": ts,
                        "event": event,
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    count += 1
                    if ts >= stop_ts:
                        log.info("Symbol %s reached duration limit (events=%d)", symbol, count)
                        stop_run = True
                        stream.stop()
                        break
                    if ts >= rotate_at:
                        handle.close()
                        log.info("Symbol %s rotating file after %.1fs (events=%d)", symbol, ts - start_ts, count)
                        handle, current_path, rotate_at = _open_writer()
                        count = 0
                    if ts >= next_log:
                        log.info(
                            "Symbol %s progress events=%d elapsed=%.1fs current_file=%s",
                            symbol,
                            count,
                            ts - start_ts,
                            current_path.name,
                        )
                        next_log = ts + log_interval
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "Symbol %s stream error, restarting in %.1fs: %s",
                    symbol,
                    restart_delay,
                    exc,
                )
                try:
                    stream.stop()
                except Exception:
                    pass
                await asyncio.sleep(restart_delay)
                continue
            else:
                if not stop_run:
                    log.warning(
                        "Symbol %s stream ended unexpectedly, restarting in %.1fs",
                        symbol,
                        restart_delay,
                    )
                    await asyncio.sleep(restart_delay)
                    continue
        stream.stop()
    finally:
        try:
            if handle:
                handle.close()
        except Exception:
            pass
    log.info("Symbol %s completed (last file=%s)", symbol, current_path if 'current_path' in locals() else "n/a")


def _open_gzip(path: Path):
    import gzip

    return gzip.open(path, mode="wt", encoding="utf-8")


async def _main_async(args: argparse.Namespace) -> None:
    tasks: List[asyncio.Task[None]] = []
    for sym in args.symbols:
        out_root = args.out
        tasks.append(
            asyncio.create_task(
                _record_symbol(
                    sym,
                    duration=args.duration,
                    rotate_sec=args.rotate_sec,
                    out_root=out_root,
                    testnet=args.testnet,
                    depth=args.depth,
                    compress=args.compress,
                    log_interval=args.log_interval,
                )
            )
        )
    await asyncio.gather(*tasks)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
