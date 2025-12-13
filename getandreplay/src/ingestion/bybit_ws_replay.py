from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

log = logging.getLogger(__name__)


class BybitLinearReplayStream:
    """
    Async generator that replays previously recorded Bybit websocket dumps.

    The JSONL input is expected to contain objects of the form
    {"symbol": "...", "ts_recorded": <epoch_seconds>, "event": {...}} as written
    by `scripts/collect_bybit_ws.py`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        speed: float = 1.0,
        loop: bool = False,
        symbol: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        self.speed = max(1e-6, float(speed))
        self.loop = loop
        self.symbol = symbol.upper() if symbol else None

    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"Replay file not found: {self.path}")
        while True:
            async for event in self._iterate_once():
                yield event
            if not self.loop:
                break

    async def _iterate_once(self) -> AsyncIterator[Dict[str, Any]]:
        base_recorded_ts: Optional[float] = None
        base_wall_ts: Optional[float] = None
        with self._open_handle() as handle:
            for line in handle:
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = payload.get("event")
                if not isinstance(event, dict):
                    continue
                if self.symbol and event.get("symbol", "").upper() != self.symbol:
                    continue
                recorded_ts = float(payload.get("ts_recorded") or 0.0)
                now = time.time()
                if base_recorded_ts is None:
                    base_recorded_ts = recorded_ts or now
                    base_wall_ts = now
                else:
                    delta = (recorded_ts - base_recorded_ts) / self.speed
                    elapsed = now - base_wall_ts  # type: ignore[arg-type]
                    wait = delta - elapsed
                    if wait > 0:
                        await asyncio.sleep(wait)
                yield event

    def _open_handle(self):
        if self.path.suffix.endswith("gz"):
            import gzip

            return gzip.open(self.path, "rt", encoding="utf-8")
        return self.path.open("r", encoding="utf-8")

    def stop(self) -> None:
        self.loop = False
