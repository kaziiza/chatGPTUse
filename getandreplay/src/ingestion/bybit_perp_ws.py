import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import websockets

from utils.orderbook import L2Orderbook

log = logging.getLogger(__name__)

LINEAR_MAINNET = "wss://stream.bybit.com/v5/public/linear"
LINEAR_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"


class BybitLinearPerpStream:
    """
    Async generator for Bybit v5 USDT perpetual (linear) orderbook + ticker stream.

    Emits orderbook events enriched with mark/index/last prices for downstream runners.
    """

    def __init__(
        self,
        symbol: str,
        *,
        testnet: bool = False,
        depth: int = 50,
        ping_interval: float = 15.0,
        max_reconnect_attempts: int = 10,
        reconnect_base_delay: float = 2.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.ws_url = LINEAR_TESTNET if testnet else LINEAR_MAINNET
        self._rest_base = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.depth = depth
        self.ping_interval = ping_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_base_delay = reconnect_base_delay

        self._ob = L2Orderbook(depth=depth)
        self._stop = False

        self._mark_price: float = 0.0
        self._index_price: float = 0.0
        self._last_price: float = 0.0
        self._reconnect_count: int = 0
        self._last_sequence: int = 0
        self._last_snapshot_ts: float = 0.0
        self._funding_rate: float = 0.0
        self._next_funding_time: Optional[int] = None
        self._resync_lock = asyncio.Lock()

    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        while not self._stop:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=None,
                    max_size=10_000_000,
                ) as ws:
                    log.info(
                        "Connected to Bybit linear WS: %s (symbol=%s)", self.ws_url, self.symbol
                    )
                    # Reset reconnect counter on successful connection
                    self._reconnect_count = 0

                    sub = {
                        "op": "subscribe",
                        "args": [
                            f"orderbook.{self.depth}.{self.symbol}",
                            f"tickers.{self.symbol}",
                        ],
                    }
                    await ws.send(json.dumps(sub))
                    log.info("Subscribe sent: %s", sub)

                    last_pong = time.time()
                    while True:
                        if time.time() - last_pong > self.ping_interval:
                            await ws.send(json.dumps({"op": "ping"}))
                            last_pong = time.time()

                        msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        data = json.loads(msg)

                        if data.get("op") == "pong":
                            last_pong = time.time()
                            continue

                        topic = data.get("topic", "")
                        if topic.startswith("tickers"):
                            self._handle_ticker(data)
                            continue

                        if topic.startswith("orderbook"):
                            event = await self._handle_orderbook(data)
                            if event:
                                yield event
                            continue
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                self._reconnect_count += 1
                if self._reconnect_count >= self.max_reconnect_attempts:
                    log.error(
                        "Max reconnect attempts (%d) reached for %s. Stopping.",
                        self.max_reconnect_attempts,
                        self.symbol,
                    )
                    raise

                # Exponential backoff with cap at 60 seconds
                delay = min(self.reconnect_base_delay * (2 ** (self._reconnect_count - 1)), 60.0)
                log.warning(
                    "Perp WS error (attempt %d/%d), reconnecting in %.1fs: %s",
                    self._reconnect_count,
                    self.max_reconnect_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------ internal

    def _handle_ticker(self, payload: Dict[str, Any]) -> None:
        ticker = payload.get("data")
        if isinstance(ticker, list):
            ticker = ticker[0] if ticker else {}
        if not isinstance(ticker, dict):
            return
        try:
            self._mark_price = float(ticker.get("markPrice") or self._mark_price or 0.0)
            self._index_price = float(ticker.get("indexPrice") or self._index_price or 0.0)
            self._last_price = float(ticker.get("lastPrice") or self._last_price or 0.0)
            funding_rate = ticker.get("fundingRate")
            if funding_rate is not None:
                self._funding_rate = float(funding_rate)
            funding_time = ticker.get("nextFundingTime") or ticker.get("fundingTime")
            if funding_time is not None:
                try:
                    self._next_funding_time = int(funding_time)
                except (TypeError, ValueError):
                    self._next_funding_time = None
        except (TypeError, ValueError):
            log.debug("Failed to parse ticker payload: %s", ticker)

    async def _handle_orderbook(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        typ = payload.get("type")
        ts = payload.get("ts")
        data = payload.get("data") or {}
        sequence = _safe_int(data.get("u") or data.get("seq") or data.get("sequence"))
        prev_sequence = _safe_int(data.get("pu") or data.get("prevSeq") or data.get("prev_sequence"))

        bids = data.get("b", [])
        asks = data.get("a", [])
        if typ == "snapshot":
            if not bids and not asks:
                # Older servers occasionally send empty snapshots; refresh via REST.
                await self._resync_orderbook("empty_snapshot")
                return None
            bid_pairs = [(float(p), float(q)) for p, q in bids]
            ask_pairs = [(float(p), float(q)) for p, q in asks]
            self._ob.reset_from_snapshot(bid_pairs, ask_pairs)
            if sequence:
                self._last_sequence = sequence
                self._last_snapshot_ts = time.time()
        elif typ == "delta":
            if self._last_sequence == 0:
                await self._resync_orderbook("delta_before_snapshot")
                return None
            if prev_sequence and self._last_sequence and prev_sequence != self._last_sequence:
                log.warning(
                    "Sequence gap detected for %s (expected %s, got %s). Resyncing...",
                    self.symbol,
                    self._last_sequence,
                    prev_sequence,
                )
                await self._resync_orderbook("sequence_gap")
                return None
            bid_pairs = [(float(p), float(q)) for p, q in bids]
            ask_pairs = [(float(p), float(q)) for p, q in asks]
            self._ob.apply_delta(bid_pairs, ask_pairs)
            if sequence:
                self._last_sequence = sequence
            else:
                self._last_sequence += 1
        else:
            return None

        return {
            "event": typ,
            "ts": ts,
            "symbol": self.symbol,
            "bids": self._ob.bids[:],
            "asks": self._ob.asks[:],
            "mid": self._ob.mid(),
            "spread": self._ob.spread(),
            "mark_price": self._mark_price or self._ob.mid(),
            "index_price": self._index_price,
            "last_price": self._last_price,
            "funding_rate": self._funding_rate,
            "next_funding_time": self._next_funding_time,
        }

    async def _resync_orderbook(self, reason: str) -> None:
        async with self._resync_lock:
            try:
                bid_pairs, ask_pairs, sequence = await self._fetch_orderbook_snapshot()
            except Exception as exc:
                log.error("Failed to resync orderbook for %s (%s): %s", self.symbol, reason, exc)
                return
            self._ob.reset_from_snapshot(bid_pairs, ask_pairs)
            if sequence:
                self._last_sequence = sequence
            self._last_snapshot_ts = time.time()
            log.info("Resynchronised %s orderbook via REST (reason=%s sequence=%s)", self.symbol, reason, sequence)

    async def _fetch_orderbook_snapshot(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], int]:
        url = f"{self._rest_base}/v5/market/orderbook"
        params = {"category": "linear", "symbol": self.symbol, "limit": self.depth}
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

        result = (payload.get("result") or {}).get("list", [])
        if not result:
            raise RuntimeError(f"Snapshot payload missing list for {self.symbol}")
        snapshot = result[0]
        bids_raw = snapshot.get("b", [])
        asks_raw = snapshot.get("a", [])
        sequence = _safe_int(snapshot.get("u") or snapshot.get("seq") or snapshot.get("sequence"))
        bid_pairs = [(float(price), float(size)) for price, size in bids_raw]
        ask_pairs = [(float(price), float(size)) for price, size in asks_raw]
        return bid_pairs, ask_pairs, sequence


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
