
from bisect import bisect_left
from typing import List, Tuple

class L2Orderbook:
    """
    Minimal level-2 orderbook builder from Bybit v5 snapshot/delta.
    Keeps top N levels for bids/asks. Prices are floats, sizes are floats.
    """
    def __init__(self, depth: int = 50):
        self.depth = depth
        self.bids: List[Tuple[float, float]] = []  # sorted desc by price
        self.asks: List[Tuple[float, float]] = []  # sorted asc by price

    def reset_from_snapshot(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        self.bids = sorted([(float(p), float(q)) for p, q in bids], key=lambda x: x[0], reverse=True)[:self.depth]
        self.asks = sorted([(float(p), float(q)) for p, q in asks], key=lambda x: x[0])[:self.depth]

    def apply_delta(self, bids_delta: List[Tuple[float, float]], asks_delta: List[Tuple[float, float]]):
        for p, q in bids_delta:
            self._apply(self.bids, float(p), float(q), side="bid")
        for p, q in asks_delta:
            self._apply(self.asks, float(p), float(q), side="ask")

    def _apply(self, side_levels: List[Tuple[float, float]], price: float, qty: float, side: str):
        # qty == 0 → remove, else upsert
        idx = None
        if side == "bid":
            # list sorted desc
            for i, (p, _) in enumerate(side_levels):
                if p == price:
                    idx = i
                    break
            if qty == 0:
                if idx is not None:
                    side_levels.pop(idx)
            else:
                if idx is not None:
                    side_levels[idx] = (price, qty)
                else:
                    side_levels.append((price, qty))
                    side_levels.sort(key=lambda x: x[0], reverse=True)
        else:
            # ask side sorted asc
            for i, (p, _) in enumerate(side_levels):
                if p == price:
                    idx = i
                    break
            if qty == 0:
                if idx is not None:
                    side_levels.pop(idx)
            else:
                if idx is not None:
                    side_levels[idx] = (price, qty)
                else:
                    side_levels.append((price, qty))
                    side_levels.sort(key=lambda x: x[0])
        # truncate to depth
        del side_levels[self.depth:]

    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2.0

    def spread(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return self.asks[0][0] - self.bids[0][0]
