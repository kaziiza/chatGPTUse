import ccxt
import pandas as pd
from datetime import datetime, timezone
import time


def fetch_mexc_ohlcv_full(symbol: str, listing_datetime: datetime, minutes: int = 1440,
                           output_file: str = "output_ohlcv.csv") -> None:
    """Fetch minute OHLCV data from MEXC and store in CSV.

    Parameters
    ----------
    symbol : str
        Market symbol like ``"AIH/USDT"``.
    listing_datetime : datetime
        Timestamp (UTC) from which to start fetching data.
    minutes : int
        Number of minutes of data to retrieve.
    output_file : str
        File path where the CSV data will be stored.
    """
    exchange = ccxt.mexc({"enableRateLimit": True})
    # Ensure market information is loaded so the symbol is recognised
    exchange.load_markets()

    timeframe = "1m"
    since = int(listing_datetime.timestamp() * 1000)
    end_time = since + minutes * 60 * 1000
    all_bars = []
    limit = 1000

    while since < end_time:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as exc:
            print(f"fetch_ohlcv failed: {exc}")
            break

        if not bars:
            print("No more data fetched, breaking.")
            break

        all_bars.extend(bars)
        since = bars[-1][0] + 60 * 1000
        time.sleep(exchange.rateLimit / 1000.0)

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        print("No data fetched.")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}. rows={len(df)}")


if __name__ == "__main__":
    symbol = "AIH/USDT"
    listing_datetime = datetime(2024, 7, 13, 14, 0, tzinfo=timezone.utc)
    fetch_mexc_ohlcv_full(symbol, listing_datetime, minutes=1440,
                           output_file="aih_1min_ohlcv_mexc.csv")
