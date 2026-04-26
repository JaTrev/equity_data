from turtle import st

from ticker_mapping import TICKER_MAPPING
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta
from time import sleep
from typing import Optional


# yfinance interval constraints:
# 1m, 2m, 5m, 15m, 30m, 60m, 90m → max 60 days history (1m limited to 7 days per request)
# 1h                               → max 730 days history
# 1d, 5d, 1wk, 1mo, 3mo           → unlimited history
INTERVAL_MAX_DAYS = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 60,
    "1h":  730,
    "1d":  None,
}

# Chunk sizes chosen to stay safely within API limits while minimizing requests
INTERVAL_CHUNK_DAYS = {
    "1m":  6,
    "2m":  55,
    "5m":  55,
    "15m": 55,
    "30m": 55,
    "60m": 700,
    "90m": 55,
    "1h":  700,
    "1d":  3650,
}


class YahooFinanceClient:
    """
    Client for Yahoo Finance OHLCV data via yfinance.

    Handles chunked downloading to work around per-request history limits,
    with automatic retries on transient failures.

    """

    def __init__(
        self,
        request_delay: float = 0.1,
        max_retries: int = 5,
        retry_delay: float = 1.0,
    ):
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _fetch_chunk(
        self,
        ticker: yf.Ticker,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> Optional[pd.DataFrame]:
        """Fetch a single chunk with retry logic. Returns None on permanent failure."""
        for attempt in range(self.max_retries):
            try:
                df = ticker.history(
                    interval=interval,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    prepost=False,
                )
                sleep(self.request_delay)
                return df

            except Exception as e:
                wait = self.retry_delay * (2 ** attempt)
                if attempt < self.max_retries - 1:
                    print(f"  Attempt {attempt + 1}/{self.max_retries} failed: {e} — retrying in {wait:.1f}s")
                    sleep(wait)
                else:
                    print(f"  Failed after {self.max_retries} attempts: {e}")
                    return None


    def download_ohlc_list(
        self,
        symbols: list[str],
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Download complete OHLCV history for a date range, chunking automatically.

        Yahoo Finance imposes per-request history caps (e.g. 7 days for 1m data).
        This method splits the range into safe chunks and stitches the results.

        Args:
            symbols:     List of ticker symbols
            interval:   Candle interval (e.g. '1m', '5m', '1h', '1d')
            start_date: Start date 'YYYY-MM-DD'
            end_date:   End date 'YYYY-MM-DD' (inclusive)

        Returns:
            DataFrame indexed by UTC timestamp with columns: open, high, low, close, volume.
            Returns an empty DataFrame if no data could be collected.
        """
        if interval not in INTERVAL_CHUNK_DAYS:
            raise ValueError(f"Unsupported interval '{interval}'. Choose from: {list(INTERVAL_CHUNK_DAYS)}")
        
        print(f"Downloading {symbols} {interval} from {start.strftime('%Y-%m-%d %H:%M UTC')} to {end.strftime('%Y-%m-%d %H:%M UTC')} ...")

        now = datetime.now(timezone.utc)
        end = min(end, now)
        chunk_days = INTERVAL_CHUNK_DAYS[interval]
        chunk_delta = timedelta(days=chunk_days)
        
        
        all_chunks: list[pd.DataFrame] = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + chunk_delta, end)

            print(f"Fetching chunk: {chunk_start.strftime('%Y-%m-%d %H:%M UTC')} → {chunk_end.strftime('%Y-%m-%d %H:%M UTC')} ... ")

            df_chunk = yf.download(
                symbols,
                interval=interval, 
                start=chunk_start, 
                end=chunk_end, 
                auto_adjust=True, 
                group_by='ticker',
                progress=False
            )
            
            if df_chunk is not None and not df_chunk.empty:
                all_chunks.append(df_chunk)
                chunk_start = chunk_end
            else:
                chunk_start = end  # skip empty window

        if not all_chunks:
            print(f"No data collected for {symbols}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df_combined = pd.concat(all_chunks)
        all_frames = []
        for name, info in TICKER_MAPPING.items():
            ticker = info['ticker']
            if ticker in df_combined.columns.levels[0]:
                df = df_combined[ticker].copy()
                df['name'] = name
                df['ticker'] = ticker
                df['currency'] = info['currency']
                all_frames.append(df)
        
        final_df = pd.concat(all_frames).sort_index()
        final_df = final_df.tz_convert('UTC')
        print(
            f"Downloaded {len(final_df):,} candles: "
            f"{final_df.index.min()} → {final_df.index.max()}"
        )
        return final_df