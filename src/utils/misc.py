

from pathlib import Path
import time
import pandas as pd

from typing import Dict
import requests


def load_existing_latest_by_symbol(path: Path) -> Dict[str, pd.Timestamp]:
    if not path.exists():
        return {}
    existing = pd.read_parquet(path)
    if existing.empty:
        return {}
    if "ticker" not in existing.columns or "timestamp" not in existing.columns:
        raise ValueError(
            f"Existing parquet at {path} must contain 'ticker' and 'timestamp' columns."
        )
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    latest = existing.groupby("ticker")["timestamp"].max()
    return latest.to_dict()

def merge_into_output(base_path: Path, update_path: Path) -> None:
    base_df = pd.read_parquet(base_path)
    upd_df = pd.read_parquet(update_path)

    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"], utc=True)
    upd_df["timestamp"] = pd.to_datetime(upd_df["timestamp"], utc=True)

    merged = pd.concat([base_df, upd_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["ticker", "timestamp"]).sort_values(
        ["ticker", "timestamp"]
    )

    final_tmp = base_path.with_suffix(".merge.tmp.parquet")
    merged.to_parquet(final_tmp, index=False)
    final_tmp.replace(base_path)


def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    resampled = (
        df_1m.resample(rule, on="timestamp", closed="left", label="left")
        .agg({
            "ticker": "first", "open": "first", "high": "max", 
            "low": "min", "close": "last", "volume": "sum"
        })
        .dropna().reset_index()
    )

    if resampled.empty:
        return resampled

    # Logic: If the 'next' candle should have started by now, 
    # then the 'current' last candle is definitely complete.
    last_bar_start = resampled["timestamp"].iloc[-1]
    next_bar_expected_start = last_bar_start + pd.Timedelta(rule)
    
    # Check the actual wall-clock end of your data
    data_end_time = df_1m["timestamp"].max()

    # If our data hasn't even reached the start time of the NEXT potential candle,
    # the current one is still 'open' and subject to change.
    if data_end_time < next_bar_expected_start:
        return resampled.iloc[:-1]
    
    return resampled
