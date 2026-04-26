

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

