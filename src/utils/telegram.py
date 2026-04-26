
from pathlib import Path
import time
import pandas as pd
from typing import Dict

import requests

def send_telegram_log(parquet_path: Path, title: str, token: str, chat_id: str) -> None:
    df = pd.read_parquet(parquet_path, columns=["name", "ticker", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    stats = df.groupby(["name", "ticker"])["timestamp"].max().sort_index()
    fmt = "%Y-%m-%d %H:%M"
    median_ts = stats.quantile(0.5, interpolation="nearest")
    ticker_w = max(len(ticker) for name, ticker in stats.index)
    name_w = max(len(name) for name, ticker in stats.index)
    table_lines = [f"{'Name':<{name_w}} {'Ticker':<{ticker_w}}  Max Timestamp", "-" * (name_w + ticker_w + 17)]
    
    for (name, ticker), ts in stats.items():
        prefix = "✅" if ts >= median_ts else "⚠️"
        table_lines.append(f"{prefix} {name:<{name_w}} {ticker:<{ticker_w}}  {ts.floor('s').strftime(fmt)}")
    text = f"{title}\n```\n" + "\n".join(table_lines) + "\n```"
    
    for attempt in range(3):
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "MarkdownV2"},
                timeout=10,
            )
            if resp.status_code != 200:
                print(f"Telegram Error Body: {resp.json()}")    
            if resp.status_code == 429:
                time.sleep(2)
            resp.raise_for_status()  # raises on 400, 500, etc. → caught by except → retried
            return
        except Exception as e:
            print(f"Telegram notification failed (attempt {attempt + 1}/3): {e}")
            time.sleep(2)
            continue
    print("Failed to send Telegram notification after 3 attempts.")