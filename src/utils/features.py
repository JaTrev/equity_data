import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA) for a given series."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> float:
    """Calculate Relative Strength Index (RSI) for a given series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(com=period - 1, adjust=False).mean() / loss.ewm(com=period - 1, adjust=False).mean()
    return float((100 - 100 / (1 + rs)).iloc[-1])

def slow_stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slow_k: int = 3) -> tuple[float, float]:
    """Calculate Stochastic Oscillator slow %K and %D for a given DataFrame."""
    low14  = df["low"].rolling(k_period).min()
    high14 = df["high"].rolling(k_period).max()
    raw_k  = (df["close"] - low14) / (high14 - low14) * 100
    sk     = raw_k.rolling(slow_k).mean()   # slow K = 3-bar SMA of raw %K
    k      = float(sk.iloc[-1])
    d      = float(sk.rolling(d_period).mean().iloc[-1])
    return k, d

def slow_stoch_series(df: pd.DataFrame, k_period: int = 14, slow_k: int = 3, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Return full slow-%K and %D series."""
    low_n  = df["low"].rolling(k_period).min()
    high_n = df["high"].rolling(k_period).max()
    raw_k  = (df["close"] - low_n) / (high_n - low_n) * 100
    sk     = raw_k.rolling(slow_k).mean()
    d      = sk.rolling(d_period).mean()
    return sk, d