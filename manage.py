#!/usr/bin/env python3
"""Keep the daily snapshots current and alert on oversold daily indicators.

Usage:
    python manage.py --update                        # refresh data/*.js from Yahoo Finance
    python manage.py --alert                         # default thresholds on latest local candle
    python manage.py --alert --optimize              # grid-search best strategy per asset, alert matches
    python manage.py --alert --optimize --loopback-days 365
    python manage.py --update --alert --optimize     # refresh then optimize-alert

`--alert` reads the joined daily history from data/*.js (kept current by --update).
`--loopback-days` (default 365) sets the analysis window for both modes.
In --optimize mode, the grid search uses a chronological 70/30 train/test split,
picks the best non-baseline config by train ROI (min 2% buy rate), and only
alerts when the latest day's indicator values match that optimized strategy.
Assets whose optimized config has a non-positive out-of-sample ROI are skipped.

Telegram delivery needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment;
without them alerts are printed to the console only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_FILE = ROOT / "index.html"
OVERLAP_DAYS = 10
RETRIES = 4
MAX_WORKERS = 4
USER_AGENT = "DCA-snapshot-updater/1.0 (daily historical price data)"

THRESHOLDS = {
    "RSI": 30.0,
    "MFI": 20.0,
    "CCI": -100.0,
}
FEAR_GREED_THRESHOLD = 25.0
TELEGRAM_LIMIT = 4000

SPLIT_RATIO = 0.70
MIN_FRAME_DAYS = 60
MIN_BUY_PCT = 2.0

# Mirrors index.html GRIDS: each config enables exactly one condition.
INDICATOR_GRIDS = {
    "RSI": {
        "configs": (
            [{"period": p, "useLower": False, "useUpper": True, "upper": u, "lower": 30}
             for p in [7, 14, 21] for u in [60, 65, 70, 75, 80]] +
            [{"period": p, "useLower": True, "useUpper": False, "lower": l, "upper": 70}
             for p in [7, 14, 21] for l in [20, 25, 30, 35, 40, 45]]
        ),
    },
    "CCI": {
        "configs": (
            [{"period": p, "useLower": False, "useUpper": True, "upper": u, "lower": -100}
             for p in [7, 14, 20] for u in [50, 100, 150, 200]] +
            [{"period": p, "useLower": True, "useUpper": False, "lower": l, "upper": 100}
             for p in [7, 14, 20] for l in [-250, -200, -150, -100, -50, 0]]
        ),
    },
    "MFI": {
        "configs": (
            [{"period": p, "useLower": False, "useUpper": True, "upper": u, "lower": 20}
             for p in [7, 14, 21] for u in [60, 70, 80, 90]] +
            [{"period": p, "useLower": True, "useUpper": False, "lower": l, "upper": 80}
             for p in [7, 14, 21] for l in [10, 15, 20, 25, 30, 35]]
        ),
    },
    "Fear & Greed": {
        "configs": (
            [{"useLower": False, "useUpper": True, "upper": u, "lower": 25}
             for u in [50, 60, 75, 90]] +
            [{"useLower": True, "useUpper": False, "lower": l, "upper": 75}
             for l in [10, 15, 20, 25, 30, 40, 50]]
        ),
    },
}


# ---------------------------------------------------------------------------
# Catalogue + snapshot I/O
# ---------------------------------------------------------------------------
def load_catalogue() -> dict[str, str]:
    """Return ticker -> category, preserving catalogue order."""
    source = INDEX_FILE.read_text(encoding="utf-8")
    match = re.search(r"const ASSET_CATALOGUE = (\{.*?\n\});", source, re.S)
    if not match:
        raise RuntimeError("Could not find ASSET_CATALOGUE in index.html")
    catalogue_json = re.sub(r",\s*([}\]])", r"\1", match.group(1))
    catalogue = json.loads(catalogue_json)

    categories: dict[str, str] = {}
    for group, assets in catalogue.items():
        category = group.strip("─ ").strip()
        for ticker in assets.values():
            if ticker in categories:
                raise RuntimeError(f"ASSET_CATALOGUE contains duplicate ticker: {ticker}")
            categories[ticker] = category
    return categories


def read_snapshot(path: Path, expected_ticker: str) -> dict:
    source = path.read_text(encoding="utf-8")
    pattern = r"window\.DCA_DATA\.market\[(\"(?:\\.|[^\"\\])*\")\]=(.*);\s*"
    match = re.fullmatch(pattern, source, re.S)
    if not match:
        raise RuntimeError(f"Unrecognized market snapshot format: {path}")
    ticker = json.loads(match.group(1))
    if ticker != expected_ticker:
        raise RuntimeError(f"Ticker mismatch in {path}: expected {expected_ticker}, found {ticker}")
    history = json.loads(match.group(2))
    required = ("d", "h", "l", "c", "v")
    if any(not isinstance(history.get(key), list) for key in required):
        raise RuntimeError(f"Missing OHLCV arrays in {path}")
    lengths = {len(history[key]) for key in required}
    if len(lengths) != 1 or not history["d"]:
        raise RuntimeError(f"Invalid OHLCV array lengths in {path}")
    return history


def write_snapshot(path: Path, ticker: str, history: dict) -> None:
    encoded_ticker = json.dumps(ticker, ensure_ascii=False)
    payload = json.dumps(history, separators=(",", ":"), allow_nan=False)
    path.write_text(
        f"window.DCA_DATA.market[{encoded_ticker}]={payload};\n",
        encoding="utf-8",
    )


def load_local_frame(ticker: str, loopback_days: int) -> pd.DataFrame | None:
    path = DATA_DIR / f"{ticker}.js"
    if not path.exists():
        return None
    try:
        history = read_snapshot(path, ticker)
    except Exception:
        return None
    if not history["d"]:
        return None
    frame = pd.DataFrame({
        "High": history["h"],
        "Low": history["l"],
        "Close": history["c"],
        "Volume": history["v"],
    }, index=pd.to_datetime(history["d"]))
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if loopback_days and len(frame) > loopback_days:
        frame = frame.iloc[-loopback_days:]
    return frame


# ---------------------------------------------------------------------------
# Fetcher (--update)
# ---------------------------------------------------------------------------
def history_rows(ticker: str, start: date, end: date) -> list[tuple[str, float, float, float, float]]:
    with warnings.catch_warnings():
        pandas4_warning = getattr(pd.errors, "Pandas4Warning", None)
        if pandas4_warning:
            warnings.simplefilter("ignore", pandas4_warning)
        frame = yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            raise_errors=True,
        )
    if frame.empty:
        raise RuntimeError("Yahoo Finance returned no daily prices")

    rows = []
    for timestamp, bar in frame.iterrows():
        day = timestamp.date()
        if day < start or day > end:
            continue
        high, low, close = (float(bar[column]) for column in ("High", "Low", "Close"))
        raw_volume = bar.get("Volume", 0)
        volume = float(raw_volume) if raw_volume is not None else 0.0
        if not all(math.isfinite(value) for value in (high, low, close, volume)):
            continue
        if high <= 0 or low <= 0 or close <= 0 or volume < 0:
            continue
        rows.append((day.isoformat(), high, low, close, volume))
    return rows


def fetch_with_retries(ticker: str, start: date, end: date) -> list[tuple[str, float, float, float, float]]:
    last_error = None
    for attempt in range(RETRIES):
        try:
            rows = history_rows(ticker, start, end)
            if not rows:
                raise RuntimeError("No complete daily price bars in the requested date range")
            return rows
        except Exception as error:
            last_error = error
            if attempt + 1 < RETRIES:
                time.sleep(min(2**attempt, 16) + random.random())
    raise RuntimeError(f"failed after {RETRIES} attempts: {last_error}") from last_error


def update_market_ticker(ticker: str, end: date) -> tuple[str, int, str | None]:
    path = DATA_DIR / f"{ticker}.js"
    try:
        history = read_snapshot(path, ticker)
        last_day = date.fromisoformat(history["d"][-1])
        start = max(date.fromisoformat(history["d"][0]), last_day - timedelta(days=OVERLAP_DAYS))
        if start > end:
            return ticker, 0, None
        rows = fetch_with_retries(ticker, start, end)

        by_date = {
            day: [high, low, close, volume]
            for day, high, low, close, volume in zip(
                history["d"], history["h"], history["l"], history["c"], history["v"]
            )
        }
        before = len(by_date)
        for day, high, low, close, volume in rows:
            by_date[day] = [high, low, close, volume]
        dates = sorted(by_date)
        updated = {
            "d": dates,
            "h": [by_date[day][0] for day in dates],
            "l": [by_date[day][1] for day in dates],
            "c": [by_date[day][2] for day in dates],
            "v": [by_date[day][3] for day in dates],
        }
        if updated != history:
            write_snapshot(path, ticker, updated)
        return ticker, len(dates) - before, None
    except Exception as error:
        return ticker, 0, str(error)


def update_fear_greed(end: date) -> int:
    path = DATA_DIR / "fear-greed.js"
    source = path.read_text(encoding="utf-8")
    match = re.fullmatch(r"window\.DCA_DATA\.fearGreed=(\{.*\});\s*", source, re.S)
    if not match:
        raise RuntimeError(f"Unrecognized Fear & Greed snapshot format: {path}")
    values = json.loads(match.group(1))

    request = Request(
        "https://api.alternative.me/fng/?limit=0&format=json",
        headers={"User-Agent": USER_AGENT},
    )
    payload = None
    for attempt in range(RETRIES):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
            break
        except (OSError, ValueError) as error:
            if attempt + 1 == RETRIES:
                raise RuntimeError(f"Fear & Greed request failed after {RETRIES} attempts: {error}") from error
            time.sleep(min(2**attempt, 16) + random.random())
    if not isinstance(payload.get("data"), list) or not payload["data"]:
        raise RuntimeError("Fear & Greed API returned no usable data")

    original_size = len(values)
    for item in payload["data"]:
        day = datetime.fromtimestamp(int(item["timestamp"]), timezone.utc).date()
        if day > end:
            continue
        value = int(item["value"])
        if 0 <= value <= 100:
            values[day.isoformat()] = value
    values = dict(sorted(values.items()))
    if values != json.loads(match.group(1)):
        path.write_text(
            "window.DCA_DATA.fearGreed="
            + json.dumps(values, separators=(",", ":"), allow_nan=False)
            + ";\n",
            encoding="utf-8",
        )
    return len(values) - original_size


def read_fear_greed() -> dict[str, int]:
    path = DATA_DIR / "fear-greed.js"
    source = path.read_text(encoding="utf-8")
    match = re.fullmatch(r"window\.DCA_DATA\.fearGreed=(\{.*\});\s*", source, re.S)
    if not match:
        raise RuntimeError(f"Unrecognized Fear & Greed snapshot format: {path}")
    values = json.loads(match.group(1))
    if not isinstance(values, dict):
        raise RuntimeError(f"Fear & Greed snapshot is not an object: {path}")

    cleaned: dict[str, int] = {}
    for day, value in values.items():
        try:
            parsed_day = date.fromisoformat(day)
            parsed_value = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= parsed_value <= 100:
            cleaned[parsed_day.isoformat()] = parsed_value
    return dict(sorted(cleaned.items()))


def build_fng_series(values: dict[str, int]) -> pd.Series | None:
    if not values:
        return None
    items = sorted(values.items())
    idx = pd.to_datetime([k for k, _ in items])
    vals = [v for _, v in items]
    return pd.Series(vals, index=idx)


def run_update(tickers: list[str], end: date, dry_run: bool, verbose: bool) -> int:
    print(f"Updating {len(tickers)} market snapshots through {end.isoformat()}")
    if dry_run:
        for ticker in tickers:
            read_snapshot(DATA_DIR / f"{ticker}.js", ticker)
        print(f"Validated {len(tickers)} market snapshots; no files changed.")
        print("Fear & Greed: would refresh through " + end.isoformat())
        return 0

    errors = []
    added_days = 0
    successful_markets = 0
    updated_markets = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(update_market_ticker, ticker, end): ticker for ticker in tickers}
        for completed, future in enumerate(as_completed(futures), start=1):
            ticker, added, error = future.result()
            added_days += added
            if error:
                errors.append((ticker, error))
                print(f"[{completed}/{len(tickers)}] {ticker}: ERROR: {error}", file=sys.stderr)
            else:
                successful_markets += 1
                updated_markets += added > 0
                if verbose:
                    suffix = f" (+{added} days)" if added else ""
                    print(f"[{completed}/{len(tickers)}] {ticker}: OK{suffix}")

    try:
        fng_added = update_fear_greed(end)
        print(f"Fear & Greed: OK (+{fng_added} days)")
    except (OSError, ValueError, KeyError, TypeError, URLError, RuntimeError) as error:
        errors.append(("Fear & Greed", str(error)))
        print(f"Fear & Greed: ERROR: {error}", file=sys.stderr)

    if errors:
        print(f"Processed {successful_markets}/{len(tickers)} market snapshots; "
              f"{len(errors)} source(s) failed; {updated_markets} ticker(s) gained new dates. "
              f"Successfully added {added_days} market days.",
              file=sys.stderr)
        return 1
    print(f"Updated {len(tickers)} market snapshots; {updated_markets} ticker(s) gained new dates. "
          f"Successfully added {added_days} market days.")
    return 0


# ---------------------------------------------------------------------------
# Indicator series (numpy outputs, aligned with the input arrays)
# ---------------------------------------------------------------------------
def indicator_rsi(close: np.ndarray, period: int) -> np.ndarray:
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    return rsi.to_numpy(dtype=float)


def indicator_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  volume: np.ndarray, period: int) -> np.ndarray:
    typical = (high + low + close) / 3.0
    money_flow = typical * volume
    typical_s = pd.Series(typical)
    mf_s = pd.Series(money_flow)
    positive = mf_s.where(typical_s > typical_s.shift(1), 0).rolling(period).sum()
    negative = mf_s.where(typical_s < typical_s.shift(1), 0).rolling(period).sum()
    mfi = 100 - 100 / (1 + positive / negative.replace(0, np.nan))
    mfi = mfi.mask((negative == 0) & (positive > 0), 100)
    mfi = mfi.mask((negative == 0) & (positive == 0), 50)
    return mfi.to_numpy(dtype=float)


def indicator_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  period: int) -> np.ndarray:
    typical = (high + low + close) / 3.0
    ts = pd.Series(typical)
    mean = ts.rolling(period).mean()
    deviation = ts.rolling(period).apply(lambda x: float(np.abs(x - x.mean()).mean()), raw=True)
    cci = (ts - mean) / (0.015 * deviation.replace(0, np.nan))
    return cci.to_numpy(dtype=float)


def calculate_latest(frame: pd.DataFrame) -> dict[str, float]:
    close = frame["Close"].to_numpy(dtype=float)
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    volume = frame["Volume"].to_numpy(dtype=float)
    rsi = indicator_rsi(close, 14)
    mfi = indicator_mfi(high, low, close, volume, 14)
    cci = indicator_cci(high, low, close, 20)
    return {
        "RSI": float(rsi[-1]) if np.isfinite(rsi[-1]) else float("nan"),
        "MFI": float(mfi[-1]) if np.isfinite(mfi[-1]) else float("nan"),
        "CCI": float(cci[-1]) if np.isfinite(cci[-1]) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Simulation helpers (numpy, ROI is scale-invariant to daily_amount)
# ---------------------------------------------------------------------------
def _segment_roi(close: np.ndarray, signal: np.ndarray, daily_amount: float = 1.0) -> float:
    n_buys = int(signal.sum())
    if n_buys == 0 or len(close) == 0:
        return 0.0
    buy_prices = close[signal]
    if (buy_prices <= 0).any():
        return 0.0
    shares = float((daily_amount / buy_prices).sum())
    invested = daily_amount * n_buys
    value = shares * close[-1]
    return float((value / invested - 1) * 100)


def evaluate_signal(close: np.ndarray, signal: np.ndarray, split_idx: int,
                    daily_amount: float = 1.0) -> dict:
    train_days = split_idx
    test_days = len(close) - split_idx
    train_sig = signal[:split_idx]
    test_sig = signal[split_idx:]
    return {
        "train_roi": _segment_roi(close[:split_idx], train_sig, daily_amount),
        "test_roi": _segment_roi(close[split_idx:], test_sig, daily_amount),
        "train_buys": int(train_sig.sum()),
        "test_buys": int(test_sig.sum()),
        "train_days": train_days,
        "test_days": test_days,
        "train_buy_pct": float(train_sig.mean() * 100) if train_days else 0.0,
        "test_buy_pct": float(test_sig.mean() * 100) if test_days else 0.0,
    }


def format_label(indicator: str, config: dict) -> str:
    base = "F&G" if indicator == "Fear & Greed" else f"{indicator}({config.get('period', '')})"
    parts = []
    if config.get("useLower"):
        parts.append(f"{base} < {config['lower']} (oversold)")
    if config.get("useUpper"):
        parts.append(f"{base} < {config['upper']} (not overbought)")
    return " & ".join(parts) if parts else base


def plain_rule(indicator: str, config: dict) -> str:
    """One-line plain-English description of what the rule does."""
    base = "Fear & Greed" if indicator == "Fear & Greed" else f"{indicator}({config.get('period', '')})"
    lower = config.get("lower")
    upper = config.get("upper")
    use_lower = config.get("useLower")
    use_upper = config.get("useUpper")
    if use_lower and not use_upper:
        return f"Buy only when {base} drops below {lower} (deep-dip filter)"
    if use_upper and not use_lower:
        return f"Buy daily unless {base} is above {upper} (skip overbought days)"
    if use_lower and use_upper:
        return f"Buy only when {base} is below {lower} and below {upper}"
    return "Buy every day (no filter)"


def verdict(train_roi: float, test_roi: float) -> str:
    """Plain-language confidence rating combining OOS profit and retention."""
    if test_roi <= 0:
        return "❌ Negative out-of-sample — skip"
    ratio = test_roi / train_roi if train_roi > 0 else 1.0
    if test_roi >= 10 and ratio >= 0.20:
        return f"✓✓ Trustworthy — {test_roi:+.1f}% out-of-sample"
    if test_roi >= 5 and ratio >= 0.15:
        return f"✓ Reasonable — {test_roi:+.1f}% out-of-sample"
    return f"⚠ Marginal — {test_roi:+.1f}% out-of-sample (train {train_roi:+.1f}%)"


# ---------------------------------------------------------------------------
# Grid search + latest-day match check (mirrors index.html)
# ---------------------------------------------------------------------------
def find_best_config(frame: pd.DataFrame, fng_series: pd.Series | None) -> dict | None:
    close = frame["Close"].to_numpy(dtype=float)
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    volume = frame["Volume"].to_numpy(dtype=float)
    n = len(close)
    split_idx = int(n * SPLIT_RATIO)
    if split_idx < 15 or (n - split_idx) < 5:
        return None

    cache = {}
    for p in (7, 14, 21):
        cache[("RSI", p)] = indicator_rsi(close, p)
        cache[("MFI", p)] = indicator_mfi(high, low, close, volume, p)
    for p in (7, 14, 20):
        cache[("CCI", p)] = indicator_cci(high, low, close, p)
    if fng_series is not None:
        aligned = fng_series.reindex(frame.index, method="ffill").to_numpy(dtype=float)
        cache[("Fear & Greed", None)] = aligned

    best = None
    best_train_roi = float("-inf")

    for ind_name, grid in INDICATOR_GRIDS.items():
        if ind_name == "Fear & Greed" and fng_series is None:
            continue
        for config in grid["configs"]:
            key = (ind_name, config.get("period"))
            values = cache.get(key)
            if values is None:
                continue
            signal = np.ones(n, dtype=bool)
            if config.get("useLower"):
                signal &= values < config["lower"]
            if config.get("useUpper"):
                signal &= values < config["upper"]
            ev = evaluate_signal(close, signal, split_idx, 1.0)
            if ev["train_buy_pct"] < MIN_BUY_PCT:
                continue
            if ev["train_roi"] <= best_train_roi:
                continue
            if ev["train_roi"] <= 0:
                continue
            best_train_roi = ev["train_roi"]
            best = {
                "indicator": ind_name,
                "params": config,
                "label": format_label(ind_name, config),
                **ev,
            }
    return best


def latest_signal_check(best: dict, frame: pd.DataFrame,
                        fng_series: pd.Series | None
                        ) -> tuple[bool, float | None, str | None]:
    """Return (matches, current_value, indicator_name)."""
    ind_name = best["indicator"]
    if ind_name == "none":
        return True, None, None
    config = best["params"]
    close = frame["Close"].to_numpy(dtype=float)
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    volume = frame["Volume"].to_numpy(dtype=float)
    p = config.get("period")

    if ind_name == "RSI":
        values, name = indicator_rsi(close, p), f"RSI({p})"
    elif ind_name == "MFI":
        values, name = indicator_mfi(high, low, close, volume, p), f"MFI({p})"
    elif ind_name == "CCI":
        values, name = indicator_cci(high, low, close, p), f"CCI({p})"
    elif ind_name == "Fear & Greed":
        if fng_series is None:
            return False, None, "Fear & Greed"
        values = fng_series.reindex(frame.index, method="ffill").to_numpy(dtype=float)
        name = "Fear & Greed"
    else:
        return False, None, None

    if len(values) == 0 or not np.isfinite(values[-1]):
        return False, None, name
    v = float(values[-1])
    matched = True
    if config.get("useLower") and not (v < config["lower"]):
        matched = False
    if config.get("useUpper") and not (v < config["upper"]):
        matched = False
    return matched, v, name


def loopback_return(frame: pd.DataFrame) -> float:
    close = frame["Close"].dropna()
    if len(close) < 2:
        return 0.0
    first = float(close.iloc[0])
    if first <= 0:
        return 0.0
    return float((float(close.iloc[-1]) / first - 1) * 100)


# ---------------------------------------------------------------------------
# Message rendering + Telegram delivery
# ---------------------------------------------------------------------------
def chunks(header: str, blocks: list[str], limit: int = TELEGRAM_LIMIT):
    current = header
    for block in blocks:
        candidate = current + "\n" + block
        if len(candidate) > limit and current != header:
            yield current
            current = header + "\n" + block
        else:
            current = candidate
    yield current


def build_messages(
    as_of: str,
    alerts_by_category: dict[str, list[dict]],
    market_alerts: list[dict],
    header_label: str,
    score_label: str,
    limit: int = TELEGRAM_LIMIT,
) -> list[str]:
    scored: list[tuple[str, list[dict], float]] = []
    for category, entries in alerts_by_category.items():
        avg_profit = sum(e["profit"] for e in entries) / len(entries)
        scored.append((category, entries, avg_profit))
    scored.sort(key=lambda x: x[2], reverse=True)

    messages: list[str] = []
    for category, entries, avg_profit in scored:
        entries_sorted = sorted(entries, key=lambda e: e["profit"], reverse=True)
        n = len(entries_sorted)
        plural = "s" if n != 1 else ""
        header = (
            f"{header_label} ({as_of})\n"
            f"{category} — {n} asset{plural} — {score_label} {avg_profit:+.1f}%"
        )
        blocks = []
        for entry in entries_sorted:
            lines = "\n".join(f"  • {line}" for line in entry["lines"])
            blocks.append(f"- {entry['ticker']}\n{lines}")
        messages.extend(chunks(header, blocks, limit))

    if market_alerts:
        n = len(market_alerts)
        plural = "s" if n != 1 else ""
        header = f"{header_label} ({as_of})\nMarket — {n} signal{plural}"
        blocks = [
            f"- Fear & Greed\n  • {m['value']} ({m['day']})"
            for m in market_alerts
        ]
        messages.extend(chunks(header, blocks, limit))

    return messages


def send_telegram(messages: list[str]) -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(
            "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set; showing alerts in the console only.",
            file=sys.stderr,
        )
        for message in messages:
            print(message)
            print()
        return 0

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        for message in messages:
            body = urlencode(
                {"chat_id": chat_id, "text": message, "disable_web_page_preview": "true"}
            ).encode()
            with urlopen(Request(url, data=body), timeout=30) as response:
                payload = json.load(response)
            if not payload.get("ok"):
                raise RuntimeError(payload.get("description", "Telegram rejected the message"))
    except Exception as error:
        print(f"Telegram error: {error}", file=sys.stderr)
        return 1
    print(f"Telegram: sent {len(messages)} message(s).")
    return 0


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------
def run_alert(
    tickers: list[str],
    categories: dict[str, str],
    loopback_days: int,
    optimize: bool,
    dry_run: bool,
) -> int:
    fng_series: pd.Series | None = None
    try:
        fng_values = read_fear_greed()
        fng_series = build_fng_series(fng_values)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Fear & Greed: {error}", file=sys.stderr)

    alerts_by_category: dict[str, list[dict]] = {}
    market_alerts: list[dict] = []
    scanned = 0
    latest_days: list[str] = []

    for ticker in tickers:
        frame = load_local_frame(ticker, loopback_days)
        if frame is None:
            print(f"{ticker}: no local data (run --update first)", file=sys.stderr)
            continue
        if len(frame) < MIN_FRAME_DAYS:
            print(f"{ticker}: only {len(frame)} rows (< {MIN_FRAME_DAYS})", file=sys.stderr)
            continue

        scanned += 1
        latest_day = frame.index[-1].date().isoformat()
        latest_days.append(latest_day)
        category = categories.get(ticker, "Other")

        is_crypto = (category == "Crypto"
                     or ticker.upper().endswith("-USD")
                     or ticker.upper().endswith("-USDT"))
        ticker_fng = fng_series if is_crypto else None

        if optimize:
            best = find_best_config(frame, ticker_fng)
            if best is None:
                print(f"{ticker} {latest_day} -> no optimized strategy")
                continue

            if best["test_roi"] <= 0:
                print(f"{ticker} {latest_day} -> weak out-of-sample "
                      f"(test {best['test_roi']:+.1f}%), skipped")
                continue

            matched, current_value, ind_name = latest_signal_check(best, frame, ticker_fng)
            if matched:
                now_line = (f"Now {ind_name} = {current_value:.1f}"
                            if current_value is not None else "Now — (no live value)")
                entry = {
                    "ticker": ticker,
                    "profit": best["test_roi"],
                    "lines": [
                        plain_rule(best["indicator"], best["params"]),
                        now_line,
                        f"Backtest: {best['train_roi']:+.1f}% in-sample → "
                        f"{best['test_roi']:+.1f}% out-of-sample",
                        f"Triggers on ~{best['test_buy_pct']:.0f}% of days",
                        verdict(best["train_roi"], best["test_roi"]),
                    ],
                }
                alerts_by_category.setdefault(category, []).append(entry)
                print(f"{ticker} {latest_day} -> MATCH: "
                      f"{plain_rule(best['indicator'], best['params'])} "
                      f"(train {best['train_roi']:+.1f}% → test {best['test_roi']:+.1f}%)")
            else:
                print(f"{ticker} {latest_day} -> no match ({best['label']})")
        else:
            values = calculate_latest(frame)
            hits = [
                f"{name} {value:.1f}"
                for name, value in values.items()
                if math.isfinite(value) and value < THRESHOLDS[name]
            ]
            if hits:
                profit = loopback_return(frame)
                alerts_by_category.setdefault(category, []).append({
                    "ticker": ticker,
                    "profit": profit,
                    "lines": hits,
                })
                print(f"{ticker} {latest_day} -> ALERT: {', '.join(hits)} ({profit:+.1f}%)")
            else:
                print(f"{ticker} {latest_day} -> no alert")

    if scanned == 0:
        print("No local data found. Run --update first.", file=sys.stderr)
        return 1

    as_of = max(latest_days) if latest_days else date.today().isoformat()

    if not optimize and fng_series is not None:
        eligible = fng_series[fng_series.index <= pd.Timestamp(as_of)]
        if not eligible.empty:
            fear_day = eligible.index[-1].date().isoformat()
            fear_value = int(eligible.iloc[-1])
            print(f"Fear & Greed {fear_day}: {fear_value}")
            if fear_value < FEAR_GREED_THRESHOLD:
                market_alerts.append({"value": fear_value, "day": fear_day})
                print(f"Fear & Greed {fear_day} {fear_value} -> ALERT")

    total = sum(len(v) for v in alerts_by_category.values()) + len(market_alerts)
    print(f"\nScanned {scanned} assets; {total} alert(s).")
    if total == 0:
        return 0

    if optimize:
        header_label = "Daily strategy matches"
        score_label = "avg out-of-sample"
    else:
        header_label = "Daily oversold alerts"
        score_label = f"avg {loopback_days}d return"

    messages = build_messages(as_of, alerts_by_category, market_alerts,
                              header_label, score_label)

    scored = sorted(
        ((cat, sum(e["profit"] for e in entries) / len(entries), len(entries))
         for cat, entries in alerts_by_category.items()),
        key=lambda x: x[1], reverse=True,
    )
    if scored:
        print("Priority order (most profitable first):")
        for rank, (cat, avg, n) in enumerate(scored, start=1):
            print(f"  {rank}. {cat} — {n} asset(s) — {score_label} {avg:+.1f}%")
        if market_alerts:
            print(f"  {len(scored) + 1}. Market — Fear & Greed")

    if dry_run:
        print(f"\nDry run: would send {len(messages)} message(s).")
        for msg in messages:
            print(msg)
            print()
        return 0
    return send_telegram(messages)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="Refresh the daily snapshots in data/.")
    parser.add_argument("--alert", action="store_true",
                        help="Scan the latest local candle and notify.")
    parser.add_argument("--optimize", action="store_true",
                        help="With --alert: grid-search the best strategy per asset and alert only matches.")
    parser.add_argument("--loopback-days", type=int, default=365,
                        help="Analysis window in days (default: 365).")
    parser.add_argument("--ticker", action="append",
                        help="Limit --update and --alert to this ticker (can be repeated).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate without changing files and print alerts without sending Telegram.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print a status line for every ticker during --update.")
    args = parser.parse_args()

    if not args.update and not args.alert:
        parser.error("choose --update, --alert, or both")
    if args.optimize and not args.alert:
        parser.error("--optimize requires --alert")
    if args.alert and args.loopback_days < 30:
        parser.error("--loopback-days must be >= 30")
    if args.alert and args.optimize and args.loopback_days < 90:
        parser.error("--optimize requires --loopback-days >= 90")

    catalogue = load_catalogue()
    tickers = args.ticker or list(catalogue)
    unknown = sorted(set(tickers) - set(catalogue))
    if unknown:
        parser.error("Unknown ticker(s): " + ", ".join(unknown))

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    status = 0
    if args.update:
        status |= run_update(tickers, end, args.dry_run, args.verbose)
    if args.alert:
        status |= run_alert(tickers, catalogue, args.loopback_days,
                            args.optimize, args.dry_run)
    return status


if __name__ == "__main__":
    raise SystemExit(main())