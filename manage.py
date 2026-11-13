#!/usr/bin/env python3
"""Keep the daily snapshots current and alert on oversold daily indicators.

Usage:
    python manage.py --update              # refresh data/*.js from Yahoo Finance
    python manage.py --alert               # scan 1D candles and notify Telegram
    python manage.py --update --alert      # refresh snapshots, then alert

Telegram delivery needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the
environment; without them alerts are printed to the console only.
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


def latest_fear_greed(values: dict[str, int], as_of: str) -> tuple[str, int] | None:
    eligible = [(day, value) for day, value in values.items() if day <= as_of]
    if not eligible:
        return None
    return eligible[-1]


def run_update(tickers: list[str], end: date, dry_run: bool, verbose: bool) -> int:
    print(f"Updating {len(tickers)} market snapshots through {end.isoformat()}")
    if dry_run:
        snapshot_ends = []
        for ticker in tickers:
            snapshot_ends.append((ticker, read_snapshot(DATA_DIR / f"{ticker}.js", ticker)["d"][-1]))
        print(f"Validated {len(tickers)} market snapshots; no files changed.")
        if verbose:
            for ticker, latest in snapshot_ends:
                print(f"{ticker}: {latest} -> {end.isoformat()}")
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


def calculate(frame: pd.DataFrame) -> dict[str, float]:
    high, low, close, volume = (
        frame[column].astype(float) for column in ("High", "Low", "Close", "Volume")
    )
    typical = (high + low + close) / 3

    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, math.nan))
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50).mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )

    money_flow = typical * volume
    positive = money_flow.where(typical > typical.shift(1), 0).rolling(14).sum()
    negative = money_flow.where(typical < typical.shift(1), 0).rolling(14).sum()
    mfi = 100 - 100 / (1 + positive / negative.replace(0, math.nan))
    mfi = mfi.mask((negative == 0) & (positive > 0), 100).mask(
        (negative == 0) & (positive == 0), 50
    )

    mean = typical.rolling(20).mean()
    deviation = typical.rolling(20).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    cci = (typical - mean) / (0.015 * deviation.replace(0, math.nan))

    return {
        "RSI": rsi.iloc[-1],
        "MFI": mfi.iloc[-1],
        "CCI": cci.iloc[-1],
    }


def chunks(header: str, blocks: list[str], limit: int = TELEGRAM_LIMIT):
    """Yield messages built from header + multi-line blocks, respecting limit."""
    current = header
    for block in blocks:
        candidate = current + "\n" + block
        if len(candidate) > limit and current != header:
            yield current
            current = header + "\n" + block
        else:
            current = candidate
    yield current


def build_category_messages(
    as_of: str,
    grouped: dict[str, list[tuple[str, list[str]]]],
    limit: int = TELEGRAM_LIMIT,
) -> list[str]:
    """Render one or more Telegram messages per category."""
    messages: list[str] = []
    for category, assets in grouped.items():
        blocks = []
        for ticker, hits in assets:
            indicator_lines = "\n".join(f"  • {hit}" for hit in hits)
            blocks.append(f"- {ticker}\n{indicator_lines}")

        asset_count = len(assets)
        plural = "s" if asset_count != 1 else ""
        header = (
            f"({as_of}) {category} — {asset_count} asset{plural}"
        )
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


def run_alert(tickers: list[str], categories: dict[str, str], dry_run: bool) -> int:
    with warnings.catch_warnings():
        pandas4_warning = getattr(pd.errors, "Pandas4Warning", None)
        if pandas4_warning:
            warnings.simplefilter("ignore", pandas4_warning)
        try:
            data = yf.download(
                tickers,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as error:
            print(f"Daily data download failed: {error}", file=sys.stderr)
            return 1
    if data.empty:
        print("No daily data returned.", file=sys.stderr)
        return 1

    as_of = data.index[-1].date().isoformat()
    grouped: dict[str, list[tuple[str, list[str]]]] = {}
    scanned = 0

    for ticker in tickers:
        if (ticker, "Close") not in data.columns:
            print(f"{ticker}: no data", file=sys.stderr)
            continue
        frame = data[ticker].dropna(subset=["Close"])
        if len(frame) < 20:
            print(f"{ticker}: not enough daily bars", file=sys.stderr)
            continue

        values = calculate(frame)
        scanned += 1
        day = frame.index[-1].date().isoformat()
        detail = "  ".join(
            f"{name} {value:.1f}" for name, value in values.items() if math.isfinite(value)
        )
        hits = [
            f"{name} {value:.1f}"
            for name, value in values.items()
            if math.isfinite(value) and value < THRESHOLDS[name]
        ]
        if hits:
            print(f"{ticker} {day}  {detail}  -> ALERT: {', '.join(hits)}")
            category = categories.get(ticker, "Other")
            grouped.setdefault(category, []).append((ticker, hits))
        else:
            print(f"{ticker} {day}  {detail}  -> no alert")

    try:
        fear_values = read_fear_greed()
        latest = latest_fear_greed(fear_values, as_of)
        if latest is None:
            print("Fear & Greed: no value on or before " + as_of, file=sys.stderr)
        else:
            fear_day, fear_value = latest
            print(f"Fear & Greed {fear_day}: {fear_value}")
            if fear_value < FEAR_GREED_THRESHOLD:
                grouped.setdefault("Market", []).append(
                    ("Fear & Greed", [f"Fear & Greed {fear_value} ({fear_day})"])
                )
                print(f"Fear & Greed {fear_day} {fear_value} -> ALERT")
            else:
                print(f"Fear & Greed {fear_day} {fear_value} -> no alert")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Fear & Greed: ERROR: {error}", file=sys.stderr)

    alert_count = sum(len(assets) for assets in grouped.values())
    print(f"\nScanned {scanned} pairs on 1D; {alert_count} asset(s) hit at least one threshold.")
    if scanned == 0:
        return 1
    if not grouped:
        return 0

    messages = build_category_messages(as_of, grouped)
    if dry_run:
        print(
            f"Dry run: would send {len(messages)} message(s) covering "
            f"{alert_count} alert(s) to Telegram."
        )
        for message in messages:
            print(message)
            print()
        return 0
    return send_telegram(messages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Refresh the daily snapshots in data/.")
    parser.add_argument("--alert", action="store_true", help="Scan the 1D timeframe and send oversold alerts.")
    parser.add_argument(
        "--ticker",
        action="append",
        help="Limit --update and --alert to this ticker (can be specified more than once).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate --update without changing files and show --alert results without sending Telegram.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a status line for every ticker during --update.",
    )
    args = parser.parse_args()
    if not args.update and not args.alert:
        parser.error("choose --update, --alert, or both")

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
        status |= run_alert(tickers, catalogue, args.dry_run)
    return status


if __name__ == "__main__":
    raise SystemExit(main())