"""CoinGecko crypto market vendor.

Prices, market capitalisation and 24-hour trading volume for crypto assets from
CoinGecko's free public API (https://api.coingecko.com/api/v3). No key is
needed; with ``COINGECKO_API_KEY`` set, it is sent as a Demo key in the
``x-cg-demo-api-key`` header for a steadier rate limit.

Two methods:

- ``get_crypto_market_data``: a snapshot as of a date (USD price, 24-hour and
  7-day change, market cap, 24-hour volume), served by the tool of that name.
- ``get_crypto_prices``: daily USD prices, volumes and market caps over a
  window, a ``get_stock_data`` vendor for crypto pairs. Yahoo stays the default;
  ``data_vendors["core_stock_apis"] = "yfinance,coingecko"`` falls back to
  CoinGecko when Yahoo cannot serve a crypto pair.

Point in time: both read ``/coins/{id}/market_chart/range`` bounded to end at
the close of the run date (midnight UTC after it), never the live-only
``/coins/markets``, so a past date sees only observations made by then.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime, timedelta

import requests

from tradingagents import __version__
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.net import get_scrubbed
from tradingagents.dataflows.vendors.yahoo.common import price_text

logger = logging.getLogger(__name__)

API_BASE = "https://api.coingecko.com/api/v3"

# Network timeout (seconds), consistent with the other vendors.
REQUEST_TIMEOUT = 30

# The keyless tier allows a handful of calls a minute and answers 429 past it.
# A short wait usually clears it; a limit that outlasts the retries is reported
# as the vendor being unavailable, never as an absence of data.
MAX_RETRIES = 2
RETRY_BASE_DELAY = 2.0
MAX_RETRY_DELAY = 15.0

# A snapshot reads a little over a week back, for the 7-day change.
SNAPSHOT_WINDOW_DAYS = 8

# A comparison price must lie within this distance of the time it stands for,
# or the change is reported as unavailable instead of spanning a gap.
_COMPARISON_TOLERANCE = timedelta(hours=36)

# The latest observation must be this recent relative to the end of the window,
# or the series is treated as stale rather than served as current.
_STALE_AFTER = timedelta(days=3)

# Ticker base -> CoinGecko coin id. CoinGecko ids are not the ticker (XRP is
# "ripple", AVAX is "avalanche-2"), so only listed bases resolve. Extend by
# adding rows.
COIN_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "BNB": "binancecoin",
    "TRX": "tron",
    "TON": "the-open-network",
    "XLM": "stellar",
    "ATOM": "cosmos",
    "UNI": "uniswap",
    "NEAR": "near",
    "SHIB": "shiba-inu",
    "XMR": "monero",
    "ETC": "ethereum-classic",
    "HBAR": "hedera-hashgraph",
    "SUI": "sui",
    "APT": "aptos",
}

# Quotes read as US dollars, as the rest of the pipeline reads them (#982).
# Longest first so USDT/USDC match before the USD inside them.
_USD_QUOTES = ("USDT", "USDC", "USD")


class CoinGeckoNotConfiguredError(VendorNotConfiguredError):
    """CoinGecko refused the configured ``COINGECKO_API_KEY``.

    A VendorNotConfiguredError (and thus still a ValueError), so the routing
    layer treats it as "vendor unavailable" like the other vendors' key errors.
    """


def coin_id(symbol: str) -> str | None:
    """The CoinGecko coin id for a USD-quoted crypto pair, else None.

    Accepts the forms the pipeline holds: ``BTC-USD``, ``BTCUSD``, ``BTC-USDT``,
    ``BTCUSDC``. A bare base (``SOL``) is not read as crypto: it is also an
    equity ticker. Purely syntactic, no network.
    """
    if not isinstance(symbol, str):
        return None
    compact = symbol.strip().upper().rstrip("+")
    if "-" in compact:
        base, _, quote = compact.partition("-")
        return COIN_IDS.get(base) if quote in _USD_QUOTES else None
    for quote in _USD_QUOTES:
        if compact.endswith(quote) and compact[: -len(quote)] in COIN_IDS:
            return COIN_IDS[compact[: -len(quote)]]
    return None


def _not_covered(symbol: str) -> str:
    return (
        f"CoinGecko covers USD-quoted crypto pairs only, and {symbol!r} is not one it "
        f"resolves (supported: {', '.join(f'{b}-USD' for b in COIN_IDS)}). No crypto "
        f"market data applies to this instrument; do not estimate any."
    )


def _headers() -> tuple[dict, str]:
    key = os.getenv("COINGECKO_API_KEY", "").strip()
    headers = {"Accept": "application/json", "User-Agent": f"TradingAgents/{__version__}"}
    if key:
        headers["x-cg-demo-api-key"] = key
    return headers, key


def _error_message(response, key: str) -> str:
    """CoinGecko's own explanation of a refusal, wherever its body puts it."""
    try:
        body = response.json()
    except ValueError:
        body = getattr(response, "text", "") or ""

    def find(node) -> str:
        if isinstance(node, str):
            return node
        if isinstance(node, dict):
            for name in ("error_message", "error", "status", "message"):
                found = find(node.get(name))
                if found:
                    return found
        return ""

    message = find(body)[:300]
    return message.replace(key, "***") if key else message


def _retry_delay(response, attempt: int) -> float:
    try:
        delay = float(response.headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        delay = RETRY_BASE_DELAY * (2 ** attempt)
    return max(0.0, min(delay, MAX_RETRY_DELAY))


def _request(path: str, params: dict, symbol: str, coin: str) -> dict:
    """GET a CoinGecko endpoint, mapping every failure onto the vendor error types."""
    headers, key = _headers()
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = get_scrubbed(
                f"{API_BASE}/{path}",
                params=params,
                timeout=REQUEST_TIMEOUT,
                secret=key,
                passthrough=(400, 401, 403, 404, 429),
                headers=headers,
            )
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise VendorUnavailableError(
                f"CoinGecko request failed ({status or type(exc).__name__})"
            ) from exc

        status = response.status_code
        if status == 429:
            if attempt < MAX_RETRIES:
                delay = _retry_delay(response, attempt)
                logger.warning("CoinGecko rate limited; retrying in %.0fs (attempt %d/%d)",
                               delay, attempt + 1, MAX_RETRIES)
                time.sleep(delay)
                continue
            raise VendorUnavailableError(
                f"CoinGecko rate limit reached (HTTP 429) after {MAX_RETRIES} retries; "
                f"a free Demo key in COINGECKO_API_KEY raises the limit"
            )
        if status == 404:
            raise NoMarketDataError(symbol, coin, "CoinGecko lists no coin under this id")
        if status in (400, 401, 403):
            message = _error_message(response, key)
            if "time range" in message.lower() or "365 days" in message.lower():
                # The keyless and Demo tiers serve about a year of history.
                raise NoMarketDataError(
                    symbol, coin,
                    "CoinGecko's free tier serves only recent history (about the past year)",
                )
            if key and status in (401, 403):
                raise CoinGeckoNotConfiguredError(
                    f"CoinGecko refused COINGECKO_API_KEY (HTTP {status}): {message}"
                )
            raise VendorUnavailableError(f"CoinGecko refused the request (HTTP {status}): {message}")
        try:
            return response.json()
        except ValueError as exc:
            raise VendorUnavailableError("CoinGecko returned an unreadable response") from exc
    raise VendorUnavailableError("CoinGecko request was not answered")  # pragma: no cover


def _day_bounds(day: str) -> tuple[datetime, datetime]:
    """``[day 00:00 UTC, next day 00:00 UTC)``: a run dated ``day`` sees up to its close."""
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
    return start, start + timedelta(days=1)


def _points(series, start: datetime, cutoff: datetime) -> list[tuple[datetime, float]]:
    """``[[ms, value], ...]`` as (UTC time, value) in ``[start, cutoff)``, oldest first.

    The request is already bounded, but the bound is applied again here: it is
    the only thing keeping an observation after the run date out of a backtest.
    """
    points = []
    for row in series or []:
        try:
            when = datetime.fromtimestamp(float(row[0]) / 1000, tz=UTC)
            value = float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if start <= when < cutoff:
            points.append((when, value))
    return sorted(points)


def _market_chart(symbol: str, coin: str, start: datetime, cutoff: datetime) -> dict:
    end = min(cutoff, datetime.now(UTC))
    return _request(
        f"coins/{coin}/market_chart/range",
        {"vs_currency": "usd", "from": int(start.timestamp()), "to": int(end.timestamp())},
        symbol,
        coin,
    )


def _at_or_before(points: list[tuple[datetime, float]], when: datetime) -> float | None:
    """The value observed at ``when`` or the latest before it, if close enough."""
    earlier = [(t, v) for t, v in points if t <= when]
    if not earlier or when - earlier[-1][0] > _COMPARISON_TOLERANCE:
        return None
    return earlier[-1][1]


def _compact_usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"${value / size:,.2f}{suffix}"
    return f"${value:,.0f}"


def _change(latest: float, earlier: float | None) -> str:
    if earlier is None or earlier == 0:
        return "n/a (no observation that far back)"
    return f"{(latest / earlier - 1) * 100:+.2f}% (from ${price_text(earlier)})"


def _assert_fresh(symbol: str, coin: str, latest: datetime, window_end: datetime, as_of: str) -> None:
    if window_end - latest > _STALE_AFTER:
        raise NoMarketDataError(
            symbol, coin,
            f"latest CoinGecko observation is from {latest:%Y-%m-%d}, stale for {as_of}",
        )


def get_crypto_market_data(symbol: str, as_of_date: str) -> str:
    """A crypto market snapshot as of ``as_of_date``, from observations made by its close.

    Args:
        symbol: The instrument, e.g. ``BTC-USD``. Anything CoinGecko does not
            resolve gets a short notice instead of data.
        as_of_date: The analysis date (yyyy-mm-dd). Only observations made
            before midnight UTC after it are read, so a past date never sees a
            later price, market cap or volume.

    Returns:
        A markdown summary: USD price, 24-hour and 7-day change, market
        capitalisation and 24-hour trading volume.
    """
    coin = coin_id(symbol)
    if coin is None:
        return _not_covered(symbol)
    try:
        _, cutoff = _day_bounds(as_of_date)
    except (TypeError, ValueError):
        return f"CoinGecko: as_of_date {as_of_date!r} is not a date; give it as YYYY-MM-DD."

    window_end = min(cutoff, datetime.now(UTC))
    start = window_end - timedelta(days=SNAPSHOT_WINDOW_DAYS)
    data = _market_chart(symbol, coin, start, cutoff)
    prices = _points(data.get("prices"), start, cutoff)
    if not prices:
        raise NoMarketDataError(symbol, coin, f"no CoinGecko prices in the week to {as_of_date}")
    latest_at, latest = prices[-1]
    _assert_fresh(symbol, coin, latest_at, window_end, as_of_date)

    caps = _points(data.get("market_caps"), start, cutoff)
    volumes = _points(data.get("total_volumes"), start, cutoff)
    market_cap = _at_or_before(caps, latest_at)
    volume = _at_or_before(volumes, latest_at)

    return (
        f"## CoinGecko market data: {symbol.strip().upper()} (CoinGecko id: {coin})\n"
        f"- As of: {as_of_date}; latest observation {latest_at:%Y-%m-%d %H:%M} UTC\n"
        f"- Price: ${price_text(latest)}\n"
        f"- 24h change: {_change(latest, _at_or_before(prices, latest_at - timedelta(days=1)))}\n"
        f"- 7d change: {_change(latest, _at_or_before(prices, latest_at - timedelta(days=7)))}\n"
        f"- Market cap: {_compact_usd(market_cap)}\n"
        f"- 24h trading volume: {_compact_usd(volume)}\n"
        f"\nOnly observations made by the end of {as_of_date} (UTC) are used.\n"
    )


def get_crypto_prices(symbol: str, start_date: str, end_date: str) -> str:
    """Daily USD prices for a crypto pair over ``[start_date, end_date]``, as CSV.

    A ``get_stock_data`` vendor: same arguments, and the same typed errors so the
    router can fall through to the next configured vendor. Each row is the last
    observation CoinGecko made on that UTC day. CoinGecko samples hourly for a
    window of up to 90 days and once a day (at 00:00 UTC) beyond that, and
    serves no open, high or low.

    Raises:
        NoMarketDataError: not a crypto pair CoinGecko resolves, or no
            observations in the window.
        VendorUnavailableError: CoinGecko could not be reached or throttled.
    """
    coin = coin_id(symbol)
    if coin is None:
        raise NoMarketDataError(symbol, symbol, "not a USD-quoted crypto pair CoinGecko covers")
    start, _ = _day_bounds(start_date)
    _, cutoff = _day_bounds(end_date)
    if start >= min(cutoff, datetime.now(UTC)):
        raise NoMarketDataError(symbol, coin, f"no CoinGecko prices between {start_date} and {end_date}")

    data = _market_chart(symbol, coin, start, cutoff)
    prices = _points(data.get("prices"), start, cutoff)
    if not prices:
        raise NoMarketDataError(symbol, coin, f"no CoinGecko prices between {start_date} and {end_date}")
    _assert_fresh(symbol, coin, prices[-1][0], min(cutoff, datetime.now(UTC)), end_date)

    daily: dict[str, dict] = {}
    for name, series in (("Close", prices),
                         ("Volume", _points(data.get("total_volumes"), start, cutoff)),
                         ("Market Cap", _points(data.get("market_caps"), start, cutoff))):
        for when, value in series:   # oldest first, so the day's last observation wins
            daily.setdefault(f"{when:%Y-%m-%d}", {})[name] = value

    rows = ["Date,Close,Volume,Market Cap"]
    for day in sorted(daily):
        values = daily[day]
        if "Close" not in values:
            continue
        volume = values.get("Volume")
        cap = values.get("Market Cap")
        rows.append(",".join([
            day,
            price_text(values["Close"]),
            "" if volume is None else f"{volume:.0f}",
            "" if cap is None else f"{cap:.0f}",
        ]))

    label = symbol.strip().upper()
    header = (
        f"# Crypto price data for {label} (CoinGecko: {coin}) from {start_date} to {end_date}\n"
        f"# Total records: {len(rows) - 1}\n"
        f"# Close is the last USD price CoinGecko observed on each UTC day; Volume is the "
        f"24-hour USD volume at that time. No open, high or low is available from this vendor.\n\n"
    )
    return header + "\n".join(rows) + "\n"
