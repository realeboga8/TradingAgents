"""ForexFactory economic calendar vendor.

Lists the scheduled macro releases in the days ahead (central-bank decisions,
CPI, payrolls, PMIs, GDP) with ForexFactory's impact rating, the consensus
forecast and the previous reading, so the news analyst can say which events
fall inside a decision's horizon instead of guessing at the calendar.

Reads ForexFactory's public weekly JSON export
(https://nfs.faireconomy.media/ff_calendar_thisweek.json): no key, no account,
nothing scraped. Each event carries ``title``, ``country`` (the currency it
moves, e.g. ``USD``), ``date`` (ISO 8601 with a UTC offset), ``impact`` (High,
Medium, Low or Holiday), ``forecast`` and ``previous``.

Point in time: the export covers only the current week (Sunday to Saturday) and
keeps no archive. A date outside that week is reported as unavailable, never as
a week without events, and a result (``actual``) released after the run date is
never shown, should the feed carry one.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import requests

from tradingagents import __version__
from tradingagents.dataflows.date_window import is_historical, to_utc
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.net import get_scrubbed

logger = logging.getLogger(__name__)

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Network timeout (seconds), consistent with the other vendors.
REQUEST_TIMEOUT = 30

# The export throttles clients that download it often, and it changes only when
# an event is added or moved, so one download serves a run's calls for a while.
CACHE_TTL_SECONDS = 15 * 60

DEFAULT_LOOK_AHEAD_DAYS = 7
MAX_LOOK_AHEAD_DAYS = 14
DEFAULT_MIN_IMPACT = "medium"

# Row cap for the rendered table: a week with every low-impact release for every
# currency runs to a few hundred rows and would flood the analyst's context.
MAX_ROWS = 60

# min_impact keeps events at or above its rank. Holidays (bank and exchange
# closures) are listed whatever the level: they move liquidity, not a forecast.
_IMPACT_RANK = {"all": 0, "low": 1, "medium": 2, "high": 3}
_HOLIDAY = "holiday"

_cache: tuple[float, list] | None = None


def clear_cache() -> None:
    """Forget the cached export, so the next call downloads it again."""
    global _cache
    _cache = None


def _fetch_feed() -> list:
    """The week's events as the export serves them, from the cache while fresh.

    Every failure to get a readable answer is a ``VendorUnavailableError``: it
    says nothing about the calendar, and the router reports it as unavailable.
    """
    global _cache
    if _cache is not None and time.monotonic() - _cache[0] < CACHE_TTL_SECONDS:
        return _cache[1]
    try:
        response = get_scrubbed(
            FEED_URL,
            params={},
            timeout=REQUEST_TIMEOUT,
            secret="",
            passthrough=(429,),
            headers={"User-Agent": f"TradingAgents/{__version__}", "Accept": "application/json"},
        )
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise VendorUnavailableError(
            f"ForexFactory calendar request failed ({status or type(exc).__name__})"
        ) from exc
    if response.status_code == 429:
        raise VendorUnavailableError(
            "ForexFactory calendar is rate limiting this client (HTTP 429); "
            "it allows only a few downloads every few minutes"
        )
    try:
        events = response.json()
    except ValueError as exc:
        # A throttled or failing endpoint can answer with an HTML page.
        raise VendorUnavailableError("ForexFactory calendar returned an unreadable response") from exc
    if not isinstance(events, list):
        raise VendorUnavailableError("ForexFactory calendar returned an unexpected payload")
    _cache = (time.monotonic(), events)
    return events


@dataclass(frozen=True)
class _Event:
    local: datetime     # as published, with the feed's own UTC offset
    when: datetime      # the same instant in UTC
    currency: str
    title: str
    impact: str
    forecast: str
    previous: str
    actual: str


def _text(raw: dict, key: str) -> str:
    value = raw.get(key)
    return "" if value is None else str(value).strip()


def _parse_event(raw) -> _Event | None:
    """One export row, or None when it has no readable date."""
    if not isinstance(raw, dict):
        return None
    try:
        local = datetime.fromisoformat(str(raw["date"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    return _Event(
        local=local,
        when=to_utc(local),
        currency=_text(raw, "country").upper(),
        title=_text(raw, "title"),
        impact=_text(raw, "impact"),
        forecast=_text(raw, "forecast"),
        previous=_text(raw, "previous"),
        actual=_text(raw, "actual"),
    )


def _feed_week(events: list[_Event]) -> tuple[date, date]:
    """The Sunday-to-Saturday week the export covers, from its earliest event."""
    first = min(e.local.date() for e in events)
    start = first - timedelta(days=(first.weekday() + 1) % 7)
    return start, start + timedelta(days=6)


def _cell(value: str) -> str:
    """A table cell: pipes would split the row, and an empty cell reads as a dash."""
    return value.replace("|", "/").strip() or "-"


def _unavailable(curr_date: str, reason: str) -> str:
    return (
        f"DATA_UNAVAILABLE: no economic calendar is available for {curr_date}. {reason} "
        f"This is not an absence of scheduled events; proceed without the calendar "
        f"and do not reconstruct it from memory."
    )


def get_economic_calendar(
    curr_date: str,
    look_ahead_days: int | None = None,
    min_impact: str | None = None,
    currencies: str | None = None,
) -> str:
    """Scheduled economic releases from ``curr_date`` through ``look_ahead_days`` ahead.

    Args:
        curr_date: The analysis date (yyyy-mm-dd); the window starts on this day.
        look_ahead_days: Days after ``curr_date`` to include (0 = that day only);
            ``None`` uses DEFAULT_LOOK_AHEAD_DAYS, and it is capped at
            MAX_LOOK_AHEAD_DAYS.
        min_impact: ``"high"``, ``"medium"``, ``"low"`` or ``"all"``
            (case-insensitive); ``None`` uses DEFAULT_MIN_IMPACT. Holidays are
            always listed.
        currencies: Comma- or space-separated currency codes to keep (e.g.
            ``"USD,EUR"``); ``None`` keeps every currency. Events ForexFactory
            files under ``All`` are kept with any selection.

    Returns:
        A markdown table of the events in the window (UTC times), or a
        ``DATA_UNAVAILABLE`` notice when the export does not cover ``curr_date``.

    Raises:
        VendorUnavailableError: the export could not be downloaded or read.
    """
    try:
        run_day = datetime.strptime(curr_date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return f"ForexFactory: curr_date {curr_date!r} is not a date; give it as YYYY-MM-DD."

    impact_key = (min_impact or DEFAULT_MIN_IMPACT).strip().lower()
    if impact_key not in _IMPACT_RANK:
        return (
            f"ForexFactory: min_impact {min_impact!r} is not a level; use 'high', "
            f"'medium', 'low' or 'all'."
        )
    try:
        days = DEFAULT_LOOK_AHEAD_DAYS if look_ahead_days is None else int(look_ahead_days)
    except (TypeError, ValueError):
        return f"ForexFactory: look_ahead_days {look_ahead_days!r} is not a number of days."
    days = max(0, min(days, MAX_LOOK_AHEAD_DAYS))

    wanted = {c for c in re.split(r"[\s,;/]+", (currencies or "").upper()) if c}
    if wanted & {"ALL", "ANY"}:
        wanted = set()

    events = [e for e in (_parse_event(raw) for raw in _fetch_feed()) if e is not None]
    if not events:
        raise VendorUnavailableError("ForexFactory calendar export carried no readable events")

    week_start, week_end = _feed_week(events)
    if run_day < week_start:
        # Name no date after the run's own: the export's week would tell a
        # backtest when it is really running.
        return _unavailable(
            curr_date,
            "ForexFactory's free export serves only the current week's calendar and "
            "keeps no archive, so the events scheduled around this date cannot be listed.",
        )
    if run_day > week_end:
        return _unavailable(
            curr_date,
            f"ForexFactory's free export currently covers {week_start} to {week_end} "
            f"only, and has not yet published the calendar for this date.",
        )

    window_end = run_day + timedelta(days=days)
    start_dt = datetime.combine(run_day, datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(window_end + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    # A release is known from the run date only if it came out by the end of
    # that day (UTC): the same half-open bound every dated feed uses.
    known_by = start_dt + timedelta(days=1)

    min_rank = _IMPACT_RANK[impact_key]
    selected = sorted(
        (
            e for e in events
            if start_dt <= e.when < end_dt
            and (e.impact.lower() == _HOLIDAY
                 or _IMPACT_RANK.get(e.impact.lower(), 0) >= min_rank)
            and (not wanted or e.currency in wanted or e.currency == "ALL")
        ),
        key=lambda e: (e.when, e.currency, e.title),
    )
    actuals = {e: e.actual if e.when < known_by else "" for e in selected}
    show_actual = any(actuals.values())

    level = "all levels" if impact_key == "all" else f"{impact_key.capitalize()} and above"
    header = (
        f"## Economic calendar (ForexFactory): {run_day} to {window_end}\n"
        f"- Impact: {level} (holidays always listed)\n"
        f"- Currencies: {', '.join(sorted(wanted)) if wanted else 'all'}\n"
        f"- Times are UTC. Forecast is the consensus estimate; Previous is the prior reading.\n"
    )
    notes = []
    if window_end > week_end:
        notes.append(
            f"The export covers the calendar through {week_end} only; days after it "
            f"are not listed, which is not an absence of events."
        )
    if is_historical(curr_date):
        notes.append(
            f"Forecast and previous values are as ForexFactory publishes them now and "
            f"may include revisions made after {curr_date}; results released after "
            f"{curr_date} are withheld."
        )
    note_text = "".join(f"\n_{n}_\n" for n in notes)

    if not selected:
        return header + note_text + (
            f"\nNo events at this impact level are scheduled in this window"
            f"{' for ' + ', '.join(sorted(wanted)) if wanted else ''}.\n"
        )

    shown = selected[:MAX_ROWS]
    if len(selected) > MAX_ROWS:
        note_text += (
            f"\n_(showing the first {MAX_ROWS} of {len(selected)} events; raise "
            f"min_impact or name currencies to narrow the list)_\n"
        )

    columns = ["Date", "Time (UTC)", "Currency", "Impact", "Event", "Forecast", "Previous"]
    if show_actual:
        columns.append("Actual")
    rows = []
    for e in shown:
        holiday = e.impact.lower() == _HOLIDAY
        cells = [
            e.local.date().isoformat() if holiday else e.when.date().isoformat(),
            "All day" if holiday else e.when.strftime("%H:%M"),
            e.currency, e.impact, e.title, e.forecast, e.previous,
        ]
        if show_actual:
            cells.append(actuals[e])
        rows.append("| " + " | ".join(_cell(c) for c in cells) + " |")

    table = (
        "\n| " + " | ".join(columns) + " |\n"
        + "| " + " | ".join("---" for _ in columns) + " |\n"
        + "\n".join(rows) + "\n"
    )
    return header + note_text + table
