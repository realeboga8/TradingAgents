"""ForexFactory economic calendar vendor: the window, impact and currency
filters, the current-week-only coverage rule, point-in-time handling of
released results, graceful failure, router integration and agent wiring.

All HTTP is mocked, so these run without a network connection. The feed is a
fixed week in 2030, so nothing depends on the real clock unless a test pins it.
"""
import re
from unittest import mock

import pytest
import requests

import tradingagents.dataflows.date_window as date_window
import tradingagents.dataflows.net as net
from tradingagents.agents import tools
from tradingagents.agents.analysts import news_analyst
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors import forexfactory as ff
from tradingagents.default_config import DEFAULT_CONFIG


def _event(title, country, date, impact, forecast="", previous="", **extra):
    return {"title": title, "country": country, "date": date, "impact": impact,
            "forecast": forecast, "previous": previous, **extra}


# The week of Sunday 2030-06-02 to Saturday 2030-06-08, in ForexFactory's
# US Eastern offset, as the export publishes it.
FEED = [
    _event("Prelim GDP q/q", "JPY", "2030-06-02T19:50:00-04:00", "High", "0.3%", "-0.1%"),
    _event("German Final Manufacturing PMI", "EUR", "2030-06-03T04:00:00-04:00", "Low", "45.4", "45.4"),
    _event("ISM Manufacturing PMI", "USD", "2030-06-03T10:00:00-04:00", "High", "49.5", "48.7",
           actual="50.1"),
    _event("JOLTS Job Openings", "USD", "2030-06-04T10:00:00-04:00", "Medium", "8.10M", "8.49M",
           actual="7.92M"),
    _event("Bank Holiday", "CAD", "2030-06-05T00:00:00-04:00", "Holiday"),
    _event("Main Refinancing Rate", "EUR", "2030-06-06T08:15:00-04:00", "High", "3.75%", "4.00%"),
    _event("OPEC-JMMC Meetings", "All", "2030-06-06T05:00:00-04:00", "Medium"),
    _event("Non-Farm Employment Change", "USD", "2030-06-07T08:30:00-04:00", "High", "180K", "175K"),
    _event("Fed Chair Speaks | Q&A", "USD", "2030-06-07T13:00:00-04:00", "Medium"),
    _event("Unreadable", "USD", "not a date", "High"),
]


@pytest.fixture(autouse=True)
def _fresh_cache():
    ff.clear_cache()
    yield
    ff.clear_cache()


@pytest.fixture
def feed():
    with mock.patch.object(ff, "_fetch_feed", return_value=list(FEED)) as fetch:
        yield fetch


def _response(status=200, payload=None, json_error=False):
    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.headers = {}
    response.text = "<html>Request Denied</html>" if json_error else ""
    if json_error:
        response.json.side_effect = ValueError("Expecting value")
    else:
        response.json.return_value = payload
    response.raise_for_status.side_effect = (
        requests.HTTPError(f"{status} Server Error", response=response) if status >= 400 else None
    )
    return response


def _dates_after(text: str, cutoff: str) -> list[str]:
    return [d for d in re.findall(r"\d{4}-\d{2}-\d{2}", text) if d > cutoff]


@pytest.mark.unit
class TestCalendarTable:
    def test_lists_the_window_in_utc_time_order(self, feed):
        out = ff.get_economic_calendar("2030-06-03", 3, "all")
        assert "## Economic calendar (ForexFactory): 2030-06-03 to 2030-06-06" in out
        assert "| 2030-06-03 | 14:00 | USD | High | ISM Manufacturing PMI | 49.5 | 48.7 |" in out
        assert out.index("German Final Manufacturing PMI") < out.index("ISM Manufacturing PMI")
        assert out.index("ISM Manufacturing PMI") < out.index("JOLTS Job Openings")
        # The window starts on the run date and ends look_ahead_days later.
        assert "Prelim GDP" not in out          # Sunday 23:50 UTC, before the window
        assert "Non-Farm" not in out            # Friday, after it

    def test_an_evening_release_is_dated_by_its_utc_day(self, feed):
        out = ff.get_economic_calendar("2030-06-02", 0, "all")
        assert "| 2030-06-02 | 23:50 | JPY | High | Prelim GDP q/q |" in out

    def test_min_impact_filters_but_holidays_stay(self, feed):
        out = ff.get_economic_calendar("2030-06-03", 6, "HIGH")
        for kept in ("ISM Manufacturing PMI", "Main Refinancing Rate", "Non-Farm Employment Change"):
            assert kept in out
        for dropped in ("JOLTS Job Openings", "German Final Manufacturing PMI", "OPEC-JMMC"):
            assert dropped not in out
        assert "| 2030-06-05 | All day | CAD | Holiday | Bank Holiday |" in out

    def test_default_impact_is_medium(self, feed):
        out = ff.get_economic_calendar("2030-06-03")
        assert "Medium and above" in out
        assert "JOLTS Job Openings" in out and "German Final Manufacturing PMI" not in out

    def test_currency_filter_keeps_events_filed_under_all(self, feed):
        out = ff.get_economic_calendar("2030-06-03", 6, "all", "usd")
        assert "ISM Manufacturing PMI" in out and "OPEC-JMMC Meetings" in out
        for other in ("German Final Manufacturing PMI", "Main Refinancing Rate", "Bank Holiday"):
            assert other not in out
        assert "- Currencies: USD" in out

    def test_several_currencies(self, feed):
        out = ff.get_economic_calendar("2030-06-03", 6, "high", "USD, EUR")
        assert "Main Refinancing Rate" in out and "ISM Manufacturing PMI" in out
        assert "- Currencies: EUR, USD" in out

    def test_a_pipe_in_a_title_does_not_split_the_row(self, feed):
        out = ff.get_economic_calendar("2030-06-07", 0, "all")
        assert "Fed Chair Speaks / Q&A" in out

    def test_an_empty_value_reads_as_a_dash(self, feed):
        out = ff.get_economic_calendar("2030-06-06", 0, "all")
        assert "| OPEC-JMMC Meetings | - | - |" in out

    def test_no_matching_events_says_so(self, feed):
        out = ff.get_economic_calendar("2030-06-08", 0, "high")
        assert "No events at this impact level" in out

    def test_row_cap(self, monkeypatch):
        many = [_event(f"Event {i}", "USD", f"2030-06-03T{i % 24:02d}:00:00+00:00", "High")
                for i in range(ff.MAX_ROWS + 5)]
        monkeypatch.setattr(ff, "_fetch_feed", lambda: many)
        out = ff.get_economic_calendar("2030-06-03", 0, "high")
        assert f"showing the first {ff.MAX_ROWS} of {ff.MAX_ROWS + 5}" in out
        assert len([ln for ln in out.splitlines() if ln.startswith("| 2030-06-03")]) == ff.MAX_ROWS

    @pytest.mark.parametrize("look_ahead, end", [(100, "2030-06-17"), (-3, "2030-06-03"), ("2", "2030-06-05")])
    def test_look_ahead_is_bounded(self, feed, look_ahead, end):
        out = ff.get_economic_calendar("2030-06-03", look_ahead, "all")
        assert f"2030-06-03 to {end}" in out


@pytest.mark.unit
class TestCoverage:
    def test_a_window_past_the_feed_week_is_flagged_not_called_empty(self, feed):
        out = ff.get_economic_calendar("2030-06-06", 7, "high")
        assert "through 2030-06-08 only" in out and "not an absence of events" in out

    def test_a_window_inside_the_week_has_no_coverage_note(self, feed):
        out = ff.get_economic_calendar("2030-06-03", 2, "high")
        assert "through 2030-06-08" not in out

    def test_a_date_before_the_feed_week_is_unavailable(self, feed):
        out = ff.get_economic_calendar("2030-05-20", 7, "high")
        assert out.startswith("DATA_UNAVAILABLE")
        assert "current week" in out and "not an absence of scheduled events" in out
        assert "ISM" not in out

    def test_an_unavailable_notice_names_no_date_after_the_run(self, feed):
        # The feed's own week would tell a backtest when it is really running.
        out = ff.get_economic_calendar("2030-05-20", 7, "high")
        assert _dates_after(out, "2030-05-20") == []

    def test_a_date_after_the_feed_week_is_unavailable(self, feed):
        out = ff.get_economic_calendar("2030-06-12", 7, "high")
        assert out.startswith("DATA_UNAVAILABLE")
        assert "2030-06-02 to 2030-06-08" in out

    def test_the_week_starts_on_sunday_even_if_its_first_event_is_later(self, monkeypatch):
        wednesday_only = [_event("CPI m/m", "USD", "2030-06-05T08:30:00-04:00", "High")]
        monkeypatch.setattr(ff, "_fetch_feed", lambda: wednesday_only)
        out = ff.get_economic_calendar("2030-06-02", 7, "high")   # the Sunday
        assert not out.startswith("DATA_UNAVAILABLE")
        assert "CPI m/m" in out


@pytest.mark.unit
class TestPointInTime:
    def test_a_result_released_after_the_run_date_is_withheld(self, feed):
        out = ff.get_economic_calendar("2030-06-03", 6, "all")
        assert "Actual" in out
        assert "| 50.1 |" in out          # released on the run date
        assert "7.92M" not in out         # released the day after

    def test_no_actual_column_when_nothing_was_known(self, feed):
        out = ff.get_economic_calendar("2030-06-02", 6, "all")
        assert "Actual" not in out and "50.1" not in out and "7.92M" not in out

    def test_a_past_date_in_the_week_carries_the_revision_caveat(self, feed, monkeypatch):
        monkeypatch.setattr(date_window, "get_current_date", lambda: "2030-06-06")
        out = ff.get_economic_calendar("2030-06-03", 6, "all")
        assert "may include revisions made after 2030-06-03" in out

    def test_a_run_dated_today_has_no_caveat(self, feed, monkeypatch):
        monkeypatch.setattr(date_window, "get_current_date", lambda: "2030-06-03")
        out = ff.get_economic_calendar("2030-06-03", 6, "all")
        assert "revisions" not in out

    def test_the_tool_clamps_a_later_date_to_the_run_date(self):
        with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
            tools.get_economic_calendar.func("2030-06-09", 3, "high", "USD", trade_date="2030-06-03")
        assert routed.call_args.args == ("get_economic_calendar", "2030-06-03", 3, "high", "USD")


@pytest.mark.unit
class TestArguments:
    def test_bad_min_impact_is_sent_back_without_a_download(self):
        with mock.patch.object(ff, "_fetch_feed", side_effect=AssertionError("must not fetch")):
            out = ff.get_economic_calendar("2030-06-03", 7, "critical")
        assert "'high', 'medium', 'low' or 'all'" in out

    def test_bad_date_is_sent_back(self):
        with mock.patch.object(ff, "_fetch_feed", side_effect=AssertionError("must not fetch")):
            out = ff.get_economic_calendar("June 3", 7, "high")
        assert "YYYY-MM-DD" in out


@pytest.mark.unit
class TestFetching:
    def test_request_identifies_the_client_and_has_a_timeout(self):
        with mock.patch.object(net.requests, "get", return_value=_response(payload=FEED)) as get:
            ff.get_economic_calendar("2030-06-03", 1, "high")
        args, kwargs = get.call_args
        assert args[0] == ff.FEED_URL
        assert kwargs["timeout"] == ff.REQUEST_TIMEOUT
        assert kwargs["headers"]["User-Agent"].startswith("TradingAgents/")

    def test_the_export_is_downloaded_once_per_cache_period(self):
        with mock.patch.object(net.requests, "get", return_value=_response(payload=FEED)) as get:
            ff.get_economic_calendar("2030-06-03", 1, "high")
            ff.get_economic_calendar("2030-06-04", 1, "medium")
        assert get.call_count == 1

    def test_a_stale_cache_is_refreshed(self, monkeypatch):
        monkeypatch.setattr(ff, "_cache", (ff.time.monotonic() - ff.CACHE_TTL_SECONDS - 1, []))
        with mock.patch.object(net.requests, "get", return_value=_response(payload=FEED)) as get:
            out = ff.get_economic_calendar("2030-06-03", 1, "high")
        assert get.call_count == 1 and "ISM Manufacturing PMI" in out

    @pytest.mark.parametrize("failure, match", [
        ({"return_value": _response(429)}, "rate limiting"),
        ({"return_value": _response(503)}, "503"),
        ({"side_effect": requests.Timeout("Read timed out.")}, "Timeout"),
        ({"side_effect": requests.ConnectionError("refused")}, "ConnectionError"),
        ({"return_value": _response(json_error=True)}, "unreadable"),
        ({"return_value": _response(payload={"error": "nope"})}, "unexpected payload"),
        ({"return_value": _response(payload=[])}, "no readable events"),
    ], ids=["429", "5xx", "timeout", "connection", "html", "not-a-list", "empty"])
    def test_every_failure_is_vendor_unavailable(self, failure, match):
        with mock.patch.object(net.requests, "get", **failure), \
                pytest.raises(VendorUnavailableError, match=match):
            ff.get_economic_calendar("2030-06-03", 1, "high")

    def test_a_failure_is_not_cached(self):
        with mock.patch.object(net.requests, "get", side_effect=requests.Timeout("slow")), \
                pytest.raises(VendorUnavailableError):
            ff.get_economic_calendar("2030-06-03", 1, "high")
        with mock.patch.object(net.requests, "get", return_value=_response(payload=FEED)):
            assert "ISM Manufacturing PMI" in ff.get_economic_calendar("2030-06-03", 1, "high")


@pytest.mark.unit
class TestRouting:
    def test_category_and_default_vendor(self):
        assert router.get_category_for_method("get_economic_calendar") == "economic_calendar"
        assert DEFAULT_CONFIG["data_vendors"]["economic_calendar"] == "forexfactory"
        assert "economic_calendar" in router.OPTIONAL_CATEGORIES

    def test_routes_to_forexfactory(self, feed):
        out = router.route_to_vendor("get_economic_calendar", "2030-06-03", 1, "high", None)
        assert "ISM Manufacturing PMI" in out

    def test_an_outage_degrades_to_a_sentinel_instead_of_ending_the_run(self, monkeypatch):
        set_config({"data_vendors": {"economic_calendar": "forexfactory"}})
        with mock.patch.object(net.requests, "get", side_effect=requests.Timeout("slow")):
            out = router.route_to_vendor("get_economic_calendar", "2030-06-03", 1, "high", None)
        assert out.startswith("DATA_UNAVAILABLE")

    def test_an_unexpected_error_degrades_too(self, monkeypatch):
        monkeypatch.setitem(router.VENDOR_METHODS["get_economic_calendar"], "forexfactory",
                            mock.Mock(side_effect=KeyError("boom")))
        out = router.route_to_vendor("get_economic_calendar", "2030-06-03", 1, "high", None)
        assert out.startswith("DATA_UNAVAILABLE") and "economic_calendar" in out


@pytest.mark.unit
def test_the_news_analyst_is_offered_and_told_about_the_calendar():
    import inspect

    assert tools.get_economic_calendar in news_analyst.TOOLS
    assert "get_economic_calendar(curr_date, look_ahead_days, min_impact)" in inspect.getsource(news_analyst)
