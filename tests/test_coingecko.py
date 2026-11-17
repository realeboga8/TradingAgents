"""CoinGecko crypto vendor: symbol to coin-id mapping, the point-in-time bound
on every observation, the snapshot and daily-price formats, error mapping
(rate limits, timeouts, refusals), the optional Demo key, router integration
(Yahoo stays the default price vendor) and the market analyst's wiring.

All HTTP is mocked, so these run without a network connection or a key.
"""
from datetime import UTC, datetime, timedelta
from unittest import mock

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

import tradingagents.dataflows.net as net
from tradingagents.agents import tools
from tradingagents.agents.analysts import market_analyst
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import coingecko as cg
from tradingagents.default_config import DEFAULT_CONFIG

POISON = 999_999.0   # the value every observation after the cutoff carries


def _ms(when: datetime) -> int:
    return int(when.timestamp() * 1000)


def _chart(start: datetime, end: datetime, cutoff: datetime, *, base=100.0,
           cap=1.2e12, volume=4.5e10) -> dict:
    """Hourly observations in [start, end); price = base + hours since start,
    and every value at or after ``cutoff`` is POISON."""
    prices, caps, volumes = [], [], []
    when, hour = start, 0
    while when < end:
        late = when >= cutoff
        prices.append([_ms(when), POISON if late else base + hour])
        caps.append([_ms(when), POISON if late else cap])
        volumes.append([_ms(when), POISON if late else volume])
        when += timedelta(hours=1)
        hour += 1
    return {"prices": prices, "market_caps": caps, "total_volumes": volumes}


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def _response(status=200, payload=None, json_error=False, headers=None):
    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {}
    response.text = "<html>oops</html>" if json_error else ""
    if json_error:
        response.json.side_effect = ValueError("Expecting value")
    else:
        response.json.return_value = payload
    response.raise_for_status.side_effect = (
        requests.HTTPError(f"{status} Server Error", response=response)
        if status >= 400 and status not in (400, 401, 403, 404, 429) else None
    )
    return response


# A past run: observations from a week before through two days after it.
AS_OF = "2025-03-10"
CUTOFF = _utc(2025, 3, 11)
PAST_CHART = _chart(_utc(2025, 3, 2), _utc(2025, 3, 13), CUTOFF)


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    monkeypatch.delenv("COINGECKO_API_KEY", raising=False)


@pytest.mark.unit
@pytest.mark.parametrize("symbol, expected", [
    ("BTC-USD", "bitcoin"), ("btc-usd", "bitcoin"), ("BTCUSD", "bitcoin"),
    ("ETH-USDT", "ethereum"), ("ETHUSDC", "ethereum"), ("SOL-USD", "solana"),
    ("XRP-USD", "ripple"), ("AVAX-USD", "avalanche-2"), ("BTCUSD+", "bitcoin"),
    ("SOL", None),          # a bare base is also an equity ticker
    ("AAPL", None), ("BTC-EUR", None), ("EURUSD", None), ("", None), (None, None),
])
def test_coin_id(symbol, expected):
    assert cg.coin_id(symbol) == expected


@pytest.mark.unit
class TestSnapshot:
    def test_a_past_date_reads_only_what_was_observed_by_its_close(self):
        with mock.patch.object(cg, "_request", return_value=PAST_CHART):
            out = cg.get_crypto_market_data("BTC-USD", AS_OF)
        assert "## CoinGecko market data: BTC-USD (CoinGecko id: bitcoin)" in out
        assert "latest observation 2025-03-10 23:00 UTC" in out
        assert "- Price: $315.00" in out                       # 03-10 23:00 is hour 215
        assert "- 24h change: +8.25% (from $291.00)" in out    # 03-09 23:00
        assert "- 7d change: +114.29% (from $147.00)" in out   # 03-03 23:00
        assert "- Market cap: $1.20T" in out
        assert "- 24h trading volume: $45.00B" in out
        assert "999" not in out and "2025-03-11" not in out

    def test_the_request_ends_at_the_close_of_the_run_date(self):
        with mock.patch.object(cg, "_request", return_value=PAST_CHART) as request:
            cg.get_crypto_market_data("ETH-USD", AS_OF)
        path, params = request.call_args.args[:2]
        assert path == "coins/ethereum/market_chart/range"
        assert params["vs_currency"] == "usd"
        assert params["to"] == int(CUTOFF.timestamp())
        assert params["from"] == int((CUTOFF - timedelta(days=cg.SNAPSHOT_WINDOW_DAYS)).timestamp())

    def test_a_run_dated_today_never_asks_for_the_future(self):
        now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        chart = _chart(now - timedelta(days=9), now + timedelta(hours=1), now + timedelta(days=1))
        with mock.patch.object(cg, "_request", return_value=chart) as request:
            out = cg.get_crypto_market_data("BTC-USD", f"{now:%Y-%m-%d}")
        assert request.call_args.args[1]["to"] <= datetime.now(UTC).timestamp() + 1
        assert f"latest observation {now:%Y-%m-%d %H:%M} UTC" in out

    def test_a_gap_reports_the_change_as_unavailable(self):
        chart = _chart(_utc(2025, 3, 9, 12), _utc(2025, 3, 11), CUTOFF)
        with mock.patch.object(cg, "_request", return_value=chart):
            out = cg.get_crypto_market_data("BTC-USD", AS_OF)
        assert "- 7d change: n/a" in out and "- 24h change: +" in out

    def test_a_stale_series_is_no_data_not_a_current_price(self):
        chart = _chart(_utc(2025, 3, 3), _utc(2025, 3, 6), CUTOFF)
        with mock.patch.object(cg, "_request", return_value=chart), \
                pytest.raises(NoMarketDataError, match="stale"):
            cg.get_crypto_market_data("BTC-USD", AS_OF)

    def test_no_prices_is_no_data(self):
        with mock.patch.object(cg, "_request", return_value={"prices": []}), \
                pytest.raises(NoMarketDataError):
            cg.get_crypto_market_data("BTC-USD", AS_OF)

    def test_a_non_crypto_instrument_gets_a_notice_and_no_request(self):
        with mock.patch.object(cg, "_request", side_effect=AssertionError("must not fetch")):
            out = cg.get_crypto_market_data("NVDA", AS_OF)
        assert "crypto pairs only" in out and "BTC-USD" in out

    def test_a_bad_date_is_sent_back(self):
        with mock.patch.object(cg, "_request", side_effect=AssertionError("must not fetch")):
            assert "YYYY-MM-DD" in cg.get_crypto_market_data("BTC-USD", "yesterday")


@pytest.mark.unit
class TestDailyPrices:
    def _prices(self):
        chart = _chart(_utc(2025, 2, 28), _utc(2025, 3, 8), _utc(2025, 3, 6), base=1000.0,
                       cap=2e11, volume=5e9)
        with mock.patch.object(cg, "_request", return_value=chart) as request:
            out = cg.get_crypto_prices("BTC-USD", "2025-03-01", "2025-03-05")
        return out, request

    def test_one_row_per_utc_day_from_its_last_observation(self):
        out, _ = self._prices()
        assert out.startswith("# Crypto price data for BTC-USD (CoinGecko: bitcoin) from 2025-03-01 to 2025-03-05")
        assert "# Total records: 5" in out
        assert "Date,Close,Volume,Market Cap" in out
        assert "2025-03-01,1047.00,5000000000,200000000000" in out   # 03-01 23:00 is hour 47
        assert "2025-03-05,1143.00,5000000000,200000000000" in out

    def test_nothing_outside_the_window_is_served(self):
        out, request = self._prices()
        assert "2025-02-28" not in out and "2025-03-06" not in out and "999999" not in out
        params = request.call_args.args[1]
        assert params["from"] == int(_utc(2025, 3, 1).timestamp())
        assert params["to"] == int(_utc(2025, 3, 6).timestamp())

    def test_a_non_crypto_symbol_is_no_data_so_the_chain_moves_on(self):
        with mock.patch.object(cg, "_request", side_effect=AssertionError("must not fetch")), \
                pytest.raises(NoMarketDataError):
            cg.get_crypto_prices("AAPL", "2025-03-01", "2025-03-05")

    def test_an_empty_window_is_no_data(self):
        with mock.patch.object(cg, "_request", return_value={"prices": []}), \
                pytest.raises(NoMarketDataError):
            cg.get_crypto_prices("BTC-USD", "2025-03-01", "2025-03-05")


def _call():
    return cg.get_crypto_market_data("BTC-USD", AS_OF)


@pytest.mark.unit
class TestHttp:
    def test_keyless_request(self):
        with mock.patch.object(net.requests, "get", return_value=_response(payload=PAST_CHART)) as get:
            _call()
        args, kwargs = get.call_args
        assert args[0] == f"{cg.API_BASE}/coins/bitcoin/market_chart/range"
        assert kwargs["timeout"] == cg.REQUEST_TIMEOUT
        assert "x-cg-demo-api-key" not in kwargs["headers"]

    def test_the_demo_key_goes_in_its_header(self, monkeypatch):
        monkeypatch.setenv("COINGECKO_API_KEY", "CG-demo-key-123")
        with mock.patch.object(net.requests, "get", return_value=_response(payload=PAST_CHART)) as get:
            _call()
        kwargs = get.call_args.kwargs
        assert kwargs["headers"]["x-cg-demo-api-key"] == "CG-demo-key-123"
        assert "CG-demo-key-123" not in str(kwargs["params"])

    def test_a_rate_limit_is_waited_out(self):
        limited = _response(429, {"status": {"error_code": 429}}, headers={"Retry-After": "7"})
        with mock.patch.object(net.requests, "get", side_effect=[limited, _response(payload=PAST_CHART)]), \
                mock.patch.object(cg.time, "sleep") as sleep:
            out = _call()
        sleep.assert_called_once_with(7.0)
        assert "- Price: $315.00" in out

    def test_a_long_retry_after_is_capped(self):
        limited = _response(429, {}, headers={"Retry-After": "120"})
        with mock.patch.object(net.requests, "get", side_effect=[limited, _response(payload=PAST_CHART)]), \
                mock.patch.object(cg.time, "sleep") as sleep:
            _call()
        sleep.assert_called_once_with(cg.MAX_RETRY_DELAY)

    def test_a_rate_limit_that_outlasts_the_retries_is_vendor_unavailable(self):
        with mock.patch.object(net.requests, "get", return_value=_response(429, {})), \
                mock.patch.object(cg.time, "sleep") as sleep, \
                pytest.raises(VendorUnavailableError, match="rate limit"):
            _call()
        assert sleep.call_count == cg.MAX_RETRIES

    @pytest.mark.parametrize("failure, error, match", [
        ({"side_effect": requests.Timeout("Read timed out.")}, VendorUnavailableError, "Timeout"),
        ({"side_effect": requests.ConnectionError("refused")}, VendorUnavailableError, "ConnectionError"),
        ({"return_value": _response(500)}, VendorUnavailableError, "500"),
        ({"return_value": _response(json_error=True)}, VendorUnavailableError, "unreadable"),
        ({"return_value": _response(404, {"error": "coin not found"})}, NoMarketDataError, "no coin"),
        ({"return_value": _response(401, {"error": {"status": {
            "error_code": 10012,
            "error_message": "Your request exceeds the allowed time range. Public API users are "
                             "limited to querying historical data within the past 365 days."}}})},
         NoMarketDataError, "recent history"),
        ({"return_value": _response(401, {"status": {"error_message": "Missing API key"}})},
         VendorUnavailableError, "HTTP 401"),
    ], ids=["timeout", "connection", "5xx", "html", "404", "time-range", "401-keyless"])
    def test_failures_map_to_vendor_errors(self, failure, error, match):
        with mock.patch.object(net.requests, "get", **failure), pytest.raises(error, match=match):
            _call()

    def test_a_refused_key_is_a_configuration_error_without_the_key(self, monkeypatch):
        key = "CG-bad-key-999"
        monkeypatch.setenv("COINGECKO_API_KEY", key)
        refused = _response(401, {"status": {"error_message": f"Invalid API key {key}"}})
        with mock.patch.object(net.requests, "get", return_value=refused), \
                pytest.raises(cg.CoinGeckoNotConfiguredError) as caught:
            _call()
        assert key not in str(caught.value) and "***" in str(caught.value)
        assert isinstance(caught.value, ValueError)   # the router's "not configured" path


@pytest.mark.unit
class TestRouting:
    def test_yahoo_stays_the_default_price_vendor(self):
        assert DEFAULT_CONFIG["data_vendors"]["core_stock_apis"] == "yfinance"
        assert DEFAULT_CONFIG["data_vendors"]["crypto_data"] == "coingecko"
        assert router.get_category_for_method("get_crypto_market_data") == "crypto_data"
        assert "crypto_data" in router.OPTIONAL_CATEGORIES
        assert "coingecko" in router.VENDOR_METHODS["get_stock_data"]

    def _stock_vendors(self, monkeypatch, yfinance, coingecko):
        monkeypatch.setitem(router.VENDOR_METHODS["get_stock_data"], "yfinance", yfinance)
        monkeypatch.setitem(router.VENDOR_METHODS["get_stock_data"], "coingecko", coingecko)

    def test_the_default_chain_never_reaches_coingecko(self, monkeypatch):
        coingecko = mock.Mock(return_value="CG_DATA")

        def no_data(symbol, *a):
            raise NoMarketDataError(symbol, symbol, "no rows")

        self._stock_vendors(monkeypatch, no_data, coingecko)
        out = router.route_to_vendor("get_stock_data", "BTC-USD", "2025-03-01", "2025-03-05")
        assert out.startswith("NO_DATA_AVAILABLE")
        coingecko.assert_not_called()

    def test_listing_coingecko_after_yahoo_falls_back_for_crypto(self, monkeypatch):
        set_config({"data_vendors": {"core_stock_apis": "yfinance,coingecko"}})

        def down(*a):
            raise VendorUnavailableError("Yahoo Finance is unreachable")

        self._stock_vendors(monkeypatch, down, mock.Mock(return_value="CG_DATA"))
        assert router.route_to_vendor("get_stock_data", "BTC-USD", "2025-03-01", "2025-03-05") == "CG_DATA"

    def test_listing_coingecko_leaves_a_stock_to_yahoo(self, monkeypatch):
        set_config({"data_vendors": {"core_stock_apis": "yfinance,coingecko"}})

        def no_data(symbol, *a):
            raise NoMarketDataError(symbol, symbol, "no rows")

        monkeypatch.setitem(router.VENDOR_METHODS["get_stock_data"], "yfinance", no_data)
        with mock.patch.object(net.requests, "get", side_effect=AssertionError("must not fetch")):
            out = router.route_to_vendor("get_stock_data", "AAPL", "2025-03-01", "2025-03-05")
        assert out.startswith("NO_DATA_AVAILABLE")

    def test_crypto_data_routes_to_coingecko(self):
        with mock.patch.object(cg, "_request", return_value=PAST_CHART):
            out = router.route_to_vendor("get_crypto_market_data", "BTC-USD", AS_OF)
        assert "- Price: $315.00" in out

    def test_a_crypto_data_outage_degrades_to_a_sentinel(self):
        with mock.patch.object(net.requests, "get", side_effect=requests.Timeout("slow")):
            out = router.route_to_vendor("get_crypto_market_data", "BTC-USD", AS_OF)
        assert out.startswith("DATA_UNAVAILABLE")


@pytest.mark.unit
class TestTool:
    def test_the_tool_clamps_a_later_date_to_the_run_date(self):
        with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
            tools.get_crypto_market_data.func("BTC-USD", "2025-04-01", trade_date=AS_OF)
        assert routed.call_args.args == ("get_crypto_market_data", "BTC-USD", AS_OF)

    def test_a_stock_run_gets_a_notice_without_a_request(self):
        with mock.patch.object(net.requests, "get", side_effect=AssertionError("must not fetch")):
            out = tools.get_crypto_market_data.func("NVDA", AS_OF, trade_date=AS_OF)
        assert "crypto pairs only" in out


class _CapturingLLM:
    def __init__(self):
        self.prompts = []

    def bind_tools(self, bound):
        self.tools = bound

        def answer(prompt_value):
            self.prompts.append(prompt_value.to_string())
            return AIMessage(content="report")

        return RunnableLambda(answer)


def _market_prompt(asset_type: str, ticker: str) -> tuple[str, tuple]:
    llm = _CapturingLLM()
    market_analyst.create_market_analyst(llm)({
        "trade_date": AS_OF, "company_of_interest": ticker, "asset_type": asset_type,
        "instrument_context": f"Instrument: {ticker}.", "messages": [HumanMessage(ticker)],
    })
    return llm.prompts[0], llm.tools


@pytest.mark.unit
def test_the_market_analyst_is_told_about_crypto_data_on_a_crypto_run_only():
    crypto_prompt, offered = _market_prompt("crypto", "BTC-USD")
    assert tools.get_crypto_market_data in offered
    assert market_analyst.CRYPTO_BRIEF.strip() in crypto_prompt

    stock_prompt, _ = _market_prompt("stock", "NVDA")
    assert market_analyst.CRYPTO_BRIEF.strip() not in stock_prompt
