"""
MCP Server for finance / market data (FastMCP).

Exposes a small, free, no-API-key set of financial-data tools backed by
``yfinance`` (Yahoo! Finance public endpoints) and ``stockstats`` for
classic technical indicators (MACD, RSI, Bollinger Bands, MAs, etc.).

Tool groups (resolved from `src.tools.registry.get_all_tools`):
    - finance_market        — price + indicator tools
    - finance_news          — company + macro news
    - finance_fundamentals  — financial statements + earnings

Why these libraries:
    * Both are MIT/Apache and require no account or API key.
    * Yahoo OHLCV is the same source the TradingAgents reference uses by default.
    * stockstats covers the indicator menu (MACD/RSI/BBANDS) without writing any
      math ourselves — a key prompt-engineering best-practice principle: keep
      tools narrow and battle-tested, leave reasoning to the LLM.

Note on rate limits: Yahoo throttles aggressively on hot loops. Each tool is
defensive (catches exceptions, returns a clear text error string) so an agent
that hits a transient failure can simply retry with a different ticker /
timeframe rather than crashing the whole graph.

Transports:
  stdio (default):  python -m src.mcp_servers.finance_server
  HTTP:             MCP_TRANSPORT=http MCP_PORT=8011 python -m src.mcp_servers.finance_server
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from typing import Any, Callable

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


# ── In-process TTL cache (60s) ────────────────────────────────────────────────
#
# Yahoo Finance throttles aggressively when an agent fan-out hits five
# parallel analysts at once. A small TTL cache de-duplicates identical
# requests inside the parallel_coordinator window without ever returning
# data older than 60s. This is intentionally a process-local dict — RAG
# / DeepEval workers run in separate processes and each maintains its own
# tiny cache, which is the right boundary (no cross-team leakage).

_CACHE: dict[tuple, tuple[float, str]] = {}
_CACHE_TTL = float(os.getenv("FINANCE_CACHE_TTL_SEC", "60"))


def _cached(key: tuple, fn: Callable[[], str]) -> str:
    """Return cached value for ``key`` if fresh; otherwise compute + store."""
    now = time.time()
    hit = _CACHE.get(key)
    if hit is not None and (now - hit[0]) < _CACHE_TTL:
        return hit[1]
    value = fn()
    _CACHE[key] = (now, value)
    # Very simple cap to keep the dict small in long-running processes.
    if len(_CACHE) > 512:
        # Drop the 64 oldest entries by insertion order.
        for k in list(_CACHE.keys())[:64]:
            _CACHE.pop(k, None)
    return value

mcp = FastMCP(
    "finance-mcp-server",
    instructions=(
        "Fetch OHLCV price history, technical indicators, news headlines, and "
        "fundamental financials for any publicly listed ticker (US + most ADRs)."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _silenced(fn, *args, **kwargs):
    """Run a yfinance call while swallowing its noisy stdout/stderr.

    ``yfinance`` prints progress bars and HTTP warnings directly to stderr —
    those bytes leak into the FastMCP response payload otherwise.
    """
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def _normalize_ticker(symbol: str) -> str:
    return (symbol or "").strip().upper()


def _date_range(days: int) -> tuple[str, str]:
    end = datetime.utcnow()
    start = end - timedelta(days=max(1, int(days)))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _fmt_price_summary(symbol: str, df) -> str:
    """Compact human-readable summary of an OHLCV DataFrame."""
    if df is None or df.empty:
        return f"No price data returned for {symbol}."
    last = df.iloc[-1]
    first = df.iloc[0]
    change = float(last["Close"]) - float(first["Close"])
    pct = 100.0 * change / float(first["Close"]) if float(first["Close"]) else 0.0
    high = float(df["High"].max())
    low = float(df["Low"].min())
    avg_vol = float(df["Volume"].mean()) if "Volume" in df else 0.0
    return (
        f"{symbol} — {df.index[0].date()} to {df.index[-1].date()} "
        f"({len(df)} bars)\n"
        f"  Open:  {float(first['Open']):.2f}\n"
        f"  Close: {float(last['Close']):.2f}  "
        f"({'+' if change >= 0 else ''}{change:.2f}, {pct:+.2f}%)\n"
        f"  High:  {high:.2f}\n"
        f"  Low:   {low:.2f}\n"
        f"  AvgVol: {avg_vol:,.0f}"
    )


def _df_tail_csv(df, n: int = 30) -> str:
    if df is None or df.empty:
        return ""
    tail = df.tail(n).round(4)
    return tail.to_csv()


# ── Market data tools ─────────────────────────────────────────────────────────

@mcp.tool()
async def get_stock_data(symbol: str, days: int = 90, interval: str = "1d") -> str:
    """Return OHLCV price history + a compact summary for a ticker.

    Source: Yahoo! Finance via yfinance (no API key required).

    Args:
        symbol:   Ticker symbol, e.g. "AAPL", "MSFT", "TSLA". Case-insensitive.
        days:     Lookback window in calendar days. Capped to 365.
        interval: yfinance interval ("1d" daily | "1h" hourly | "1wk" weekly).
                  Hourly limited to ~60 days by Yahoo.
    """
    symbol = _normalize_ticker(symbol)
    if not symbol:
        return "Error: symbol is required."
    days = max(1, min(int(days), 365))

    def _impl() -> str:
        try:
            import yfinance as yf
            start, end = _date_range(days)
            df = _silenced(
                yf.download,
                tickers=symbol, start=start, end=end, interval=interval,
                progress=False, auto_adjust=False, threads=False,
            )
            # yfinance ≥ 0.2.40 returns a column-multiindex when threads=False; flatten it.
            if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
                df = df.copy()
                df.columns = [c[0] for c in df.columns]
            summary = _fmt_price_summary(symbol, df)
            csv_tail = _df_tail_csv(df, n=20)
            return f"{summary}\n\nMost recent 20 bars (CSV):\n{csv_tail}"
        except Exception as exc:
            return f"Error fetching price data for {symbol}: {type(exc).__name__}: {exc}"

    return _cached(("stock", symbol, days, interval), _impl)


@mcp.tool()
async def get_indicators(
    symbol: str,
    indicators: str = "macd,rsi_14,boll,close_20_sma,close_50_sma",
    days: int = 180,
) -> str:
    """Return classic technical indicators (MACD / RSI / Bollinger Bands / SMAs / etc.).

    Indicator names are stockstats column expressions — the most useful ones:
        macd, macds, macdh           MACD line, signal, histogram
        rsi_14                       14-period Relative Strength Index
        boll, boll_ub, boll_lb       Bollinger Bands (mid, upper, lower)
        close_20_sma, close_50_sma   Simple moving averages
        atr                          Average True Range
        kdjk, kdjd, kdjj             Stochastic KDJ

    Args:
        symbol:     Ticker, e.g. "AAPL".
        indicators: Comma-separated list of stockstats column expressions.
        days:       Lookback for the underlying price data (capped at 365).
    """
    symbol = _normalize_ticker(symbol)
    if not symbol:
        return "Error: symbol is required."
    days = max(30, min(int(days), 365))
    cols = [c.strip() for c in (indicators or "").split(",") if c.strip()]
    if not cols:
        return "Error: at least one indicator column is required."

    def _impl() -> str:
        try:
            import yfinance as yf
            from stockstats import wrap as ss_wrap
            start, end = _date_range(days)
            df = _silenced(
                yf.download,
                tickers=symbol, start=start, end=end, interval="1d",
                progress=False, auto_adjust=False, threads=False,
            )
            if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
                df = df.copy()
                df.columns = [c[0] for c in df.columns]
            if df.empty:
                return f"No price data for {symbol}; cannot compute indicators."
            df = df.rename(columns={c: c.lower() for c in df.columns})
            sdf = ss_wrap(df.copy())
            rows = []
            for col in cols:
                try:
                    _ = sdf[col]
                except Exception as e_col:
                    rows.append(f"  ! {col} — failed: {type(e_col).__name__}: {e_col}")
            usable = [c for c in cols if c in sdf.columns]
            if not usable:
                return f"Error: no requested indicator could be computed for {symbol}."
            tail = sdf[usable].tail(20).round(4)
            last = sdf[usable].iloc[-1].round(4).to_dict()
            last_str = ", ".join(f"{k}={v}" for k, v in last.items())
            return (
                f"{symbol} indicators ({df.index[-1].date()})\n"
                f"  Latest: {last_str}\n"
                + ("\nWarnings:\n" + "\n".join(rows) if rows else "")
                + f"\n\nLast 20 bars of indicators (CSV):\n{tail.to_csv()}"
            )
        except Exception as exc:
            return f"Error computing indicators for {symbol}: {type(exc).__name__}: {exc}"

    return _cached(("indicators", symbol, ",".join(sorted(cols)), days), _impl)


# ── News tools ────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_news(symbol: str, limit: int = 10) -> str:
    """Return the most recent news headlines for a ticker (Yahoo-curated feed).

    Args:
        symbol: Ticker symbol, e.g. "TSLA".
        limit:  Number of headlines to return (capped at 25).
    """
    symbol = _normalize_ticker(symbol)
    if not symbol:
        return "Error: symbol is required."
    limit = max(1, min(int(limit), 25))

    def _impl() -> str:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            items = _silenced(lambda: ticker.news) or []
            if not items:
                return f"No news returned for {symbol}."
            out_lines = [f"News for {symbol} (top {min(limit, len(items))}):"]
            for it in items[:limit]:
                content = it.get("content") or it
                title = content.get("title") or it.get("title") or "(no title)"
                publisher = (
                    (content.get("provider") or {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else content.get("publisher") or it.get("publisher") or "?"
                )
                link = (
                    (content.get("canonicalUrl") or {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else content.get("link") or it.get("link") or ""
                )
                pub_time = content.get("pubDate") or it.get("providerPublishTime") or ""
                if isinstance(pub_time, (int, float)):
                    pub_time = datetime.utcfromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
                out_lines.append(f"  • [{publisher}] {title} ({pub_time})\n    {link}")
            return "\n".join(out_lines)
        except Exception as exc:
            return f"Error fetching news for {symbol}: {type(exc).__name__}: {exc}"

    return _cached(("news", symbol, limit), _impl)


@mcp.tool()
async def get_global_news(limit: int = 10) -> str:
    """Return macroeconomic / market-moving headlines from broad index proxies.

    Pulls news for SPY (S&P 500 ETF) — the most reliable free proxy for
    "what's happening in the market today" via yfinance's free feed.

    Args:
        limit: Total headlines to return (capped at 25).
    """
    limit = max(1, min(int(limit), 25))

    def _impl() -> str:
        try:
            import yfinance as yf
            seen_titles: set[str] = set()
            out_lines = ["Macro / market news (sources: SPY, QQQ, ^VIX):"]
            for sym in ("SPY", "QQQ", "^VIX"):
                ticker = yf.Ticker(sym)
                items = _silenced(lambda: ticker.news) or []
                for it in items:
                    content = it.get("content") or it
                    title = content.get("title") or it.get("title") or ""
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    publisher = (
                        (content.get("provider") or {}).get("displayName")
                        if isinstance(content.get("provider"), dict)
                        else content.get("publisher") or it.get("publisher") or "?"
                    )
                    pub_time = content.get("pubDate") or it.get("providerPublishTime") or ""
                    if isinstance(pub_time, (int, float)):
                        pub_time = datetime.utcfromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
                    out_lines.append(f"  • [{sym}/{publisher}] {title} ({pub_time})")
                    if len(out_lines) - 1 >= limit:
                        return "\n".join(out_lines)
            return "\n".join(out_lines) if len(out_lines) > 1 else "No macro headlines returned."
        except Exception as exc:
            return f"Error fetching global news: {type(exc).__name__}: {exc}"

    return _cached(("global_news", limit), _impl)


# ── Fundamentals tools ────────────────────────────────────────────────────────

@mcp.tool()
async def get_fundamentals(symbol: str) -> str:
    """Return company profile + key fundamental ratios (cap, P/E, margins, etc.).

    Args:
        symbol: Ticker, e.g. "MSFT".
    """
    symbol = _normalize_ticker(symbol)
    if not symbol:
        return "Error: symbol is required."

    def _impl() -> str:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = _silenced(lambda: ticker.info) or {}
            if not info:
                return f"No fundamentals returned for {symbol}."
            keys = [
                ("longName", "Name"),
                ("sector", "Sector"),
                ("industry", "Industry"),
                ("country", "Country"),
                ("marketCap", "Market Cap"),
                ("trailingPE", "P/E (TTM)"),
                ("forwardPE", "P/E (Forward)"),
                ("priceToBook", "P/B"),
                ("dividendYield", "Dividend Yield"),
                ("beta", "Beta"),
                ("profitMargins", "Profit Margin"),
                ("operatingMargins", "Operating Margin"),
                ("returnOnEquity", "ROE"),
                ("totalRevenue", "Total Revenue"),
                ("revenueGrowth", "Revenue Growth"),
                ("debtToEquity", "Debt/Equity"),
                ("freeCashflow", "Free Cash Flow"),
                ("recommendationKey", "Analyst Recommendation"),
            ]
            out_lines = [f"Fundamentals for {symbol}:"]
            for k, label in keys:
                v = info.get(k)
                if v is None:
                    continue
                if isinstance(v, float):
                    v = f"{v:,.4f}" if abs(v) < 1 else f"{v:,.2f}"
                elif isinstance(v, int):
                    v = f"{v:,}"
                out_lines.append(f"  {label}: {v}")
            summary = info.get("longBusinessSummary") or ""
            if summary:
                out_lines.append("\nBusiness Summary:")
                out_lines.append(f"  {summary[:1200]}")
            return "\n".join(out_lines)
        except Exception as exc:
            return f"Error fetching fundamentals for {symbol}: {type(exc).__name__}: {exc}"

    return _cached(("fundamentals", symbol), _impl)


@mcp.tool()
async def get_balance_sheet(symbol: str) -> str:
    """Return the most recent annual balance sheet rows for a company."""
    return _fundamentals_table(symbol, "balance_sheet")


@mcp.tool()
async def get_cashflow(symbol: str) -> str:
    """Return the most recent annual cash-flow statement rows for a company."""
    return _fundamentals_table(symbol, "cashflow")


@mcp.tool()
async def get_income_statement(symbol: str) -> str:
    """Return the most recent annual income-statement rows for a company."""
    return _fundamentals_table(symbol, "financials")


def _fundamentals_table(symbol: str, table: str) -> str:
    """Shared helper for get_balance_sheet / get_cashflow / get_income_statement."""
    symbol = _normalize_ticker(symbol)
    if not symbol:
        return "Error: symbol is required."

    def _impl() -> str:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = _silenced(lambda: getattr(ticker, table))
            if df is None or df.empty:
                return f"No {table.replace('_', ' ')} data for {symbol}."
            df = df.iloc[:, :4]
            return f"{symbol} — {table.replace('_', ' ').title()}:\n{df.round(0).to_csv()}"
        except Exception as exc:
            return f"Error fetching {table} for {symbol}: {type(exc).__name__}: {exc}"

    return _cached(("statement", symbol, table), _impl)


# ── Entrypoint ───────────────────────────────────────────────────────────────

def _run() -> None:
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "http":
        port = int(os.getenv("MCP_PORT", "8011"))
        mcp.run(transport="http", port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    _run()
