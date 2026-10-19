"""
NSE Movers + Next-day Bullish Scanner (Streamlit, mobile friendly)

What's new vs the old version
  * Uses only FULLY CLOSED N-minute candles (the still-forming candle is dropped).
  * ~5-10x faster: Nifty 200 default, one bulk download per 200 symbols, daily
    indicators cached separately, and threshold/filter changes never re-download.
  * Volume surge (candle + daily RVOL), VWAP, EMA trend, RSI, MACD, 20D breakout,
    close-strength -> 0-100 "next-day bullish" score with trigger / stop-loss.
"""

import io
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="NSE Movers & Bullish Setups", page_icon="📈", layout="wide")

IST = "Asia/Kolkata"
BASE = "https://nsearchives.nseindia.com/content/indices/"
UNIVERSES = {
    "Nifty 100": "ind_nifty100list.csv",
    "Nifty 200": "ind_nifty200list.csv",
    "Nifty 500": "ind_nifty500list.csv",
}
CHUNK = 200  # one bulk download covers a whole Nifty 200

SNAP_COLS = ["Symbol", "LTP", "Move %", "Vol x", "VWAP", "vs VWAP %",
             "Candle", "Date", "Lag", "Live"]
DAILY_COLS = ["Symbol", "Close", "Day %", "RVOL", "CLV", "RSI", "EMA20", "EMA50",
              "MACDh", "MACDh_prev", "HH20", "Hi52", "High", "Low"]


# =====================================================================
#  Pure logic (no UI) - kept above any widget code
# =====================================================================
def is_market_open():
    now = pd.Timestamp.now(tz=IST)
    return now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)


def should_poll():
    """Auto-refresh window: slightly wider than market hours."""
    now = pd.Timestamp.now(tz=IST)
    return now.weekday() < 5 and (9, 10) <= (now.hour, now.minute) <= (15, 35)


# Approx. cumulative share of a day's volume traded by minute-of-session
# (NSE volume is U-shaped: heavy open and close).
_PROFILE_X = [0, 30, 120, 240, 330, 375]
_PROFILE_Y = [0.0, 0.18, 0.42, 0.65, 0.82, 1.0]


def elapsed_vol_fraction(now):
    mins = (now.hour * 60 + now.minute) - (9 * 60 + 15)
    return max(float(np.interp(mins, _PROFILE_X, _PROFILE_Y)), 0.05)


def _download(symbols, period, interval):
    return yf.download(list(symbols), period=period, interval=interval,
                       group_by="ticker", threads=True, progress=False,
                       auto_adjust=False, timeout=20)


def _sym_frame(data, sym):
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        if sym not in data.columns.get_level_values(0):
            return None
        return data[sym]
    return data


def _to_ist(df):
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    return df


def symbol_row(sym, df, window, live, now):
    """Build one snapshot row from 1-minute bars using the LAST CLOSED candle."""
    if df is None:
        return None
    df = df.dropna(subset=["Close"])
    if len(df) < 2:
        return None
    df = _to_ist(df)
    df = df[df.index.date == df.index[-1].date()]          # latest session only
    sym_live = live and df.index[-1].date() == now.date()  # False on holidays/pre-open

    if sym_live:  # drop the 1-min bar that is still forming
        df = df[df.index + pd.Timedelta(minutes=1) <= now]
        if df.empty:
            return None

    open_t = df.index[0].normalize() + pd.Timedelta(hours=9, minutes=15)
    agg = df.resample(f"{window}min", origin=open_t, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Close"])

    last_1m_end = df.index[-1] + pd.Timedelta(minutes=1)
    if sym_live:  # only candles whose full window has elapsed
        agg = agg[agg.index + pd.Timedelta(minutes=window) <= last_1m_end]
    if agg.empty:
        return None

    cand, start = agg.iloc[-1], agg.index[-1]
    if not cand["Open"] > 0:
        return None
    end = min(start + pd.Timedelta(minutes=window), last_1m_end)

    # candle volume vs median of earlier candles in this session
    prev_vol = agg["Volume"].iloc[:-1]
    prev_vol = prev_vol[prev_vol > 0]
    vol_x = cand["Volume"] / prev_vol.median() if len(prev_vol) >= 3 else np.nan

    # session VWAP up to the candle close
    d = df[df.index < end]
    vol_sum = d["Volume"].sum()
    vwap = ((d["High"] + d["Low"] + d["Close"]) / 3 * d["Volume"]).sum() / vol_sum if vol_sum > 0 else np.nan
    ltp = float(cand["Close"])

    return {
        "Symbol": sym.replace(".NS", ""),
        "LTP": ltp,
        "Move %": (ltp / float(cand["Open"]) - 1) * 100,
        "Vol x": float(vol_x),
        "VWAP": float(vwap),
        "vs VWAP %": (ltp / vwap - 1) * 100 if vwap == vwap and vwap > 0 else np.nan,
        "Candle": end.strftime("%H:%M"),
        "Date": end.strftime("%d %b"),
        "Lag": (now - end).total_seconds() / 60,
        "Live": bool(sym_live),
    }


def daily_row(sym, d, live, now):
    """Daily-timeframe indicators for one symbol."""
    if d is None:
        return None
    d = d.dropna(subset=["Close"]).copy()
    if len(d) < 60:
        return None
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    d = d[~d.index.duplicated(keep="last")]

    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    partial = live and d.index[-1].date() == now.date()

    # RVOL: today's (projected if still trading) volume vs prior 20-day average
    avg_vol = v.iloc[:-1].tail(20).mean()
    mins = (now.hour * 60 + now.minute) - (9 * 60 + 15)
    if partial:
        rvol = v.iloc[-1] / elapsed_vol_fraction(now) / avg_vol if (avg_vol > 0 and mins >= 20) else np.nan
    else:
        rvol = v.iloc[-1] / avg_vol if avg_vol > 0 else np.nan

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()

    rng = h.iloc[-1] - l.iloc[-1]
    return {
        "Symbol": sym.replace(".NS", ""),
        "Close": float(c.iloc[-1]),
        "Day %": float((c.iloc[-1] / c.iloc[-2] - 1) * 100),
        "RVOL": float(rvol),
        "CLV": float((c.iloc[-1] - l.iloc[-1]) / rng) if rng > 0 else 0.5,
        "RSI": float(rsi.iloc[-1]),
        "EMA20": float(ema20.iloc[-1]),
        "EMA50": float(ema50.iloc[-1]),
        "MACDh": float(hist.iloc[-1]),
        "MACDh_prev": float(hist.iloc[-2]),
        "HH20": float(h.iloc[-21:-1].max()),   # prior 20 days' high (excl. today)
        "Hi52": float(h.tail(250).max()),
        "High": float(h.iloc[-1]),
        "Low": float(l.iloc[-1]),
    }


def score_row(r):
    """0-100 next-day bullish score + readable signals."""
    s, tags = 0, []

    rv = r["RVOL"]
    if rv >= 2:
        s += 25; tags.append(f"Vol {rv:.1f}x")
    elif rv >= 1.5:
        s += 18; tags.append(f"Vol {rv:.1f}x")
    elif rv >= 1.2:
        s += 10; tags.append(f"Vol {rv:.1f}x")

    if r["CLV"] >= 0.8:
        s += 15; tags.append("Closing near high")
    elif r["CLV"] >= 0.65:
        s += 8

    vw = r.get("vs VWAP %", np.nan)
    if pd.notna(vw) and vw > 0:
        s += 10; tags.append("Above VWAP")

    if r["Close"] > r["EMA20"] > r["EMA50"]:
        s += 15; tags.append("Uptrend (20>50 EMA)")
    elif r["Close"] > r["EMA20"]:
        s += 7

    rsi = r["RSI"]
    if 55 <= rsi <= 70:
        s += 10; tags.append(f"RSI {rsi:.0f}")
    elif 50 <= rsi < 55 or 70 < rsi <= 75:
        s += 5

    if r["Close"] > r["HH20"]:
        s += 15; tags.append("20D breakout")
    elif r["Close"] >= 0.97 * r["Hi52"]:
        s += 8; tags.append("Near 52W high")

    if r["MACDh"] > 0 and r["MACDh"] > r["MACDh_prev"]:
        s += 10; tags.append("MACD rising")
    elif r["MACDh"] > 0:
        s += 5

    if rsi > 78 or r["Close"] > 1.12 * r["EMA20"]:
        s -= 10; tags.append("⚠ Overextended")

    return max(0, min(100, s)), " · ".join(tags)


def build_bullish(daily, snap):
    if daily.empty:
        return pd.DataFrame()
    df = daily.copy()
    if not snap.empty:
        df = df.merge(snap[["Symbol", "vs VWAP %"]], on="Symbol", how="left")
    else:
        df["vs VWAP %"] = np.nan
    out = [score_row(r) for r in df.to_dict("records")]
    df["Score"] = [o[0] for o in out]
    df["Signals"] = [o[1] for o in out]
    df["Setup"] = np.select([df["Score"] >= 70, df["Score"] >= 55],
                            ["🔥 Strong", "✅ Good"], "👀 Watch")
    df["Trigger"] = df["High"]                                   # buy above today's high
    df["SL"] = df["Low"]                                         # stop below today's low
    df["Risk %"] = (df["High"] - df["Low"]) / df["High"] * 100
    return df


# ---------------------------------------------------------------------
#  Cached data layer
# ---------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def load_symbols(fname):
    r = requests.get(BASE + fname, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return tuple(s.strip() + ".NS" for s in df["Symbol"].dropna())


@st.cache_data(ttl=30, show_spinner=False)
def intraday_snapshot(symbols: tuple, window: int, live: bool):
    """ONE row per stock (no threshold filtering) -> filters never trigger a re-download."""
    now = pd.Timestamp.now(tz=IST)
    rows = []
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        data = None
        for period in ("1d", "5d"):  # 5d fallback for weekends / pre-open
            try:
                data = _download(chunk, period, "1m")
            except Exception:
                data = None
            if data is not None and not data.dropna(how="all").empty:
                break
        if data is None or data.empty:
            continue
        for sym in chunk:
            try:
                row = symbol_row(sym, _sym_frame(data, sym), window, live, now)
            except (KeyError, IndexError, ValueError):
                row = None
            if row:
                rows.append(row)
    return pd.DataFrame(rows, columns=SNAP_COLS)


@st.cache_data(ttl=600, show_spinner=False)
def daily_features(symbols: tuple, live: bool):
    now = pd.Timestamp.now(tz=IST)
    rows = []
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        try:
            data = _download(chunk, "1y", "1d")
        except Exception:
            continue
        for sym in chunk:
            try:
                row = daily_row(sym, _sym_frame(data, sym), live, now)
            except (KeyError, IndexError, ValueError):
                row = None
            if row:
                rows.append(row)
    return pd.DataFrame(rows, columns=DAILY_COLS)


# =====================================================================
#  UI
# =====================================================================
market_open = is_market_open()

st.sidebar.header("Settings")
universe = st.sidebar.selectbox("Universe", list(UNIVERSES), index=1,
                                help="Nifty 200 is the best speed/coverage balance.")
window = st.sidebar.selectbox("Candle size (min)", [1, 2, 3, 5, 10, 15], index=2)
threshold = st.sidebar.number_input("Min candle move %", 0.1, 20.0, 1.0, 0.1)
only_up = st.sidebar.checkbox("Only gainers", False)
min_vol_x = st.sidebar.number_input("Min candle volume surge (x, 0 = off)", 0.0, 20.0, 0.0, 0.5,
                                    help="Candle volume vs median candle volume earlier today.")
max_lag = st.sidebar.number_input("Max extra data lag (min, live only)", 1, 30, 3, 1,
                                  help="Skip stocks whose last closed candle is older than "
                                       "candle size + this many minutes.")
min_score = st.sidebar.slider("Next-day min score", 0, 100, 60, 5)

st.sidebar.divider()
auto_refresh = st.sidebar.toggle("Auto refresh", value=True,
                                 help="Only runs during market hours.")
refresh_sec = st.sidebar.selectbox("Refresh every (sec)", [30, 60, 120, 300], index=1)
if st.sidebar.button("🔄 Refresh now"):
    st.cache_data.clear()

try:
    symbols = load_symbols(UNIVERSES[universe])
except Exception:
    st.warning(f"Couldn't fetch the {universe} list from NSE. Upload the index CSV instead.")
    up = st.file_uploader("Index constituents CSV (needs a 'Symbol' column)", type="csv")
    if not up:
        st.stop()
    symbols = tuple(s.strip() + ".NS" for s in pd.read_csv(up)["Symbol"].dropna())

st.title("📈 NSE Movers & Bullish Setups")


def _color(v):
    if pd.isna(v):
        return ""
    return "color: #16a34a" if v > 0 else ("color: #dc2626" if v < 0 else "")


def show(df, fmt, color_cols):
    sty = df.style.format(fmt, na_rep="–")
    mapper = getattr(sty, "map", None) or sty.applymap
    sty = mapper(_color, subset=color_cols)
    try:
        st.dataframe(sty, hide_index=True, width="stretch")
    except Exception:  # older Streamlit
        st.dataframe(sty, hide_index=True, use_container_width=True)


def render():
    t0 = time.time()
    now = pd.Timestamp.now(tz=IST)
    live = is_market_open()

    with st.spinner(f"Scanning {len(symbols)} stocks..."):
        snap = intraday_snapshot(symbols, window, live)
        daily = daily_features(symbols, live)
    took = time.time() - t0

    live_eff = bool(live and not snap.empty and snap["Live"].any())
    status = "🟢 Market open (live)" if live_eff else "🔴 Market closed (last session)"
    note = f" · auto refresh {refresh_sec}s" if (auto_refresh and live_eff) else " · auto refresh off"
    st.caption(f"{now:%d %b %Y, %H:%M:%S} IST · {status}{note} · ⏱ {took:.1f}s")

    if snap.empty:
        st.error("No data returned from Yahoo Finance. Try 'Refresh now' in a minute.")
        return

    last_candle = snap["Candle"].mode().iloc[0]
    last_date = snap["Date"].mode().iloc[0]
    lag_txt = f"data lag ≈ {max(snap['Lag'].median(), 0):.1f} min · " if live_eff else ""
    st.caption(f"Latest closed {window}-min candle: **{last_candle}** ({last_date}) · "
               f"{lag_txt}{len(snap)}/{len(symbols)} stocks loaded")

    bull = build_bullish(daily, snap)

    tab1, tab2 = st.tabs(["⚡ Candle movers", "🎯 Next-day bullish"])

    # ---------------- tab 1: movers ----------------
    with tab1:
        m = snap.copy()
        if live_eff:
            m = m[(~m["Live"]) | (m["Lag"] <= window + max_lag)]
        m = m[m["Move %"].abs() >= threshold]
        if only_up:
            m = m[m["Move %"] > 0]
        if min_vol_x > 0:
            m = m[m["Vol x"] >= min_vol_x]
        if not bull.empty:
            m = m.merge(bull[["Symbol", "Day %", "RVOL", "Score"]], on="Symbol", how="left")
        else:
            m["Day %"], m["RVOL"], m["Score"] = np.nan, np.nan, np.nan
        m = m.sort_values("Move %", key=abs, ascending=False).reset_index(drop=True)

        st.caption(f"≥{threshold}% move in the last closed {window}-min candle"
                   + ("" if live_eff else " of the last session"))
        if m.empty:
            st.info("No stocks matched. Try a lower % or smaller volume-surge filter.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Gainers", int((m["Move %"] > 0).sum()))
            c2.metric("Losers", int((m["Move %"] < 0).sum()))
            c3.metric("With vol surge ≥2x", int((m["Vol x"] >= 2).sum()))
            cols = ["Symbol", "LTP", "Move %", "Vol x", "vs VWAP %", "Day %", "RVOL", "Score", "Candle"]
            show(m[cols],
                 {"LTP": "{:.2f}", "Move %": "{:+.2f}", "Vol x": "{:.1f}x", "vs VWAP %": "{:+.2f}",
                  "Day %": "{:+.2f}", "RVOL": "{:.1f}x", "Score": "{:.0f}"},
                 ["Move %", "vs VWAP %", "Day %"])

    # ---------------- tab 2: next-day bullish ----------------
    with tab2:
        st.caption("Ranks stocks most likely to follow through tomorrow. Best used after 3:30 PM, "
                   "when close-strength and volume are final.")
        if bull.empty:
            st.info("Daily data unavailable right now. Try 'Refresh now'.")
        else:
            b = bull[bull["Score"] >= min_score].sort_values(["Score", "RVOL"], ascending=False)
            b = b.reset_index(drop=True)
            if b.empty:
                st.info(f"No stocks scored ≥ {min_score}. Lower the score in the sidebar.")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Strong (≥70)", int((bull["Score"] >= 70).sum()))
                c2.metric(f"Shown (≥{min_score})", len(b))
                cols = ["Symbol", "Setup", "Score", "Close", "Day %", "RVOL", "RSI",
                        "Trigger", "SL", "Risk %", "Signals"]
                show(b[cols],
                     {"Score": "{:.0f}", "Close": "{:.2f}", "Day %": "{:+.2f}", "RVOL": "{:.1f}x",
                      "RSI": "{:.0f}", "Trigger": "{:.2f}", "SL": "{:.2f}", "Risk %": "{:.1f}"},
                     ["Day %"])
        with st.expander("How the score works"):
            st.markdown(
                "**Volume surge (25)** RVOL ≥2x / 1.5x / 1.2x vs 20-day average (projected for the full day while live) · "
                "**Close strength (15)** close in top 20% of day's range · "
                "**Above VWAP (10)** buyers in control at the close · "
                "**Trend (15)** Close > EMA20 > EMA50 · "
                "**RSI (10)** 55–70 = momentum without exhaustion · "
                "**Breakout (15)** close above prior 20-day high (or within 3% of 52W high) · "
                "**MACD (10)** histogram positive and rising · "
                "**Penalty (−10)** RSI > 78 or >12% above EMA20.\n\n"
                "**Trigger** = buy above today's high, **SL** = today's low. "
                "A heuristic ranking, not a prediction. Check the chart, news and market trend first. "
                "Not investment advice."
            )

    st.caption("Data from Yahoo Finance (can lag the exchange by 1–2 min). Not investment advice.")


# auto refresh only around market hours and only if the toggle is on
run_every = refresh_sec if (auto_refresh and should_poll()) else None
st.fragment(run_every=run_every)(render)()
