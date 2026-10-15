"""
Nifty 500 - N-minute mover screener (Streamlit, mobile friendly)
Deploy free on Streamlit Community Cloud, then open the link on your phone.
"""

import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Nifty 500 Movers", page_icon="📈", layout="centered")

NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
CHUNK = 100
IST = "Asia/Kolkata"

# ---------------- sidebar settings ----------------
st.sidebar.header("Settings")
threshold = st.sidebar.number_input("Min move %", 0.1, 20.0, 1.0, 0.1)
window = st.sidebar.number_input("Window (minutes)", 1, 30, 3, 1)
only_up = st.sidebar.checkbox("Only gainers", False)
max_stale = st.sidebar.number_input("Ignore data older than (min)", 1, 30, 5, 1)
refresh = st.sidebar.selectbox("Auto refresh (sec)", [30, 60, 120, 300], index=1)


# ---------------- data ----------------
@st.cache_data(ttl=86400, show_spinner=False)
def load_symbols():
    r = requests.get(NSE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return [s.strip() + ".NS" for s in df["Symbol"].dropna()]


@st.cache_data(ttl=45, show_spinner=False)
def scan(symbols: tuple, threshold: float, window: int, only_up: bool, max_stale: int):
    rows = []
    now = pd.Timestamp.now(tz=IST)
    for i in range(0, len(symbols), CHUNK):
        chunk = list(symbols[i:i + CHUNK])
        try:
            data = yf.download(chunk, period="1d", interval="1m", group_by="ticker",
                               threads=True, progress=False, auto_adjust=False)
        except Exception:
            continue
        if data is None or data.empty:
            continue
        for sym in chunk:
            try:
                close = data[sym]["Close"].dropna()
                if len(close) < 2:
                    continue
                last_ts, last_price = close.index[-1], close.iloc[-1]
                if (now - last_ts).total_seconds() / 60 > max_stale:
                    continue
                past = close[close.index <= last_ts - pd.Timedelta(minutes=window)]
                if past.empty:
                    continue
                past_price = past.iloc[-1]
                pct = (last_price - past_price) / past_price * 100
                if abs(pct) >= threshold and (pct > 0 or not only_up):
                    rows.append({
                        "Symbol": sym.replace(".NS", ""),
                        "LTP": round(float(last_price), 2),
                        "Change %": round(float(pct), 2),
                        "Prev": round(float(past_price), 2),
                        "Time": last_ts.strftime("%H:%M"),
                    })
            except (KeyError, IndexError):
                continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Change %", key=abs, ascending=False).reset_index(drop=True)


# ---------------- symbols ----------------
try:
    symbols = load_symbols()
except Exception:
    st.warning("Couldn't fetch the Nifty 500 list from NSE. Upload ind_nifty500list.csv instead.")
    up = st.file_uploader("Nifty 500 CSV", type="csv")
    if not up:
        st.stop()
    symbols = [s.strip() + ".NS" for s in pd.read_csv(up)["Symbol"].dropna()]


# ---------------- UI ----------------
st.title("📈 Nifty 500 Movers")


def render():
    now = pd.Timestamp.now(tz=IST)
    market_open = now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)
    st.caption(f"{now:%d %b %Y, %H:%M:%S} IST · "
               f"{'🟢 Market open' if market_open else '🔴 Market closed'} · "
               f"≥{threshold}% in {window} min")

    with st.spinner("Scanning 500 stocks..."):
        df = scan(tuple(symbols), threshold, window, only_up, max_stale)

    if df.empty:
        st.info("No stocks matched right now." if market_open
                else "Market is closed, so there is no fresh data.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Gainers", int((df["Change %"] > 0).sum()))
    c2.metric("Losers", int((df["Change %"] < 0).sum()))

    def color(v):
        return "color: #16a34a" if v > 0 else "color: #dc2626"

    st.dataframe(df.style.map(color, subset=["Change %"]),
                 use_container_width=True, hide_index=True)
    st.caption("Data from Yahoo Finance (may be delayed). Not investment advice.")
