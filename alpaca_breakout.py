#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alpaca Krypto-Bot: Volatilitäts-/Donchian-Breakout auf ETH/USDT (nur Long)
- Daten & Trading komplett über alpaca-py
- Timeframe standardmäßig 1h
- Einstieg: Close > Donchian-High (N) UND Close > EMA200 UND ATR-Filter
- Stop: Donchian-Low (N) (dynamisch nachgezogen)
- Positionsgröße: risikobasiert (Risk % des Kontos / Stop-Distanz)
- Ausstieg:
    * Stop-Hit (Market)
    * Zeit-Exit nach MAX_HOLD_H
    * (Optional) Teilgewinn bei R=+2, Rest per Trailing-Stop (deaktiviert in dieser Version)
Hinweis: Exit/Stops werden per Logik (Market Orders) verwaltet, nicht als Bracket-Order.
"""

import os
import time
import json
import math
import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# --- Alpaca SDK ---
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# =========================
# Konfiguration aus ENV
# =========================
SYMBOL          = os.getenv("SYMBOL", "ETH/USDT")
TIMEFRAME_STR   = os.getenv("TIMEFRAME", "1h")     # "15m", "1h", "4h"
EMA_N           = int(os.getenv("EMA_N", "200"))
ATR_N           = int(os.getenv("ATR_N", "14"))
DONCHIAN_N      = int(os.getenv("DONCHIAN_N", "24"))   # 24 Kerzen ~ 1d bei 1h
ATR_MIN_PCT     = float(os.getenv("ATR_MIN_PCT", "0.004"))  # min. Volatilität z.B. 0.4%
RISK_PCT        = float(os.getenv("RISK_PCT", "0.01"))      # 1% Kontorisiko
LONG_ONLY       = os.getenv("LONG_ONLY", "true").lower() in ("1", "true", "yes")
MAX_HOLD_H      = int(os.getenv("MAX_HOLD_H", "48"))        # Zeit-Exit
LOOP_INTERVAL   = int(os.getenv("LOOP_INTERVAL_SEC", "60"))

APCA_KEY        = os.getenv("APCA_API_KEY_ID", "")
APCA_SECRET     = os.getenv("APCA_API_SECRET_KEY", "")
APCA_BASE_URL   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

STATE_PATH      = os.getenv("STATE_PATH", "/mnt/data/eth_usdt_state.json")  # Position-Metadaten


# =========================
# Helpers
# =========================
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def tf_from_string(s: str) -> TimeFrame:
    s = s.strip().lower()
    if s in ("1m", "1min", "minute"):
        return TimeFrame.Minute
    if s in ("5m", "5min"):
        return TimeFrame(5, "Minute")
    if s in ("15m", "15min"):
        return TimeFrame(15, "Minute")
    if s in ("30m", "30min"):
        return TimeFrame(30, "Minute")
    if s in ("1h", "hour"):
        return TimeFrame.Hour
    if s in ("4h",):
        return TimeFrame(4, "Hour")
    if s in ("1d", "day"):
        return TimeFrame.Day
    raise ValueError(f"Unsupported TIMEFRAME: {s}")

def load_state() -> dict:
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(d: dict) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(d, f)
    except Exception:
        pass


# =========================
# Indikatoren
# =========================
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(n, min_periods=n).mean()


# =========================
# Alpaca Clients
# =========================
def get_market_client() -> CryptoHistoricalDataClient:
    # Für den Daten-Client sind Key/Secret optional – wir geben sie mit
    return CryptoHistoricalDataClient(APCA_KEY, APCA_SECRET)

def get_trading_client() -> TradingClient:
    # Base URL wird aus ENV genommen (paper/live)
    return TradingClient(APCA_KEY, APCA_SECRET, paper=("paper" in APCA_BASE_URL))


# =========================
# Daten holen
# =========================
def fetch_bars(symbol: str, timeframe: TimeFrame, lookback: int = 1000) -> pd.DataFrame:
    client = get_market_client()
    end = now_utc()
    # großzügiger Startzeitraum (90 Tage), Alpaca limitiert über 'limit' ohnehin
    start = end - dt.timedelta(days=90)

    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        limit=lookback,
    )
    resp = client.get_crypto_bars(req)
    # Response: resp[symbol] -> list of Bar
    bars = resp.data.get(symbol, [])
    if not bars:
        return pd.DataFrame()

    rows = []
    for b in bars:
        rows.append({
            "t": pd.Timestamp(b.timestamp).tz_convert("UTC"),
            "o": float(b.open),
            "h": float(b.high),
            "l": float(b.low),
            "c": float(b.close),
            "v": float(b.volume or 0.0),
        })
    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    return df


# =========================
# Trading-Logik
# =========================
def compute_signals(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Gibt (close, ema200, don_up, don_lo, atr_val) der letzten ABGESCHLOSSENEN Kerze zurück.
    """
    if df.empty or len(df) < max(EMA_N, DONCHIAN_N, ATR_N) + 1:
        return None, None, None, None, None

    df = df.copy()
    df["ema"] = ema(df["c"], EMA_N)
    df["atr"] = atr(df["h"], df["l"], df["c"], ATR_N)
    df["don_up"] = df["h"].rolling(DONCHIAN_N, min_periods=DONCHIAN_N).max()
    df["don_lo"] = df["l"].rolling(DONCHIAN_N, min_periods=DONCHIAN_N).min()

    # letzte ABGESCHLOSSENE Kerze = vorletzte Zeile
    last = df.iloc[-2]
    return float(last["c"]), float(last["ema"]), float(last["don_up"]), float(last["don_lo"]), float(last["atr"])


def position_for_symbol(tc: TradingClient, symbol: str):
    try:
        poss = tc.get_open_positions()
        for p in poss:
            if p.symbol.replace('-', '/').upper() == symbol.upper():
                return p
    except Exception:
        pass
    return None


def dollar_risk_per_trade(tc: TradingClient) -> float:
    acct = tc.get_account()
    eq = float(acct.equity)
    return eq * max(0.0, min(0.05, RISK_PCT))  # clamp 0…5%


def calc_qty(risk_amount: float, entry: float, stop: float) -> float:
    stop_dist = max(1e-6, entry - stop)  # Long
    qty = risk_amount / stop_dist
    # Runde auf 6 Nachkommastellen runter (ETH fractional):
    qty = math.floor(qty * 1e6) / 1e6
    return max(0.0001, qty)


def place_market_order(tc: TradingClient, symbol: str, side: OrderSide, qty: float):
    req = MarketOrderRequest(
        symbol=symbol.replace('/', '-'),  # Alpaca nutzt "ETH-USDT"
        qty=str(qty),                     # als string senden (fractional ok)
        side=side,
        time_in_force=TimeInForce.GTC,
    )
    return tc.submit_order(req)


def seconds_since(ts_iso: str) -> float:
    try:
        t = dt.datetime.fromisoformat(ts_iso)
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return (now_utc() - t).total_seconds()
    except Exception:
        return 1e12


def trade_once() -> None:
    tf = tf_from_string(TIMEFRAME_STR)
    df = fetch_bars(SYMBOL, tf, lookback=1200)
    close, ema200, don_up, don_lo, atr_val = compute_signals(df)

    if any(x is None or np.isnan(x) for x in (close, ema200, don_up, don_lo, atr_val)):
        print("[BOT] Nicht genug Daten.")
        return

    atr_pct = atr_val / close if close else 0.0
    print(f"[BOT] • Close={close:.2f} • EMA200={ema200:.2f} • Up={don_up:.2f} • Lo={don_lo:.2f} • ATR={atr_val:.2f} ({atr_pct:.3%})")

    trend_ok   = close > ema200
    vola_ok    = atr_pct >= ATR_MIN_PCT
    breakout   = close > don_up
    stop_price = float(don_lo)

    tc = get_trading_client()
    pos = position_for_symbol(tc, SYMBOL)

    # --------------------
    # Exit / Management
    # --------------------
    st = load_state()
    st_pos = st.get("position", {})

    if pos:
        qty = float(pos.qty)
        avg = float(pos.avg_entry_price)
        # dynamischen Stop auf jüngstes Donchian-Low ziehen (nie absenken)
        prev_stop = float(st_pos.get("stop", stop_price))
        stop_dyn  = max(prev_stop, stop_price)
        st_pos["stop"] = stop_dyn

        # Zeit-Exit?
        opened_iso = st_pos.get("opened_at")
        if not opened_iso:
            st_pos["opened_at"] = now_utc().isoformat()

        hold_sec = seconds_since(st_pos["opened_at"])
        time_exit = hold_sec > MAX_HOLD_H * 3600

        # Stop-Hit?
        stop_hit = close < stop_dyn

        if stop_hit or time_exit:
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            print(f"[BOT] EXIT: {'Stop' if stop_hit else 'Zeit-Exit'} • qty={qty} • stop={stop_dyn:.2f}")
            try:
                place_market_order(tc, SYMBOL, side, abs(qty))
            except Exception as e:
                print(f"[BOT] Exit-Order Fehler: {e}")
            # State löschen
            st["position"] = {}
            save_state(st)
            return

        # Noch im Trade: State aktualisieren
        st["position"] = st_pos
        save_state(st)
        print(f"[BOT] HOLD • qty={qty} • stop={stop_dyn:.2f}")
        return

    # --------------------
    # Entry
    # --------------------
    # Nur Long in dieser Version
    if LONG_ONLY:
        if breakout and trend_ok and vola_ok:
            risk = dollar_risk_per_trade(tc)
            qty  = calc_qty(risk, close, stop_price)
            if qty <= 0:
                print("[BOT] Größe <= 0 – kein Entry.")
                return

            print(f"[BOT] BUY • qty={qty} • risk≈${risk:.2f} • stop={stop_price:.2f}")
            try:
                place_market_order(tc, SYMBOL, OrderSide.BUY, qty)
                # State speichern
                st["position"] = {
                    "symbol": SYMBOL,
                    "opened_at": now_utc().isoformat(),
                    "stop": stop_price
                }
                save_state(st)
            except Exception as e:
                print(f"[BOT] Buy-Order Fehler: {e}")
        else:
            reason = []
            if not breakout: reason.append("kein Breakout")
            if not trend_ok: reason.append("Trend<EMA200")
            if not vola_ok:  reason.append("ATR zu klein")
            print(f"[BOT] Kein Entry ({', '.join(reason)})")
    else:
        print("[BOT] SHORT/Long-Mix nicht implementiert (LONG_ONLY).")


# =========================
# Runner
# =========================
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="Eine Runde und beenden.")
    p.add_argument("--loop", action="store_true", help="Dauerschleife.")
    args = p.parse_args()

    print(f"[BOT] Starte • Exchange=Alpaca Crypto • Symbol={SYMBOL} • TF={TIMEFRAME_STR}")
    if args.once:
        trade_once()
        return
    if args.loop:
        while True:
            try:
                trade_once()
            except Exception as e:
                print(f"[BOT] Fehler: {e}")
            time.sleep(LOOP_INTERVAL)
        return

    # default einmal
    trade_once()


if __name__ == "__main__":
    main()
