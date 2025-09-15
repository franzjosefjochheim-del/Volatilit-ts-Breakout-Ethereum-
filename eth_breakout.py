#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETH Volatility Breakout (Donchian + ATR) – live/paper mit ccxt
- Timeframe: 1h
- Breakout über 24h-Range mit ATR-Puffer
- Volatilitätsfilter, Trendfilter (EMA200)
- Positionsgröße & Stop auf ATR-Basis
- Teilgewinn bei +2R, Rest per Chandelier (3*ATR)
- Zeit-Exit nach 48h ohne +1R
- 1 Position gleichzeitig (kein Pyramiding)
"""

import os
import time
import json
import math
import datetime as dt
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import ccxt

# ------------------------ Konfiguration aus ENV ------------------------
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower()
SYMBOL        = os.getenv("SYMBOL", "ETH/USDT")
RISK_PCT      = float(os.getenv("RISK_PCT", "0.75"))           # % vom Equity pro Trade
LONG_ONLY     = os.getenv("LONG_ONLY", "0") == "1"
PAPER         = os.getenv("PAPER", "1") == "1"

TIMEFRAME     = "1h"
DONCHIAN_N    = 24
ATR_N         = 14
ATR_ENTRY_K   = 0.2     # Puffer * ATR
ATR_STOP_K    = 1.5
ATR_TRAIL_K   = 3.0
VOL_FILTER    = 0.01    # ATR/Close >= 1%
EMA_N         = 200
PARTIAL_R     = 2.0     # 50% raus bei 2R
TIME_EXIT_H   = 48
LOOP_SEC      = 60

STATE_FILE    = "eth_breakout_state.json"  # einfacher lokaler State

# ------------------------ Utils ------------------------

def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "position": None,   # {"side":"long/short","entry":float,"size":float,"stop":float,"risk_per_unit":float,"t_entry":iso,"tp_half_done":bool}
        "last_bar_time": None
    }

def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_FILE)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def get_exchange() -> ccxt.Exchange:
    klass = getattr(ccxt, EXCHANGE_NAME)
    kwargs = {"enableRateLimit": True}
    if not PAPER:
        api_key = os.getenv("API_KEY", "")
        api_secret = os.getenv("API_SECRET", "")
        if api_key and api_secret:
            kwargs.update({"apiKey": api_key, "secret": api_secret})
        # ggf. weitere Felder: password, uid etc. je nach Börse
    ex = klass(kwargs)
    # für einige Börsen Testnet konfigurieren (optional, hier generisch ausgelassen)
    return ex

def fetch_ohlcv_df(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def can_trade_now(df: pd.DataFrame) -> Tuple[bool, Optional[pd.Timestamp]]:
    """Nur auf bar-close reagieren. Gibt True zurück, wenn neue geschlossene Bar existiert."""
    if df.empty:
        return False, None
    last = df["timestamp"].iloc[-1]
    # Bei ccxt ist letzte Zeile die aktuell laufende Kerze ODER die letzte geschlossene – je nach Börse.
    # Sicherer Ansatz: Wir handeln die VORLETZTE Zeile (die ist sicher geschlossen).
    if len(df) < 2:
        return False, None
    closed_bar_time = df["timestamp"].iloc[-2]
    return True, closed_bar_time

def price_precision(ex: ccxt.Exchange, symbol: str) -> Tuple[int, int]:
    ex.load_markets()
    m = ex.markets.get(symbol, {})
    price_decimals = 6
    amount_decimals = 6
    if m:
        price_decimals  = m.get("precision", {}).get("price", 6)
        amount_decimals = m.get("precision", {}).get("amount", 6)
    return price_decimals, amount_decimals

def round_to(x: float, decimals: int) -> float:
    if decimals >= 0:
        p = 10.0**decimals
        return math.floor(x * p) / p
    return x

# ------------------------ Handelslogik ------------------------

def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = atr(df, ATR_N)
    df["ema200"] = ema(df["close"], EMA_N)
    # Donchian über die letzten DONCHIAN_N Bars, exkl. aktueller Bar: shift(1)
    df["donch_high"] = df["high"].rolling(DONCHIAN_N).max().shift(1)
    df["donch_low"]  = df["low"].rolling(DONCHIAN_N).min().shift(1)
    return df

def signal(df: pd.DataFrame, long_only: bool) -> Dict[str, Any]:
    """Ermittelt Signal auf der letzten geschlossenen Bar (Index -2)."""
    if len(df) < max(DONCHIAN_N + 2, ATR_N + 2, EMA_N + 2):
        return {"action":"NONE"}
    i = -2  # letzte abgeschlossene
    row = df.iloc[i]
    close = row["close"]
    a = row["atr"]
    ema200 = row["ema200"]
    donch_high = row["donch_high"]
    donch_low  = row["donch_low"]

    if pd.isna(a) or pd.isna(ema200) or pd.isna(donch_high) or pd.isna(donch_low):
        return {"action":"NONE"}

    # Volatilitätsfilter
    if (a / close) < VOL_FILTER:
        return {"action":"NONE"}

    # Trendfilter + Breakout mit ATR-Puffer
    up_break   = close > (donch_high + ATR_ENTRY_K * a)
    down_break = close < (donch_low  - ATR_ENTRY_K * a)

    # Optional: nur Long handeln
    if long_only:
        if up_break and close > ema200:
            stop = close - ATR_STOP_K * a
            return {"action":"LONG", "entry": close, "stop": stop, "atr": a}
        return {"action":"NONE"}

    # Long + Short
    if up_break and close > ema200:
        stop = close - ATR_STOP_K * a
        return {"action":"LONG", "entry": close, "stop": stop, "atr": a}
    if down_break and close < ema200:
        stop = close + ATR_STOP_K * a
        return {"action":"SHORT","entry": close, "stop": stop, "atr": a}

    return {"action":"NONE"}

def calc_position_size(equity_usd: float, entry: float, stop: float) -> float:
    risk_usd = equity_usd * (RISK_PCT / 100.0)
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return 0.0
    size = risk_usd / per_unit_risk
    return max(0.0, size)

def current_equity_usd(ex: ccxt.Exchange) -> float:
    """Für PAPER nehmen wir einen fiktiven Account mit 10.000 USD. Live: Free-Cash + Wert?"""
    if PAPER:
        return 10000.0
    # Minimalvariante (Spot): fiat-Saldo + ETH*Preis. Für echte Portfoliobewertung bitte ausbauen.
    balances = ex.fetch_balance()
    total_usdt = balances.get("total", {}).get("USDT", 0.0)
    # ETH-Wert grob dazurechnen
    ticker = ex.fetch_ticker(SYMBOL)
    last   = float(ticker["last"])
    total_eth = balances.get("total", {}).get("ETH", 0.0)
    return float(total_usdt + total_eth * last)

def place_order(ex: ccxt.Exchange, side: str, amount: float) -> Dict[str, Any]:
    if PAPER:
        return {"id":"paper-order", "status":"filled"}
    try:
        ord = ex.create_order(SYMBOL, "market", side, amount)
        return ord
    except Exception as e:
        print(f"[ORDER] Fehler: {e}")
        return {"id":"error", "status":"rejected", "error": str(e)}

def flatten_position(ex: ccxt.Exchange, pos: Dict[str, Any]) -> None:
    if pos is None: 
        return
    side = pos["side"]
    amount = pos["size"]
    if amount <= 0:
        return
    exit_side = "sell" if side == "long" else "buy"
    print(f"[EXIT] Schließe Position: {exit_side} {amount}")
    _ = place_order(ex, exit_side, amount)

def manage_open_position(ex: ccxt.Exchange, df: pd.DataFrame, state: Dict[str, Any]) -> None:
    """Regelt Trailing, Teilgewinn, Zeit-Exit auf Basis der letzten geschlossenen Bar."""
    pos = state.get("position")
    if not pos:
        return
    i = -2
    row = df.iloc[i]
    close = float(row["close"])
    a = float(row["atr"])
    entry = float(pos["entry"])
    side  = pos["side"]
    stop  = float(pos["stop"])
    t_entry = dt.datetime.fromisoformat(pos["t_entry"])
    r_per_unit = float(pos["risk_per_unit"])  # = abs(entry - initial_stop)
    tp_half_done = bool(pos.get("tp_half_done", False))

    # 1) Zeit-Exit nach TIME_EXIT_H
    if (utcnow() - t_entry).total_seconds() >= TIME_EXIT_H * 3600:
        if abs(close - entry) < r_per_unit:  # grobe Bedingung "kein +1R in TimeExit"
            print("[TIME EXIT] Keine +1R innerhalb Zeitfenster -> Flat")
            flatten_position(ex, pos)
            state["position"] = None
            save_state(state)
            return

    # 2) Teilgewinn bei +2R (50% schließen einmalig)
    pnl_per_unit = (close - entry) if side == "long" else (entry - close)
    R = pnl_per_unit / r_per_unit if r_per_unit > 0 else 0.0
    if (not tp_half_done) and (R >= PARTIAL_R):
        half = pos["size"] * 0.5
        print(f"[TP 50%] Realisiere halbe Position bei ~{close}")
        _ = place_order(ex, "sell" if side=="long" else "buy", half)
        pos["size"] = pos["size"] - half
        pos["tp_half_done"] = True
        save_state(state)

    # 3) Trailing-Stop (Chandelier: 3*ATR)
    if side == "long":
        trail = close - ATR_TRAIL_K * a
        new_stop = max(stop, trail)  # Stop nur anheben
        if new_stop != stop:
            print(f"[TRAIL] Stop anheben: {stop:.2f} -> {new_stop:.2f}")
            pos["stop"] = new_stop
            save_state(state)
        # Wenn close unter Stop -> aussteigen
        if close <= pos["stop"]:
            print(f"[STOP OUT] LONG @ {close:.2f} unter Stop {pos['stop']:.2f}")
            flatten_position(ex, pos)
            state["position"] = None
            save_state(state)
    else:
        trail = close + ATR_TRAIL_K * a
        new_stop = min(stop, trail)  # Stop nur senken (bei Short)
        if new_stop != stop:
            print(f"[TRAIL] Stop senken: {stop:.2f} -> {new_stop:.2f}")
            pos["stop"] = new_stop
            save_state(state)
        if close >= pos["stop"]:
            print(f"[STOP OUT] SHORT @ {close:.2f} über Stop {pos['stop']:.2f}")
            flatten_position(ex, pos)
            state["position"] = None
            save_state(state)

def try_new_entry(ex: ccxt.Exchange, df: pd.DataFrame, state: Dict[str, Any]) -> None:
    if state.get("position"):
        return  # kein Pyramiding
    sig = signal(df, LONG_ONLY)
    if sig["action"] == "NONE":
        return

    entry = float(sig["entry"])
    stop  = float(sig["stop"])
    a     = float(sig["atr"])

    # Spread/Plumps-Check (einfach)
    ticker = ex.fetch_ticker(SYMBOL)
    ask = float(ticker["ask"] or entry)
    bid = float(ticker["bid"] or entry)
    if ask <= 0 or bid <= 0:
        return
    spread = (ask - bid) / ((ask + bid) / 2.0)
    if spread > 0.0005:  # > 5 bps -> nicht handeln
        print(f"[SKIP] Spread zu groß: {spread:.5f}")
        return

    eq = current_equity_usd(ex)
    size = calc_position_size(eq, entry, stop)
    if size <= 0:
        print("[SKIP] Größe=0")
        return

    # Rundung nach Börsenpräzision
    price_dec, amt_dec = price_precision(ex, SYMBOL)
    size = round_to(size, amt_dec)
    if size <= 0:
        print("[SKIP] Größe nach Rundung=0")
        return

    side = "buy" if sig["action"] == "LONG" else "sell"
    print(f"[ENTRY] {sig['action']} {size} @ ~{entry:.2f}, Stop {stop:.2f}, ATR {a:.2f}")
    ord = place_order(ex, side, size)
    if ord.get("status") == "rejected":
        print("[ENTRY] Order abgelehnt")
        return

    # Position in State anlegen
    state["position"] = {
        "side": "long" if side == "buy" else "short",
        "entry": entry,
        "size": size,
        "stop": stop,
        "risk_per_unit": abs(entry - stop),
        "t_entry": utcnow().isoformat(),
        "tp_half_done": False
    }
    save_state(state)

# ------------------------ Main Loop ------------------------

def trade_once(ex: ccxt.Exchange, state: Dict[str, Any]) -> None:
    df = fetch_ohlcv_df(ex, SYMBOL, TIMEFRAME, limit=max(EMA_N + 50, DONCHIAN_N + 50))
    if df.empty:
        print("[WARN] Keine Daten")
        return

    # Neue Bar prüfen
    ok, closed_time = can_trade_now(df)
    if not ok:
        return
    last_seen = state.get("last_bar_time")
    if last_seen == (closed_time.isoformat() if closed_time else None):
        # nichts neues
        return

    # Indikatoren
    ind = build_indicators(df)

    # Offene Position managen (Stop/Trail/TP/TimeExit) – immer auf Basis letzter geschlossener Bar
    manage_open_position(ex, ind, state)

    # Neuer Einstieg?
    try_new_entry(ex, ind, state)

    # Bar-Marker aktualisieren
    state["last_bar_time"] = closed_time.isoformat()
    save_state(state)

def loop_forever():
    print(f"[BOT] ETH Breakout gestartet – TF={TIMEFRAME} – PAPER={PAPER}")
    ex = get_exchange()
    state = load_state()
    while True:
        try:
            trade_once(ex, state)
        except Exception as e:
            print(f"[ERR] {e}")
        time.sleep(LOOP_SEC)

def run_once():
    ex = get_exchange()
    state = load_state()
    trade_once(ex, state)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="Eine Runde ausführen")
    p.add_argument("--loop", action="store_true", help="Endlosschleife")
    args = p.parse_args()

    if args.once:
        run_once()
    else:
        loop_forever()
