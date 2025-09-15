#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ethereum Volatility Breakout Bot
- Donchian-Kanal (obere/untere 24-Periode) + ATR-Puffer
- Trendfilter: EMA200 (nur Longs oberhalb EMA200)
- 1 Position max, Trailing Stop nach Chandelier (3*ATR), Zeit-Exit nach 48h
- Timeframe per ENV (TIMEFRAME), z.B. 15m / 1h / 4h / 1d
- PAPER=true (Default) => nur Logs, keine echten Orders
- Exchanges via ccxt (nutze z.B. 'kraken', 'bybit', 'coinbase', 'binanceus' – Binance global ist in vielen Regionen gesperrt)
"""

import os
import sys
import time
import math
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import ccxt


# =========================
# -------- ENV ------------
# =========================

EXCHANGE_ID = os.getenv("EXCHANGE", "kraken").strip().lower()
SYMBOL = os.getenv("SYMBOL", "ETH/USDT").strip().upper()

# Timeframe aus ENV (Fallback 1h); harte Whitelist zur Sicherheit:
TIMEFRAME = os.getenv("TIMEFRAME", "1h").strip().lower()
_ALLOWED_TF = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"}
if TIMEFRAME not in _ALLOWED_TF:
    print(f"[WARN] Ungültiges TIMEFRAME='{TIMEFRAME}'. Erlaubt: {_ALLOWED_TF}. Fallback auf '1h'.")
    TIMEFRAME = "1h"

RISK_PCT = float(os.getenv("RISK_PCT", "0.01"))  # 1% vom verfügbaren Kapital
LONG_ONLY = os.getenv("LONG_ONLY", "true").strip().lower() == "true"
PAPER = os.getenv("PAPER", "true").strip().lower() == "true"

API_KEY = os.getenv("API_KEY", "").strip()
API_SECRET = os.getenv("API_SECRET", "").strip()

# Strategie-Parameter (können bei Bedarf ebenfalls per ENV steuerbar gemacht werden)
EMA_N = int(os.getenv("EMA_N", "200"))
DONCHIAN_N = int(os.getenv("DONCHIAN_N", "24"))
ATR_N = int(os.getenv("ATR_N", "14"))
ATR_BUFFER = float(os.getenv("ATR_BUFFER", "0.5"))  # 0.5 * ATR Zusatzpuffer auf Breakout
TRAIL_MULT = float(os.getenv("TRAIL_MULT", "3.0"))  # Chandelier-Stop: 3*ATR
MAX_POSITIONS = 1
TIME_EXIT_HOURS = int(os.getenv("TIME_EXIT_HOURS", "48"))

# Loop-Intervall: kurze Wartezeit, wir handeln auf fertige Kerzen (Polling)
LOOP_SLEEP_SEC = int(os.getenv("LOOP_SLEEP_SEC", "60"))


# =========================
# ----- Utilities ----------
# =========================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def tf_to_minutes(tf: str) -> int:
    mapping = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
        "1d": 1440,
    }
    return mapping[tf]

def print_header():
    print(
        f"[BOT] Exchange={EXCHANGE_ID} • Symbol={SYMBOL} • TF={TIMEFRAME} "
        f"• PAPER={PAPER} • RISK_PCT={RISK_PCT:.2%} • LONG_ONLY={LONG_ONLY}"
    )

# =========================
# ---- Exchange-Init ------
# =========================

def init_exchange() -> ccxt.Exchange:
    kwargs: Dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": 30000,
    }
    if API_KEY and API_SECRET:
        kwargs["apiKey"] = API_KEY
        kwargs["secret"] = API_SECRET

    if not hasattr(ccxt, EXCHANGE_ID):
        raise RuntimeError(f"Unbekannte Exchange '{EXCHANGE_ID}'. Prüfe ccxt Doku.")

    ex: ccxt.Exchange = getattr(ccxt, EXCHANGE_ID)(kwargs)

    # Für einige Börsen sinnvolle Defaults:
    # (keine harten Anforderungen – ccxt übernimmt viel)
    ex.options = ex.options or {}
    return ex

# =========================
# ---- Daten & Indikatoren
# =========================

def fetch_ohlcv_df(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 600) -> pd.DataFrame:
    """Holt OHLCV und liefert DataFrame mit UTC-Timestamps und Spalten [t,o,h,l,c,v]."""
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["mts", "o", "h", "l", "c", "v"])
    df["t"] = pd.to_datetime(df["mts"], unit="ms", utc=True)
    df = df[["t", "o", "h", "l", "c", "v"]].sort_values("t").reset_index(drop=True)
    return df

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["h"]
    low = df["l"]
    close = df["c"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def donchian(df: pd.DataFrame, n: int = 20):
    upper = df["h"].rolling(n).max()
    lower = df["l"].rolling(n).min()
    return upper, lower

# =========================
# ---- Positions-State ----
# =========================

class Position:
    def __init__(self):
        self.side: Optional[str] = None   # "long" / "short"
        self.entry: Optional[float] = None
        self.qty: float = 0.0
        self.stop: Optional[float] = None
        self.open_time: Optional[datetime] = None
        self.highest_since_entry: Optional[float] = None
        self.lowest_since_entry: Optional[float] = None

    def reset(self):
        self.__init__()

STATE = {
    "pos": Position()
}

# =========================
# ---- Order-Helfer --------
# =========================

def get_quote_from_symbol(symbol: str) -> str:
    # "ETH/USDT" -> "USDT"
    if "/" in symbol:
        return symbol.split("/")[1]
    return symbol

def fetch_equity_quote(ex: ccxt.Exchange, symbol: str) -> float:
    """Lädt verfügbares Quote-Wallet (z.B. USDT). Bei PAPER wird ein fester Wert angenommen."""
    if PAPER or not API_KEY:
        return 1000.0  # Paper-Einsatz
    quote = get_quote_from_symbol(symbol)
    bal = ex.fetch_balance()
    return float(bal.get(quote, {}).get("free", 0.0))

def send_market_order(ex: ccxt.Exchange, symbol: str, side: str, qty: float) -> Optional[Dict[str, Any]]:
    if qty <= 0:
        return None
    if PAPER:
        print(f"[PAPER] {side.upper()} {qty:.6f} {symbol} (MARKET)")
        return {"id": "paper-order", "symbol": symbol, "side": side, "amount": qty, "type": "market"}

    try:
        order = ex.create_order(symbol, "market", side, qty)
        print(f"[LIVE]  {side.upper()} {qty:.6f} {symbol} (MARKET) -> {order.get('id')}")
        return order
    except Exception as e:
        print(f"[ERR] Order fehlgeschlagen: {e}")
        return None

# =========================
# ---- Logik / Signale ----
# =========================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema"] = ema(df["c"], EMA_N)
    df["atr"] = atr(df, ATR_N)
    dc_up, dc_lo = donchian(df, DONCHIAN_N)
    df["dc_up"] = dc_up
    df["dc_lo"] = dc_lo
    return df

def decide_and_trade(ex: ccxt.Exchange, df: pd.DataFrame):
    """Kernlogik: Breakout + ATR-Puffer + EMA200-Filter. 1 Position max; Trailstop + Zeit-Exit."""
    if df.empty or len(df) < max(EMA_N, DONCHIAN_N, ATR_N) + 2:
        print("[BOT] Zu wenig Daten – warte …")
        return

    pos: Position = STATE["pos"]

    # Wir verwenden IMMER die letzte ABGESCHLOSSENE Kerze
    # -> index -2 (die letzte könnte noch unvollständig sein)
    row = df.iloc[-2]
    close = float(row["c"])
    ema200 = float(row["ema"])
    atr_val = float(row["atr"])
    dc_up = float(row["dc_up"])
    dc_lo = float(row["dc_lo"])

    # Entry-Bedingungen (nur long, falls LONG_ONLY)
    long_break = close > (dc_up + ATR_BUFFER * atr_val)
    short_break = (not LONG_ONLY) and (close < (dc_lo - ATR_BUFFER * atr_val))

    # --- Trailstop & Zeit-Exit, wenn Position offen ---
    if pos.side is not None:
        # Extremwerte seit Entry für Chandelier
        if pos.side == "long":
            pos.highest_since_entry = max(pos.highest_since_entry or close, close)
            trail = pos.highest_since_entry - TRAIL_MULT * atr_val
            if pos.stop is None:
                pos.stop = trail
            else:
                pos.stop = max(pos.stop, trail)  # Stop nur nachziehen

            # Zeit-Exit?
            if pos.open_time and now_utc() - pos.open_time >= timedelta(hours=TIME_EXIT_HOURS):
                print("[BOT] Zeit-Exit ausgelöst. Schließe LONG.")
                qty_close = pos.qty
                send_market_order(ex, SYMBOL, "sell", qty_close)
                pos.reset()
                return

            # Stop ausgelöst?
            if close <= (pos.stop or -1):
                print(f"[BOT] Stop ausgelöst @ {pos.stop:.2f}. Schließe LONG.")
                qty_close = pos.qty
                send_market_order(ex, SYMBOL, "sell", qty_close)
                pos.reset()
                return

        elif pos.side == "short":
            pos.lowest_since_entry = min(pos.lowest_since_entry or close, close)
            trail = pos.lowest_since_entry + TRAIL_MULT * atr_val
            if pos.stop is None:
                pos.stop = trail
            else:
                pos.stop = min(pos.stop, trail)
            if pos.open_time and now_utc() - pos.open_time >= timedelta(hours=TIME_EXIT_HOURS):
                print("[BOT] Zeit-Exit ausgelöst. Schließe SHORT.")
                qty_close = pos.qty
                send_market_order(ex, SYMBOL, "buy", qty_close)
                pos.reset()
                return
            if close >= (pos.stop or 1e9):
                print(f"[BOT] Stop ausgelöst @ {pos.stop:.2f}. Schließe SHORT.")
                qty_close = pos.qty
                send_market_order(ex, SYMBOL, "buy", qty_close)
                pos.reset()
                return

        # Wenn Position offen ist, keine neuen Entries prüfen (nur 1 Position)
        return

    # --- Keine Position offen: neue Entry-Signale prüfen ---
    # Trendfilter: nur Long wenn über EMA200, nur Short wenn unter EMA200
    can_long = close > ema200
    can_short = close < ema200 and (not LONG_ONLY)

    # Positionsgröße: risikobasiert anhand ATR – sehr konservativ:
    # Wir riskieren RISK_PCT des Quote-Kapitals mit Stop-Abstand ~ (TRAIL_MULT * ATR)
    quote_capital = fetch_equity_quote(ex, SYMBOL)
    risk_amount = max(0.0, quote_capital * RISK_PCT)
    stop_distance = max(atr_val * TRAIL_MULT, 1e-6)
    # Grobe Annäherung der Stückzahl:
    qty_long = risk_amount / stop_distance

    # Exchange-Minimumgrößen sind unterschiedlich – wir lassen es hier grob.
    # Alternative: ex.markets[SYMBOL]['limits']['amount']['min'] berücksichtigen (wenn verfügbar).
    qty_long = max(0.0, round(qty_long, 6))

    if long_break and can_long and qty_long > 0:
        print(f"[BOT] LONG-Signal • Close={close:.2f} > DonchianUp+ATRpuffer ({dc_up + ATR_BUFFER*atr_val:.2f}) • EMA200={ema200:.2f}")
        order = send_market_order(ex, SYMBOL, "buy", qty_long)
        if order:
            STATE["pos"] = Position()
            STATE["pos"].side = "long"
            STATE["pos"].entry = close
            STATE["pos"].qty = qty_long
            STATE["pos"].open_time = now_utc()
            STATE["pos"].highest_since_entry = close
            STATE["pos"].stop = close - TRAIL_MULT * atr_val
            print(f"[BOT] LONG eröffnet @ {close:.2f} • Qty={qty_long:.6f} • initialer Stop={STATE['pos'].stop:.2f}")
        return

    if short_break and can_short:
        # Für Short analoge Stückzahl (bei Spot nicht immer verfügbar)
        qty_short = qty_long
        if qty_short <= 0:
            return
        print(f"[BOT] SHORT-Signal • Close={close:.2f} < DonchianLo-ATRpuffer ({dc_lo - ATR_BUFFER*atr_val:.2f}) • EMA200={ema200:.2f}")
        order = send_market_order(ex, SYMBOL, "sell", qty_short)
        if order:
            STATE["pos"] = Position()
            STATE["pos"].side = "short"
            STATE["pos"].entry = close
            STATE["pos"].qty = qty_short
            STATE["pos"].open_time = now_utc()
            STATE["pos"].lowest_since_entry = close
            STATE["pos"].stop = close + TRAIL_MULT * atr_val
            print(f"[BOT] SHORT eröffnet @ {close:.2f} • Qty={qty_short:.6f} • initialer Stop={STATE['pos'].stop:.2f}")
        return

    # Kein Signal:
    print(
        f"[BOT] Kein Entry • Close={close:.2f} • EMA200={ema200:.2f} • "
        f"Up={dc_up:.2f} • Lo={dc_lo:.2f} • ATR={atr_val:.2f}"
    )

# =========================
# ---- Run-Wrapper --------
# =========================

def trade_once(ex: ccxt.Exchange):
    print(f"[BOT] Starte Runde • Exchange={EXCHANGE_ID} • TF={TIMEFRAME}")
    try:
        ex.load_markets()
        df = fetch_ohlcv_df(ex, SYMBOL, TIMEFRAME, limit=max(EMA_N + 50, DONCHIAN_N + 50, ATR_N + 50))
        if df.empty:
            print("[BOT] Keine Daten vom Feed.")
            return
        df = compute_indicators(df)
        decide_and_trade(ex, df)
        print("[BOT] Runde fertig.")
    except ccxt.base.errors.ExchangeNotAvailable as e:
        print(f"[ERR] Exchange nicht verfügbar: {e}")
    except Exception as e:
        print(f"[ERR] Unerwarteter Fehler: {e}")

def loop_forever(ex: ccxt.Exchange):
    print_header()
    while True:
        trade_once(ex)
        time.sleep(LOOP_SLEEP_SEC)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Eine Runde ausführen.")
    parser.add_argument("--loop", action="store_true", help="Endlosschleife.")
    args = parser.parse_args()

    # Binance global ist oft regional gesperrt (HTTP 451). Nutze z.B. 'binanceus', 'kraken', 'bybit', 'coinbase'.
    if EXCHANGE_ID == "binance":
        print("[WARN] 'binance' ist in vielen Regionen gesperrt (HTTP 451). "
              "Bitte stattdessen z.B. 'binanceus', 'kraken', 'bybit' oder 'coinbase' verwenden.")

    ex = init_exchange()

    if args.once:
        print_header()
        trade_once(ex)
        return
    if args.loop:
        loop_forever(ex)
        return

    # Default: einmal ausführen
    print_header()
    trade_once(ex)


if __name__ == "__main__":
    main()
