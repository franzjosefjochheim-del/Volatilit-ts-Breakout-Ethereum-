# Ethereum Volatility Breakout Bot

Automatisierter Trading-Bot für **Ethereum (ETH/USDT)** basierend auf einer
**Volatilitäts-Breakout-Strategie** mit **Donchian-Kanal + ATR**.

## Features
- Timeframe: 1h
- Breakout über 24h-Range mit ATR-Puffer
- Volatilitätsfilter (ATR/Close ≥ 1 %)
- Trendfilter (EMA200)
- Positionsgröße und Stop-Loss auf ATR-Basis
- Teilgewinn bei +2R, Rest per Trailing Stop (Chandelier 3*ATR)
- Zeit-Exit nach 48h ohne +1R
- Kein Pyramiding (max. 1 Position)

## Installation
```bash
pip install -r requirements.txt
