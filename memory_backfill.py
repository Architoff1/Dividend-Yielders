# =========================================================
# SYNTHETIC MEMORY BACKFILL ENGINE
# OPTION A — HYBRID C+
# WEEKLY SNAPSHOTS | 5Y ROLLING WINDOW
# =========================================================

import yfinance as yf
import json
import os
from datetime import datetime, timedelta

# ================= CONFIG =================
MEMORY_DIR = "memory"
LOOKBACK_YEARS = 5
SNAPSHOT_INTERVAL_DAYS = 7   # weekly

# ================= MEMORY HELPERS =================
def ensure_memory_dir():
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)

def memory_path(stock):
    return os.path.join(MEMORY_DIR, f"{stock}.json")

def save_memory(stock, data):
    ensure_memory_dir()
    with open(memory_path(stock), "w") as f:
        json.dump(data, f, indent=4)

# ================= CORE BACKFILL =================
def build_synthetic_memory(
    stock,
    compute_indicators,
    extract_signals,
    confidence_engine
):
    """
    Builds historical synthetic memory using past price data.
    Uses the SAME logic as live analysis (no cheating).
    """

    print(f"🧠 Building synthetic memory for {stock}...")

    ticker = yf.Ticker(stock + ".NS")
    df = ticker.history(period=f"{LOOKBACK_YEARS}y")

    if df.empty:
        print(f"⚠️ No historical data for {stock}")
        return

    df = compute_indicators(df)

    memory = []
    start_date = df.index.min()
    end_date = df.index.max()

    current_date = start_date

    while current_date <= end_date:
        try:
            window = df[df.index <= current_date].tail(260)  # ~1 trading year

            if len(window) < 200:
                current_date += timedelta(days=SNAPSHOT_INTERVAL_DAYS)
                continue

            latest = window.iloc[-1]
            signals, _, _ = extract_signals(window, {})

            confidence = confidence_engine(signals)

            memory.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "confidence": confidence,
                "synthetic": True
            })

        except Exception:
            pass

        current_date += timedelta(days=SNAPSHOT_INTERVAL_DAYS)

    save_memory(stock, memory)
    print(f"✅ Synthetic memory created for {stock} ({len(memory)} entries)")
