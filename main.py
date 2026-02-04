# =========================================================
# DIVIDEND + VALUE + MULTI-TIMEFRAME ENGINE
# PHASE 1 — STRUCTURAL FOUNDATION (DECISION-GRADE)
# PHASE 2.1 — MEMORY ENGINE (INTELLIGENCE LAYER)
# PHASE 2.2 — CONFIDENCE TREND & DECAY ENGINE
# PHASE 2.3 — CAPITAL PACING & ENTRY GUARDRAILS
# PLATFORM-INDEPENDENT DISPLAY ENGINE (RESTORED)
# =========================================================

import yfinance as yf
import pandas as pd
import json
import os
import sys
from datetime import datetime

# ================= ENVIRONMENT DETECTION =================
def detect_environment():
    if "ipykernel" in sys.modules:
        return "notebook"
    if os.environ.get("VSCODE_PID"):
        return "notebook"
    return "terminal"

ENV = detect_environment()

# ================= USER SETTINGS =================
TOTAL_CAPITAL = 10000
MIN_HOLD_YEARS = 3
IDEAL_HOLD_YEARS = 7
WATCHLIST_FILE = "watchlist.json"

# ================= PHASE 2.1 — MEMORY ENGINE =================
MEMORY_DIR = "memory"

def ensure_memory_dir():
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)

def memory_path(stock):
    return os.path.join(MEMORY_DIR, f"{stock}.json")

def load_memory(stock):
    if not os.path.exists(memory_path(stock)):
        return []
    with open(memory_path(stock), "r") as f:
        return json.load(f)

def save_memory(stock, data):
    with open(memory_path(stock), "w") as f:
        json.dump(data, f, indent=4)

def write_memory(snapshot):
    ensure_memory_dir()
    history = load_memory(snapshot["Stock"])
    history.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "confidence": snapshot["Confidence Score"]
    })
    save_memory(snapshot["Stock"], history)

# ================= PHASE 2.2 — CONFIDENCE TREND =================
def confidence_trend(stock, lookback=7):
    mem = load_memory(stock)[-lookback:]
    if len(mem) < 2:
        return {"trend": "FLAT", "delta": 0, "decay": False, "stagnant": True}

    scores = [m["confidence"] for m in mem]
    delta = scores[-1] - scores[0]

    trend = "IMPROVING" if delta >= 5 else "WEAKENING" if delta <= -5 else "FLAT"
    decay = scores[-1] >= 60 and delta < 0
    stagnant = len(set(scores[-3:])) == 1 if len(scores) >= 3 else False

    return {"trend": trend, "delta": delta, "decay": decay, "stagnant": stagnant}

# ================= PHASE 2.3 — ENTRY & CAPITAL =================
def entry_guardrails(df):
    latest = df.iloc[-1]
    if latest["RSI"] > 70:
        return "WAIT"
    if latest["Close"] > latest["200DMA"] * 1.12:
        return "WAIT"
    return "BUY NOW"

def capital_pacing(score, trend):
    if trend["decay"] or trend["stagnant"]:
        return "PAUSED", 0.0
    if trend["trend"] == "IMPROVING":
        return "FAST", 0.50
    if trend["trend"] == "FLAT":
        return "SLOW", 0.25
    return "PAUSED", 0.0

# ================= WATCHLIST =================
def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE):
        return []
    with open(WATCHLIST_FILE, "r") as f:
        return [s.replace(".NS", "") for s in json.load(f).get("stocks", [])]

def save_watchlist(stocks):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump({"stocks": stocks}, f, indent=4)

def add_to_watchlist(stock):
    wl = load_watchlist()
    if stock not in wl:
        wl.append(stock)
        save_watchlist(wl)

def remove_from_watchlist(stock):
    wl = load_watchlist()
    if stock in wl:
        wl.remove(stock)
        save_watchlist(wl)

# ================= DATA =================
def fetch_data(stock):
    t = yf.Ticker(stock + ".NS")
    return t.history(period="10y"), t.info

# ================= INDICATORS =================
def compute_indicators(df):
    df["200DMA"] = df["Close"].rolling(200).mean()
    df["20WMA"] = df["Close"].rolling(100).mean()
    df["10MMA"] = df["Close"].rolling(210).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Volatility"] = df["Close"].pct_change().rolling(30).std()
    return df

# ================= BUSINESS CLASSIFICATION =================
def classify_business(info):
    industry = (info.get("industry") or "").lower()
    summary = (info.get("longBusinessSummary") or "").lower()

    if "government" in summary:
        return "PSU"
    if "bank" in industry:
        return "BANKING"
    if any(x in industry for x in ["oil", "coal", "mining"]):
        return "COMMODITY"
    if "fmcg" in industry:
        return "FMCG"
    return "GENERAL"

# ================= SIGNALS =================
def extract_signals(df, info):
    latest = df.iloc[-1]
    pe = info.get("trailingPE")
    div = info.get("dividendYield") or 0
    if div > 1:
        div /= 100

    return {
        "price_above_200dma": latest["Close"] > latest["200DMA"],
        "rsi_ok": 45 <= latest["RSI"] <= 65,
        "pe_ok": pe is not None and pe < 20,
        "dividend_ok": div >= 0.025,
        "low_volatility": latest["Volatility"] < 0.025,
        "weekly_trend": "UP" if latest["Close"] > latest["20WMA"] else "DOWN",
        "monthly_trend": "UP" if latest["Close"] > latest["10MMA"] else "DOWN"
    }, pe, div

# ================= CONFIDENCE =================
def confidence_engine(s):
    return min(
        (20 if s["price_above_200dma"] else 0) +
        (15 if s["rsi_ok"] else 0) +
        (15 if s["pe_ok"] else 0) +
        (20 if s["dividend_ok"] else 0) +
        (15 if s["low_volatility"] else 0),
        100
    )

# ================= DECISION =================
def decision_engine(score, w, m):
    if score >= 75 and w == "UP" and m == "UP":
        return "ACCUMULATE"
    if score >= 55:
        return "ACCUMULATE_SLOWLY"
    if score >= 40:
        return "HOLD"
    return "AVOID"

def position_size(score):
    return TOTAL_CAPITAL * (
        0.30 if score >= 75 else
        0.20 if score >= 55 else
        0.10 if score >= 40 else 0
    )

# ================= ANALYSIS CORE =================
def analyze_single_stock(stock):
    df, info = fetch_data(stock)
    df = compute_indicators(df)

    sector = classify_business(info)
    signals, pe, div = extract_signals(df, info)
    score = confidence_engine(signals)
    trend = confidence_trend(stock)

    entry = entry_guardrails(df)
    pace, frac = capital_pacing(score, trend)
    if pace == "PAUSED":
        entry = "WAIT"

    max_buy = position_size(score)
    latest = df.iloc[-1]

    return {
        "Stock": stock,
        "Sector": sector,
        "Price": latest["Close"],
        "Dividend Yield %": div * 100,
        "P/E": pe,
        "Weekly Trend": signals["weekly_trend"],
        "Monthly Trend": signals["monthly_trend"],
        "1Y Return %": ((df["Close"].iloc[-1] / df["Close"].iloc[-252]) - 1) * 100,
        "Seasonal Bias": "POSITIVE" if df["Close"].pct_change().mean() > 0 else "NEGATIVE",
        "Confidence Score": score,
        "Confidence Trend": trend["trend"],
        "Confidence Δ": trend["delta"],
        "Decay Warning": "YES" if trend["decay"] else "NO",
        "Stagnation": "YES" if trend["stagnant"] else "NO",
        "Action": decision_engine(score, signals["weekly_trend"], signals["monthly_trend"]),
        "Entry Timing": entry,
        "Capital Pace": pace,
        "Deploy Now (₹)": max_buy * frac,
        "Remaining Allocation (₹)": max_buy * (1 - frac),
        "Max Buy (₹)": max_buy,
        "Holding Period": f"{MIN_HOLD_YEARS}–{IDEAL_HOLD_YEARS} yrs"
    }

# ================= DISPLAY ENGINE =================
def show_table(df, title):
    if ENV == "notebook":
        from IPython.display import display, HTML
        display(HTML(f"<h3>{title}</h3>"))
        display(df)
    else:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        console = Console()
        console.print(f"\n[bold cyan]{title}[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold magenta")
        for c in df.columns:
            table.add_column(c, justify="center")

        for _, row in df.iterrows():
            styled = []
            for val, col in zip(row.values, df.columns):
                if col == "Confidence Score":
                    styled.append(Text(str(val),
                        style="bold green" if val >= 75 else
                              "bold yellow" if val >= 55 else
                              "bold red"))
                elif col == "Action":
                    styled.append(Text(val,
                        style="bold green" if "ACCUMULATE" in val else
                              "yellow" if val == "HOLD" else
                              "bold red"))
                else:
                    styled.append(Text(str(val)))
            table.add_row(*styled)

        console.print(table)

# ================= MAIN =================
def main():
    cols = [
        "Stock","Sector","Price","Dividend Yield %","P/E",
        "Weekly Trend","Monthly Trend","1Y Return %","Seasonal Bias",
        "Confidence Score","Confidence Trend","Confidence Δ",
        "Decay Warning","Stagnation","Action",
        "Entry Timing","Capital Pace","Deploy Now (₹)",
        "Remaining Allocation (₹)","Max Buy (₹)","Holding Period"
    ]

    wl = load_watchlist()
    if wl:
        res = [analyze_single_stock(s) for s in wl]
        for r in res:
            write_memory(r)
        show_table(pd.DataFrame(res)[cols], "📈 WATCHLIST ANALYSIS")

    search = input("\n🔍 Enter stock ticker to analyze (or press Enter to skip): ").strip().upper()
    if search:
        clean = search.replace(".NS", "")
        r = analyze_single_stock(clean)
        write_memory(r)
        show_table(pd.DataFrame([r])[cols], "🔎 STOCK ANALYSIS")

        wl = load_watchlist()
        if clean in wl:
            if input("🗑 Remove from watchlist? (y/n): ").lower() == "y":
                remove_from_watchlist(clean)
        else:
            if input("➕ Add to watchlist? (y/n): ").lower() == "y":
                add_to_watchlist(clean)

if __name__ == "__main__":
    main()

