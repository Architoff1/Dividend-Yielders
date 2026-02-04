# =========================================================
# DIVIDEND + VALUE + MULTI-TIMEFRAME ENGINE
# PHASE 1 — STRUCTURAL FOUNDATION (DECISION-GRADE)
# PHASE 2.1 — MEMORY ENGINE (INTELLIGENCE LAYER)
# PHASE 2.2 — CONFIDENCE TREND & DECAY ENGINE
# PHASE 2.3 — CAPITAL PACING & ENTRY GUARDRAILS
# PLATFORM-INDEPENDENT DISPLAY ENGINE
# =========================================================

import yfinance as yf
import pandas as pd
import json
import os
import sys
from datetime import datetime

pd.options.display.float_format = "{:,.2f}".format

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
    path = memory_path(stock)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_memory(stock, data):
    with open(memory_path(stock), "w") as f:
        json.dump(data, f, indent=4)

def write_memory(snapshot):
    ensure_memory_dir()
    stock = snapshot["Stock"]
    history = load_memory(stock)

    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "price": float(snapshot["Price"]),
        "confidence": snapshot["Confidence Score"],
        "action": snapshot["Action"],
        "weekly_trend": snapshot["Weekly Trend"],
        "monthly_trend": snapshot["Monthly Trend"],
        "dividend_yield": snapshot["Dividend Yield %"],
        "pe": snapshot["P/E"],
        "volatility": snapshot["Confidence Breakdown"].get("volatility", 0) / 15,
        "sector": snapshot["Sector"]
    }

    history.append(entry)
    save_memory(stock, history)

# ================= PHASE 2.2 — CONFIDENCE TREND ENGINE =================
def get_recent_memory(stock, lookback=7):
    history = load_memory(stock)
    if not history:
        return []
    return history[-lookback:]

def confidence_trend(stock, lookback=7):
    mem = get_recent_memory(stock, lookback)

    if len(mem) < 2:
        return {"trend": "INSUFFICIENT_DATA", "delta": 0, "decay": False, "stagnant": False}

    confidences = [m["confidence"] for m in mem if m.get("confidence") is not None]
    if len(confidences) < 2:
        return {"trend": "INSUFFICIENT_DATA", "delta": 0, "decay": False, "stagnant": False}

    delta = confidences[-1] - confidences[0]

    if delta >= 5:
        trend = "IMPROVING"
    elif delta <= -5:
        trend = "WEAKENING"
    else:
        trend = "FLAT"

    decay = confidences[-1] >= 60 and delta < 0
    stagnant = len(set(confidences[-3:])) == 1 if len(confidences) >= 3 else False

    return {"trend": trend, "delta": delta, "decay": decay, "stagnant": stagnant}

# ================= PHASE 2.3 — CAPITAL PACING & ENTRY GUARDRAILS =================
def entry_guardrails(df):
    latest = df.iloc[-1]
    if latest["RSI"] > 70:
        return "WAIT"
    if latest["Close"] > latest["200DMA"] * 1.12:
        return "WAIT"
    return "BUY NOW"

def capital_pacing(conf_score, trend_meta):
    if trend_meta["trend"] == "IMPROVING":
        pace, frac = "FAST", 0.50
    elif trend_meta["trend"] == "FLAT":
        pace, frac = "SLOW", 0.25
    else:
        pace, frac = "PAUSED", 0.0

    if trend_meta["decay"] or trend_meta["stagnant"]:
        pace, frac = "PAUSED", 0.0

    return pace, frac

# ================= WATCHLIST =================
def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE):
        return []
    with open(WATCHLIST_FILE, "r") as f:
        return json.load(f).get("stocks", [])

def save_watchlist(stocks):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump({"stocks": stocks}, f, indent=4)

def add_to_watchlist(ticker):
    stocks = load_watchlist()
    if ticker not in stocks:
        stocks.append(ticker)
        save_watchlist(stocks)

def remove_from_watchlist(ticker):
    stocks = load_watchlist()
    if ticker in stocks:
        stocks.remove(ticker)
        save_watchlist(stocks)

# ================= DATA =================
def fetch_data(ticker, period="10y"):
    stock = yf.Ticker(ticker)
    return stock.history(period=period), stock.info

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

# ================= DIVIDEND NORMALIZATION =================
def normalize_dividend_yield(raw, price):
    if raw is None or raw <= 0:
        return 0.0
    if raw > 5:
        inferred = raw / price
        return inferred if inferred < 0.08 else 0.0
    return raw if raw < 1 else raw / 100

# ================= BUSINESS CLASSIFICATION =================
def classify_business(info):
    industry = info.get("industry", "").lower()
    summary = info.get("longBusinessSummary", "").lower()

    if "reliance" in summary or "diversified" in industry:
        return "CONGLOMERATE"
    if "government" in summary:
        return "PSU"
    if any(x in industry for x in ["oil", "coal", "mining"]):
        return "COMMODITY"
    if "bank" in industry:
        return "BANKING"
    if "fmcg" in industry:
        return "FMCG"
    return "GENERAL"

# ================= TREND ENGINE =================
def trend_state(price, ma):
    if price > ma * 1.02:
        return "UP"
    if price < ma * 0.98:
        return "DOWN"
    return "FLAT"

def yearly_performance(df):
    return ((df["Close"].iloc[-1] / df["Close"].iloc[-252]) - 1) * 100

def seasonal_bias(df):
    month = datetime.now().month
    df["Month"] = df.index.month
    past = df[df["Month"] == month].tail(10)
    return "POSITIVE" if past["Close"].pct_change().mean() > 0 else "NEGATIVE"

# ================= SIGNAL EXTRACTION =================
def extract_signals(df, info):
    latest = df.iloc[-1]
    pe = info.get("trailingPE", None)
    div = normalize_dividend_yield(info.get("dividendYield", 0), latest["Close"])

    signals = {
        "price_above_200dma": latest["Close"] > latest["200DMA"],
        "rsi_ok": 45 <= latest["RSI"] <= 65,
        "pe_ok": pe is not None and pe < 20,
        "dividend_ok": div >= 0.025,
        "low_volatility": latest["Volatility"] < 0.025,
        "weekly_trend": trend_state(latest["Close"], latest["20WMA"]),
        "monthly_trend": trend_state(latest["Close"], latest["10MMA"])
    }

    return signals, pe, div

# ================= CONFIDENCE ENGINE =================
def confidence_engine(signals):
    components = {
        "trend": 20 if signals["price_above_200dma"] else 0,
        "rsi": 15 if signals["rsi_ok"] else 0,
        "valuation": 15 if signals["pe_ok"] else 0,
        "dividend": 20 if signals["dividend_ok"] else 0,
        "volatility": 15 if signals["low_volatility"] else 0
    }

    return {"score": max(min(sum(components.values()), 100), 0), "components": components}

# ================= DECISION ENGINE =================
def decision_engine(conf_score, weekly, monthly):
    if conf_score >= 75 and weekly == "UP" and monthly != "DOWN":
        return "ACCUMULATE"
    if conf_score >= 55 and weekly != "DOWN":
        return "ACCUMULATE_SLOWLY"
    if conf_score >= 40:
        return "HOLD"
    return "AVOID"

def position_size(conf):
    return TOTAL_CAPITAL * (
        0.30 if conf >= 75 else
        0.20 if conf >= 55 else
        0.10 if conf >= 40 else 0
    )

# ================= ANALYSIS CORE =================
def analyze_single_stock(ticker):
    df, info = fetch_data(ticker)
    df = compute_indicators(df)

    sector = classify_business(info)
    signals, pe, div = extract_signals(df, info)
    conf = confidence_engine(signals)

    trend_meta = confidence_trend(ticker.replace(".NS", ""))
    entry = entry_guardrails(df)
    pace, frac = capital_pacing(conf["score"], trend_meta)

    max_buy = position_size(conf["score"])
    deploy_now = round(max_buy * frac, 2)
    remaining = round(max_buy - deploy_now, 2)

    action = decision_engine(conf["score"], signals["weekly_trend"], signals["monthly_trend"])
    latest = df.iloc[-1]

    return {
        "Stock": ticker.replace(".NS", ""),
        "Sector": sector,
        "Price": latest["Close"],
        "Dividend Yield %": div * 100,
        "P/E": pe,
        "Weekly Trend": signals["weekly_trend"],
        "Monthly Trend": signals["monthly_trend"],
        "1Y Return %": yearly_performance(df),
        "Seasonal Bias": seasonal_bias(df),
        "Confidence Score": conf["score"],
        "Confidence Trend": trend_meta["trend"],
        "Confidence Δ": trend_meta["delta"],
        "Decay Warning": "YES" if trend_meta["decay"] else "NO",
        "Stagnation": "YES" if trend_meta["stagnant"] else "NO",
        "Action": action,
        "Entry Timing": entry,
        "Capital Pace": pace,
        "Deploy Now (₹)": deploy_now,
        "Remaining Allocation (₹)": remaining,
        "Max Buy (₹)": max_buy,
        "Holding Period": f"{MIN_HOLD_YEARS}–{IDEAL_HOLD_YEARS} yrs",
        "Confidence Breakdown": conf["components"]
    }

# ================= DISPLAY ENGINE =================
def show_table(df, title=None):
    if ENV == "notebook":
        from IPython.display import display, HTML
        if title:
            display(HTML(f"<h3>{title}</h3>"))
        display(df)
    else:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        console = Console()
        if title:
            console.print(f"\n[bold cyan]{title}[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold magenta")
        for col in df.columns:
            table.add_column(col, justify="center")

        for _, row in df.iterrows():
            styled_row = []
            for val, col in zip(row.values, df.columns):
                if col == "Confidence Score":
                    styled_row.append(Text(str(val),
                        style="bold green" if val >= 75 else
                              "bold yellow" if val >= 55 else
                              "bold red"))
                elif col == "Action":
                    styled_row.append(Text(val,
                        style="bold green" if "ACCUMULATE" in val else
                              "yellow" if val == "HOLD" else
                              "bold red"))
                else:
                    styled_row.append(Text(str(val)))
            table.add_row(*styled_row)

        console.print(table)

# ================= MAIN =================
def main():
    cols = [
        "Stock", "Sector", "Price", "Dividend Yield %",
        "P/E", "Weekly Trend", "Monthly Trend",
        "1Y Return %", "Seasonal Bias",
        "Confidence Score", "Confidence Trend", "Confidence Δ",
        "Decay Warning", "Stagnation",
        "Action", "Entry Timing", "Capital Pace",
        "Deploy Now (₹)", "Remaining Allocation (₹)",
        "Max Buy (₹)", "Holding Period"
    ]

    watchlist = load_watchlist()
    if watchlist:
        results = [analyze_single_stock(s) for s in watchlist]
        for r in results:
            write_memory(r)
        df = pd.DataFrame(results)
        show_table(df[cols].sort_values("Confidence Score", ascending=False),
                   title="📈 WATCHLIST ANALYSIS")
    else:
        print("ℹ️ Watchlist empty")

    search = input("\n🔍 Enter stock ticker to analyze (or press Enter to skip): ").strip().upper()
    if search:
        stock_data = analyze_single_stock(search)
        write_memory(stock_data)
        sdf = pd.DataFrame([stock_data])
        show_table(sdf[cols], title="🔎 STOCK ANALYSIS")

        if search in watchlist:
            if input("Remove from watchlist? (y/n): ").lower() == "y":
                remove_from_watchlist(search)
        else:
            if input("Add to watchlist? (y/n): ").lower() == "y":
                add_to_watchlist(search)

if __name__ == "__main__":
    main()

