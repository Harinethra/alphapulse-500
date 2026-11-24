import os
import json
import math
import time
import requests
import yfinance as yf
import numpy as np
from datetime import datetime
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE = "live_data.json"

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=60000)
) if GEMINI_API_KEY else None

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash"
]

HIGH_IMPACT_KEYWORDS = [
    "order", "contract", "award", "letter of intent", "loi", "acquisition",
    "amalgamation", "demerger", "buyback", "capacity expansion", "commercial production",
    "usfda", "clearance", "resignation", "auditor", "cfo", "default",
    "credit rating", "sebi", "joint venture", "mou", "preferential", "financial results",
    "outcome", "board meeting", "dividend", "press release", "investor presentation"
]

def is_valid_num(val):
    return val is not None and not math.isnan(float(val)) and not math.isinf(float(val))

def fetch_nse_announcements():
    """NSE India வெப்சைட்டில் இருந்து Live அறிவிப்புகளை எடுக்கும்"""
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
    }
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        res = session.get("https://www.nseindia.com/api/corporate-announcements?index=equities", headers=headers, timeout=15)
        res.raise_for_status()
        return res.json()[:25]
    except Exception as e:
        print(f"NSE Fetch Error: {e}")
        return []

def get_stock_metrics(symbol):
    """Yahoo Finance-ல் இருந்து Live ஷேர் விலை மற்றும் அடிப்படை விவரங்களை எடுக்கும்"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="1y")
        info = ticker.info
        if hist.empty or len(hist) < 30:
            return None

        close = hist["Close"].dropna()
        volume = hist["Volume"].dropna()
        if len(close) < 15 or len(volume) < 21:
            return None

        current_price = float(close.iloc[-1])
        if not is_valid_num(current_price) or current_price <= 0:
            return None

        sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else current_price
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma_50

        avg_vol_20 = float(volume.iloc[-21:-1].mean())
        vol_ratio = round(float(volume.iloc[-1]) / avg_vol_20, 2) if (is_valid_num(avg_vol_20) and avg_vol_20 > 0) else 1.0

        price_10d_ago = float(close.iloc[-11]) if len(close) >= 11 else current_price
        two_week_return_pct = round(((current_price - price_10d_ago) / price_10d_ago) * 100, 1) if price_10d_ago > 0 else 0.0
        is_overextended = two_week_return_pct > 20.0

        high_low = (hist["High"] - hist["Low"]).dropna()
        atr_14 = float(high_low.rolling(14).mean().iloc[-1]) if len(high_low) >= 14 else (current_price * 0.03)
        if not is_valid_num(atr_14):
            atr_14 = current_price * 0.03

        return {
            "symbol": symbol,
            "cmp": round(current_price, 2),
            "target": round(current_price * 1.12, 2),
            "sl": round(current_price - (1.8 * atr_14), 2),
            "above_200dma": bool(current_price > sma_200),
            "vol_ratio": vol_ratio if is_valid_num(vol_ratio) else 1.0,
            "two_week_return_pct": two_week_return_pct if is_valid_num(two_week_return_pct) else 0.0,
            "is_overextended": bool(is_overextended),
            "revenue_cr": round((info.get("totalRevenue", 0) or 0) / 1e7, 1),
            "mcap_cr": round((info.get("marketCap", 0) or 0) / 1e7, 1),
            "roe": round((info.get("returnOnEquity", 0) or 0) * 100, 1),
            "de": round((info.get("debtToEquity", 0) or 0) / 100, 2),
            "pe": round(info.get("trailingPE", 0) or 0, 1),
            "sector": info.get("sector", "Nifty 500")
        }
    except Exception as e:
        print(f"Metrics error for {symbol}: {e}")
        return None

def analyze_with_gemini(filing, metrics):
    """Fallback மாடல்களுடன் தமிழ் மற்றும் ஆங்கிலத்தில் செய்தியின் தாக்கத்தை அலசும்"""
    if not client:
        print("GEMINI_API_KEY missing!")
        return None

    prompt = f"""
    Analyze this NSE exchange announcement as an Institutional Research Analyst:
    Company: {filing.get('sm_name')} ({metrics['symbol']})
    Annual Revenue: ₹{metrics['revenue_cr']} Cr | Market Cap: ₹{metrics['mcap_cr']} Cr
    2-Week Price Change: {metrics['two_week_return_pct']}% | Volume Spike: {metrics['vol_ratio']}x
    Headline: {filing.get('desc')}
    Details: {filing.get('attchmntText')}

    Return strictly valid JSON:
    {{
        "catalyst_score_out_of_30": int (0 to 30),
        "sentiment": "BUY" | "WATCH" | "AVOID",
        "news_ta": "எளிய தமிழில் 1-2 வரியில் என்ன செய்தி மற்றும் கம்பெனி வருமானத்தில் இது எத்தனை சதவீதம்",
        "news_en": "Crisp 1-2 lines in English on catalyst & revenue materiality",
        "buy_check_ta": "இப்போ வாங்கலாமா? (Priced-in செக் மற்றும் வால்யூம் நிலவரம் தமிழில்)",
        "buy_check_en": "Is it priced-in? Volume & setup check in English",
        "domino_ta": "இதனால் பயன்பெறும் மற்ற செக்டார்/பங்குகள் (தமிழில்)",
        "domino_en": "Second-order beneficiary stocks/sectors in English"
    }}
    """

    for model_name in GEMINI_MODELS:
        for attempt in range(2):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                if res and res.text:
                    print(f"Success using {model_name} for {metrics['symbol']}")
                    return json.loads(res.text)
            except Exception as e:
                print(f"[{model_name}] Attempt {attempt + 1} failed for {metrics['symbol']}: {e}")
                time.sleep(2)

    return None

def escape_md(text):
    if not text:
        return ""
    for ch in ["_", "*", "`", "["]:
        text = str(text).replace(ch, "")
    return text

def send_telegram_alert(card):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    msg = (
        f"🚨 *LIVE NIFTY 500 CALL: {escape_md(card['symbol'])}*\n"
        f"🏆 *Score:* `{card['score']}/100` | *Verdict:* *{escape_md(card['verdict'])}*\n\n"
        f"📰 *என்ன செய்தி?*\n{escape_md(card['news_ta'])}\n\n"
        f"💡 *இப்போ வாங்கலாமா?*\n{escape_md(card['buy_check_ta'])}\n\n"
        f"🎯 *Price Levels:*\n"
        f"• *CMP:* ₹{card['cmp']} | *Target:* ₹{card['target']} | *SL:* ₹{card['sl']}\n"
        f"🔗 [NSE Official PDF]({card['pdf_url']})"
    )
    try:
        res = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=15
        )
        print(f"Alert sent for {card['symbol']} - Telegram Status: {res.status_code}")
    except Exception as e:
        print(f"Telegram send error for {card['symbol']}: {e}")

def sanitize_cards(cards):
    """ஏற்கனவே உள்ள NaN மதிப்புகளை சுத்தம் செய்து Valid JSON ஆக மாற்றும்"""
    clean = []
    for c in cards:
        cmp_val = c.get("cmp")
        if not is_valid_num(cmp_val):
            continue
        clean.append(c)
    return clean

def main():
    existing_cards = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                existing_cards = sanitize_cards(json.load(f))
        except Exception:
            existing_cards = []

    seen_ids = {str(c.get("id")) for c in existing_cards}
    raw_filings = fetch_nse_announcements()

    new_count = 0
    for item in raw_filings:
        seq_id = str(item.get("seq_id", ""))
        symbol = item.get("symbol", "")
        text = f"{item.get('desc', '')} {item.get('attchmntText', '')}".lower()

        if not seq_id or seq_id in seen_ids:
            continue
        if not any(kw in text for kw in HIGH_IMPACT_KEYWORDS):
            continue

        metrics = get_stock_metrics(symbol)
        if not metrics:
            continue

        ai_data = analyze_with_gemini(item, metrics)
        if not ai_data:
            continue

        cat_pts = min(max(int(ai_data.get("catalyst_score_out_of_30", 15)), 0), 30)
        tech_pts = (10 if metrics["above_200dma"] else 3) + (15 if metrics["vol_ratio"] >= 2.0 else 7) - (8 if metrics["is_overextended"] else 0)
        fund_pts = (15 if metrics["roe"] >= 15 else 8) + (10 if metrics["de"] <= 0.6 else 4)
        smart_pts = 18 if ai_data.get("sentiment") == "BUY" else 8
        total_score = min(max(cat_pts + tech_pts + fund_pts + smart_pts, 15), 98)

        card = {
            "id": seq_id,
            "symbol": symbol,
            "name": item.get("sm_name", symbol),
            "sector": metrics["sector"],
            "time": item.get("an_dt", datetime.now().strftime("%d-%b-%Y %H:%M")),
            "score": total_score,
            "verdict": ai_data.get("sentiment", "WATCH"),
            "cmp": metrics["cmp"],
            "target": metrics["target"],
            "sl": metrics["sl"],
            "vol_spike": f"{metrics['vol_ratio']}x",
            "two_wk_rally": f"{metrics['two_week_return_pct']}%",
            "roe": f"{metrics['roe']}%",
            "de": str(metrics["de"]),
            "pdf_url": item.get("attchmntFile", "https://www.nseindia.com"),
            "news_ta": ai_data.get("news_ta", ""),
            "news_en": ai_data.get("news_en", ""),
            "buy_check_ta": ai_data.get("buy_check_ta", ""),
            "buy_check_en": ai_data.get("buy_check_en", ""),
            "domino_ta": ai_data.get("domino_ta", ""),
            "domino_en": ai_data.get("domino_en", "")
        }

        existing_cards.insert(0, card)
        seen_ids.add(seq_id)
        new_count += 1

        send_telegram_alert(card)

        if new_count >= 4:
            break

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_cards[:30], f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Added {new_count} new live calls.")

if __name__ == "__main__":
    main()
