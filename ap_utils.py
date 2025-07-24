import requests, json, csv, os, hmac, hashlib, time
from datetime import datetime
import numpy as np
import pandas as pd

FAV_FILE = "ap_favorites.json"
HISTORY_FILE = "ap_history.csv"
ALERT_FILE = "ap_alerts.json"

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BASE_URL = "https://api.binance.com"

# ====================================================
# ✅ Binance Spot Order Fonksiyonları
# ====================================================
def sign_params(params):
    query = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return f"{query}&signature={signature}"

def place_order(symbol, side, quantity):
    url = f"{BASE_URL}/api/v3/order"
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": int(time.time() * 1000)
    }
    headers = {"X-MBX-APIKEY": API_KEY}
    signed = sign_params(params)
    res = requests.post(f"{url}?{signed}", headers=headers)
    return res.json()

def get_balance(asset):
    url = f"{BASE_URL}/api/v3/account"
    headers = {"X-MBX-APIKEY": API_KEY}
    res = requests.get(url, headers=headers, params={"timestamp": int(time.time() * 1000)})
    data = res.json()
    for b in data.get("balances", []):
        if b["asset"] == asset:
            return float(b["free"])
    return 0.0

def get_price(symbol="BTCUSDT"):
    url = f"{BASE_URL}/api/v3/ticker/price?symbol={symbol}"
    return float(requests.get(url).json()["price"])

def auto_trade(symbol="BTCUSDT", signal="BEKLE"):
    if not API_KEY or not API_SECRET:
        return {"status": "NO_API_KEY"}
    if signal == "ALIM":
        usdt = get_balance("USDT")
        qty = round((usdt * 0.1) / get_price(symbol), 6)
        if qty > 0:
            return place_order(symbol, "BUY", qty)
    elif signal == "SATIŞ":
        base_asset = symbol.replace("USDT", "")
        bal = get_balance(base_asset)
        qty = round(bal * 0.1, 6)
        if qty > 0:
            return place_order(symbol, "SELL", qty)
    return {"status": "NO_TRADE"}

# ====================================================
# ✅ VWAP Skor Hesaplama ( /ap için )
# ====================================================
def get_24h_tickers():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    return requests.get(url).json()

def calculate_scores():
    data = get_24h_tickers()
    alt_vwap_sum, alt_vol_sum = 0, 0
    alt_vs_btc_sum, alt_vs_btc_vol = 0, 0
    long_term_volumes = []
    btc_vwap = 0

    for coin in data:
        sym = coin['symbol']
        if not sym.endswith("USDT") or sym in ["BUSDUSDT","USDCUSDT","FDUSDUSDT"]:
            continue
        price_change = float(coin['priceChangePercent'])
        volume = float(coin['quoteVolume'])

        if sym == "BTCUSDT":
            btc_vwap = price_change
            continue

        alt_vwap_sum += price_change * volume
        alt_vol_sum += volume
        if price_change > btc_vwap:
            alt_vs_btc_sum += (price_change - btc_vwap) * volume
            alt_vs_btc_vol += volume

        long_term_volumes.append(volume)

    alt_vwap = alt_vwap_sum / alt_vol_sum if alt_vol_sum else 0
    alt_vs_btc_vwap = alt_vs_btc_sum / alt_vs_btc_vol if alt_vs_btc_vol else 0
    long_term_score = sum(long_term_volumes) / len(long_term_volumes) / 1_000_000 if long_term_volumes else 0

    return (round(min(100, max(0, alt_vs_btc_vwap + 50)), 1),
            round(min(100, max(0, alt_vwap + 50)), 1),
            round(min(100, max(0, long_term_score)), 1))

def save_daily_history():
    vsbtc, alt, longt = calculate_scores()
    date = datetime.now().strftime("%Y-%m-%d")
    rows = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            rows = list(csv.reader(f))
    rows.append([date, vsbtc, alt, longt])
    if len(rows) > 60:
        rows = rows[-60:]
    with open(HISTORY_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return vsbtc, alt, longt

def compare_with_history(days=1):
    if not os.path.exists(HISTORY_FILE):
        return calculate_scores(), [0,0,0], ["➡️","➡️","➡️"]
    with open(HISTORY_FILE, "r") as f:
        rows = list(csv.reader(f))
    if len(rows) < days:
        return calculate_scores(), [0,0,0], ["➡️","➡️","➡️"]

    now = calculate_scores()
    base = [float(rows[-1][1]), float(rows[-1][2]), float(rows[-1][3])] if days == 1 else \
           [sum(float(r[i]) for r in rows[-days:]) / days for i in range(1,4)]

    diff = [round(((n - b) / b) * 100, 1) if b else 0 for n, b in zip(now, base)]
    arrows = ["🔼" if d > 0 else "🔻" if d < 0 else "➡️" for d in diff]
    return now, diff, arrows

# ====================================================
# ✅ CVD + AutoTrade ( /ap )
# ====================================================
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

def get_klines(symbol="BTCUSDT", interval="1h", limit=100):
    url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    return [float(x[4]) for x in data], data

def calculate_cvd(kline_data):
    cvd = 0
    for k in kline_data:
        open_p, close_p = float(k[1]), float(k[4])
        vol = float(k[5])
        if close_p > open_p:
            cvd += vol
        else:
            cvd -= vol
    return round(cvd / 1_000_000, 1)

def get_autotrade_signal(score, cvd):
    if score > 65 and cvd > 0:
        return "ALIM"
    elif (100 - score) > 65 and cvd < 0:
        return "SATIŞ"
    else:
        return "BEKLE"

def ap_command(period="24h", days=None):
    now, diff, arr = compare_with_history(days if days else 1)
    closes, raw = get_klines("BTCUSDT")
    cvd = calculate_cvd(raw)
    avg_score = (now[0] + now[1] + now[2]) / 3
    auto_signal = get_autotrade_signal(avg_score, cvd)
    trade_result = auto_trade("BTCUSDT", auto_signal)

    return (f"Ap({period}) raporu\n"
            f"Altların Kısa Vadede Btc'ye Karşı Gücü(0-100): {now[0]} {arr[0]}%{abs(diff[0])}\n"
            f"Altların Kısa Vadede Gücü(0-100): {now[1]} {arr[1]}%{abs(diff[1])}\n"
            f"Coinlerin Uzun Vadede Gücü(0-100): {now[2]} {arr[2]}%{abs(diff[2])}\n"
            f"Balina Para Girişi (CVD): {cvd}M → {'Pozitif' if cvd>0 else 'Negatif'}\n"
            f"AutoTrade Sinyali: {auto_signal}\n"
            f"Emir Sonucu: {trade_result}")

# ====================================================
# ✅ RSI, MACD, EMA, ADX Hesaplamaları (/trend)
# ====================================================
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - (100. / (1. + rs))
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        upval = max(delta, 0)
        downval = -min(delta, 0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - (100. / (1. + rs))
    return round(rsi[-1], 1)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return "Yükseliş" if macd.iloc[-1] > signal_line.iloc[-1] else "Düşüş"

def calculate_ema_trend(prices, period=50):
    ema = pd.Series(prices).ewm(span=period, adjust=False).mean()
    return "Yükseliş" if prices[-1] > ema.iloc[-1] else "Düşüş", period

def calculate_adx(high, low, close, period=14):
    df = pd.DataFrame({"High": high, "Low": low, "Close": close})
    df["TR"] = (df["High"] - df["Low"]).abs()
    df["+DM"] = df["High"].diff()
    df["-DM"] = df["Low"].diff().abs()
    df["+DM"] = np.where((df["+DM"] > df["-DM"]) & (df["+DM"] > 0), df["+DM"], 0.0)
    df["-DM"] = np.where((df["-DM"] > df["+DM"]) & (df["-DM"] > 0), df["-DM"], 0.0)
    tr14 = df["TR"].rolling(window=period).sum()
    plus_dm14 = df["+DM"].rolling(window=period).sum()
    minus_dm14 = df["-DM"].rolling(window=period).sum()
    plus_di14 = 100 * (plus_dm14 / tr14)
    minus_di14 = 100 * (minus_dm14 / tr14)
    dx = (abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)) * 100
    adx = dx.rolling(window=period).mean()
    return round(adx.iloc[-1], 1)

def calculate_trend_score(rsi, macd, ema_trend, adx):
    score = 50
    score += 10 if rsi < 30 else -10 if rsi > 70 else 0
    score += 15 if macd == "Yükseliş" else -15
    score += 15 if ema_trend == "Yükseliş" else -15
    score += 10 if adx > 25 else -5
    return min(100, max(0, score))

def rsi_macd_command(coins):
    msg = "📊 RSI & MACD & EMA-ADX Trend Analizi\n"
    for coin in coins:
        symbol = coin.upper() + "USDT"
        closes, raw = get_klines(symbol, "1h", 100)
        high = [float(x[2]) for x in raw]
        low = [float(x[3]) for x in raw]
        close = [float(x[4]) for x in raw]
        rsi = calculate_rsi(closes)
        macd = calculate_macd(closes)
        ema_trend, ema_period = calculate_ema_trend(closes)
        adx = calculate_adx(high, low, close)
        score = calculate_trend_score(rsi, macd, ema_trend, adx)
        signal = get_autotrade_signal(score, calculate_cvd(raw))
        msg += (f"\n{coin.upper()}:\n"
                f"RSI={rsi} | MACD={macd} | EMA Trend: {ema_trend}({ema_period}) | "
                f"ADX: {adx} | alış:%{score} (sat:%{100-score}) | {signal}")
    return msg

# ====================================================
# ✅ P Komutu
# ====================================================
def p_command(coins):
    data = get_24h_tickers()
    lookup = {d['symbol']: d for d in data}
    msg = ""
    for c in coins:
        s = (c.upper() + "USDT")
        if s not in lookup:
            msg += f"{c.upper()}: bulunamadı\n"
            continue
        d = lookup[s]
        price = float(d['lastPrice'])
        change = float(d['priceChangePercent'])
        vol = float(d['quoteVolume']) / 1_000_000
        pf = f"{price:.2f}" if price >= 1 else f"{price:.8f}"
        arrow = "🔼" if change > 0 else "🔻" if change < 0 else "➡️"
        msg += f"{c.upper()}: {pf} {arrow}{change}% (Vol: {vol:.1f}M$)\n"
    return msg

# ====================================================
# ✅ Favoriler & Alert
# ====================================================
def load_favorites():
    if os.path.exists(FAV_FILE):
        with open(FAV_FILE, "r") as f:
            return json.load(f)
    return {}

def save_favorites(data):
    with open(FAV_FILE, "w") as f:
        json.dump(data, f)

def add_favorite(fav, coins):
    favs = load_favorites()
    favs[fav] = coins
    save_favorites(favs)
    return f"{fav} güncellendi: {' '.join(coins)}"

def delete_favorite(fav):
    favs = load_favorites()
    if fav in favs:
        del favs[fav]
        save_favorites(favs)
        return f"{fav} silindi"
    return "Favori bulunamadı"

def set_alert_threshold(level1, level2):
    alert = {"level1": float(level1), "level2": float(level2)}
    with open(ALERT_FILE, "w") as f:
        json.dump(alert, f)
    return f"Alertler güncellendi: {level1}, {level2}"
