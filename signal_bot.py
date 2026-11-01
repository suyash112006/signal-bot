"""
Telegram signal bot: XAUUSD + major forex pairs
Strategy stack: HTF trend (EMA50/200) + liquidity sweep + MSS/CHoCH + FVG retest
                + EMA/RSI/ATR/volume/premium-discount confluence scoring.
Rules: min RR 1:2, risk 0.7%, max 3 signals/day, all sessions, high-score setups only.
Signals only. The bot never places trades.
"""
import os, json, time, math, logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------- CONFIG ---------------------------------
TOKEN      = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]
BALANCE    = float(os.getenv("ACCOUNT_BALANCE", "1000"))   # for lot-size calc
RISK_PCT   = 0.7
MIN_RR     = 2.0
MAX_DAILY  = 3
MIN_SCORE  = int(os.getenv("MIN_SCORE", "10"))              # out of 12
DAY_TZ     = os.getenv("DAY_TZ", "Asia/Kolkata")            # when the "day" resets
STATE_FILE = os.getenv("STATE_FILE", "state.json")

# name: (yahoo tickers in priority order, contract size, USD is base currency, decimals)
PAIRS = {
    "XAUUSD": (["XAUUSD=X", "GC=F"], 100,    False, 2),
    "EURUSD": (["EURUSD=X"],         100000, False, 5),
    "GBPUSD": (["GBPUSD=X"],         100000, False, 5),
    "AUDUSD": (["AUDUSD=X"],         100000, False, 5),
    "NZDUSD": (["NZDUSD=X"],         100000, False, 5),
    "USDJPY": (["USDJPY=X"],         100000, True,  3),
    "USDCAD": (["USDCAD=X"],         100000, True,  5),
    "USDCHF": (["USDCHF=X"],         100000, True,  5),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

# ----------------------------- TELEGRAM -------------------------------
def tg(text):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=20,
        )
        if not r.ok:
            log.error("Telegram error: %s", r.text)
    except Exception as e:
        log.error("Telegram send failed: %s", e)

# ----------------------------- STATE ----------------------------------
def today():
    return datetime.now(ZoneInfo(DAY_TZ)).date().isoformat()

def load_state():
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
    except Exception:
        s = {"day": today(), "count": 0, "sent": [], "open": []}
    if s.get("day") != today():
        s["day"], s["count"] = today(), 0
    return s

def save_state(s):
    s["sent"] = s["sent"][-300:]
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)

# ----------------------------- DATA -----------------------------------
def get_candles(tickers, interval, period, minutes):
    for t in tickers:
        try:
            df = yf.download(t, interval=interval, period=period,
                             progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            # keep closed candles only
            df = df[df.index + pd.Timedelta(minutes=minutes) <= pd.Timestamp.now(tz="UTC")]
            if len(df) >= 100:
                return df
        except Exception as e:
            log.warning("data fail %s: %s", t, e)
    return None

# ----------------------------- INDICATORS -----------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def atr(df, n=14):
    tr = pd.concat([df.high - df.low,
                    (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def swings(df, k=3):
    h, l = df.high.values, df.low.values
    hi, lo = [], []
    for i in range(k, len(df) - k):
        if h[i] == max(h[i - k:i + k + 1]): hi.append(i)
        if l[i] == min(l[i - k:i + k + 1]): lo.append(i)
    return hi, lo

def htf_bias(df1h):
    if df1h is None or len(df1h) < 210:
        return None
    c = df1h.close
    e50, e200 = ema(c, 50).iloc[-1], ema(c, 200).iloc[-1]
    if c.iloc[-1] > e200 and e50 > e200: return "bull"
    if c.iloc[-1] < e200 and e50 < e200: return "bear"
    return None

# ----------------------------- STRATEGY -------------------------------
def find_setup(df, bias, side):
    """Sweep -> MSS -> FVG -> retest confirmation. Returns dict or None."""
    L = side == "long"
    if bias != ("bull" if L else "bear"):
        return None
    sign = 1 if L else -1
    d = df.copy()
    d["atr"], d["ema20"], d["ema50"], d["rsi"] = atr(d), ema(d.close, 20), ema(d.close, 50), rsi(d.close)
    has_vol = d.volume.tail(50).sum() > 0                  # forex spot often has no real volume
    d["vol"] = d.volume if has_vol else (d.high - d.low)   # fallback: range as activity proxy
    d["vavg"] = d.vol.rolling(20).mean()

    o, h, l, c = d.open.values, d.high.values, d.low.values, d.close.values
    n = len(d); last = n - 1
    a = d.atr.iloc[last]
    if not a or np.isnan(a): return None
    hi, lo = swings(d)
    piv = lo if L else hi

    # 1) liquidity sweep: wick through a swing level, close back inside
    found = None
    for s in range(last - 1, max(last - 16, 70), -1):
        for p in sorted([p for p in piv if s - 60 < p < s - 3], reverse=True):
            lvl = l[p] if L else h[p]
            swept = (l[s] < lvl and c[s] > lvl) if L else (h[s] > lvl and c[s] < lvl)
            intact = (c[p + 1:s] > lvl).all() if L else (c[p + 1:s] < lvl).all()
            if swept and intact:
                found = (s, p); break
        if found: break
    if not found: return None
    s, p = found

    # 2) market structure shift: close beyond the recent counter-swing
    ref = h[s - 6:s].max() if L else l[s - 6:s].min()
    if not any((c[m] > ref) if L else (c[m] < ref) for m in range(s, n)):
        return None

    # 3) latest unmitigated fair value gap formed after the sweep
    fvg = None
    for i in range(last - 1, s, -1):
        if L and l[i] > h[i - 2]:       bot, top = h[i - 2], l[i]
        elif (not L) and h[i] < l[i - 2]: bot, top = h[i], l[i - 2]
        else: continue
        if top - bot < 0.1 * a: continue
        if L and (c[i + 1:] < bot).any(): continue
        if (not L) and (c[i + 1:] > top).any(): continue
        fvg = (bot, top); break
    if not fvg: return None
    bot, top = fvg

    # 4) retest + confirmation candle on the last closed bar
    ok = (l[last] <= top and c[last] > bot and c[last] > o[last]) if L else \
         (h[last] >= bot and c[last] < top and c[last] < o[last])
    if not ok: return None

    # 5) levels
    entry = c[last]
    ext = l[s:].min() if L else h[s:].max()
    sl = ext - 0.25 * a if L else ext + 0.25 * a
    risk = abs(entry - sl)
    if not (0.5 * a <= risk <= 3 * a): return None
    tp1, tp2 = entry + sign * MIN_RR * risk, entry + sign * 3 * risk

    # room to target: next opposing liquidity must be at least 2R away
    opp = [h[i] if L else l[i] for i in (hi if L else lo)
           if i >= n - 80 and ((h[i] > entry) if L else (l[i] < entry))]
    if opp:
        nearest = min(opp) if L else max(opp)
        if abs(nearest - entry) < MIN_RR * risk: return None

    # 6) scoring (8 required + 4 optional = 12)
    score, notes = 8, ["1H trend aligned", "Liquidity sweep", "MSS / CHoCH", "FVG retest + confirmation candle"]
    e20, e50, r = d.ema20.iloc[last], d.ema50.iloc[last], d.rsi.iloc[last]
    if (e20 > e50 and c[last] > e50) if L else (e20 < e50 and c[last] < e50):
        score += 1; notes.append("EMA 20/50 alignment")
    if (40 <= r <= 65) if L else (35 <= r <= 60):
        score += 1; notes.append(f"RSI healthy ({r:.0f})")
    if (d.vol.iloc[s:last + 1] > 1.2 * d.vavg.iloc[s:last + 1]).any():
        score += 1; notes.append("Volume expansion" + ("" if has_vol else " (range proxy)"))
    mid = (h[-50:].max() + l[-50:].min()) / 2
    if (entry < mid) if L else (entry > mid):
        score += 1; notes.append("Discount zone" if L else "Premium zone")

    return {"side": side, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "score": score,
            "notes": notes, "key": f"{side}|{d.index[s].isoformat()}",
            "time": (d.index[last] + pd.Timedelta(minutes=15)).isoformat()}

# ----------------------------- SIZING / FORMAT ------------------------
def lot_size(pair, entry, sl):
    _, cs, usd_base, _ = PAIRS[pair]
    per_unit = cs / entry if usd_base else cs      # USD P/L per 1.0 price move per 1 lot
    lot = BALANCE * RISK_PCT / 100 / (abs(entry - sl) * per_unit)
    return max(0.01, math.floor(lot * 100) / 100)

def session_name():
    h = datetime.now(timezone.utc).hour
    if 0 <= h < 7: return "Asian"
    if 7 <= h < 12: return "London"
    if 12 <= h < 16: return "London/NY overlap"
    return "New York"

def market_open():
    n = datetime.now(timezone.utc); wd, hr = n.weekday(), n.hour
    if wd == 5: return False
    if wd == 4 and hr >= 21: return False
    if wd == 6 and hr < 22: return False
    if hr == 21: return False          # daily rollover: wide spreads, skip
    return True

def send_signal(pair, st):
    dec = PAIRS[pair][3]; f = lambda x: f"{x:.{dec}f}"
    em = "🟢" if st["side"] == "long" else "🔴"
    lot = lot_size(pair, st["entry"], st["sl"])
    now_local = datetime.now(ZoneInfo(DAY_TZ)).strftime("%d %b %H:%M")
    tg(
        f"{em} <b>{st['side'].upper()} {pair}</b>  |  Score {st['score']}/12\n"
        f"Session: {session_name()}  |  {now_local}\n\n"
        f"Entry: <b>{f(st['entry'])}</b>\n"
        f"SL: <b>{f(st['sl'])}</b>\n"
        f"TP1 (1:{MIN_RR:g}): <b>{f(st['tp1'])}</b>\n"
        f"TP2 (1:3): <b>{f(st['tp2'])}</b>\n\n"
        f"Risk {RISK_PCT}% of ${BALANCE:,.0f} = ${BALANCE * RISK_PCT / 100:.2f}  |  Lot: <b>{lot}</b>\n"
        f"Tip: close part at TP1, move SL to breakeven.\n\n"
        + "\n".join("✅ " + x for x in st["notes"])
    )

# ----------------------------- TRACKING -------------------------------
def check_open(state):
    still = []
    for t in state["open"]:
        cfg = PAIRS[t["pair"]]
        df = get_candles(cfg[0], "15m", "5d", 15)
        if df is None:
            still.append(t); continue
        bars = df[df.index >= pd.Timestamp(t["time"])]
        res = None
        for _, r in bars.iterrows():
            if t["side"] == "long":
                if r.low <= t["sl"]: res = "SL"; break      # same-bar conflict counted as loss
                if r.high >= t["tp1"]: res = "TP"; break
            else:
                if r.high >= t["sl"]: res = "SL"; break
                if r.low <= t["tp1"]: res = "TP"; break
        if res == "TP":
            tg(f"🎯 <b>{t['pair']} {t['side'].upper()}</b>: TP1 hit (+{MIN_RR:g}R)")
        elif res == "SL":
            tg(f"🛑 <b>{t['pair']} {t['side'].upper()}</b>: stopped out (-1R)")
        elif datetime.now(timezone.utc) - pd.Timestamp(t["time"]).to_pydatetime() > timedelta(hours=48):
            tg(f"⌛ {t['pair']} {t['side'].upper()} signal expired (no TP/SL in 48h)")
        else:
            still.append(t)
    state["open"] = still

# ----------------------------- SCAN LOOP ------------------------------
def scan(state):
    cands = []
    for pair, cfg in PAIRS.items():
        if any(t["pair"] == pair for t in state["open"]):
            continue                                    # one live signal per pair
        df15 = get_candles(cfg[0], "15m", "5d", 15)
        df1h = get_candles(cfg[0], "1h", "30d", 60)
        if df15 is None or df1h is None:
            continue
        bias = htf_bias(df1h)
        for side in ("long", "short"):
            st = find_setup(df15, bias, side)
            if st and st["score"] >= MIN_SCORE and f"{pair}|{st['key']}" not in state["sent"]:
                cands.append((pair, st))
    cands.sort(key=lambda x: -x[1]["score"])            # best setups first
    for pair, st in cands:
        if state["count"] >= MAX_DAILY:
            break
        send_signal(pair, st)
        state["count"] += 1
        state["sent"].append(f"{pair}|{st['key']}")
        state["open"].append({"pair": pair, "side": st["side"], "sl": st["sl"],
                              "tp1": st["tp1"], "time": st["time"]})

def run_once():
    state = load_state()
    if market_open():
        check_open(state)
        if state["count"] < MAX_DAILY:
            scan(state)
    save_state(state)

def main():
    banner = (f"🤖 Signal bot online\nPairs: {', '.join(PAIRS)}\n"
              f"Risk {RISK_PCT}% | Min RR 1:{MIN_RR:g} | Max {MAX_DAILY}/day | Min score {MIN_SCORE}/12")
    if os.getenv("RUN_ONCE") == "1":            # GitHub Actions mode: one scan, then exit
        if os.getenv("SEND_STARTUP") == "1":
            tg(banner)
        run_once()
        return
    tg(banner)
    while True:
        try:
            run_once()
        except Exception as e:
            log.exception("loop error: %s", e)
        now = datetime.now(timezone.utc)
        nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=15 - now.minute % 15)
        time.sleep(max(30, (nxt - now).total_seconds() + 25))   # wake just after 15m candle closes

if __name__ == "__main__":
    main()
