#!/usr/bin/env python3
"""
Multi-Indicator Confirmation — MA setup'larını çoklu indikatör ile doğrula.

Algoritma "BRSAN HMA 21 saygılı" diyor. Bu script ekler:
"Şu an BRSAN MA'ya yaklaşıyor + RSI oversold + BB lower band + MACD dönüş = 4/6 confirmation"

İndikatörler:
- RSI (14): Aşırı alım/satım dönüşü
- MACD (12,26,9): Histogram momentum
- Bollinger Bands (20, 2): Volatilite/sınır
- İchimoku (9,26,52): Trend ve cloud
- OBV: Hacim/fiyat divergence
- SMI (Stochastic Momentum Index, 14,3,3): Momentum

Her indikatör vote verir: +1 (long), -1 (short), 0 (neutral)
Toplam confirmation score = -6 ile +6 arası

Kullanim:
    python multi_indicator_confirm.py --csv reports/scan.csv --output reports/confirm.html
"""

import argparse, os, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import borsapy as bp
    HAS_BORSAPY = True
except ImportError:
    HAS_BORSAPY = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# === İndikatör Hesaplamaları (manuel, library bağımlılığı yok) ===

def rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(close, period=20, std_mult=2):
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper, middle, lower


def ichimoku(high, low, close):
    """Tenkan, Kijun, Senkou A, Senkou B, Chikou"""
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    return tenkan, kijun, senkou_a, senkou_b, chikou


def obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


def smi(close, high, low, period=14, smooth1=3, smooth2=3):
    """Stochastic Momentum Index"""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    midpoint = (hh + ll) / 2
    diff = close - midpoint
    range_diff = (hh - ll) / 2
    sm1_diff = diff.ewm(span=smooth1).mean()
    sm1_range = range_diff.ewm(span=smooth1).mean()
    sm2_diff = sm1_diff.ewm(span=smooth2).mean()
    sm2_range = sm1_range.ewm(span=smooth2).mean()
    smi_val = 100 * sm2_diff / sm2_range.replace(0, np.nan)
    smi_signal = smi_val.ewm(span=smooth1).mean()
    return smi_val, smi_signal


def atr(high, low, close, period=14):
    h = high.values
    l = low.values
    c = close.values
    tr = np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])
    atr_s = pd.Series(tr).rolling(period).mean()
    return pd.concat([pd.Series([np.nan]), atr_s]).reset_index(drop=True)


# === Veri çekme ===

def is_bist_index(symbol):
    """BIST endeks sembolu algilama (X ile baslar, 4+ karakter)."""
    s = symbol.upper().replace('.IS', '')
    return s.startswith('X') and len(s) >= 4


def fetch_data(ticker, source='borsapy', days=400, interval='1d'):
    """borsapy v0.10+ ile veri çek.

    Native interval'ler: 1m, 5m, 15m, 30m, 1h, 1d
    Türetilen: 4h (1h+resample), 1wk (1d+W), 1mo (1d+ME)
    """
    base = ticker.replace('.IS', '')
    is_index = is_bist_index(base)

    # Interval → borsapy period mapping
    def _bp_period(intv, def_days):
        mp = {'1m':'1g','5m':'5g','15m':'5g','30m':'5g',
              '1h':'1ay','4h':'3ay'}
        if intv in mp:
            return mp[intv]
        # Günlük+ için days'i 'Xy' veya max yap
        if def_days >= 1825:
            return '5y'
        if def_days >= 1100:
            return '3y'
        return f"{def_days}g"

    BORSAPY_NATIVE = {'1m','5m','15m','30m','1h','1d'}

    try:
        if source == 'borsapy' and HAS_BORSAPY:
            # Hangi interval'ı çekelim?
            if interval in BORSAPY_NATIVE:
                bp_int = interval
                resample = None
            elif interval == '4h':
                bp_int, resample = '1h', '4h'
            elif interval == '1wk':
                bp_int, resample = '1d', '1wk'
            elif interval == '1mo':
                bp_int, resample = '1d', '1mo'
            else:
                bp_int, resample = '1d', None

            bp_period = _bp_period(interval, days)

            # Endeks için Index() önce, Ticker() fallback
            df = None
            if is_index and hasattr(bp, 'Index'):
                try:
                    df = bp.Index(base).history(period=bp_period, interval=bp_int)
                except Exception:
                    df = None
                if df is None or df.empty:
                    try:
                        df = bp.Ticker(base).history(period=bp_period, interval=bp_int)
                    except Exception:
                        df = None
            else:
                df = bp.Ticker(base).history(period=bp_period, interval=bp_int)

            if df is None or df.empty:
                return None

            # Resample
            if resample == '4h':
                df = df.resample('4h').agg({'Open':'first','High':'max','Low':'min',
                    'Close':'last','Volume':'sum'}).dropna()
            elif resample == '1wk':
                df = df.resample('W').agg({'Open':'first','High':'max','Low':'min',
                    'Close':'last','Volume':'sum'}).dropna()
            elif resample == '1mo':
                df = df.resample('ME').agg({'Open':'first','High':'max','Low':'min',
                    'Close':'last','Volume':'sum'}).dropna()

        elif HAS_YFINANCE:
            symbol = f"^{base}" if is_index else f"{base}.IS"
            df = yf.download(symbol, period='1y', interval=interval, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
        else:
            return None

        if df is None or len(df) < 60:
            return None
        if 'Volume' not in df.columns or df['Volume'].isna().all():
            df['Volume'] = 1.0
        return df
    except Exception:
        return None


# === Vote sistemi ===

def rsi_ma(rsi_series, period=9):
    """RSI'in hareketli ortalamasi (sinyal cizgisi gibi)."""
    return rsi_series.ewm(span=period, adjust=False).mean()


def detect_rsi_divergence(close, rsi_series, lookback=20):
    """Bullish/Bearish divergence tespiti."""
    if len(close) < lookback + 5:
        return None
    p = close.values[-lookback:]
    r = rsi_series.values[-lookback:]
    half = lookback // 2
    p_recent_low = p[-half:].min()
    p_old_low = p[:half].min()
    r_recent_low = r[-half + np.argmin(p[-half:])]
    r_old_low = r[np.argmin(p[:half])]
    if p_recent_low < p_old_low * 0.98 and r_recent_low > r_old_low + 3:
        return 'bullish_divergence'
    p_recent_high = p[-half:].max()
    p_old_high = p[:half].max()
    r_recent_high = r[-half + np.argmax(p[-half:])]
    r_old_high = r[np.argmax(p[:half])]
    if p_recent_high > p_old_high * 1.02 and r_recent_high < r_old_high - 3:
        return 'bearish_divergence'
    return None


def vote_rsi(rsi_now, rsi_prev, rsi_ma_now=None, rsi_ma_prev=None,
              divergence=None, side='LONG'):
    """RSI profesyonel vote - cross, divergence, asiri al/sat dahil."""
    if pd.isna(rsi_now) or pd.isna(rsi_prev):
        return 0, 'N/A'

    # Temel seviye yorumu
    if rsi_now > 80:
        level = f"{rsi_now:.0f} EXTREM ASIRI ALIM"
    elif rsi_now > 70:
        level = f"{rsi_now:.0f} asiri alim (isiniyor)"
    elif rsi_now < 20:
        level = f"{rsi_now:.0f} EXTREM ASIRI SATIM"
    elif rsi_now < 30:
        level = f"{rsi_now:.0f} asiri satim"
    elif rsi_now > 50:
        level = f"{rsi_now:.0f} bullish bolge"
    else:
        level = f"{rsi_now:.0f} bearish bolge"

    # MA kesişimi
    cross = ""
    if rsi_ma_now is not None and not pd.isna(rsi_ma_now):
        if rsi_prev <= rsi_ma_prev and rsi_now > rsi_ma_now:
            cross = " + MA yukari kesti"
        elif rsi_prev >= rsi_ma_prev and rsi_now < rsi_ma_now:
            cross = " + MA asagi kesti"

    # Divergence
    div = ""
    if divergence == 'bullish_divergence':
        div = " + BULLISH DIVERGENCE"
    elif divergence == 'bearish_divergence':
        div = " + BEARISH DIVERGENCE"

    desc = level + cross + div

    # Vote logic - detayli
    if side == 'LONG':
        # Guclu LONG: bullish div veya extrem oversold + yukari kesim
        if divergence == 'bullish_divergence':
            return 2, desc  # ekstra puan
        if rsi_now < 30 and "yukari kesti" in cross:
            return 2, desc
        if rsi_now < 35 and rsi_now > rsi_prev:
            return 1, desc
        if "yukari kesti" in cross and rsi_now > 45:
            return 1, desc
        if rsi_now > 70:
            return -1, desc
        return 0, desc
    else:  # SHORT
        if divergence == 'bearish_divergence':
            return 2, desc
        if rsi_now > 70 and "asagi kesti" in cross:
            return 2, desc
        if rsi_now > 65 and rsi_now < rsi_prev:
            return 1, desc
        if "asagi kesti" in cross and rsi_now < 55:
            return 1, desc
        if rsi_now < 30:
            return -1, desc
        return 0, desc


def vote_macd(macd_line_now, macd_line_prev, signal_now, signal_prev,
               hist_now, hist_prev, hist_3ago=None, side='LONG'):
    """MACD profesyonel vote - cross, sifir bolgesi, histogram pikleri."""
    if pd.isna(hist_now) or pd.isna(hist_prev):
        return 0, 'N/A'

    above_signal = macd_line_now > signal_now
    macd_pos = macd_line_now > 0
    signal_pos = signal_now > 0
    hist_pos = hist_now > 0

    # Cross detection
    bull_cross = macd_line_prev <= signal_prev and macd_line_now > signal_now
    bear_cross = macd_line_prev >= signal_prev and macd_line_now < signal_now

    # Histogram pikleri (sadece 3 bar varsa)
    hist_growing = hist_3ago is not None and abs(hist_now) > abs(hist_prev) > abs(hist_3ago)
    hist_shrinking = hist_3ago is not None and abs(hist_now) < abs(hist_prev) < abs(hist_3ago)

    # Durum etiketi
    if above_signal and macd_pos and signal_pos and hist_pos:
        state = "TUM POZITIF (guclu bullish)"
        long_score = 2
        short_score = -2
    elif not above_signal and not macd_pos and not signal_pos and not hist_pos:
        state = "TUM NEGATIF (guclu bearish)"
        long_score = -2
        short_score = 2
    elif above_signal and not macd_pos:
        state = "Dipten donus baslangici"
        long_score = 1
        short_score = -1
    elif not above_signal and macd_pos:
        state = "Tepeden dusus baslangici"
        long_score = -1
        short_score = 1
    elif above_signal and macd_pos:
        state = "Bullish gelisiyor"
        long_score = 1
        short_score = -1
    elif not above_signal and not macd_pos:
        state = "Bearish gelisiyor"
        long_score = -1
        short_score = 1
    else:
        state = "Karisik"
        long_score = 0
        short_score = 0

    # Cross bonusu
    extras = []
    if bull_cross:
        extras.append("yukari cross")
        if side == 'LONG': long_score += 1
        else: short_score -= 1
    if bear_cross:
        extras.append("asagi cross")
        if side == 'LONG': long_score -= 1
        else: short_score += 1

    # Histogram pikleri - momentum yorumu
    if hist_pos and hist_shrinking:
        extras.append("Hist pikler azaliyor (bullish zayifliyor)")
        if side == 'LONG': long_score -= 1
    elif not hist_pos and hist_shrinking:
        extras.append("Hist neg pikler azaliyor (LONG firsati)")
        if side == 'LONG': long_score += 1
    elif hist_pos and hist_growing:
        extras.append("Hist buyuyor (bullish guclu)")
    elif not hist_pos and hist_growing:
        extras.append("Hist neg buyuyor (bearish guclu)")
        if side == 'SHORT': short_score += 1

    desc = state
    if extras:
        desc += " | " + ", ".join(extras)

    # Vote (clamp -2..+2)
    if side == 'LONG':
        v = max(-2, min(2, long_score))
    else:
        v = max(-2, min(2, short_score))
    return v, desc


def vote_bb(close, upper, middle, lower, side='LONG'):
    """Bollinger Bands: alt banda touch (long) veya üst banda touch (short)"""
    if pd.isna(upper) or pd.isna(lower):
        return 0, 'N/A'
    if side == 'LONG':
        if close < lower * 1.02:
            return 1, f"Alt banda yakın ({close:.2f} < {lower:.2f})"
        if close > upper * 0.98:
            return -1, f"Üst banda yakın"
    else:
        if close > upper * 0.98:
            return 1, f"Üst banda yakın"
        if close < lower * 1.02:
            return -1, f"Alt banda yakın"
    pos_pct = (close - lower) / (upper - lower) * 100 if upper > lower else 50
    return 0, f"Orta ({pos_pct:.0f}%)"


def vote_ichimoku(close, tenkan, kijun, senkou_a, senkou_b, side='LONG'):
    """İchimoku: Cloud üstünde (long) veya altında (short) + TK cross"""
    if any(pd.isna([tenkan, kijun, senkou_a, senkou_b])):
        return 0, 'N/A'
    cloud_top = max(senkou_a, senkou_b)
    cloud_bot = min(senkou_a, senkou_b)
    if side == 'LONG':
        if close > cloud_top and tenkan > kijun:
            return 1, "Bulutun üstü + TK bullish"
        if close < cloud_bot:
            return -1, "Bulutun altı"
        if cloud_bot <= close <= cloud_top:
            return 0, "Bulutun içi"
    else:
        if close < cloud_bot and tenkan < kijun:
            return 1, "Bulutun altı + TK bearish"
        if close > cloud_top:
            return -1, "Bulutun üstü"
    return 0, "Karışık sinyal"


def vote_obv(obv_now, obv_prev, close_now, close_prev, side='LONG'):
    """OBV divergence: fiyat ve OBV aynı yönde mi"""
    if pd.isna(obv_now) or pd.isna(obv_prev):
        return 0, 'N/A'
    price_dir = np.sign(close_now - close_prev)
    obv_dir = np.sign(obv_now - obv_prev)
    if side == 'LONG':
        if obv_dir > 0 and price_dir >= 0:
            return 1, "Hacim destekli yukseliş"
        if obv_dir < 0 and price_dir > 0:
            return -1, "Bearish divergence (hacim destekli değil)"
    else:
        if obv_dir < 0 and price_dir <= 0:
            return 1, "Hacim destekli düşüş"
        if obv_dir > 0 and price_dir < 0:
            return -1, "Bullish divergence"
    return 0, "Tutarlı"


def vote_smi(smi_now, smi_prev, smi_sig_now, side='LONG'):
    """SMI momentum dönüşü"""
    if pd.isna(smi_now) or pd.isna(smi_prev):
        return 0, 'N/A'
    if side == 'LONG':
        if smi_now < -40 and smi_now > smi_prev:
            return 1, f"Aşırı satımda dönüş ({smi_now:.0f})"
        if smi_now > smi_sig_now and smi_now > -20:
            return 1, f"Sinyal üstü ({smi_now:.0f})"
        if smi_now > 40:
            return -1, f"Aşırı alım ({smi_now:.0f})"
    else:
        if smi_now > 40 and smi_now < smi_prev:
            return 1, f"Aşırı alımda dönüş ({smi_now:.0f})"
    return 0, f"Nötr ({smi_now:.0f})"


def analyze_ticker(ticker, ma_value, side, source='borsapy'):
    """Hisse için tüm indikatörleri hesapla ve vote'la."""
    df = fetch_data(ticker, source=source)
    if df is None:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Tüm indikatörleri hesapla
    rsi_v = rsi(close)
    macd_l, macd_s, macd_h = macd(close)
    bb_u, bb_m, bb_l = bollinger(close)
    tenkan, kijun, sa, sb, _ = ichimoku(high, low, close)
    obv_v = obv(close, volume)
    smi_v, smi_s = smi(close, high, low)
    atr_v = atr(high, low, close)

    # Son bar değerleri
    last = -1
    prev = -2
    current_price = close.iloc[last]
    current_atr = atr_v.iloc[last]

    if pd.isna(current_atr) or current_atr <= 0:
        return None

    # Distance to MA
    distance_atr = (current_price - ma_value) / current_atr if not pd.isna(ma_value) else None

    # RSI MA ve divergence hesabi (profesyonel analiz)
    rsi_ma_v = rsi_ma(rsi_v)
    divergence = detect_rsi_divergence(close, rsi_v, lookback=20)

    # Vote'lar - detayli
    votes = {}
    votes['RSI'] = vote_rsi(rsi_v.iloc[last], rsi_v.iloc[prev],
                              rsi_ma_v.iloc[last] if not pd.isna(rsi_ma_v.iloc[last]) else None,
                              rsi_ma_v.iloc[prev] if not pd.isna(rsi_ma_v.iloc[prev]) else None,
                              divergence, side)
    votes['MACD'] = vote_macd(macd_l.iloc[last], macd_l.iloc[prev],
                                macd_s.iloc[last], macd_s.iloc[prev],
                                macd_h.iloc[last], macd_h.iloc[prev],
                                macd_h.iloc[-4] if len(macd_h) > 3 else None,
                                side)
    votes['Bollinger'] = vote_bb(close.iloc[last], bb_u.iloc[last], bb_m.iloc[last], bb_l.iloc[last], side)
    votes['Ichimoku'] = vote_ichimoku(close.iloc[last], tenkan.iloc[last], kijun.iloc[last],
                                       sa.iloc[last], sb.iloc[last], side)
    votes['OBV'] = vote_obv(obv_v.iloc[last], obv_v.iloc[prev],
                             close.iloc[last], close.iloc[prev], side)
    votes['SMI'] = vote_smi(smi_v.iloc[last], smi_v.iloc[prev], smi_s.iloc[last], side)

    total_score = sum(v[0] for v in votes.values())

    return {
        'ticker': ticker,
        'current_price': current_price,
        'atr': current_atr,
        'distance_atr': distance_atr,
        'side': side,
        'votes': votes,
        'total_score': total_score,
    }


def generate_html(setups: list, output: str):
    html = [
        '<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">',
        '<title>Multi-Indicator Confirmation</title>',
        '<style>',
        'body{font-family:-apple-system,sans-serif;background:#0a0e14;color:#e6e6e6;'
        'padding:20px;max-width:1500px;margin:auto;}',
        'h1{color:#5fb3ff;border-bottom:2px solid #5fb3ff;padding-bottom:8px;}',
        '.setup{background:#1a1f29;padding:15px;border-radius:8px;margin:15px 0;}',
        '.score-high{border-left:4px solid #ffd700;background:#2a2400;}',
        '.score-mid{border-left:4px solid #7fc97f;}',
        '.score-low{border-left:4px solid #555;}',
        '.score-neg{border-left:4px solid #e74c3c;background:#2a1a1a;}',
        'table{border-collapse:collapse;width:100%;font-family:monospace;font-size:13px;}',
        'th,td{padding:6px 10px;border-bottom:1px solid #2a2f39;}',
        'th{background:#2a2f39;color:#5fb3ff;}',
        '.pos{color:#7fc97f;}.neg{color:#e74c3c;}.neut{color:#888;}',
        '.score-badge{display:inline-block;padding:4px 12px;border-radius:12px;font-weight:bold;}',
        '.badge-high{background:#ffd700;color:#000;}',
        '.badge-mid{background:#7fc97f;color:#000;}',
        '.badge-low{background:#444;color:#aaa;}',
        '.badge-neg{background:#e74c3c;color:#fff;}',
        '</style></head><body>',
        f'<h1>Multi-Indicator Confirmation — {datetime.now():%Y-%m-%d %H:%M}</h1>',
        f'<p>{len(setups)} setup analiz edildi. Score = (-8, +8) aralığında. RSI ve MACD profesyonel analizle +2/-2 verebiliyor.</p>',
    ]

    # Skora göre sırala (yüksekten düşüğe)
    setups_sorted = sorted(setups, key=lambda s: -s.get('total_score', -10))

    for s in setups_sorted:
        if s is None:
            continue
        score = s['total_score']
        if score >= 4:
            cls, badge = 'score-high', 'badge-high'
        elif score >= 2:
            cls, badge = 'score-mid', 'badge-mid'
        elif score >= 0:
            cls, badge = 'score-low', 'badge-low'
        else:
            cls, badge = 'score-neg', 'badge-neg'

        html.append(f'<div class="setup {cls}">')
        html.append(f'<h2>{s["ticker"]} <span class="score-badge {badge}">'
                    f'Score: {score:+d} / +8</span></h2>')
        dist = s.get('distance_atr', 0)
        dist_str = f"{dist:+.2f} ATR" if dist is not None else "—"
        curr_p = s["current_price"]
        atr_v = s["atr"]

        # Fiyat formati
        def fmt_p(p):
            if curr_p < 10: return f"{p:.4f}"
            elif curr_p < 100: return f"{p:.3f}"
            else: return f"{p:.2f}"

        # Top MA'nin etiketi (DESTEK/DIRENC)
        top_ma_val = s.get('top_ma_value', None)
        if top_ma_val is not None:
            top_etiket = 'DESTEK' if top_ma_val < curr_p else 'DIRENC'
            top_etiket_cls = 'pos' if top_ma_val < curr_p else 'neg'
        else:
            top_etiket = '—'
            top_etiket_cls = ''

        html.append(f'<p><strong>Fiyat:</strong> {fmt_p(curr_p)} | '
                    f'<strong>Top MA:</strong> {s.get("top_ma", "—")} @{fmt_p(top_ma_val) if top_ma_val else "—"} '
                    f'<span class="{top_etiket_cls}">({top_etiket})</span> | '
                    f'<strong>Yön:</strong> <span class="{"pos" if s["side"]=="LONG" else "neg"}">{s["side"]}</span> | '
                    f'<strong>Uzaklık:</strong> {dist_str} | '
                    f'<strong>ATR:</strong> {atr_v:.2f}</p>')

        # YENI: MA Kümeleri (Destek/Direnc) - tum robust MA'lardan
        all_mas = s.get('all_robust_mas', [])
        if all_mas and len(all_mas) >= 2:
            destek_mas = [m for m in all_mas if m['value'] < curr_p]
            direnc_mas = [m for m in all_mas if m['value'] >= curr_p]

            html.append('<div style="background:#0f1419;padding:10px;border-radius:6px;font-family:monospace;margin:10px 0;">')
            html.append('<strong style="color:#7be2ff;">🎯 MA Kümeleri (Destek/Direnç)</strong><br>')

            # Cluster algoritmasi
            def cluster_mas_local(mas, threshold_atr=0.5):
                if not mas: return []
                sorted_mas = sorted(mas, key=lambda x: x['value'])
                clusters = [[sorted_mas[0]]]
                for ma in sorted_mas[1:]:
                    if abs(ma['value'] - clusters[-1][-1]['value']) <= threshold_atr * atr_v:
                        clusters[-1].append(ma)
                    else:
                        clusters.append([ma])
                return clusters

            # Direncler (yukaridakiler)
            direnc_clusters = cluster_mas_local(direnc_mas)
            for cl in sorted(direnc_clusters, key=lambda c: c[0]['value']):
                vals = [m['value'] for m in cl]
                tags = ', '.join(f"{m['ma_type']}{m['period']}" for m in cl)
                if len(cl) > 1:
                    html.append(f'<div style="color:#ff8c69;">🔴 {fmt_p(min(vals))}-{fmt_p(max(vals))}: {tags} <strong>({len(cl)} MA)</strong></div>')
                else:
                    html.append(f'<div style="color:#ff8c69;">🔴 {fmt_p(vals[0])}: {tags}</div>')

            html.append(f'<div style="color:#7be2ff;font-weight:bold;border-top:1px solid #5fb3ff;border-bottom:1px solid #5fb3ff;padding:4px 0;margin:6px 0;">━━ FIYAT: {fmt_p(curr_p)} ━━</div>')

            # Destekler (asagidakiler)
            destek_clusters = cluster_mas_local(destek_mas)
            for cl in sorted(destek_clusters, key=lambda c: -c[0]['value']):
                vals = [m['value'] for m in cl]
                tags = ', '.join(f"{m['ma_type']}{m['period']}" for m in cl)
                if len(cl) > 1:
                    html.append(f'<div style="color:#7fc97f;">🟢 {fmt_p(min(vals))}-{fmt_p(max(vals))}: {tags} <strong>({len(cl)} MA)</strong></div>')
                else:
                    html.append(f'<div style="color:#7fc97f;">🟢 {fmt_p(vals[0])}: {tags}</div>')
            html.append('</div>')

        html.append('<table><tr><th>Indikatör</th><th>Vote</th><th>Açıklama</th></tr>')
        for ind, (v, expl) in s['votes'].items():
            sym = '✓' if v > 0 else ('✗' if v < 0 else '○')
            cls_v = 'pos' if v > 0 else ('neg' if v < 0 else 'neut')
            html.append(f'<tr><td>{ind}</td><td class="{cls_v}">{sym} {v:+d}</td><td>{expl}</td></tr>')
        html.append('</table>')
        html.append('</div>')

    html.append('<div style="margin-top:30px;padding:15px;background:#1a1f29;border-radius:8px;">')
    html.append('<strong>Yorum kılavuzu:</strong><br>')
    html.append('• Score +5..+8 (🥇 altın): RSI/MACD profesyonel + 3+ indikatör onayı → güçlü setup<br>')
    html.append('• Score +2..+4 (🥈 gümüş): Çoğunluk onaylıyor → trade edilebilir, ek inceleme yap<br>')
    html.append('• Score 0..+1: Zayıf veya karışık sinyal → beklemek daha iyi<br>')
    html.append('• Score negatif: Indikatörler ters yönde → BU TRADE\'I ALMA<br>')
    html.append('<br><strong>RSI:</strong> divergence varsa +2, sadece kesim +1, asiri al/sat -1<br>')
    html.append('<strong>MACD:</strong> TUM POZITIF +2, dipten donus +1, cross bonus +1, hist zayifliyor -1<br>')
    html.append('</div></body></html>')

    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"✓ HTML: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--tickers', type=str, default='')
    p.add_argument('--all-robust', action='store_true')
    p.add_argument('--n_tickers', type=int, default=15, help='En çok kaç hisse analiz edilsin')
    p.add_argument('--source', type=str, default='borsapy')
    p.add_argument('--output', type=str, default='confirm.html')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"CSV: {len(df)} satir")

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    elif args.all_robust:
        if 'wf_robust' in df.columns:
            tickers = df[df['wf_robust']==True].groupby('ticker').size().nlargest(args.n_tickers).index.tolist()
        else:
            tickers = df.nlargest(args.n_tickers, 'composite_score')['ticker'].unique().tolist()
    else:
        print("HATA: --tickers veya --all-robust gerekli")
        sys.exit(1)

    setups = []
    for tk in tickers:
        sub = df[df['ticker'] == tk]
        if 'wf_robust' in sub.columns:
            sub = sub[sub['wf_robust'] == True]
        if sub.empty:
            continue
        top = sub.nlargest(1, 'composite_score').iloc[0]
        ma_val = top.get('current_ma_value', np.nan)

        # Yön belirleme: composite_score pozitif ise mevcut tarafa LONG
        # Mevcut MA değeri yoksa skip
        if pd.isna(ma_val):
            print(f"  {tk}: MA değeri yok, atlanıyor")
            continue

        # Hisse'nin mevcut fiyatı vs MA -> yön
        price_data = fetch_data(tk, source=args.source)
        if price_data is None:
            continue
        current_price = price_data['Close'].iloc[-1]
        side = 'LONG' if current_price > ma_val else 'SHORT'

        # YENI: Bu hissenin TOP 5 robust MA'sini topla (cluster + DESTEK/DIRENC icin)
        top5 = sub.nlargest(5, 'composite_score')
        all_mas = []
        for _, r in top5.iterrows():
            mv = r.get('current_ma_value', np.nan)
            if not pd.isna(mv):
                all_mas.append({
                    'ma_type': r['ma_type'],
                    'period': int(r['period']),
                    'value': float(mv),
                    'wr_pct': r['wr_pct'],
                    'touches': int(r.get('touches', 0)),
                    'composite_score': r['composite_score'],
                })

        result = analyze_ticker(tk, ma_val, side, source=args.source)
        if result:
            result['top_ma'] = f"{top['ma_type']} {top['period']}"
            result['top_ma_value'] = float(ma_val)
            result['all_robust_mas'] = all_mas  # YENI: cluster gosterim icin
            setups.append(result)
            print(f"  {tk}: score={result['total_score']:+d} ({side}), {len(all_mas)} robust MA")

    generate_html(setups, args.output)
    print(f"\n✓ {len(setups)} hisse analiz edildi")


if __name__ == '__main__':
    main()
