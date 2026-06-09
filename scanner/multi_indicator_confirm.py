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

def fetch_data(ticker, source='borsapy', days=400):
    base = ticker.replace('.IS', '')
    try:
        if source == 'borsapy' and HAS_BORSAPY:
            start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            df = bp.Ticker(base).history(start=start)
        elif HAS_YFINANCE:
            df = yf.download(f"{base}.IS", period='1y', progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
        else:
            return None
        if df is None or len(df) < 60:
            return None
        return df
    except Exception:
        return None


# === Vote sistemi ===

def vote_rsi(rsi_now, rsi_prev, side='LONG'):
    """RSI vote:
    LONG: oversold (<30) ve yukselmeye basladi -> +1
    SHORT: overbought (>70) ve dusmeye basladi -> -1
    """
    if pd.isna(rsi_now) or pd.isna(rsi_prev):
        return 0, 'N/A'
    if side == 'LONG':
        if rsi_now < 35 and rsi_now > rsi_prev:
            return 1, f"Oversold dönüş ({rsi_now:.0f})"
        if rsi_now > 70:
            return -1, f"Overbought ({rsi_now:.0f})"
        return 0, f"Nötr ({rsi_now:.0f})"
    else:  # SHORT
        if rsi_now > 65 and rsi_now < rsi_prev:
            return 1, f"Overbought dönüş ({rsi_now:.0f})"
        if rsi_now < 30:
            return -1, f"Oversold ({rsi_now:.0f})"
        return 0, f"Nötr ({rsi_now:.0f})"


def vote_macd(hist_now, hist_prev, side='LONG'):
    """MACD histogram dönüş işareti"""
    if pd.isna(hist_now) or pd.isna(hist_prev):
        return 0, 'N/A'
    if side == 'LONG':
        if hist_now > hist_prev and hist_now > 0:
            return 1, f"Pozitif momentum ({hist_now:+.2f})"
        if hist_now < hist_prev and hist_now < 0:
            return -1, f"Negatif momentum ({hist_now:+.2f})"
    else:
        if hist_now < hist_prev and hist_now < 0:
            return 1, f"Negatif momentum ({hist_now:+.2f})"
        if hist_now > hist_prev and hist_now > 0:
            return -1, f"Pozitif momentum ({hist_now:+.2f})"
    return 0, f"Nötr ({hist_now:+.2f})"


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

    # Vote'lar
    votes = {}
    votes['RSI'] = vote_rsi(rsi_v.iloc[last], rsi_v.iloc[prev], side)
    votes['MACD'] = vote_macd(macd_h.iloc[last], macd_h.iloc[prev], side)
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
        f'<p>{len(setups)} setup analiz edildi. Score = (-6, +6) aralığında.</p>',
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
                    f'Score: {score:+d} / +6</span></h2>')
        dist = s.get('distance_atr', 0)
        dist_str = f"{dist:+.2f} ATR" if dist is not None else "—"
        html.append(f'<p><strong>Fiyat:</strong> {s["current_price"]:.2f} | '
                    f'<strong>Yön önerisi:</strong> <span class="{"pos" if s["side"]=="LONG" else "neg"}">{s["side"]}</span> | '
                    f'<strong>MA Uzaklık:</strong> {dist_str} | '
                    f'<strong>ATR:</strong> {s["atr"]:.2f}</p>')

        html.append('<table><tr><th>Indikatör</th><th>Vote</th><th>Açıklama</th></tr>')
        for ind, (v, expl) in s['votes'].items():
            sym = '✓' if v > 0 else ('✗' if v < 0 else '○')
            cls_v = 'pos' if v > 0 else ('neg' if v < 0 else 'neut')
            html.append(f'<tr><td>{ind}</td><td class="{cls_v}">{sym} {v:+d}</td><td>{expl}</td></tr>')
        html.append('</table>')
        html.append('</div>')

    html.append('<div style="margin-top:30px;padding:15px;background:#1a1f29;border-radius:8px;">')
    html.append('<strong>Yorum kılavuzu:</strong><br>')
    html.append('• Score +4..+6 (🥇 altın): 4+ indikatör onaylıyor → güçlü setup<br>')
    html.append('• Score +2..+3 (🥈 gümüş): 2-3 confirmation → trade edilebilir, ama ek inceleme yap<br>')
    html.append('• Score 0..+1: Zayıf veya karışık sinyal → beklemek daha iyi<br>')
    html.append('• Score negatif: Indikatörler ters yönde → BU TRADE\'I ALMA<br>')
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

        result = analyze_ticker(tk, ma_val, side, source=args.source)
        if result:
            result['top_ma'] = f"{top['ma_type']} {top['period']}"
            setups.append(result)
            print(f"  {tk}: score={result['total_score']:+d} ({side})")

    generate_html(setups, args.output)
    print(f"\n✓ {len(setups)} hisse analiz edildi")


if __name__ == '__main__':
    main()
