#!/usr/bin/env python3
"""
Tek Hisse / Enstrüman Sorgu Tool'u

Bir sembol için TÜM analizleri tek bir yerde gösterir:
- En iyi MA kombinasyonları (top 10)
- Mevcut fiyat ve setup durumu (TOUCH_ZONE / SETUP_READY / NEAR / FAR)
- Multi-indicator durumu (RSI, MACD, Bollinger, İchimoku, OBV, SMI)
- Her aktif setup için somut entry/stop/TP/lot önerisi

Kullanim:
    # CSV varsa filtreleyerek (hizli):
    python single_ticker_query.py --ticker ASELS --csv reports/scan.csv

    # CSV yoksa canli mini tarama (~30s):
    python single_ticker_query.py --ticker BVSAN --scan-fresh

    # Coklu enstruman destegi (eger borsapy destekliyorsa):
    python single_ticker_query.py --ticker BTCUSD --scan-fresh
    python single_ticker_query.py --ticker XAUUSD --scan-fresh
"""

import argparse
import os
import sys
import re
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


# === Symbol detection — multi-instrument support ===
def detect_instrument_type(symbol: str) -> str:
    """Sembol turunu tahmin et: bist, crypto, forex, metal, fund."""
    s = symbol.upper().replace('.IS', '')

    # Metals
    if s.startswith('XAU') or s.startswith('XAG') or s.startswith('XPT'):
        return 'metal'

    # Crypto patterns
    crypto_bases = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE',
                    'AVAX', 'MATIC', 'LINK', 'UNI', 'LTC', 'TRX']
    if any(s.startswith(c) for c in crypto_bases):
        return 'crypto'

    # Forex (3-letter pairs)
    forex_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'TRY', 'CHF', 'CAD', 'AUD', 'NZD']
    if len(s) == 6 and s[:3] in forex_currencies and s[3:] in forex_currencies:
        return 'forex'

    # BIST default (turkce hisse formati)
    return 'bist'


def fetch_data(ticker, source='borsapy', period_days=1100):
    """Multi-instrument veri çek."""
    base = ticker.replace('.IS', '').upper()
    inst_type = detect_instrument_type(base)

    if source == 'borsapy' and HAS_BORSAPY:
        try:
            start = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')
            # borsapy bp.Ticker tüm enstrümanları (BIST/FX/crypto) destekliyor olabilir
            # Eğer ayrı sınıflar gerekiyorsa burada dispatch yap
            t = bp.Ticker(base)
            df = t.history(start=start)

            if df is None or df.empty or len(df) < 50:
                # FX/Crypto için alternatif sınıf dene
                if inst_type == 'crypto' and hasattr(bp, 'Crypto'):
                    t = bp.Crypto(base)
                    df = t.history(start=start)
                elif inst_type == 'forex' and hasattr(bp, 'FX'):
                    t = bp.FX(base)
                    df = t.history(start=start)
                elif inst_type == 'metal' and hasattr(bp, 'Metal'):
                    t = bp.Metal(base)
                    df = t.history(start=start)

            if df is None or df.empty:
                raise RuntimeError(f"borsapy bos veri: {base}")
            if len(df) < 50:
                raise RuntimeError(f"borsapy yetersiz: {base} ({len(df)} bar)")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df, inst_type
        except Exception as e:
            print(f"  borsapy hatasi: {e}")
            if HAS_YFINANCE and inst_type == 'bist':
                source = 'yfinance'
            else:
                return None, inst_type

    if source == 'yfinance' and HAS_YFINANCE:
        # Sadece BIST için yfinance var
        if inst_type == 'bist':
            symbol = f"{base}.IS"
        elif inst_type == 'crypto':
            symbol = f"{base[:3]}-{base[3:]}" if len(base) == 6 else base
        else:
            symbol = base
        df = yf.download(symbol, period='3y', progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return (df if not df.empty else None), inst_type

    return None, inst_type


# === MA Hesaplamalari (scanner ile birebir) ===

def sma(s, p): return s.rolling(p).mean()
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def wma(s, p):
    w = np.arange(1, p+1)
    return s.rolling(p).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
def hma(s, p):
    half = int(p/2)
    sq = int(np.sqrt(p))
    return wma(2*wma(s, half) - wma(s, p), sq)
def vwma(s, v, p):
    return (s*v).rolling(p).sum() / v.rolling(p).sum()
def alma(s, p, offset=0.85, sigma=6):
    m = offset*(p-1)
    sig = p/sigma
    w = np.array([np.exp(-((i-m)**2)/(2*sig*sig)) for i in range(p)])
    w /= w.sum()
    return s.rolling(p).apply(lambda x: np.dot(x, w), raw=True)
def kama(s, p, fast=2, slow=30):
    change = abs(s - s.shift(p))
    vol_ = (s.diff().abs()).rolling(p).sum()
    er = (change/vol_).fillna(0)
    sc = (er*(2/(fast+1)-2/(slow+1)) + 2/(slow+1))**2
    out = pd.Series(index=s.index, dtype=float)
    out.iloc[p-1] = s.iloc[p-1]
    for i in range(p, len(s)):
        out.iloc[i] = out.iloc[i-1] + sc.iloc[i]*(s.iloc[i]-out.iloc[i-1])
    return out

MA_FUNCS = {'SMA':sma, 'EMA':ema, 'WMA':wma, 'HMA':hma, 'ALMA':alma}
def compute_ma(typ, close, vol, period):
    if typ == 'VWMA':
        return vwma(close, vol, period)
    elif typ == 'KAMA':
        return kama(close, period)
    return MA_FUNCS[typ](close, period)


# === İndikatorler (basit) ===

def rsi(close, p=14):
    d = close.diff()
    g = (d.where(d>0, 0)).rolling(p).mean()
    l = (-d.where(d<0, 0)).rolling(p).mean()
    return 100 - 100/(1 + g/l.replace(0, np.nan))


def macd(close):
    ef = close.ewm(span=12, adjust=False).mean()
    es = close.ewm(span=26, adjust=False).mean()
    m = ef - es
    sig = m.ewm(span=9, adjust=False).mean()
    return m, sig, m - sig


def atr_calc(df, p=14):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    tr = np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])
    s = pd.Series(tr).rolling(p).mean()
    return pd.concat([pd.Series([np.nan]), s]).reset_index(drop=True).values


# === Mini scan (CSV yoksa) ===

def mini_scan(ticker, source='borsapy'):
    """Tek hisse için tüm MA kombinasyonlarını tara."""
    print(f"  {ticker} için canlı tarama başlıyor...")
    df, inst_type = fetch_data(ticker, source=source)
    if df is None:
        return None, None, None

    print(f"  Veri: {len(df)} bar, {df.index[0].date()} - {df.index[-1].date()}")
    atr_v = atr_calc(df)
    close = df['Close'].values
    volume = df['Volume'].values
    high = df['High'].values
    low = df['Low'].values

    MA_TYPES = ['SMA', 'EMA', 'WMA', 'VWMA', 'HMA', 'ALMA', 'KAMA']

    # Veri uzunlugu kontrolu - kisa veri ise uzun periyotlari atla
    n_bars = len(df)
    if n_bars < 100:
        # Yeni listelenen hisse - sadece kisa MA'lar
        PERIODS = [3, 5, 8, 10, 13, 20, 21, 22, 34]
        min_touches_eff = 3
        min_adr_eff = 0.3
        print(f"  Uyari: Az veri ({n_bars} bar) - sadece kisa MA'lar (3-34), gevsek filtre")
    elif n_bars < 250:
        # Orta veri
        PERIODS = [3, 5, 8, 10, 13, 20, 21, 22, 34, 50, 55, 89, 100]
        min_touches_eff = 5
        min_adr_eff = 0.35
        print(f"  Orta veri ({n_bars} bar) - kisa-orta MA'lar (3-100)")
    elif n_bars < 500:
        PERIODS = [3, 5, 8, 10, 13, 20, 21, 22, 34, 50, 55, 89, 100, 144, 200]
        min_touches_eff = 8
        min_adr_eff = 0.4
    else:
        # Yeterli veri - tum periyotlar
        PERIODS = [3, 5, 8, 10, 13, 20, 21, 22, 34, 50, 55, 89, 100, 144, 200, 233, 250]
        min_touches_eff = 10
        min_adr_eff = 0.4

    results = []
    skipped_reasons = {'period_too_long': 0, 'too_few_touches': 0, 'low_adr': 0, 'nan_data': 0}
    for ma_type in MA_TYPES:
        for period in PERIODS:
            if ma_type == 'HMA' and period < 20:
                continue
            # MA periyodu veri uzunlugundan fazla ise atla
            if period * 2 > n_bars:
                skipped_reasons['period_too_long'] += 1
                continue
            try:
                ma = compute_ma(ma_type, df['Close'], df['Volume'], period)
                if ma is None or ma.isna().sum() > len(ma) * 0.5:
                    continue

                # Basit touch + tepki analizi
                ma_v = ma.values
                touches = 0
                wins = 0
                mfe_sum = 0
                was_far = False
                dist_sum = 0
                atr_count = 0
                react_pct = 1.5
                atr_mult_zone = 0.2
                sep_mult = 2.5

                for i in range(50, len(df)-5):
                    if np.isnan(ma_v[i]) or np.isnan(atr_v[i]):
                        continue
                    # ADR
                    dist_sum += abs(close[i] - ma_v[i])
                    atr_count += 1
                    # Pre-touch separation
                    if abs(close[i] - ma_v[i]) > sep_mult * atr_v[i]:
                        was_far = True
                    # Touch
                    in_zone = (low[i] <= ma_v[i] + atr_mult_zone*atr_v[i] and
                               high[i] >= ma_v[i] - atr_mult_zone*atr_v[i])
                    if in_zone and was_far:
                        touches += 1
                        # MFE 5 bar
                        prev_above = close[i-1] > ma_v[i-1] if not np.isnan(ma_v[i-1]) else True
                        from_above = prev_above
                        if from_above:
                            max_fav = max(high[i+1:i+6])
                            move = (max_fav - close[i]) / close[i] * 100
                        else:
                            min_fav = min(low[i+1:i+6])
                            move = (close[i] - min_fav) / close[i] * 100
                        if move >= react_pct:
                            wins += 1
                            mfe_sum += move
                        was_far = False

                if touches < min_touches_eff:
                    skipped_reasons['too_few_touches'] += 1
                    continue
                wr = wins/touches * 100
                avg_mfe = mfe_sum/wins if wins > 0 else 0
                avg_dist = dist_sum / atr_count if atr_count > 0 else 0
                avg_atr = np.nanmean(atr_v[50:])
                adr = avg_dist / avg_atr if avg_atr > 0 else 0
                if adr < min_adr_eff:
                    skipped_reasons['low_adr'] += 1
                    continue
                expectancy = (wr/100) * avg_mfe - (1-wr/100) * 1.5  # avg_mae varsayım 1.5
                composite = expectancy * np.sqrt(touches) * min(adr, 2.0)

                results.append({
                    'ma_type': ma_type,
                    'period': period,
                    'touches': touches,
                    'wins': wins,
                    'wr_pct': wr,
                    'avg_mfe': avg_mfe,
                    'expectancy': expectancy,
                    'adr': adr,
                    'composite_score': composite,
                    'current_ma_value': float(ma_v[-1]),
                })
            except Exception:
                continue

    if not results:
        # Hicbir sonuc yoksa neden olduğunu logla
        print(f"  Hicbir MA filtreyi gecmedi. Atlanan nedenler: {skipped_reasons}")
        print(f"  Min touches: {min_touches_eff}, Min ADR: {min_adr_eff}")

        # FALLBACK MODE: Cok gevsek filtre ile tekrar dene
        # Tek hisse merak edildiginde en azindan bir bilgi gostermek icin
        print(f"  FALLBACK: Cok gevsek filtre ile tekrar deniyor...")
        min_touches_fb = 2
        min_adr_fb = 0.15

        for ma_type in MA_TYPES:
            for period in PERIODS:
                if ma_type == 'HMA' and period < 20:
                    continue
                if period * 2 > n_bars:
                    continue
                try:
                    ma = compute_ma(ma_type, df['Close'], df['Volume'], period)
                    if ma is None or ma.isna().sum() > len(ma) * 0.5:
                        continue
                    ma_v = ma.values
                    touches = wins = 0
                    mfe_sum = 0
                    was_far = False
                    dist_sum = atr_count = 0
                    for i in range(50, len(df)-5):
                        if np.isnan(ma_v[i]) or np.isnan(atr_v[i]):
                            continue
                        dist_sum += abs(close[i] - ma_v[i])
                        atr_count += 1
                        if abs(close[i] - ma_v[i]) > 1.5 * atr_v[i]:  # gevsek separation
                            was_far = True
                        in_zone = (low[i] <= ma_v[i] + 0.3*atr_v[i] and
                                   high[i] >= ma_v[i] - 0.3*atr_v[i])  # genis zone
                        if in_zone and was_far:
                            touches += 1
                            prev_above = close[i-1] > ma_v[i-1] if not np.isnan(ma_v[i-1]) else True
                            if prev_above:
                                max_fav = max(high[i+1:i+6])
                                move = (max_fav - close[i]) / close[i] * 100
                            else:
                                min_fav = min(low[i+1:i+6])
                                move = (close[i] - min_fav) / close[i] * 100
                            if move >= 1.0:  # gevsek react_pct
                                wins += 1
                                mfe_sum += move
                            was_far = False

                    if touches < min_touches_fb:
                        continue
                    wr = wins/touches * 100
                    avg_mfe = mfe_sum/wins if wins > 0 else 0
                    avg_dist = dist_sum / atr_count if atr_count > 0 else 0
                    avg_atr_val = np.nanmean(atr_v[50:])
                    adr = avg_dist / avg_atr_val if avg_atr_val > 0 else 0
                    if adr < min_adr_fb:
                        continue
                    expectancy = (wr/100) * avg_mfe - (1-wr/100) * 1.0
                    composite = expectancy * np.sqrt(touches) * min(adr, 2.0)

                    results.append({
                        'ma_type': ma_type,
                        'period': period,
                        'touches': touches,
                        'wins': wins,
                        'wr_pct': wr,
                        'avg_mfe': avg_mfe,
                        'expectancy': expectancy,
                        'adr': adr,
                        'composite_score': composite,
                        'current_ma_value': float(ma_v[-1]),
                        'weak_signal': True,  # FALLBACK ile bulundu
                    })
                except Exception:
                    continue

        if results:
            print(f"  FALLBACK: {len(results)} MA bulundu (zayif sinyal)")

    return pd.DataFrame(results), df, inst_type


# === Ana fonksiyon ===

def analyze_ticker(ticker, csv_path=None, source='borsapy', portfolio=100000, risk_pct=1.0):
    ticker = ticker.upper().strip()

    # 1. MA listesi al (CSV varsa filtreleyerek, yoksa canli tarama)
    if csv_path and os.path.exists(csv_path):
        print(f"  CSV'den {ticker} filtreleniyor...")
        df_csv = pd.read_csv(csv_path)
        sub = df_csv[df_csv['ticker'] == ticker].copy()
        if sub.empty:
            print(f"  {ticker} CSV'de yok, canli taramaya gecilecek")
            csv_path = None
        else:
            top = sub.nlargest(15, 'composite_score')
            print(f"  CSV'de {len(sub)} kayit bulundu, top 15 alindi")

    scan_df = None
    if not csv_path:
        scan_df, price_df, inst_type = mini_scan(ticker, source=source)
        if scan_df is None:
            # Veri çekme tamamen başarısız
            print(f"  {ticker}: veri kaynağına erişilemedi")
            return None
        if scan_df.empty:
            # Veri var ama MA bulunamadı - yine de rapor üret (sadece fiyat bilgisi)
            print(f"  {ticker}: hicbir MA robust degil, ama fiyat raporu uretilecek")
            top = pd.DataFrame()  # boş
        else:
            top = scan_df.nlargest(15, 'composite_score')
    else:
        # CSV'de fiyat verisi yok, sadece güncel bilgi için çek
        price_df, inst_type = fetch_data(ticker, source=source)

    if price_df is None:
        return None

    return {
        'ticker': ticker,
        'top_mas': top,
        'price_df': price_df,
        'instrument_type': inst_type,
    }


def build_html_report(result, portfolio=100000, risk_pct=1.0, output='ticker_report.html'):
    ticker = result['ticker']
    df = result['price_df']
    top = result['top_mas']
    inst_type = result['instrument_type']

    # Güncel fiyat ve ATR
    current_price = float(df['Close'].iloc[-1])
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    tr = np.maximum.reduce([high[1:]-low[1:], np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])])
    atr_val = float(np.mean(tr[-14:]))
    lo20 = float(np.min(low[-20:]))
    hi20 = float(np.max(high[-20:]))

    # Multi-indicator
    rsi_now = float(rsi(df['Close']).iloc[-1])
    m, sig, hist = macd(df['Close'])
    macd_now = float(m.iloc[-1])
    macd_hist = float(hist.iloc[-1])

    # 20-gün hacim (TL)
    avg_volume_tl = float((df['Close'] * df['Volume']).tail(20).mean())

    inst_emoji = {'bist': '🇹🇷', 'crypto': '₿', 'forex': '💱', 'metal': '🥇'}.get(inst_type, '📊')

    html = ['<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">',
            f'<title>{ticker} Analizi</title>',
            '<style>',
            'body{font-family:-apple-system,sans-serif;background:#0a0e14;color:#e6e6e6;'
            'padding:20px;max-width:1300px;margin:auto;}',
            'h1{color:#5fb3ff;border-bottom:2px solid #5fb3ff;padding-bottom:8px;}',
            'h2{color:#7be2ff;margin-top:30px;}',
            '.box{background:#1a1f29;padding:15px;border-radius:8px;margin:15px 0;}',
            '.price-bar{font-size:28px;color:#7fc97f;}',
            '.metric{display:inline-block;margin-right:25px;}',
            '.metric strong{display:block;color:#7be2ff;font-size:12px;}',
            '.metric span{font-size:18px;font-weight:bold;}',
            'table{border-collapse:collapse;width:100%;font-family:monospace;font-size:13px;}',
            'th,td{padding:8px 10px;border-bottom:1px solid #2a2f39;text-align:left;}',
            'th{background:#2a2f39;color:#5fb3ff;}',
            '.touch{background:#2a3a4d;}.ready{background:#2a3a2a;}',
            '.long{color:#7fc97f;}.short{color:#ff8c69;}',
            '.warn{color:#ffc857;}',
            '</style></head><body>',
            f'<h1>{inst_emoji} {ticker} — Analiz Raporu</h1>',
            f'<p style="color:#888;">Üretildi: {datetime.now():%Y-%m-%d %H:%M} | '
            f'Enstrüman: {inst_type} | Veri: {len(df)} bar ({df.index[0].date()}'
            f' - {df.index[-1].date()})</p>',
            '<div class="box">',
            f'<div class="price-bar">Güncel Fiyat: <strong>{current_price:.4f}</strong></div><br>',
            f'<div class="metric"><strong>ATR (14)</strong><span>{atr_val:.4f}</span></div>',
            f'<div class="metric"><strong>ATR%</strong><span>{atr_val/current_price*100:.2f}%</span></div>',
            f'<div class="metric"><strong>20-gün Düşük</strong><span>{lo20:.4f}</span></div>',
            f'<div class="metric"><strong>20-gün Yüksek</strong><span>{hi20:.4f}</span></div>',
            f'<div class="metric"><strong>Range Pozisyon</strong><span>'
            f'{(current_price-lo20)/(hi20-lo20)*100:.0f}%</span></div>',
            '</div>',
            ]

    # Hacim (sadece BIST için anlamlı)
    if inst_type == 'bist':
        vol_cls = 'long' if avg_volume_tl > 50_000_000 else ('warn' if avg_volume_tl > 10_000_000 else 'short')
        vol_label = "Yüksek" if avg_volume_tl > 50_000_000 else ("Orta" if avg_volume_tl > 10_000_000 else "Düşük (dikkat)")
        html.append(f'<div class="box"><strong>20-gün Ort. Günlük Hacim:</strong> '
                    f'<span class="{vol_cls}">{avg_volume_tl/1e6:.1f}M TL ({vol_label})</span></div>')

    # Multi-indicator özet
    rsi_cls = 'short' if rsi_now > 70 else ('long' if rsi_now < 30 else '')
    rsi_label = 'Aşırı Alım' if rsi_now > 70 else ('Aşırı Satım' if rsi_now < 30 else 'Nötr')
    html.append(f'<div class="box"><h2>📊 İndikatör Durumu</h2>')
    html.append(f'<div class="metric"><strong>RSI(14)</strong>'
                f'<span class="{rsi_cls}">{rsi_now:.0f} ({rsi_label})</span></div>')
    html.append(f'<div class="metric"><strong>MACD</strong>'
                f'<span class="{"long" if macd_now > 0 else "short"}">{macd_now:.4f}</span></div>')
    html.append(f'<div class="metric"><strong>MACD Histogram</strong>'
                f'<span class="{"long" if macd_hist > 0 else "short"}">{macd_hist:+.4f}</span></div>')
    html.append('</div>')

    # Top MA tablosu
    html.append('<h2>🎯 En İyi Hareketli Ortalamalar (Top 15)</h2>')
    html.append('<table><tr><th>#</th><th>MA</th><th>Per</th><th>Değer</th>'
                '<th>Uzaklık (ATR)</th><th>Durum</th><th>Yön</th>'
                '<th>WR</th><th>Exp</th><th>Skor</th><th>Entry</th>'
                '<th>Stop</th><th>TP1</th><th>TP2</th><th>Lot</th></tr>')

    actionable = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        ma_val = row.get('current_ma_value', np.nan)
        if pd.isna(ma_val):
            continue

        dist = current_price - ma_val
        dist_atr = dist / atr_val if atr_val > 0 else 0

        abs_dist = abs(dist_atr)
        if abs_dist < 0.5:
            status, status_cls = 'TOUCH_ZONE', 'touch'
            action_text = '⚡ HAZIR'
        elif abs_dist < 2.0:
            status, status_cls = 'NEAR', ''
            action_text = 'Yakın'
        elif abs_dist < 3.5:
            status, status_cls = 'SETUP_READY', 'ready'
            action_text = '⏳ Touch bekle'
        else:
            status, status_cls = 'FAR', ''
            action_text = 'Çok uzak'

        side = 'LONG' if dist > 0 else 'SHORT'
        side_cls = 'long' if side == 'LONG' else 'short'

        # Trade parametreleri
        avg_mfe = row.get('avg_mfe', 5)
        if side == 'LONG':
            entry = ma_val
            stop = ma_val - 1.5 * atr_val
            tp1 = entry * (1 + avg_mfe * 0.5 / 100)
            tp2 = entry * (1 + avg_mfe / 100)
        else:
            entry = ma_val
            stop = ma_val + 1.5 * atr_val
            tp1 = entry * (1 - avg_mfe * 0.5 / 100)
            tp2 = entry * (1 - avg_mfe / 100)

        risk_per_lot = abs(entry - stop)
        n_lots = int((portfolio * risk_pct / 100) / risk_per_lot) if risk_per_lot > 0 else 0

        if status in ('TOUCH_ZONE', 'SETUP_READY'):
            actionable.append((status, row, side, entry, stop, tp1, tp2, n_lots))

        html.append(f'<tr class="{status_cls}"><td>{i}</td>')
        html.append(f'<td>{row["ma_type"]}</td><td>{row["period"]}</td>')
        html.append(f'<td>{ma_val:.4f}</td>')
        html.append(f'<td>{dist_atr:+.2f}</td>')
        html.append(f'<td>{action_text}</td>')
        html.append(f'<td class="{side_cls}">{side}</td>')
        html.append(f'<td>{row["wr_pct"]:.0f}%</td>')
        html.append(f'<td>{row["expectancy"]:+.2f}</td>')
        html.append(f'<td>{row["composite_score"]:.2f}</td>')
        html.append(f'<td>{entry:.4f}</td>')
        html.append(f'<td>{stop:.4f}</td>')
        html.append(f'<td>{tp1:.4f}</td>')
        html.append(f'<td>{tp2:.4f}</td>')
        html.append(f'<td>{n_lots:,}</td></tr>')
    html.append('</table>')

    # Aksiyon listesi
    if actionable:
        html.append('<h2>⚡ Aksiyon Alınabilir Setup\'lar</h2>')
        html.append('<div class="box">')
        for status, row, side, entry, stop, tp1, tp2, lots in actionable:
            side_cls = 'long' if side == 'LONG' else 'short'
            rr = abs(tp2 - entry) / max(abs(entry - stop), 0.0001)
            html.append(f'<p><strong>{row["ma_type"]} {row["period"]}</strong> — '
                        f'<span class="{side_cls}">{side}</span> | '
                        f'Durum: {status} | '
                        f'WR: {row["wr_pct"]:.0f}% | '
                        f'R:R: {rr:.2f} | '
                        f'Entry: {entry:.4f}, Stop: {stop:.4f}, TP2: {tp2:.4f}, Lot: {lots}</p>')
        html.append('</div>')
    else:
        html.append('<div class="box warn">Şu an aksiyon alınabilir setup yok. '
                    'Fiyat MA\'lardan uzak veya tam touch zone\'da değil.</div>')

    html.append('</body></html>')

    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"\n✓ Rapor: {output}")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', required=True, help='Hisse/sembol kodu')
    p.add_argument('--csv', type=str, default='', help='Tarama CSV (opsiyonel)')
    p.add_argument('--source', type=str, default='borsapy')
    p.add_argument('--portfolio', type=float, default=100000)
    p.add_argument('--risk_pct', type=float, default=1.0)
    p.add_argument('--scan-fresh', action='store_true', help='CSV varsa bile yeniden tara')
    p.add_argument('--output', type=str, default='')
    p.add_argument('--summary-text', type=str, default='', help='Telegram-ready özet metni için dosya yolu')
    args = p.parse_args()

    csv = '' if args.scan_fresh else args.csv
    result = analyze_ticker(args.ticker, csv_path=csv, source=args.source,
                            portfolio=args.portfolio, risk_pct=args.risk_pct)
    if not result:
        print(f"HATA: {args.ticker} için veri çekilemedi")
        sys.exit(1)

    output = args.output or f"{args.ticker.upper()}_analiz.html"
    build_html_report(result, portfolio=args.portfolio, risk_pct=args.risk_pct,
                       output=output)

    # Telegram özet dosyası (opsiyonel)
    if args.summary_text:
        summary = _build_telegram_summary(result, portfolio=args.portfolio,
                                          risk_pct=args.risk_pct)
        with open(args.summary_text, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"✓ Telegram özeti: {args.summary_text}")


def _build_telegram_summary(result, portfolio=100000, risk_pct=1.0):
    """Telegram-ready özet metni (Markdown)."""
    ticker = result['ticker']
    df = result['price_df']
    top = result['top_mas']
    inst_type = result['instrument_type']

    if df is None or len(df) == 0:
        return f"🎯 *{ticker}*\n\nVeri çekilemedi."

    current_price = float(df['Close'].iloc[-1])
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    tr = np.maximum.reduce([high[1:]-low[1:], np.abs(high[1:]-close[:-1]),
                            np.abs(low[1:]-close[:-1])])
    atr_val = float(np.mean(tr[-14:]))
    lo20 = float(np.min(low[-20:]))
    hi20 = float(np.max(high[-20:]))
    range_pos = (current_price - lo20) / (hi20 - lo20) * 100 if hi20 > lo20 else 50

    rsi_now = float(rsi(df['Close']).iloc[-1])
    m_l, m_s, m_h = macd(df['Close'])
    macd_hist = float(m_h.iloc[-1])

    avg_volume_tl = float((df['Close'] * df['Volume']).tail(20).mean())

    inst_emoji = {'bist': '🇹🇷', 'crypto': '₿', 'forex': '💱',
                  'metal': '🥇', 'bist_index': '📊'}.get(inst_type, '📊')

    lines = [f"{inst_emoji} *{ticker} Analizi*"]
    lines.append(f"Fiyat: *{current_price:.4f}* | ATR: {atr_val:.4f} ({atr_val/current_price*100:.2f}%)")
    lines.append(f"20-gün Range: *{range_pos:.0f}%* pozisyonda ({lo20:.2f} - {hi20:.2f})")

    # Hacim (sadece BIST için)
    if inst_type == 'bist':
        if avg_volume_tl > 50_000_000:
            vol_lbl = "Yüksek ✓"
        elif avg_volume_tl > 10_000_000:
            vol_lbl = "Orta"
        else:
            vol_lbl = "Düşük ⚠️"
        lines.append(f"Hacim: {avg_volume_tl/1e6:.0f}M TL ({vol_lbl})")

    # İndikatörler
    rsi_lbl = 'Aşırı Alım' if rsi_now > 70 else ('Aşırı Satım' if rsi_now < 30 else 'Nötr')
    macd_arrow = '↑' if macd_hist > 0 else '↓'
    lines.append(f"İnd: RSI {rsi_now:.0f} ({rsi_lbl}) | MACD {macd_arrow} {macd_hist:+.4f}")
    lines.append("")

    # Top 8 MA setup'ları
    if top is None or top.empty:
        lines.append("⚠️ *Hiçbir robust MA bulunamadı*")
        n_bars = len(df)
        if n_bars < 100:
            lines.append(f"Sebep: Yeni listelenen hisse ({n_bars} bar veri yeterli değil)")
            lines.append("En az 6 ay (~120 bar) veri toplandıktan sonra tekrar dene")
        else:
            lines.append(f"Veri: {n_bars} bar var ama MA'lar filtreyi gecemedi")
            lines.append("Sebep: Hisse yatay piyasada (trend yok) veya MA'lara yapisik")
        lines.append("")
        lines.append("📍 *Mevcut Durum*")
        # Detayli yorum
        lines.append(f"• Fiyat: {current_price:.4f} ({range_pos:.0f}% range pozisyonu)")
        if range_pos < 25:
            lines.append(f"• 20-gun range DIPTE - destek arayisi, bounce potansiyeli")
        elif range_pos > 75:
            lines.append(f"• 20-gun range ZIRVEDE - direnc bolgesi, dusus riski")
        else:
            lines.append(f"• 20-gun range ORTADA - yatay piyasa, beklemede kal")

        if rsi_now < 30:
            lines.append(f"• RSI {rsi_now:.0f} ASIRI SATIM - bounce ihtimali var")
        elif rsi_now > 70:
            lines.append(f"• RSI {rsi_now:.0f} ASIRI ALIM - dusus ihtimali")
        else:
            lines.append(f"• RSI {rsi_now:.0f} notr")

        macd_dir = "yukseliyor ↑" if macd_hist > 0 else "dususte ↓"
        lines.append(f"• MACD histogram {macd_dir} ({macd_hist:+.4f})")

        lines.append("")
        lines.append("💡 *Oneri:* Bu hisse su an *trade etmek icin uygun degil*.")
        lines.append("Trend olusana kadar bekle veya baska hisse incele.")

        return '\n'.join(lines)

    # Weak signal kontrolu - fallback ile bulunmus mu?
    weak_signals = top.get('weak_signal', pd.Series([False]*len(top))).any() if 'weak_signal' in top.columns else False
    if weak_signals:
        lines.append("⚠️ *Sadece ZAYIF sinyaller bulundu* (gevsek filtre)")
        lines.append("Bu MA'lar tam dogrulanmamis, dikkat!")
        lines.append("")

    lines.append("📊 *En İyi 8 MA Setup*")
    lines.append("```")
    # T = touch sayisi (kritik bilgi!)
    lines.append(f"{'#':<2} {'MA':<5} {'P':<4} {'T':<3} {'Mes':<6} {'Durum':<6} {'Yön':<5} {'WR':<5} {'Exp':<6}")
    lines.append("-" * 52)

    actionable = []
    sides_seen = set()
    suspicious_count = 0  # Az touch + yuksek WR olan setup sayisi

    for i, (_, row) in enumerate(top.head(8).iterrows(), 1):
        ma_val = row.get('current_ma_value', np.nan)
        if pd.isna(ma_val):
            continue

        dist = current_price - ma_val
        dist_atr = dist / atr_val if atr_val > 0 else 0
        abs_dist = abs(dist_atr)

        if abs_dist < 0.5:
            status, sk = 'TOUCH', 'touch'
        elif abs_dist < 2.0:
            status, sk = 'YAKIN', 'near'
        elif abs_dist < 3.5:
            status, sk = 'READY', 'ready'
        else:
            status, sk = 'UZAK', 'far'

        side = 'LONG' if dist > 0 else 'SHORT'
        sides_seen.add(side)
        wr = row['wr_pct']
        exp = row['expectancy']
        touches = int(row.get('touches', 0))

        # Şüpheli: az touch + yüksek WR (overfitting)
        is_suspicious = touches < 10 and wr >= 90
        if is_suspicious:
            suspicious_count += 1

        # Touch sayısını görsel ile işaretle
        touch_mark = '!' if is_suspicious else ' '

        lines.append(f"{i:<2} {row['ma_type']:<5} {int(row['period']):<4} "
                     f"{touches:<3}{touch_mark}{dist_atr:+5.2f} {status:<6} {side:<5} "
                     f"{wr:>3.0f}%  {exp:+5.2f}")

        if sk in ('touch', 'ready'):
            actionable.append((row, status, side, ma_val, dist_atr, touches, is_suspicious))

    lines.append("```")

    # Uyarı: Hem LONG hem SHORT setup varsa yatay piyasa
    if 'LONG' in sides_seen and 'SHORT' in sides_seen:
        lines.append("")
        lines.append("⚠️ *YATAY PIYASA UYARISI*")
        lines.append("Hem LONG hem SHORT setup var = fiyat MA'lar arasinda sikismis")
        lines.append("Trade etme, breakout bekle!")

    # Uyarı: Şüpheli setup'lar (az touch + yüksek WR)
    if suspicious_count >= 3:
        lines.append("")
        lines.append(f"⚠️ *DUSUK GUVEN UYARISI*")
        lines.append(f"{suspicious_count} setup'ta touch sayisi <10 ama WR >=90% (! isareti)")
        lines.append("Bu istatistiksel olarak anlamli degil, dikkat!")

    # Aksiyon alınabilir setup'lar
    if actionable:
        lines.append("")
        lines.append("⚡ *Aksiyon Setup'ları:*")
        for row, status, side, ma_val, dist_atr, touches, is_susp in actionable[:5]:
            emoji = '🟢' if side == 'LONG' else '🔴'
            susp_mark = ' ⚠️' if is_susp else ''
            mfe = row.get('avg_mfe', 5)
            if side == 'LONG':
                entry = ma_val
                stop = ma_val - 1.5 * atr_val
                tp2 = entry * (1 + mfe / 100)
            else:
                entry = ma_val
                stop = ma_val + 1.5 * atr_val
                tp2 = entry * (1 - mfe / 100)
            risk_per_lot = abs(entry - stop)
            n_lots = int((portfolio * risk_pct / 100) / risk_per_lot) if risk_per_lot > 0 else 0
            lines.append(f"{emoji} *{row['ma_type']} {int(row['period'])}* — {status} {side}{susp_mark}")
            lines.append(f"   Entry: `{entry:.4f}` | Stop: `{stop:.4f}` | TP2: `{tp2:.4f}` | Lot: {n_lots:,}")
            lines.append(f"   T={touches} | WR: {row['wr_pct']:.0f}% | Exp: {row['expectancy']:+.2f} | Skor: {row.get('composite_score', 0):.1f}")
    else:
        lines.append("")
        lines.append("⏸ Şu an aksiyon alınabilir setup yok — fiyat MA'lardan uzak veya yakın geçişte.")

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
