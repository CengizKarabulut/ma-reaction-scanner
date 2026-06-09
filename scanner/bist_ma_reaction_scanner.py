#!/usr/bin/env python3
"""
================================================================================
BIST MA TEPKİ PROFİLİ TARAYICI + WALK-FORWARD ANALİZ
================================================================================
Pine Script ma_reaction_profile_v5.pine'ın Python karşılığı.
BIST hisselerini tarar, tüm 140 MA kombinasyonunu (7 tip × 20 periyot) analiz eder,
walk-forward testi yapar ve raporlar üretir.

Kullanım örnekleri:
  python bist_ma_reaction_scanner.py --tickers ASELS,GARAN,THYAO
  python bist_ma_reaction_scanner.py --tickers ASELS --period 5y --report html
  python bist_ma_reaction_scanner.py --all-bist --period 3y --workers 8

Gereksinimler:
  pip install borsapy pandas numpy   # veya: pip install yfinance pandas numpy
  # opsiyonel: pip install borsapy  (alternatif veri kaynağı)
================================================================================
"""

import argparse
import sys
import os
import re
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# === VERİ KAYNAĞI ===
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import borsapy as bp
    HAS_BORSAPY = True
except ImportError:
    HAS_BORSAPY = False

# === MA HESAPLAMA FONKSİYONLARI (Pine'la birebir uyumlu) ===

def sma(series, period):
    return series.rolling(period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def wma(series, period):
    """Linear weighted MA — Pine ile aynı"""
    weights = np.arange(1, period + 1, dtype=float)
    w_sum = weights.sum()
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / w_sum, raw=True)

def vwma(close, volume, period):
    """Volume Weighted MA"""
    pv = close * volume
    return pv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)

def kama(series, period, fast=2, slow=30):
    """
    Kaufman Adaptive MA — Pine'ın f_kama() ile birebir uyumlu.
    Recursive: KAMA_t = KAMA_t-1 + SC × (Price - KAMA_t-1)
    """
    change = (series - series.shift(period)).abs()
    volatility = series.diff().abs().rolling(period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    fastest_sc = 2.0 / (fast + 1)
    slowest_sc = 2.0 / (slow + 1)
    sc = ((er * (fastest_sc - slowest_sc)) + slowest_sc) ** 2

    n = len(series)
    out = np.full(n, np.nan)
    src = series.values
    sc_v = sc.values
    started = False
    for i in range(n):
        if not started:
            if i >= period and not np.isnan(src[i]) and not np.isnan(sc_v[i]):
                out[i] = src[i]
                started = True
        else:
            if not np.isnan(sc_v[i]) and not np.isnan(out[i-1]):
                out[i] = out[i-1] + sc_v[i] * (src[i] - out[i-1])
            else:
                out[i] = out[i-1]
    return pd.Series(out, index=series.index)

def alma(series, period, offset=0.85, sigma=6):
    """Arnaud Legoux MA — Pine ile aynı (offset=0.85, sigma=6)"""
    m = offset * (period - 1)
    s = period / sigma
    indices = np.arange(period)
    weights = np.exp(-((indices - m) ** 2) / (2 * s * s))
    weights /= weights.sum()
    return series.rolling(period).apply(lambda x: np.dot(x, weights), raw=True)

def hma(series, period):
    """Hull MA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    half = max(1, period // 2)
    sqrt_p = max(1, int(np.sqrt(period)))
    wma_half = wma(series, half)
    wma_full = wma(series, period)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_p)

def true_range(high, low, close):
    """True Range — Wilder"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(high, low, close, period=14):
    """ATR — Wilder smoothing (RMA)"""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(high, low, close, period=14):
    """ADX — Pine'ın ta.adx() ile uyumlu"""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=high.index)
    tr = true_range(high, low, close)
    atr_v = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_v
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_v
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

# === MA TÜRÜ DİSPATCH ===

def compute_ma(ma_type, close, volume, period):
    if ma_type == 'SMA':   return sma(close, period)
    if ma_type == 'EMA':   return ema(close, period)
    if ma_type == 'WMA':   return wma(close, period)
    if ma_type == 'VWMA':  return vwma(close, volume, period)
    if ma_type == 'KAMA':  return kama(close, period)
    if ma_type == 'ALMA':  return alma(close, period)
    if ma_type == 'HMA':   return hma(close, period)
    raise ValueError(f"Bilinmeyen MA türü: {ma_type}")

# === REAKSİYON ANALİZİ (Pine'ın process_r() karşılığı) ===

def analyze_reactions(df, ma_series,
                     react_bars=5, react_pct=1.5,
                     atr_mult=0.2, adx_threshold=25,
                     separation_mult=2.0, breakthrough_bars=10):
    """
    Bir MA'nın tepki istatistiklerini hesapla — v6 metodolojisi.
    Pine'ın process_r() fonksiyonunun birebir Python karşılığı.

    v6 KAVRAMSAL DÜZELTMELER:
    - Pre-touch separation: bir touch sayılması için fiyatın önce MA'dan
      separation_mult × ATR uzaklaşması şart (kısa MA "sahte saygı" düzeltici)
    - Breakthrough count: fiyat MA'yı geçip breakthrough_bars boyunca
      öbür tarafta kalırsa "kırılım" sayılır

    Returns: dict {
        'trend_touches', 'trend_reacts',
        'range_touches', 'range_reacts',
        'mfe_sum', 'mae_sum',
        'brk_count'   # YENİ v6
    }
    """
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    atr_v = df['ATR'].values
    adx_v = df['ADX'].values
    ma = ma_series.values
    n = len(df)

    trend_t = trend_r = range_t = range_r = brk_count = 0
    mfe_sum = mae_sum = 0.0
    # v6.1: ADR (Average Distance Ratio) - MA'nin "yapisik" mi gercek destek/direnc mi
    dist_sum = 0.0
    atr_sum = 0.0
    valid_bars = 0

    # 1. PASS: Touch detection + Breakthrough tracking
    touches = []
    prev_in_zone = False
    was_far_enough = False  # v6: pre-separation şartı
    was_above = None        # v6: yön tracking
    bars_since_cross = 0    # v6: cross süresi

    for i in range(n):
        if np.isnan(ma[i]) or np.isnan(atr_v[i]):
            prev_in_zone = False
            continue
        zone = atr_mult * atr_v[i]
        in_zone = low[i] <= ma[i] + zone and high[i] >= ma[i] - zone

        # v6: Pre-touch separation şartı
        distance_from_ma = abs(close[i] - ma[i])
        if not in_zone and distance_from_ma > atr_v[i] * separation_mult:
            was_far_enough = True

        # Touch sadece pre-separation şartı sağlandıysa geçerli
        new_touch = in_zone and not prev_in_zone and was_far_enough
        if new_touch:
            is_trending = (not np.isnan(adx_v[i])) and (adx_v[i] > adx_threshold)
            touches.append((i, is_trending))
            if is_trending:
                trend_t += 1
            else:
                range_t += 1
            was_far_enough = False  # Reset
        prev_in_zone = in_zone

        # v6.1: ADR icin biriktirme - her gecerli bar
        if not np.isnan(close[i]) and not np.isnan(ma[i]) and not np.isnan(atr_v[i]):
            dist_sum += abs(close[i] - ma[i])
            atr_sum += atr_v[i]
            valid_bars += 1

        # v6: Breakthrough tespiti
        currently_above = close[i] > ma[i]
        if was_above is None:
            was_above = currently_above
        elif currently_above != was_above:
            bars_since_cross = 0
            was_above = currently_above
        else:
            bars_since_cross += 1
            if bars_since_cross == breakthrough_bars:
                brk_count += 1

    # 2. PASS: MFE/MAE analizi
    for touch_i, was_trending in touches:
        if touch_i + react_bars >= n or touch_i < 1:
            continue

        touch_price = close[touch_i]
        ma_prev = ma[touch_i - 1]
        ref_close = close[touch_i - 1]

        if np.isnan(ma_prev) or np.isnan(touch_price):
            continue

        from_above = ref_close > ma_prev

        if from_above:
            max_fav_price = high[touch_i + 1]
            max_adv_price = low[touch_i + 1]
        else:
            max_fav_price = low[touch_i + 1]
            max_adv_price = high[touch_i + 1]

        for fb in range(1, react_bars):
            idx = touch_i + fb
            if from_above:
                max_fav_price = max(max_fav_price, high[idx])
                max_adv_price = min(max_adv_price, low[idx])
            else:
                max_fav_price = min(max_fav_price, low[idx])
                max_adv_price = max(max_adv_price, high[idx])

        if from_above:
            move = (max_fav_price - touch_price) / touch_price * 100
            adverse = (touch_price - max_adv_price) / touch_price * 100
        else:
            move = (touch_price - max_fav_price) / touch_price * 100
            adverse = (max_adv_price - touch_price) / touch_price * 100

        mae_sum += adverse

        if move >= react_pct:
            mfe_sum += move
            if was_trending:
                trend_r += 1
            else:
                range_r += 1

    # v6.1: ADR hesapla
    avg_dist = dist_sum / valid_bars if valid_bars > 0 else 0
    avg_atr = atr_sum / valid_bars if valid_bars > 0 else 1
    adr = avg_dist / avg_atr if avg_atr > 0 else 0

    return {
        'trend_touches': trend_t,
        'trend_reacts': trend_r,
        'range_touches': range_t,
        'range_reacts': range_r,
        'mfe_sum': mfe_sum,
        'mae_sum': mae_sum,
        'brk_count': brk_count,
        'adr': adr,  # v6.1: yapisiklik metrigi
    }

# === METRİK HESAPLAMA ===

def compute_metrics(stats):
    """Raw stats'tan WR, Expectancy, Respect Ratio, Composite skor — v6."""
    tt = stats['trend_touches']
    tr = stats['trend_reacts']
    rt = stats['range_touches']
    rr = stats['range_reacts']
    brk = stats.get('brk_count', 0)
    total_t = tt + rt
    total_r = tr + rr

    if total_t == 0:
        return None

    wr = total_r / total_t
    avg_mfe = stats['mfe_sum'] / total_r if total_r > 0 else 0.0
    avg_mae = stats['mae_sum'] / total_t
    expectancy = wr * avg_mfe - (1 - wr) * avg_mae
    net_edge = avg_mfe - avg_mae  # v6: WR'siz spread
    # v6: respect ratio - MA gerçekten kırılmadan kalıyor mu?
    respect_ratio = max(0.0, 1.0 - brk / max(total_t, 1))
    # v6.1: ADR (Average Distance Ratio) - MA'nin gercek dest/dir mi yoksa yapisik mi
    adr = stats.get('adr', 1.0)
    # ADR'i clamp et: cok dusuk (yapisik) cezalandirir, cok yuksek capper
    adr_factor = max(0.3, min(adr, 2.0))
    # v6.1: composite skor = expectancy × √touches × respect_ratio × adr_factor
    composite_score = expectancy * np.sqrt(total_t) * respect_ratio * adr_factor

    trend_wr = (tr / tt * 100) if tt > 0 else np.nan
    range_wr = (rr / rt * 100) if rt > 0 else np.nan

    if total_t >= 100:
        grade = 'A+'
    elif total_t >= 50:
        grade = 'A'
    elif total_t >= 25:
        grade = 'B'
    else:
        grade = 'C'

    return {
        'touches': total_t,
        'reacts': total_r,
        'wr_pct': wr * 100,
        'avg_mfe': avg_mfe,
        'avg_mae': avg_mae,
        'expectancy': expectancy,
        'net_edge': net_edge,
        'breakthroughs': brk,
        'respect_ratio': respect_ratio,
        'adr': adr,  # v6.1
        'trend_wr_pct': trend_wr,
        'range_wr_pct': range_wr,
        'grade': grade,
        'composite_score': composite_score,
    }

# === WALK-FORWARD ANALİZİ ===

def walk_forward(df, ma_type, period, train_pct=0.7, **kwargs):
    """
    İlk %70 ile training stats, son %30 ile out-of-sample test.
    İki dönemin metriklerini ayrı döner.
    """
    split_idx = int(len(df) * train_pct)
    if split_idx < 100 or len(df) - split_idx < 50:
        return None, None  # Yetersiz veri

    ma_full = compute_ma(ma_type, df['Close'], df['Volume'], period)

    train_df = df.iloc[:split_idx]
    train_ma = ma_full.iloc[:split_idx]

    test_df = df.iloc[split_idx:].copy()
    # Test bölümünde ATR/ADX yeniden hesabı yerine zaten df'te olanı kullan
    test_ma = ma_full.iloc[split_idx:]

    train_stats = analyze_reactions(train_df, train_ma, **kwargs)
    test_stats = analyze_reactions(test_df, test_ma, **kwargs)

    return compute_metrics(train_stats), compute_metrics(test_stats)

# === HİSSE TARAMA ===

MA_TYPES = ['SMA', 'EMA', 'WMA', 'VWMA', 'KAMA', 'ALMA', 'HMA']
PERIODS = [3, 5, 8, 10, 13, 20, 21, 22, 34, 50, 55, 89, 100, 144, 200, 233, 250, 377, 610, 987]

def _period_to_start_date(period: str) -> str:
    """borsapy '3y' gibi periyotlari desteklemez, start_date'e cevir."""
    from datetime import datetime, timedelta
    period = period.lower().strip()
    # Hazir gun sayilari
    days_map = {
        '1ay': 30, '3ay': 90, '6ay': 180,
        '1y': 365, '2y': 730, '3y': 1095, '5y': 1825, '10y': 3650,
        'max': 365 * 15,
    }
    if period in days_map:
        days = days_map[period]
    else:
        # "Ny" formati - regex
        m = re.match(r'^(\d+)y$', period)
        if m:
            days = int(m.group(1)) * 365
        else:
            m = re.match(r'^(\d+)ay$', period)
            if m:
                days = int(m.group(1)) * 30
            else:
                days = 365  # Default 1 yil
    # 60 gun buffer (trading days icin)
    start = (datetime.now() - timedelta(days=days + 60)).strftime('%Y-%m-%d')
    return start


def fetch_data(ticker, period='3y', source='yfinance'):
    """Veri çek (borsapy veya yfinance)

    borsapy: TradingView/Paratic üzerinden direkt BIST verisi
    yfinance: Yahoo Finance (.IS suffix gerektirir, bazı hisselerde delisted hatası)
    """
    base_symbol = ticker.replace('.IS', '') if ticker.endswith('.IS') else ticker

    if source == 'borsapy' and HAS_BORSAPY:
        try:
            t = bp.Ticker(base_symbol)
            # borsapy "3y" gibi periyotlari desteklemez (default 30 bar doner)
            # Bu yuzden start_date kullaniyoruz
            start_date = _period_to_start_date(period)
            df = t.history(start=start_date)
            if df is None or df.empty:
                raise RuntimeError(f"borsapy bos veri dondu: {base_symbol}")
            if len(df) < 50:
                raise RuntimeError(f"borsapy yetersiz veri: {base_symbol} ({len(df)} bar)")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            if HAS_YFINANCE:
                print(f"  borsapy hatasi ({base_symbol}: {e}), yfinance fallback")
                source = 'yfinance'
            else:
                raise

    if source == 'yfinance' and HAS_YFINANCE:
        symbol = f"{base_symbol}.IS"
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df

    raise RuntimeError(f"Veri kaynagi '{source}' kullanilamiyor")

def scan_stock(ticker, period='3y', source='yfinance', interval='1d',
               min_touches=10,
               react_bars=5, react_pct=1.5, atr_mult=0.2, adx_threshold=25,
               separation_mult=2.5, breakthrough_bars=10,
               min_hma_period=20, min_adr=0.4,
               do_walk_forward=True):
    """Tek hisse için tüm 140 MA + walk-forward analizi"""
    try:
        df = fetch_data(ticker, period, source)
    except Exception as e:
        return None, f"Veri çekme hatası: {e}"

    if df is None or len(df) < 300:
        return None, f"Yetersiz veri ({0 if df is None else len(df)} bar)"

    # ATR ve ADX
    df['ATR'] = atr(df['High'], df['Low'], df['Close'], 14)
    df['ADX'] = adx(df['High'], df['Low'], df['Close'], 14)

    react_kwargs = dict(
        react_bars=react_bars, react_pct=react_pct,
        atr_mult=atr_mult, adx_threshold=adx_threshold,
        separation_mult=separation_mult, breakthrough_bars=breakthrough_bars,
    )

    results = []
    for ma_type in MA_TYPES:
        for ma_period in PERIODS:
            # v6.1: HMA icin minimum periyot filtresi (kisa HMA fiyata yapisik olur)
            if ma_type == 'HMA' and ma_period < min_hma_period:
                continue
            try:
                ma_series = compute_ma(ma_type, df['Close'], df['Volume'], ma_period)
                stats = analyze_reactions(df, ma_series, **react_kwargs)
                metrics = compute_metrics(stats)
                if metrics is None or metrics['touches'] < min_touches:
                    continue
                # v6.1: ADR (Avg Distance Ratio) filtresi - MA yapisiksa atla
                if metrics.get('adr', 1.0) < min_adr:
                    continue

                row = {
                    'ticker': ticker,
                    'ma_type': ma_type,
                    'period': ma_period,
                    **metrics,
                    'current_ma_value': float(ma_series.iloc[-1]) if not pd.isna(ma_series.iloc[-1]) else np.nan,
                    'current_close': float(df['Close'].iloc[-1]),
                }
                row['distance_pct'] = (row['current_close'] - row['current_ma_value']) / row['current_ma_value'] * 100 if not np.isnan(row['current_ma_value']) else np.nan

                # Walk-forward
                if do_walk_forward:
                    train_m, test_m = walk_forward(df, ma_type, ma_period, **react_kwargs)
                    row['wf_train_exp'] = train_m['expectancy'] if train_m else np.nan
                    row['wf_test_exp']  = test_m['expectancy']  if test_m  else np.nan
                    row['wf_train_wr']  = train_m['wr_pct']     if train_m else np.nan
                    row['wf_test_wr']   = test_m['wr_pct']      if test_m  else np.nan
                    # v6: Daha sıkı robust kriteri
                    # (1) İki dönemde de + expectancy
                    # (2) Test expectancy training'in en az %50'si
                    # (3) Test WR training'in en az %80'i
                    # (4) Min dokunma sayısı yeterli iki dönemde de
                    row['wf_robust'] = bool(train_m and test_m
                                            and train_m['expectancy'] > 0
                                            and test_m['expectancy'] > 0
                                            and test_m['expectancy'] >= 0.5 * train_m['expectancy']
                                            and test_m['wr_pct'] >= 0.8 * train_m['wr_pct']
                                            and test_m['touches'] >= 5)
                results.append(row)
            except Exception:
                continue

    if not results:
        return None, "Hiçbir MA min_touches'ı geçemedi"

    return pd.DataFrame(results), None

# === RAPORLAMA ===

def print_summary(combined_df, top_n=10):
    """Konsola özet bas"""
    print("\n" + "="*80)
    print("HİSSE BAŞI TOP MA'LAR (composite skor sıralı)")
    print("="*80)
    for ticker in combined_df['ticker'].unique():
        sub = combined_df[combined_df['ticker'] == ticker]
        print(f"\n▸ {ticker}  ({len(sub)} aday MA)")
        top = sub.nlargest(top_n, 'composite_score')[
            ['ma_type', 'period', 'touches', 'wr_pct', 'expectancy',
             'avg_mfe', 'avg_mae', 'breakthroughs', 'respect_ratio', 'grade',
             'wf_train_exp', 'wf_test_exp', 'wf_robust']
        ].copy()
        top.columns = ['MA', 'Per', 'Dok', 'WR%', 'Exp', 'MFE', 'MAE', 'Brk', 'Resp', 'Grade', 'WF-Tr', 'WF-Ts', 'WF-OK']
        for col in ['WR%','Exp','MFE','MAE','Resp','WF-Tr','WF-Ts']:
            top[col] = top[col].round(2)
        print(top.to_string(index=False))

    print("\n" + "="*80)
    print("BIST GENELİ — EN POPÜLER MA'LAR (her hissede top 10'a kaç kez girmiş)")
    print("="*80)
    top_per_stock = (
        combined_df.groupby('ticker', group_keys=False)
        .apply(lambda g: g.nlargest(top_n, 'composite_score'))
    )
    pop = (
        top_per_stock.groupby(['ma_type', 'period'])
        .size().reset_index(name='hisse_sayısı')
        .sort_values('hisse_sayısı', ascending=False).head(20)
    )
    print(pop.to_string(index=False))

    print("\n" + "="*80)
    print("WALK-FORWARD ROBUSTLUK")
    print("="*80)
    if 'wf_robust' in combined_df.columns:
        n_total = len(combined_df.dropna(subset=['wf_robust']))
        n_robust = combined_df['wf_robust'].sum()
        print(f"Robust MA sayısı: {n_robust:,} / {n_total:,} ({100*n_robust/n_total:.1f}%)")
        robust_only = combined_df[combined_df['wf_robust'] == True]
        if len(robust_only) > 0:
            print("\nEn iyi robust MA'lar (test set expectancy sıralı, top 15):")
            top_robust = robust_only.nlargest(15, 'wf_test_exp')[
                ['ticker','ma_type','period','wr_pct','expectancy',
                 'wf_train_exp','wf_test_exp','grade']
            ].copy()
            for col in ['wr_pct','expectancy','wf_train_exp','wf_test_exp']:
                top_robust[col] = top_robust[col].round(2)
            print(top_robust.to_string(index=False))

def save_html_report(combined_df, path='ma_scan_report.html'):
    """Renkli, sıralı HTML rapor"""
    style = """
    <style>
    body { font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee; padding: 20px; }
    h2 { color: #4ec9b0; border-bottom: 1px solid #444; padding-bottom: 8px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 30px; font-size: 12px; }
    th { background: #2d2d2d; padding: 6px; border: 1px solid #444; }
    td { padding: 5px; border: 1px solid #333; text-align: right; }
    td.text { text-align: left; }
    tr.robust td { background: #1a3d2e; }
    tr.top1 td { background: #2d5a4e; font-weight: bold; }
    .pos { color: #4ade80; }
    .neg { color: #f87171; }
    </style>
    """
    html = f"<html><head><meta charset='utf-8'>{style}</head><body>"
    html += f"<h1>BIST MA Tepki Profili — Tarama Raporu</h1>"
    html += f"<p>Oluşturulma: {datetime.now():%Y-%m-%d %H:%M}</p>"
    html += f"<p>Hisse sayısı: {combined_df['ticker'].nunique()} | Toplam aday MA: {len(combined_df):,}</p>"

    for ticker in combined_df['ticker'].unique():
        sub = combined_df[combined_df['ticker'] == ticker].nlargest(10, 'composite_score')
        html += f"<h2>{ticker}</h2><table><tr>"
        for col in ['Sıra','MA','Per','Dok','WR%','MFE','MAE','Exp','Brk(Resp%)','T%','Y%','Grade','Değer','Uzak%','WF-Tr','WF-Ts','Robust']:
            html += f"<th>{col}</th>"
        html += "</tr>"
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            cls = 'robust' if r.get('wf_robust') else ''
            if i == 1: cls = 'top1'
            html += f"<tr class='{cls}'>"
            html += f"<td>{i}</td><td class='text'>{r['ma_type']}</td><td>{r['period']}</td>"
            html += f"<td>{r['touches']}</td>"
            html += f"<td>{r['wr_pct']:.1f}</td>"
            html += f"<td>{r['avg_mfe']:.2f}</td>"
            html += f"<td>{r['avg_mae']:.2f}</td>"
            exp_cls = 'pos' if r['expectancy'] > 0 else 'neg'
            html += f"<td class='{exp_cls}'>{r['expectancy']:+.2f}</td>"
            # v6: avg_speed kaldırıldı, yerine Brk + Respect%
            brk = int(r.get('breakthroughs', 0))
            resp = r.get('respect_ratio', 0) * 100
            html += f"<td>{brk} ({resp:.0f}%)</td>"
            html += f"<td>{r['trend_wr_pct']:.0f}</td>" if not pd.isna(r['trend_wr_pct']) else "<td>—</td>"
            html += f"<td>{r['range_wr_pct']:.0f}</td>" if not pd.isna(r['range_wr_pct']) else "<td>—</td>"
            html += f"<td>{r['grade']}</td>"
            html += f"<td>{r['current_ma_value']:.2f}</td>"
            html += f"<td>{r['distance_pct']:+.1f}%</td>"
            html += f"<td>{r.get('wf_train_exp', np.nan):.2f}</td>" if not pd.isna(r.get('wf_train_exp', np.nan)) else "<td>—</td>"
            html += f"<td>{r.get('wf_test_exp', np.nan):.2f}</td>"  if not pd.isna(r.get('wf_test_exp', np.nan))  else "<td>—</td>"
            html += f"<td>{'✓' if r.get('wf_robust') else '✗'}</td>"
            html += "</tr>"
        html += "</table>"

    html += "</body></html>"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nHTML rapor: {path}")



# === ANA AKIŞ ===

def main():
    parser = argparse.ArgumentParser(
        description='BIST MA Tepki Profili Tarama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--tickers', type=str, default='ASELS,GARAN,THYAO,KCHOL,SISE',
                       help='Virgülle ayrı liste (ASELS,GARAN). --all-bist ile override.')
    parser.add_argument('--all-bist', action='store_true',
                       help='Endeks listesi kullan (--bist-list ile birlikte)')
    parser.add_argument('--bist-list', type=str, default='BIST_100',
                       help='Endeks adi: BIST_30, BIST_50, BIST_100, BIST_TUM, XBANK, XUTUM vs')
    parser.add_argument('--period', type=str, default='3y',
                       help='Veri periyodu (1y, 2y, 3y, 5y, max)')
    parser.add_argument('--source', type=str, default='borsapy',
                       choices=['yfinance', 'borsapy'],
                       help='Veri kaynağı: borsapy (TradingView, BIST için tavsiye) veya yfinance')
    parser.add_argument('--interval', type=str, default='1d',
                       choices=['1d', '1wk', '1mo'],
                       help='Zaman dilimi: 1d (gunluk), 1wk (haftalik), 1mo (aylik)')
    parser.add_argument('--min_touches', type=int, default=10)
    parser.add_argument('--min_hma_period', type=int, default=20,
                       help='HMA icin minimum periyot (kisa HMA fiyata yapisik olur)')
    parser.add_argument('--min_adr', type=float, default=0.4,
                       help='Minimum ADR (Avg Distance Ratio) - dusuk olursa MA yapisik kabul edilir')
    parser.add_argument('--react_bars', type=int, default=5)
    parser.add_argument('--react_pct', type=float, default=1.5)
    parser.add_argument('--atr_mult', type=float, default=0.2)
    parser.add_argument('--adx_threshold', type=float, default=25)
    parser.add_argument('--separation_mult', type=float, default=2.5,
                       help='v6: Pre-touch separation şartı (ATR çarpanı, kısa MA bias düzeltici)')
    parser.add_argument('--breakthrough_bars', type=int, default=10,
                       help='v6: Kırılım onay barı (respect ratio hesabı için)')
    parser.add_argument('--no_walk_forward', action='store_true',
                       help='Walk-forward analizini atla (3x hızlanır)')
    parser.add_argument('--output', type=str, default='ma_scan_results.csv')
    parser.add_argument('--report', type=str, choices=['none','html','both'],
                       default='both', help='Rapor formatı')
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1,
                       help='Paralel worker sayısı (yfinance rate limit nedeniyle 1 önerilir)')
    args = parser.parse_args()

    if not HAS_YFINANCE and not HAS_BORSAPY:
        print("HATA: borsapy veya yfinance gerekli. pip install borsapy")
        sys.exit(1)

    # Hisse listesini hazirla
    if args.all_bist:
        # tickers.py'dan dinamik endeks bileseni cek (borsapy varsa)
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from tickers import get_list
            tickers = get_list(args.bist_list)
            if not tickers:
                print(f"HATA: '{args.bist_list}' icin hisse listesi bos.")
                return
            print(f"Endeks listesi: {args.bist_list} -> {len(tickers)} hisse")
        except ImportError as e:
            print(f"HATA: tickers.py yuklenemedi: {e}")
            return
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    print(f"\nTarama başlıyor: {len(tickers)} hisse × {len(MA_TYPES)*len(PERIODS)} MA kombinasyonu = {len(tickers)*len(MA_TYPES)*len(PERIODS):,} aday")
    print(f"Parametreler: period={args.period}, react_bars={args.react_bars}, react_pct={args.react_pct}, atr_mult={args.atr_mult}")
    print(f"Walk-forward: {'KAPALI' if args.no_walk_forward else 'AÇIK (ilk %70 / son %30)'}\n")

    scan_kwargs = dict(
        period=args.period, source=args.source, interval=args.interval,
        min_touches=args.min_touches,
        react_bars=args.react_bars, react_pct=args.react_pct,
        atr_mult=args.atr_mult, adx_threshold=args.adx_threshold,
        separation_mult=args.separation_mult,
        breakthrough_bars=args.breakthrough_bars,
        min_hma_period=args.min_hma_period,
        min_adr=args.min_adr,
        do_walk_forward=not args.no_walk_forward,
    )

    all_results = []
    t_start = datetime.now()

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as exc:
            futures = {exc.submit(scan_stock, t, **scan_kwargs): t for t in tickers}
            for i, fut in enumerate(as_completed(futures), 1):
                ticker = futures[fut]
                try:
                    df, err = fut.result()
                    if df is not None:
                        all_results.append(df)
                        top1 = df.nlargest(1, 'composite_score').iloc[0]
                        print(f"[{i}/{len(tickers)}] ✓ {ticker} — top: {top1['ma_type']} {top1['period']} "
                              f"(skor={top1['composite_score']:.2f}, exp={top1['expectancy']:+.2f})")
                    else:
                        print(f"[{i}/{len(tickers)}] ✗ {ticker} — {err}")
                except Exception as e:
                    print(f"[{i}/{len(tickers)}] ✗ {ticker} — Exception: {e}")
    else:
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] {ticker}...", end=' ', flush=True)
            df, err = scan_stock(ticker, **scan_kwargs)
            if df is not None:
                all_results.append(df)
                top1 = df.nlargest(1, 'composite_score').iloc[0]
                print(f"✓ top: {top1['ma_type']} {top1['period']} (skor={top1['composite_score']:.2f}, exp={top1['expectancy']:+.2f})")
            else:
                print(f"✗ {err}")

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"\nTarama tamamlandı: {elapsed:.1f}s")

    if not all_results:
        print("Hiçbir hisse analiz edilemedi.")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"CSV: {args.output} ({len(combined):,} satır)")

    if args.report in ('html', 'both'):
        save_html_report(combined, path=args.output.replace('.csv', '.html'))
    if args.report in ('none',):
        pass
    else:
        print_summary(combined, top_n=args.top_n)


if __name__ == '__main__':
    main()
