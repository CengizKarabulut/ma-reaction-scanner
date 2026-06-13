#!/usr/bin/env python3
"""
Backtest Simulator — Robust MA pattern'leri için gerçek trade simülasyonu.

Algoritmadaki istatistikler (WR, MFE, MAE) gerçek trade'i temsil etmez:
- Algoritma her touch'ı sayar
- Gerçek trade: position sizing, stop, TP1/TP2/trailing, slippage, komisyon dahil

Bu script tarihi veri üzerinde TAM strateji simülasyonu yapar.

Kullanim:
    python backtest_simulator.py \\
        --csv reports/scan.csv \\
        --tickers ASELS,BRSAN,GARAN \\
        --top_ma 3 \\
        --portfolio 100000 \\
        --commission 0.001 \\
        --slippage 0.001 \\
        --output reports/backtest.html
"""

import argparse
import os
import sys
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


# MA hesaplama fonksiyonlari (scanner ile birebir aynı)
def sma(s, p): return s.rolling(p).mean()
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def wma(s, p):
    weights = np.arange(1, p+1)
    return s.rolling(p).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
def hma(s, p):
    half = int(p/2)
    sqrt_p = int(np.sqrt(p))
    return wma(2*wma(s, half) - wma(s, p), sqrt_p)
def vwma(s, vol, p):
    return (s * vol).rolling(p).sum() / vol.rolling(p).sum()

def alma(s, p, offset=0.85, sigma=6):
    m = offset * (p - 1)
    sig = p / sigma
    w = np.array([np.exp(-((i - m) ** 2) / (2 * sig ** 2)) for i in range(p)])
    w /= w.sum()
    return s.rolling(p).apply(lambda x: np.dot(x, w), raw=True)

def kama(s, p, fast=2, slow=30):
    change = abs(s - s.shift(p))
    vol = (s.diff().abs()).rolling(p).sum()
    er = (change / vol).fillna(0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    out = pd.Series(index=s.index, dtype=float)
    out.iloc[p-1] = s.iloc[p-1]
    for i in range(p, len(s)):
        out.iloc[i] = out.iloc[i-1] + sc.iloc[i] * (s.iloc[i] - out.iloc[i-1])
    return out

MA_FUNCS = {'SMA': sma, 'EMA': ema, 'WMA': wma, 'HMA': hma, 'ALMA': alma}

def compute_ma(ma_type, close, volume, period):
    if ma_type == 'VWMA':
        return vwma(close, volume, period)
    elif ma_type == 'KAMA':
        return kama(close, period)
    elif ma_type in MA_FUNCS:
        return MA_FUNCS[ma_type](close, period)
    return None


def is_bist_index(symbol):
    """BIST endeks sembolu algilama (X ile baslar, 4+ karakter)."""
    s = symbol.upper().replace('.IS', '')
    return s.startswith('X') and len(s) >= 4


def fetch_data(ticker, period_days=1100, source='borsapy'):
    """3 yil veri cek (borsapy v0.10+). BIST endeksleri icin bp.Index() kullanir."""
    base = ticker.replace('.IS', '')
    is_index = is_bist_index(base)

    # period_days → borsapy period
    if period_days >= 1825:
        bp_period = '5y'
    elif period_days >= 1100:
        bp_period = '3y'
    elif period_days >= 365:
        bp_period = '1y'
    else:
        bp_period = f"{period_days}g"

    try:
        if source == 'borsapy' and HAS_BORSAPY:
            # Borsapy period stringi yerine start_date kullan (3y'yi anlamıyor)
            from datetime import datetime as _dt, timedelta as _td
            _start = (_dt.now() - _td(days=period_days + 60)).strftime('%Y-%m-%d')

            def _bp_h(obj):
                try:
                    return obj.history(start=_start, interval='1d')
                except TypeError:
                    return obj.history(period=bp_period, interval='1d')

            df = None
            if is_index and hasattr(bp, 'Index'):
                try:
                    df = _bp_h(bp.Index(base))
                except Exception:
                    df = None
                if df is None or df.empty:
                    try:
                        df = _bp_h(bp.Ticker(base))
                    except Exception:
                        df = None
            else:
                df = _bp_h(bp.Ticker(base))
        elif HAS_YFINANCE:
            symbol = f"^{base}" if is_index else f"{base}.IS"
            df = yf.download(symbol, period='3y', progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
        else:
            return None
        if df is None or len(df) < 100:
            return None
        if 'Volume' not in df.columns or df['Volume'].isna().all():
            df['Volume'] = 1.0
        return df
    except Exception as e:
        print(f"  {ticker}: hata {e}")
        return None


def compute_atr(df, period=14):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])
    atr = pd.Series(tr).rolling(period).mean()
    atr = pd.concat([pd.Series([np.nan]), atr]).reset_index(drop=True)
    atr.index = df.index
    return atr


def simulate_trades(df, ma, ma_type, period, ticker,
                    atr_mult=0.2, separation_mult=2.5, stop_atr_mult=1.5,
                    portfolio=100000, risk_pct=1.0,
                    commission=0.001, slippage=0.001,
                    max_hold_bars=10):
    """Bir MA için tüm trade'leri simule et.

    Strateji:
    - Pre-touch separation: fiyat MA'dan 2.5 ATR uzaklasmis olmali
    - Touch zone'a girdiğinde entry (yön = fiyatın MA'ya göre konumu)
    - Stop: MA + 1.5 ATR ters yönde
    - TP1: avg_MFE × 0.5 (1/3 sat)
    - TP2: avg_MFE (1/3 sat)
    - TP3: trailing 2×ATR (son 1/3)
    - Time stop: max_hold_bars
    """
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    atr = compute_atr(df).values
    ma_v = ma.values
    n = len(df)
    dates = df.index

    trades = []
    in_position = False
    was_far_enough = False
    bars_in_pos = 0
    entry_price = stop_price = tp1_price = tp2_price = 0
    side = None  # 'LONG' or 'SHORT'
    n_lots = 0
    pos_remaining = 0  # 1.0, 0.66, 0.33
    trailing_high = trailing_low = 0
    pending_sells = []  # [(price, lots), ...]

    # avg_MFE'yi ön hesapla (basit yaklaşım — son 60 bar üzerinden)
    # Production'da algoritma istatistiğini kullanırız ama burada self-contained
    rough_mfe_pct = 5.0  # default %5 hedef
    rough_mae_pct = 1.5  # default %1.5 stop

    for i in range(50, n - 1):
        price = close[i]
        ma_val = ma_v[i]
        atr_v = atr[i]

        if np.isnan(ma_val) or np.isnan(atr_v) or atr_v <= 0:
            continue

        if not in_position:
            # Pre-touch separation tracking
            dist = abs(price - ma_val)
            if dist > separation_mult * atr_v:
                was_far_enough = True

            # Touch detection
            in_zone = abs(price - ma_val) < atr_mult * atr_v
            if in_zone and was_far_enough:
                # Yön belirle: fiyatın MA'ya göre yaklaşma yönü
                # Önceki bar üstteyse şimdi alttan yaklaşmış (LONG bias)
                prev_above = close[i-1] > ma_v[i-1] if not np.isnan(ma_v[i-1]) else False
                side = 'LONG' if prev_above else 'SHORT'

                # Entry parametreleri
                slip = price * slippage
                if side == 'LONG':
                    entry_price = price + slip  # alış kötü
                    stop_price = ma_val - stop_atr_mult * atr_v
                    tp1_price = entry_price * (1 + rough_mfe_pct * 0.5 / 100)
                    tp2_price = entry_price * (1 + rough_mfe_pct / 100)
                else:
                    entry_price = price - slip
                    stop_price = ma_val + stop_atr_mult * atr_v
                    tp1_price = entry_price * (1 - rough_mfe_pct * 0.5 / 100)
                    tp2_price = entry_price * (1 - rough_mfe_pct / 100)

                risk_per_lot = abs(entry_price - stop_price)
                if risk_per_lot <= 0:
                    was_far_enough = False
                    continue
                n_lots = int((portfolio * risk_pct / 100) / risk_per_lot)
                if n_lots < 1:
                    was_far_enough = False
                    continue

                in_position = True
                bars_in_pos = 0
                pos_remaining = 1.0
                trailing_high = high[i] if side == 'LONG' else 0
                trailing_low = low[i] if side == 'SHORT' else float('inf')
                was_far_enough = False
                entry_date = dates[i]

        else:
            bars_in_pos += 1
            high_i = high[i]
            low_i = low[i]

            # Exit kontrolü (sırayla: stop → tp → time)
            exit_reason = None
            exit_price = None

            # Trailing güncelle (TP3 için)
            if side == 'LONG':
                if high_i > trailing_high:
                    trailing_high = high_i
                trail_stop = trailing_high - 2 * atr_v
            else:
                if low_i < trailing_low:
                    trailing_low = low_i
                trail_stop = trailing_low + 2 * atr_v

            # Stop kontrolü
            if side == 'LONG':
                if low_i <= stop_price:
                    exit_price = stop_price * (1 - slippage)
                    exit_reason = 'STOP'
                elif pos_remaining < 1.0 and low_i <= trail_stop:
                    # Trailing stop (sadece TP1 sonrası aktif)
                    exit_price = trail_stop * (1 - slippage)
                    exit_reason = 'TRAILING'
                elif pos_remaining > 0.5 and high_i >= tp1_price:
                    # TP1 partial exit
                    exit_price = tp1_price * (1 - slippage)
                    sell_lots = int(n_lots / 3)
                    pnl = (exit_price - entry_price) * sell_lots
                    commission_cost = (entry_price * sell_lots + exit_price * sell_lots) * commission
                    pnl -= commission_cost
                    trades.append({
                        'ticker': ticker, 'ma': f"{ma_type} {period}",
                        'entry_date': str(entry_date)[:10], 'exit_date': str(dates[i])[:10],
                        'side': side, 'reason': 'TP1',
                        'entry': entry_price, 'exit': exit_price,
                        'lots': sell_lots, 'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * sell_lots)) * 100,
                        'bars_held': bars_in_pos,
                    })
                    pos_remaining = 0.66
                    continue
                elif pos_remaining > 0.2 and high_i >= tp2_price:
                    # TP2 partial exit
                    exit_price = tp2_price * (1 - slippage)
                    sell_lots = int(n_lots / 3)
                    pnl = (exit_price - entry_price) * sell_lots
                    commission_cost = (entry_price * sell_lots + exit_price * sell_lots) * commission
                    pnl -= commission_cost
                    trades.append({
                        'ticker': ticker, 'ma': f"{ma_type} {period}",
                        'entry_date': str(entry_date)[:10], 'exit_date': str(dates[i])[:10],
                        'side': side, 'reason': 'TP2',
                        'entry': entry_price, 'exit': exit_price,
                        'lots': sell_lots, 'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * sell_lots)) * 100,
                        'bars_held': bars_in_pos,
                    })
                    pos_remaining = 0.33
                    continue

            else:  # SHORT
                if high_i >= stop_price:
                    exit_price = stop_price * (1 + slippage)
                    exit_reason = 'STOP'

            # Time stop
            if exit_reason is None and bars_in_pos >= max_hold_bars:
                exit_price = close[i] * (1 - slippage * (1 if side == 'LONG' else -1))
                exit_reason = 'TIME'

            if exit_reason:
                # Final exit (pozisyonun kalanı)
                remaining_lots = int(n_lots * pos_remaining)
                if side == 'LONG':
                    pnl = (exit_price - entry_price) * remaining_lots
                else:
                    pnl = (entry_price - exit_price) * remaining_lots
                commission_cost = (entry_price * remaining_lots + exit_price * remaining_lots) * commission
                pnl -= commission_cost
                trades.append({
                    'ticker': ticker, 'ma': f"{ma_type} {period}",
                    'entry_date': str(entry_date)[:10], 'exit_date': str(dates[i])[:10],
                    'side': side, 'reason': exit_reason,
                    'entry': entry_price, 'exit': exit_price,
                    'lots': remaining_lots, 'pnl': pnl,
                    'pnl_pct': (pnl / (entry_price * remaining_lots)) * 100 if remaining_lots > 0 else 0,
                    'bars_held': bars_in_pos,
                })
                in_position = False
                pos_remaining = 0

    return trades


def compute_stats(trades: list, portfolio: float) -> dict:
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    total_pnl = df['pnl'].sum()

    # Cumulative equity curve
    df_sorted = df.sort_values('exit_date')
    equity = portfolio + df_sorted['pnl'].cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    max_dd = drawdown.min()

    return {
        'n_trades': len(df),
        'n_wins': len(wins),
        'n_losses': len(losses),
        'win_rate': len(wins) / len(df) * 100 if len(df) > 0 else 0,
        'total_pnl': total_pnl,
        'total_return_pct': total_pnl / portfolio * 100,
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0,
        'max_drawdown_pct': max_dd,
        'avg_bars_held': df['bars_held'].mean(),
        'by_reason': df.groupby('reason').size().to_dict(),
    }


def generate_html(all_trades: list, stats_per_combo: dict, overall_stats: dict, portfolio: float, output: str, per_ticker_info: dict = None):
    html = [
        '<!DOCTYPE html>',
        '<html lang="tr"><head><meta charset="utf-8">',
        '<title>Backtest Simulator</title>',
        '<style>',
        'body{font-family:-apple-system,sans-serif;background:#0a0e14;color:#e6e6e6;'
        'padding:20px;max-width:1400px;margin:auto;}',
        'h1{color:#5fb3ff;border-bottom:2px solid #5fb3ff;padding-bottom:8px;}',
        'h2{color:#7be2ff;}',
        '.summary{background:#1a1f29;padding:20px;border-radius:8px;margin:15px 0;}',
        '.metric{display:inline-block;margin:8px 20px 8px 0;}',
        '.metric strong{display:block;color:#7be2ff;font-size:13px;}',
        '.metric span{font-size:24px;font-weight:bold;}',
        '.pos{color:#7fc97f;}.neg{color:#e74c3c;}.neut{color:#ffc857;}',
        'table{border-collapse:collapse;width:100%;font-family:monospace;font-size:12px;}',
        'th,td{padding:6px 10px;border-bottom:1px solid #2a2f39;}',
        'th{background:#2a2f39;color:#5fb3ff;}',
        '</style></head><body>',
        '<h1>Backtest Simulator Sonuçları</h1>',
        f'<div class="summary">',
        f'<div class="metric"><strong>Portföy</strong><span>{portfolio:,.0f} TL</span></div>',
    ]

    if overall_stats:
        s = overall_stats
        ret_class = 'pos' if s['total_return_pct'] > 0 else 'neg'
        dd_class = 'neg' if abs(s['max_drawdown_pct']) > 15 else 'neut' if abs(s['max_drawdown_pct']) > 5 else 'pos'
        wr_class = 'pos' if s['win_rate'] > 60 else 'neut' if s['win_rate'] > 40 else 'neg'
        pf_class = 'pos' if s['profit_factor'] > 1.5 else 'neut' if s['profit_factor'] > 1 else 'neg'

        html.extend([
            f'<div class="metric"><strong>Trade Sayısı</strong><span>{s["n_trades"]}</span></div>',
            f'<div class="metric"><strong>Win Rate</strong><span class="{wr_class}">{s["win_rate"]:.1f}%</span></div>',
            f'<div class="metric"><strong>Toplam Getiri</strong><span class="{ret_class}">{s["total_return_pct"]:+.2f}% ({s["total_pnl"]:+,.0f} TL)</span></div>',
            f'<div class="metric"><strong>Profit Factor</strong><span class="{pf_class}">{s["profit_factor"]:.2f}</span></div>',
            f'<div class="metric"><strong>Max Drawdown</strong><span class="{dd_class}">{s["max_drawdown_pct"]:.1f}%</span></div>',
            f'<div class="metric"><strong>Avg Bars Held</strong><span>{s["avg_bars_held"]:.0f}</span></div>',
        ])
    html.append('</div>')

    # Per-combo özet
    html.append('<h2>Kombinasyon Bazlı Performans</h2>')
    html.append('<table><tr><th>Hisse</th><th>MA</th><th>Trade#</th><th>WR</th><th>PnL TL</th>'
                '<th>Return%</th><th>PF</th><th>Max DD%</th></tr>')

    for combo, s in sorted(stats_per_combo.items(), key=lambda x: -x[1].get('total_pnl', 0)):
        ticker, ma = combo
        ret_cls = 'pos' if s['total_pnl'] > 0 else 'neg'
        html.append(f'<tr><td>{ticker}</td><td>{ma}</td>'
                    f'<td>{s["n_trades"]}</td>'
                    f'<td>{s["win_rate"]:.0f}%</td>'
                    f'<td class="{ret_cls}">{s["total_pnl"]:+,.0f}</td>'
                    f'<td class="{ret_cls}">{s["total_return_pct"]:+.2f}%</td>'
                    f'<td>{s["profit_factor"]:.2f}</td>'
                    f'<td>{s["max_drawdown_pct"]:.1f}%</td></tr>')
    html.append('</table>')

    # YENI: Her hisse icin ANLIK DURUM + MA CLUSTER (Cengiz: tum taramalarda olsun)
    if per_ticker_info:
        html.append('<h2>📊 Anlık Durum — Fiyat + MA Cluster (Destek/Direnç)</h2>')
        for tk, info in per_ticker_info.items():
            curr_p = info['price']
            atr_v = info['atr']
            mas = info['robust_mas']

            def fmt_p(p, ref=curr_p):
                if ref < 10: return f"{p:.4f}"
                elif ref < 100: return f"{p:.3f}"
                else: return f"{p:.2f}"

            html.append(f'<div style="background:#1a1f29;padding:12px;border-radius:6px;margin:12px 0;">')
            html.append(f'<h3 style="margin:0 0 8px 0;color:#7be2ff;">{tk} — Fiyat: {fmt_p(curr_p)} TL | ATR: {atr_v:.2f}</h3>')

            if mas and len(mas) >= 1:
                destek_mas = [m for m in mas if m['value'] < curr_p]
                direnc_mas = [m for m in mas if m['value'] >= curr_p]

                html.append('<div style="background:#0f1419;padding:8px;border-radius:4px;font-family:monospace;font-size:13px;">')

                # Cluster algoritmasi
                def cluster_local(items, threshold_atr=0.5):
                    if not items: return []
                    sorted_i = sorted(items, key=lambda x: x['value'])
                    clusters = [[sorted_i[0]]]
                    for i in sorted_i[1:]:
                        if abs(i['value'] - clusters[-1][-1]['value']) <= threshold_atr * atr_v:
                            clusters[-1].append(i)
                        else:
                            clusters.append([i])
                    return clusters

                # Direncler
                for cl in sorted(cluster_local(direnc_mas), key=lambda c: c[0]['value']):
                    vals = [m['value'] for m in cl]
                    tags = ', '.join(f"{m['ma_type']}{m['period']}" for m in cl)
                    if len(cl) > 1:
                        html.append(f'<div style="color:#ff8c69;">🔴 {fmt_p(min(vals))}-{fmt_p(max(vals))}: {tags} <strong>({len(cl)} MA)</strong></div>')
                    else:
                        html.append(f'<div style="color:#ff8c69;">🔴 {fmt_p(vals[0])}: {tags}</div>')

                html.append(f'<div style="color:#7be2ff;font-weight:bold;border-top:1px solid #5fb3ff;border-bottom:1px solid #5fb3ff;padding:3px 0;margin:5px 0;">━━ FIYAT: {fmt_p(curr_p)} ━━</div>')

                # Destekler
                for cl in sorted(cluster_local(destek_mas), key=lambda c: -c[0]['value']):
                    vals = [m['value'] for m in cl]
                    tags = ', '.join(f"{m['ma_type']}{m['period']}" for m in cl)
                    if len(cl) > 1:
                        html.append(f'<div style="color:#7fc97f;">🟢 {fmt_p(min(vals))}-{fmt_p(max(vals))}: {tags} <strong>({len(cl)} MA)</strong></div>')
                    else:
                        html.append(f'<div style="color:#7fc97f;">🟢 {fmt_p(vals[0])}: {tags}</div>')

                html.append('</div>')

            html.append('</div>')

    # Trade listesi
    if all_trades:
        html.append('<h2>Trade Geçmişi (son 50)</h2>')
        html.append('<table><tr><th>Hisse</th><th>MA</th><th>Giriş</th><th>Çıkış</th>'
                    '<th>Yön</th><th>Sebep</th><th>Lot</th><th>PnL</th><th>%</th></tr>')
        for t in all_trades[-50:]:
            cls = 'pos' if t['pnl'] > 0 else 'neg'
            html.append(f'<tr><td>{t["ticker"]}</td><td>{t["ma"]}</td>'
                        f'<td>{t["entry_date"]}</td><td>{t["exit_date"]}</td>'
                        f'<td>{t["side"]}</td><td>{t["reason"]}</td>'
                        f'<td>{t["lots"]}</td>'
                        f'<td class="{cls}">{t["pnl"]:+,.0f}</td>'
                        f'<td class="{cls}">{t["pnl_pct"]:+.2f}%</td></tr>')
        html.append('</table>')

    html.append('</body></html>')

    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"✓ Backtest raporu: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Scanner CSV (robust MA listesi için)')
    p.add_argument('--tickers', type=str, default='')
    p.add_argument('--all-robust', action='store_true')
    p.add_argument('--top_ma', type=int, default=3)
    p.add_argument('--portfolio', type=float, default=100000)
    p.add_argument('--risk_pct', type=float, default=1.0)
    p.add_argument('--commission', type=float, default=0.001)
    p.add_argument('--slippage', type=float, default=0.001)
    p.add_argument('--source', type=str, default='borsapy')
    p.add_argument('--output', type=str, default='backtest.html')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"CSV: {len(df)} satir, {df['ticker'].nunique()} hisse")

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    elif args.all_robust:
        if 'wf_robust' in df.columns:
            tickers = df[df['wf_robust']==True].groupby('ticker').size().nlargest(10).index.tolist()
        else:
            tickers = df.nlargest(10, 'composite_score')['ticker'].unique().tolist()
    else:
        print("HATA: --tickers veya --all-robust")
        sys.exit(1)

    print(f"Backtest hisseleri: {tickers}")

    all_trades = []
    stats_per_combo = {}
    per_ticker_info = {}  # YENI: her hisse icin fiyat + robust MA degerleri (cluster icin)

    for tk in tickers:
        sub = df[df['ticker'] == tk]
        if 'wf_robust' in sub.columns:
            sub = sub[sub['wf_robust'] == True]
        if sub.empty:
            print(f"  {tk}: robust MA yok")
            continue
        top = sub.nlargest(args.top_ma, 'composite_score')

        price_df = fetch_data(tk, source=args.source)
        if price_df is None:
            continue

        # YENI: ANLIK FIYAT + ATR + Robust MA degerleri
        curr_price = float(price_df['Close'].iloc[-1])
        atr_s = compute_atr(price_df)
        curr_atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else curr_price * 0.02
        all_robust_mas = []

        for _, row in top.iterrows():
            ma_type = row['ma_type']
            period = int(row['period'])
            ma = compute_ma(ma_type, price_df['Close'], price_df['Volume'], period)
            if ma is None:
                continue

            # YENI: Bu MA'nin SON degeri (cluster icin)
            ma_last = float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else None
            if ma_last is not None:
                all_robust_mas.append({
                    'ma_type': ma_type,
                    'period': period,
                    'value': ma_last,
                    'wr_pct': row['wr_pct'],
                    'touches': int(row.get('touches', 0)),
                })

            trades = simulate_trades(
                price_df, ma, ma_type, period, tk,
                portfolio=args.portfolio, risk_pct=args.risk_pct,
                commission=args.commission, slippage=args.slippage,
            )
            if trades:
                all_trades.extend(trades)
                stats_per_combo[(tk, f"{ma_type} {period}")] = compute_stats(trades, args.portfolio)
                print(f"  {tk} {ma_type}{period}: {len(trades)} trade, "
                      f"PnL={sum(t['pnl'] for t in trades):+,.0f}")

        # Hisse seviyesi anlik bilgi
        per_ticker_info[tk] = {
            'price': curr_price,
            'atr': curr_atr,
            'robust_mas': all_robust_mas,
        }

    overall = compute_stats(all_trades, args.portfolio)
    generate_html(all_trades, stats_per_combo, overall, args.portfolio, args.output, per_ticker_info)
    print(f"\n✓ Toplam {len(all_trades)} trade, Net PnL: {overall.get('total_pnl', 0):+,.0f} TL")


if __name__ == '__main__':
    main()
