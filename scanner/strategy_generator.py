#!/usr/bin/env python3
"""
Strategy Generator — MA Reaction Scanner sonuçlarından somut trade önerileri.

Her hisse için:
- Robust top N MA'larin listesi
- Mevcut fiyat ile her MA arasi mesafe (ATR cinsinden)
- Setup durumu: "trade fırsatı var" / "bekle" / "uzakta"
- Entry/Stop/TP1/TP2 sayılı önerisi
- Position sizing (portföy %'i)
- Risk-Reward ve Expected Value

Kullanim:
    python strategy_generator.py \\
        --csv reports/scan.csv \\
        --tickers ASELS,GARAN \\
        --portfolio 100000 \\
        --risk_pct 1.0 \\
        --output reports/strategies.html

veya tum robust hisseler:
    python strategy_generator.py --csv reports/scan.csv --all-robust
"""

import argparse
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Veri kaynaklari
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


def is_bist_index(symbol):
    """BIST endeks sembolu algilama (X ile baslar, 4+ karakter)."""
    s = symbol.upper().replace('.IS', '')
    return s.startswith('X') and len(s) >= 4


def fetch_current_data(ticker: str, source: str = 'borsapy') -> dict:
    """Hissenin guncel fiyatini ve ATR'sini cek. BIST endeksleri icin bp.Index() kullanir."""
    base = ticker.replace('.IS', '')
    is_index = is_bist_index(base)

    try:
        if source == 'borsapy' and HAS_BORSAPY:
            df = None
            if is_index and hasattr(bp, 'Index'):
                try:
                    df = bp.Index(base).history(period='3ay', interval='1d')
                except Exception:
                    df = None
                if df is None or df.empty:
                    try:
                        df = bp.Ticker(base).history(period='3ay', interval='1d')
                    except Exception:
                        df = None
            else:
                df = bp.Ticker(base).history(period='3ay', interval='1d')
        elif HAS_YFINANCE:
            symbol = f"^{base}" if is_index else f"{base}.IS"
            df = yf.download(symbol, period='2mo', progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
        else:
            return None

        if df is None or len(df) < 14:
            return None

        # Endekslerde Volume olmayabilir
        if 'Volume' not in df.columns or df['Volume'].isna().all():
            df['Volume'] = 1.0

        # ATR (14 period)
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        tr = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        ])
        atr = np.mean(tr[-14:])

        return {
            'price': float(close[-1]),
            'atr': float(atr),
            'high_20': float(np.max(high[-20:])),
            'low_20': float(np.min(low[-20:])),
            'volume_20': float(np.mean(df['Volume'].values[-20:])),
        }
    except Exception as e:
        print(f"  {ticker}: veri cekilemedi ({e})")
        return None


def evaluate_setup(price: float, ma_value: float, atr: float, separation_mult: float = 2.0) -> dict:
    """MA'ya gore setup durumunu degerlendir."""
    if pd.isna(ma_value) or ma_value <= 0:
        return {'status': 'INVALID', 'distance_atr': None, 'side': None}

    distance = price - ma_value
    distance_atr = distance / atr if atr > 0 else 0

    abs_dist_atr = abs(distance_atr)

    if abs_dist_atr < 0.5:
        status = 'TOUCH_ZONE'      # Su an touch zone'da, sinyal beklemeli
        action = 'WATCH'
    elif abs_dist_atr < separation_mult:
        status = 'NEAR'             # Yakin ama henuz uzakta degil
        action = 'WAIT_FOR_SEPARATION'
    elif abs_dist_atr < separation_mult * 1.5:
        status = 'SETUP_READY'      # 2-3 ATR uzakta, geri donus icin hazir
        action = 'WAIT_FOR_TOUCH'
    else:
        status = 'FAR'              # Cok uzakta, ya breakthrough oldu ya da yeni trend
        action = 'NO_TRADE'

    side = 'LONG' if distance < 0 else 'SHORT'

    return {
        'status': status,
        'action': action,
        'side': side,
        'distance_atr': distance_atr,
        'distance_pct': (distance / price * 100) if price > 0 else 0,
    }


def calculate_trade_params(price: float, ma_value: float, atr: float,
                          avg_mfe_pct: float, avg_mae_pct: float, wr: float,
                          portfolio: float, risk_pct: float = 1.0,
                          stop_atr_mult: float = 1.5) -> dict:
    """Trade'in entry/stop/TP/pozisyon parametrelerini hesapla.

    Varsayim: Long side. (BIST'te short sinirli.)
    Entry = MA value (touch noktasi)
    Stop = MA - 1.5 * ATR (asagi)
    TP1 = Entry + avg_MFE * 0.5
    TP2 = Entry + avg_MFE
    """
    # Setup yönü
    side = 'LONG' if price > ma_value else 'SHORT'  # Mevcut konuma göre öneri yönü

    # Long taraflı varsayım (BIST'te short pratikte zor)
    if side == 'LONG':
        # Fiyat MA'nin üstünde - geri dönüş sırasında MA touch'da AL
        entry = ma_value
        stop = ma_value - stop_atr_mult * atr
        tp1 = entry * (1 + avg_mfe_pct * 0.5 / 100)
        tp2 = entry * (1 + avg_mfe_pct / 100)
    else:
        # Fiyat MA'nin altinda - SHORT setup (bilgi amaclı, BIST'te dikkat)
        entry = ma_value
        stop = ma_value + stop_atr_mult * atr
        tp1 = entry * (1 - avg_mfe_pct * 0.5 / 100)
        tp2 = entry * (1 - avg_mfe_pct / 100)

    risk_per_lot = abs(entry - stop)
    if risk_per_lot <= 0:
        return None

    risk_budget = portfolio * (risk_pct / 100)
    n_lots = int(risk_budget / risk_per_lot)
    position_value = n_lots * entry
    position_pct = (position_value / portfolio * 100) if portfolio > 0 else 0

    # Expected Value (tek trade icin, TL cinsinden)
    avg_win = (tp1 + tp2) / 2 - entry  # Iki TP ortalaması
    avg_loss = entry - stop
    if side == 'SHORT':
        avg_win = entry - (tp1 + tp2) / 2
        avg_loss = stop - entry

    ev_per_lot = wr / 100 * avg_win - (1 - wr / 100) * avg_loss
    ev_total = ev_per_lot * n_lots

    return {
        'side': side,
        'entry': entry,
        'stop': stop,
        'tp1': tp1,
        'tp2': tp2,
        'risk_per_lot': risk_per_lot,
        'n_lots': n_lots,
        'position_value': position_value,
        'position_pct': position_pct,
        'risk_total': risk_per_lot * n_lots,
        'ev_per_lot': ev_per_lot,
        'ev_total': ev_total,
        'rr_to_tp1': abs(tp1 - entry) / risk_per_lot if risk_per_lot > 0 else 0,
        'rr_to_tp2': abs(tp2 - entry) / risk_per_lot if risk_per_lot > 0 else 0,
    }


def generate_html_report(strategies: list, portfolio: float, risk_pct: float, path: str):
    """HTML strateji raporu uret."""
    html = [
        '<!DOCTYPE html>',
        '<html lang="tr"><head><meta charset="utf-8">',
        '<title>MA Reaction Strategy Report</title>',
        '<style>',
        'body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;'
        'background:#0a0e14;color:#e6e6e6;padding:20px;max-width:1400px;margin:auto;}',
        'h1{color:#5fb3ff;border-bottom:2px solid #5fb3ff;padding-bottom:8px;}',
        'h2{color:#7be2ff;margin-top:30px;}',
        '.summary{background:#1a1f29;padding:15px;border-radius:8px;margin:15px 0;'
        'border-left:4px solid #5fb3ff;}',
        '.ticker-block{background:#1a1f29;padding:20px;border-radius:8px;margin:20px 0;}',
        '.status-WATCH{color:#ffc857;}.status-WAIT_FOR_SEPARATION{color:#888;}',
        '.status-WAIT_FOR_TOUCH{color:#7fc97f;}.status-NO_TRADE{color:#e74c3c;}',
        '.long-side{color:#7fc97f;font-weight:bold;}',
        '.short-side{color:#ff8c69;font-weight:bold;}',
        'table{border-collapse:collapse;width:100%;margin:10px 0;}',
        'th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #2a2f39;}',
        'th{background:#2a2f39;color:#5fb3ff;}',
        '.trade-table{font-family:Monaco,Consolas,monospace;font-size:13px;}',
        '.highlight{background:#2a3a4d;padding:10px;border-radius:4px;margin:8px 0;}',
        '.warning{color:#ffc857;font-style:italic;font-size:13px;margin-top:8px;}',
        '</style></head><body>',
        f'<h1>BIST MA Reaction Strategy Raporu</h1>',
        f'<div class="summary">',
        f'<strong>Tarih:</strong> {datetime.now():%Y-%m-%d %H:%M}<br>',
        f'<strong>Portföy:</strong> {portfolio:,.0f} TL | ',
        f'<strong>Trade başına risk:</strong> %{risk_pct} ({portfolio*risk_pct/100:,.0f} TL)<br>',
        f'<strong>Stratej sayısı:</strong> {len(strategies)} hisse</div>'
    ]

    for s in strategies:
        ticker = s['ticker']
        current = s['current']
        rows = s['ma_rows']

        if current is None:
            html.append(f'<div class="ticker-block"><h2>{ticker}</h2>')
            html.append('<p>Güncel veri çekilemedi.</p></div>')
            continue

        html.append(f'<div class="ticker-block">')
        html.append(f'<h2>{ticker} — Fiyat: {current["price"]:.2f} TL | '
                    f'ATR: {current["atr"]:.2f} TL ({current["atr"]/current["price"]*100:.1f}%) | '
                    f'20-gün range: {current["low_20"]:.2f} - {current["high_20"]:.2f}</h2>')

        # Setup'lar tablosu
        html.append('<table class="trade-table">')
        html.append('<tr><th>#</th><th>MA</th><th>Per</th><th>Değer</th>'
                    '<th>Uzak (ATR)</th><th>Durum</th><th>Yön</th><th>Etiket</th>'
                    '<th>WR</th><th>Exp</th><th>ADR</th><th>Setup</th></tr>')

        actionable = []
        ma_clusters = {'destek': [], 'direnc': []}  # Cluster icin (Cengiz: tum dosyalarda DESTEK/DIRENC)
        for i, row in enumerate(rows, 1):
            ma_val = row['current_ma_value']
            ma_type = row['ma_type']
            period = row['period']
            setup = evaluate_setup(current['price'], ma_val, current['atr'])
            side_class = 'long-side' if setup['side'] == 'LONG' else 'short-side'

            # YENI: DESTEK/DIRENC etiketi
            if ma_val < current['price']:
                etiket = 'DESTEK'
                etiket_cls = 'long-side'
                ma_clusters['destek'].append((ma_type, period, ma_val))
            else:
                etiket = 'DIRENC'
                etiket_cls = 'short-side'
                ma_clusters['direnc'].append((ma_type, period, ma_val))

            html.append(f'<tr>')
            html.append(f'<td>{i}</td>')
            html.append(f'<td>{ma_type}</td><td>{period}</td>')
            html.append(f'<td>{ma_val:.2f}</td>')
            html.append(f'<td>{setup["distance_atr"]:+.2f}</td>')
            html.append(f'<td class="status-{setup["action"]}">{setup["status"]}</td>')
            html.append(f'<td class="{side_class}">{setup["side"]}</td>')
            html.append(f'<td class="{etiket_cls}">{etiket}</td>')
            html.append(f'<td>{row["wr_pct"]:.0f}%</td>')
            html.append(f'<td>{row["expectancy"]:+.2f}</td>')
            html.append(f'<td>{row.get("adr", 0):.2f}</td>')
            html.append(f'<td class="status-{setup["action"]}">{setup["action"]}</td>')
            html.append(f'</tr>')

            if setup['action'] in ['WATCH', 'WAIT_FOR_TOUCH']:
                actionable.append((row, setup))

        html.append('</table>')

        # MA Kumeleri haritasi (Destek/Direnc) - Cengiz: tum dosyalarda olsun
        if ma_clusters['destek'] or ma_clusters['direnc']:
            html.append('<h3 style="color:#7be2ff;margin-top:15px;">🎯 MA Kümeleri (Destek/Direnç)</h3>')
            html.append('<div style="background:#0f1419;padding:10px;border-radius:6px;font-family:monospace;">')

            atr_v = current['atr']
            curr_price = current['price']

            def fmt(p):
                if curr_price < 10: return f"{p:.4f}"
                elif curr_price < 100: return f"{p:.3f}"
                else: return f"{p:.2f}"

            def cluster_mas(mas, threshold_atr=0.5):
                if not mas: return []
                sorted_mas = sorted(mas, key=lambda x: x[2])
                clusters = [[sorted_mas[0]]]
                for ma in sorted_mas[1:]:
                    if abs(ma[2] - clusters[-1][-1][2]) <= threshold_atr * atr_v:
                        clusters[-1].append(ma)
                    else:
                        clusters.append([ma])
                return clusters

            # Direnclar
            direnc_clusters = cluster_mas(ma_clusters['direnc'])
            for cl in sorted(direnc_clusters, key=lambda c: c[0][2]):
                vals = [m[2] for m in cl]
                tags = ', '.join(f"{m[0]}{m[1]}" for m in cl)
                if len(cl) > 1:
                    html.append(f'<div style="color:#ff8c69;">🔴 {fmt(min(vals))}-{fmt(max(vals))}: {tags} <strong>({len(cl)} MA cluster)</strong></div>')
                else:
                    html.append(f'<div style="color:#ff8c69;">🔴 {fmt(vals[0])}: {tags}</div>')

            html.append(f'<div style="color:#7be2ff;font-weight:bold;border-top:1px solid #5fb3ff;border-bottom:1px solid #5fb3ff;padding:4px 0;margin:8px 0;">━━ FIYAT: {fmt(curr_price)} ━━</div>')

            # Destekler (asagidan yukariya en yakin)
            destek_clusters = cluster_mas(ma_clusters['destek'])
            for cl in sorted(destek_clusters, key=lambda c: -c[0][2]):
                vals = [m[2] for m in cl]
                tags = ', '.join(f"{m[0]}{m[1]}" for m in cl)
                if len(cl) > 1:
                    html.append(f'<div style="color:#7fc97f;">🟢 {fmt(min(vals))}-{fmt(max(vals))}: {tags} <strong>({len(cl)} MA cluster)</strong></div>')
                else:
                    html.append(f'<div style="color:#7fc97f;">🟢 {fmt(vals[0])}: {tags}</div>')

            html.append('</div>')

        # Aksiyon alinabilecek setup'lar icin detay
        if actionable:
            html.append('<h3 style="color:#7fc97f;margin-top:20px;">⚡ Aksiyon Alınabilir Setup\'lar</h3>')
            for row, setup in actionable:
                ma_label = f"{row['ma_type']} {row['period']}"
                params = calculate_trade_params(
                    current['price'], row['current_ma_value'], current['atr'],
                    row['avg_mfe'], row['avg_mae'], row['wr_pct'],
                    portfolio, risk_pct
                )
                if params is None:
                    continue

                html.append(f'<div class="highlight">')
                html.append(f'<strong>{ma_label}</strong> — {setup["status"]} ({setup["action"]})<br>')
                html.append(f'WR: <strong>{row["wr_pct"]:.0f}%</strong> | '
                            f'avg_MFE: <strong>+{row["avg_mfe"]:.2f}%</strong> | '
                            f'avg_MAE: <strong>-{row["avg_mae"]:.2f}%</strong> | '
                            f'Touches: {row["touches"]}<br><br>')

                html.append('<table style="margin-top:8px;">')
                html.append(f'<tr><th>Parametre</th><th>Değer</th><th>Açıklama</th></tr>')
                html.append(f'<tr><td>Yön</td><td class="{("long-side" if params["side"]=="LONG" else "short-side")}">{params["side"]}</td>'
                            f'<td>Fiyatın MA\'ya göre konumu</td></tr>')
                html.append(f'<tr><td>Entry</td><td>{params["entry"]:.2f} TL</td>'
                            f'<td>MA seviyesi (touch noktası)</td></tr>')
                html.append(f'<tr><td>Stop</td><td>{params["stop"]:.2f} TL</td>'
                            f'<td>MA + 1.5 × ATR ({abs(params["entry"]-params["stop"])/params["entry"]*100:.2f}%)</td></tr>')
                html.append(f'<tr><td>TP1 (1/3 sat)</td><td>{params["tp1"]:.2f} TL</td>'
                            f'<td>avg_MFE × 50% (R:R = {params["rr_to_tp1"]:.2f}:1)</td></tr>')
                html.append(f'<tr><td>TP2 (1/3 sat)</td><td>{params["tp2"]:.2f} TL</td>'
                            f'<td>avg_MFE seviyesi (R:R = {params["rr_to_tp2"]:.2f}:1)</td></tr>')
                html.append(f'<tr><td>TP3 (son 1/3)</td><td>Trailing 2×ATR</td>'
                            f'<td>Her yeni high\'tan {2*current["atr"]:.2f} TL aşağı</td></tr>')
                html.append(f'<tr><td>Lot</td><td>{params["n_lots"]:,}</td>'
                            f'<td>Risk: {params["risk_total"]:,.0f} TL ({params["position_pct"]:.1f}% portföy)</td></tr>')
                html.append(f'<tr><td>Pozisyon</td><td>{params["position_value"]:,.0f} TL</td><td></td></tr>')
                html.append(f'<tr><td>Beklenen EV</td><td>{params["ev_total"]:,.0f} TL</td>'
                            f'<td>WR × avg_win − (1-WR) × avg_loss</td></tr>')
                html.append('</table>')
                html.append('</div>')
        else:
            html.append('<p class="warning">Şu an aksiyon alınabilir setup yok — fiyat MA\'lardan uzakta veya yakın geçiş bölgesinde.</p>')

        html.append('</div>')

    html.append('<div class="warning" style="margin-top:30px;padding:15px;background:#1a1f29;border-radius:8px;">')
    html.append('<strong>⚠️ Önemli notlar:</strong><br>')
    html.append('• Bu öneriler algoritmanın tarihsel istatistiklerine dayanıyor, gelecekteki performans garanti değil.<br>')
    html.append('• Slippage ve komisyon dahil edilmedi (her trade için %0.3-0.5 ek maliyet hesapla).<br>')
    html.append('• Bir hisseye birden fazla MA için aynı anda pozisyon açma (korelasyonlu risk).<br>')
    html.append('• Time stop: 10 bar (breakthrough_bars) geçti, TP1\'e ulaşmadıysa çık.<br>')
    html.append('• Önce paper trade ile 20-30 setup\'ı test et, sonra canlı paraya geç.')
    html.append('</div></body></html>')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"✓ HTML strateji raporu: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Scanner çıktısı CSV')
    parser.add_argument('--tickers', type=str, default='',
                       help='Virgülle ayrı tickerlar (boş = all-robust)')
    parser.add_argument('--all-robust', action='store_true',
                       help='Tüm robust hisseleri al (top 20)')
    parser.add_argument('--top_ma', type=int, default=5,
                       help='Her hisse için kaç top MA göster')
    parser.add_argument('--portfolio', type=float, default=100000)
    parser.add_argument('--risk_pct', type=float, default=1.0,
                       help='Trade başına risk (portföy %)')
    parser.add_argument('--source', type=str, default='borsapy',
                       choices=['borsapy', 'yfinance'])
    parser.add_argument('--output', type=str, default='strategies.html')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"HATA: CSV bulunamadı: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"CSV yüklendi: {len(df):,} satır, {df['ticker'].nunique()} hisse")

    # Ticker listesi belirleme
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    elif args.all_robust:
        if 'wf_robust' in df.columns:
            # En çok robust MA'ya sahip 20 hisse
            robust_counts = df[df['wf_robust'] == True].groupby('ticker').size()
            tickers = robust_counts.nlargest(20).index.tolist()
        else:
            tickers = df.nlargest(20, 'composite_score')['ticker'].unique().tolist()
    else:
        print("HATA: --tickers veya --all-robust gerekli")
        sys.exit(1)

    print(f"Hisseler: {len(tickers)} ({', '.join(tickers[:5])}...)")

    strategies = []
    for tk in tickers:
        sub = df[df['ticker'] == tk].copy()
        if 'wf_robust' in sub.columns:
            sub = sub[sub['wf_robust'] == True]
        if sub.empty:
            print(f"  {tk}: robust MA yok, atlanıyor")
            continue

        top_rows = sub.nlargest(args.top_ma, 'composite_score').to_dict('records')
        current = fetch_current_data(tk, source=args.source)

        strategies.append({
            'ticker': tk,
            'current': current,
            'ma_rows': top_rows,
        })
        print(f"  {tk}: {len(top_rows)} MA, "
              f"fiyat={current['price']:.2f}" if current else f"  {tk}: veri yok")

    generate_html_report(strategies, args.portfolio, args.risk_pct, args.output)
    print(f"\n✓ {len(strategies)} hisse için strateji raporu oluşturuldu")


if __name__ == '__main__':
    main()
