"""
BIST Panorama Raporu — TÜM hisseler için tam tablo.

Scanner sadece "robust olan" hisseleri vurguluyor. Panorama tüm hisseleri
göstererek (yakın/uzak ayrımı yapmadan) genel BIST trendi sunar.

Her hisse için:
- Mevcut fiyat
- En yakın 5 MA (yukarı/aşağı yön ayırarak)
- Her MA'nın güven puanı (Wilson Score)
- Sektör bilgisi
- DESTEK/DIRENC bölgeleri
- Entry/Stop/Target önerisi (cluster bazlı)

Kullanım:
    # CSV'den (scanner çıktısı)
    python panorama_report.py --csv reports/scan_2026-06-13_1d.csv \\
        --output reports/panorama_2026-06-13.html

    # Filtreleme
    python panorama_report.py --csv ... --min-confidence 50 --top-stocks 50
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
try:
    from sector_resolver import get_sector_info
    HAS_SECTOR = True
except ImportError:
    HAS_SECTOR = False


def compute_ma_proximity(df_stock: pd.DataFrame) -> pd.DataFrame:
    """Her hisse için MA'ların mevcut fiyata uzaklığı."""
    if df_stock.empty or 'current_close' not in df_stock.columns:
        return df_stock

    df = df_stock.copy()
    df['distance_pct'] = ((df['current_ma_value'] - df['current_close']) / df['current_close'] * 100)
    df['abs_distance_pct'] = df['distance_pct'].abs()
    # Yön: positive=DIRENC (MA üzerinde), negative=DESTEK
    df['direction'] = df['distance_pct'].apply(lambda x: '🔴 DIRENC' if x > 0 else '🟢 DESTEK')
    return df


def find_ma_clusters(df_stock: pd.DataFrame, threshold_pct: float = 1.5) -> list:
    """Birbirine yakın MA'ları kümele.

    threshold_pct: %1.5 içinde olan MA'lar tek küme sayılır.
    Returns: [(center_value, [ma_rows], 'destek'/'direnc'), ...]
    """
    if df_stock.empty or 'current_ma_value' not in df_stock.columns:
        return []

    df = df_stock.dropna(subset=['current_ma_value']).copy()
    if df.empty:
        return []
    df = df.sort_values('current_ma_value')
    current_price = df['current_close'].iloc[0] if 'current_close' in df.columns else None

    clusters = []
    current_cluster = []
    cluster_start_val = None

    for _, row in df.iterrows():
        val = row['current_ma_value']
        if cluster_start_val is None:
            cluster_start_val = val
            current_cluster = [row]
        elif (val - cluster_start_val) / max(cluster_start_val, 1e-9) * 100 <= threshold_pct:
            current_cluster.append(row)
        else:
            # Yeni küme başlat
            if current_cluster:
                center = np.mean([r['current_ma_value'] for r in current_cluster])
                direction = 'direnc' if current_price and center > current_price else 'destek'
                clusters.append({
                    'center': center,
                    'mas': current_cluster,
                    'direction': direction,
                    'strength': len(current_cluster),
                })
            cluster_start_val = val
            current_cluster = [row]

    # Son küme
    if current_cluster:
        center = np.mean([r['current_ma_value'] for r in current_cluster])
        direction = 'direnc' if current_price and center > current_price else 'destek'
        clusters.append({
            'center': center,
            'mas': current_cluster,
            'direction': direction,
            'strength': len(current_cluster),
        })

    return clusters


def stock_panorama(df_stock: pd.DataFrame) -> dict:
    """Tek hisse için tam panorama özeti."""
    if df_stock.empty:
        return {}

    df = compute_ma_proximity(df_stock)
    if 'current_close' not in df.columns:
        return {}

    current_price = df['current_close'].iloc[0]
    ticker = df['ticker'].iloc[0]

    # Mevcut fiyatın altındaki en yakın 5 DESTEK (MA < fiyat)
    supports = df[df['current_ma_value'] < current_price].nsmallest(5, 'abs_distance_pct')
    # Mevcut fiyatın üzerindeki en yakın 5 DIRENC (MA > fiyat)
    resistances = df[df['current_ma_value'] > current_price].nsmallest(5, 'abs_distance_pct')

    # Cluster bul
    clusters = find_ma_clusters(df)

    # En güçlü cluster'lar (destek + direnç)
    support_clusters = [c for c in clusters if c['direction'] == 'destek']
    resistance_clusters = [c for c in clusters if c['direction'] == 'direnc']
    support_clusters.sort(key=lambda c: abs(c['center'] - current_price))
    resistance_clusters.sort(key=lambda c: abs(c['center'] - current_price))

    # Robust MA sayısı
    n_robust = int(df['wf_robust'].sum()) if 'wf_robust' in df.columns else 0
    n_total = len(df)

    # Genel trend: destek mi direnç mi baskın?
    n_support_robust = int(df[(df['wf_robust'] == True) &
                                 (df['current_ma_value'] < current_price)].shape[0]) \
        if 'wf_robust' in df.columns else 0
    n_resistance_robust = int(df[(df['wf_robust'] == True) &
                                    (df['current_ma_value'] > current_price)].shape[0]) \
        if 'wf_robust' in df.columns else 0

    # Sektör
    sector = 'Unknown'
    if HAS_SECTOR:
        info = get_sector_info(ticker)
        if info:
            sector = info.get('sector', 'Unknown')

    return {
        'ticker': ticker,
        'price': current_price,
        'sector': sector,
        'n_robust': n_robust,
        'n_total': n_total,
        'n_support_robust': n_support_robust,
        'n_resistance_robust': n_resistance_robust,
        'top_supports': supports,
        'top_resistances': resistances,
        'support_clusters': support_clusters[:3],
        'resistance_clusters': resistance_clusters[:3],
    }


def generate_panorama_html(panoramas: list, output_path: str):
    """Tüm hisseler için tam HTML rapor."""
    css = """
    <style>
        body { font-family: -apple-system, sans-serif; background: #0a0e14; color: #e6e6e6;
               padding: 20px; max-width: 1600px; margin: auto; }
        h1, h2 { color: #5fb3ff; border-bottom: 1px solid #2a2f39; padding-bottom: 8px; }
        h2 { margin-top: 24px; }
        .stock-card { background: #1a1f29; border-radius: 12px; padding: 16px;
                       margin: 16px 0; border-left: 4px solid #5fb3ff; }
        .stock-header { display: flex; justify-content: space-between; align-items: center;
                         flex-wrap: wrap; gap: 12px; }
        .ticker-name { font-size: 22px; font-weight: bold; color: #fff; }
        .price-tag { font-size: 18px; color: #5fb3ff; font-weight: bold; }
        .sector { color: #8b95a5; font-size: 13px; }
        .balance { display: flex; gap: 16px; margin: 12px 0; }
        .balance-box { background: #0f1419; padding: 8px 16px; border-radius: 6px;
                       border-left: 3px solid #2a2f39; }
        .balance-box.support { border-left-color: #7fc97f; }
        .balance-box.resistance { border-left-color: #ff8c69; }
        .balance-label { color: #8b95a5; font-size: 11px; }
        .balance-value { color: #fff; font-size: 18px; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0;
                background: #0f1419; border-radius: 6px; overflow: hidden; font-size: 13px; }
        th { background: #2a2f39; color: #5fb3ff; padding: 8px; text-align: left; }
        td { padding: 6px 8px; border-bottom: 1px solid #2a2f39; }
        .support-cell { color: #7fc97f; }
        .resistance-cell { color: #ff8c69; }
        .robust-tag { background: #1a3f2f; color: #7fc97f; padding: 2px 8px;
                      border-radius: 4px; font-size: 11px; font-weight: bold; }
        .cluster-box { background: #0a0e14; border: 1px solid #2a2f39; padding: 10px;
                       border-radius: 6px; margin: 6px 0; }
        .cluster-strong { border-color: #5fb3ff; }
        .summary-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                          gap: 12px; margin: 16px 0; }
        .stat-box { background: #1a1f29; padding: 16px; border-radius: 8px; }
        .stat-label { color: #8b95a5; font-size: 12px; text-transform: uppercase; }
        .stat-value { color: #fff; font-size: 22px; font-weight: bold; margin-top: 4px; }
    </style>
    """

    # Üst seviye istatistik
    n_stocks = len(panoramas)
    n_support_dominant = sum(1 for p in panoramas
                              if p.get('n_support_robust', 0) > p.get('n_resistance_robust', 0))
    n_resistance_dominant = sum(1 for p in panoramas
                                 if p.get('n_resistance_robust', 0) > p.get('n_support_robust', 0))
    n_no_robust = sum(1 for p in panoramas if p.get('n_robust', 0) == 0)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BIST Panorama — Tüm Hisseler</title>
{css}</head><body>
<h1>🌐 BIST Panorama Raporu</h1>
<p>Oluşturuldu: {datetime.now():%Y-%m-%d %H:%M} | {n_stocks} hisse</p>

<h2>📊 Genel Özet</h2>
<div class="summary-stats">
    <div class="stat-box">
        <div class="stat-label">Toplam Hisse</div>
        <div class="stat-value">{n_stocks}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">🟢 Destek Baskın</div>
        <div class="stat-value" style="color:#7fc97f">{n_support_dominant}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">🔴 Direnç Baskın</div>
        <div class="stat-value" style="color:#ff8c69">{n_resistance_dominant}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Robust MA Yok</div>
        <div class="stat-value" style="color:#8b95a5">{n_no_robust}</div>
    </div>
</div>

<h2>📋 Her Hisse İçin Detay</h2>
"""

    # Hisseleri robust sayısına göre sırala
    panoramas.sort(key=lambda p: -p.get('n_robust', 0))

    for p in panoramas:
        if not p:
            continue
        ticker = p['ticker']
        price = p.get('price', 0)
        sector = p.get('sector', '?')

        # Fiyat formatı
        if price < 10:
            pf = f"{price:.4f}"
        elif price < 100:
            pf = f"{price:.3f}"
        else:
            pf = f"{price:.2f}"

        n_robust = p.get('n_robust', 0)
        n_sup = p.get('n_support_robust', 0)
        n_res = p.get('n_resistance_robust', 0)

        html += f"""
<div class="stock-card">
    <div class="stock-header">
        <div>
            <span class="ticker-name">{ticker}</span>
            <span class="sector"> • {sector}</span>
        </div>
        <div class="price-tag">{pf} TL</div>
    </div>
    <div class="balance">
        <div class="balance-box support">
            <div class="balance-label">🟢 Robust DESTEK</div>
            <div class="balance-value">{n_sup}</div>
        </div>
        <div class="balance-box resistance">
            <div class="balance-label">🔴 Robust DIRENC</div>
            <div class="balance-value">{n_res}</div>
        </div>
        <div class="balance-box">
            <div class="balance-label">Toplam Robust</div>
            <div class="balance-value">{n_robust}</div>
        </div>
    </div>
"""

        # Yakın destek/direnç tabloları
        supports = p.get('top_supports', pd.DataFrame())
        resistances = p.get('top_resistances', pd.DataFrame())

        if not supports.empty or not resistances.empty:
            html += '<div style="display:flex; gap:16px; flex-wrap:wrap;">'

            if not supports.empty:
                html += '<div style="flex:1; min-width:300px;"><h3 style="color:#7fc97f">🟢 En Yakın 5 DESTEK</h3>'
                html += '<table><tr><th>MA</th><th>Per</th><th>Değer</th><th>Uzaklık</th><th>WR</th><th>Robust</th></tr>'
                for _, r in supports.iterrows():
                    is_robust = r.get('wf_robust', False)
                    robust_tag = '<span class="robust-tag">✓</span>' if is_robust else ''
                    wr = r.get('wr_pct', 0)
                    ma_val = r.get('current_ma_value', 0)
                    val_str = f"{ma_val:.4f}" if price < 10 else f"{ma_val:.2f}"
                    html += f'<tr class="support-cell"><td>{r["ma_type"]}</td>'
                    html += f'<td>{int(r["period"])}</td><td>{val_str}</td>'
                    html += f'<td>{r.get("distance_pct", 0):+.2f}%</td>'
                    html += f'<td>{wr:.0f}%</td><td>{robust_tag}</td></tr>'
                html += '</table></div>'

            if not resistances.empty:
                html += '<div style="flex:1; min-width:300px;"><h3 style="color:#ff8c69">🔴 En Yakın 5 DIRENC</h3>'
                html += '<table><tr><th>MA</th><th>Per</th><th>Değer</th><th>Uzaklık</th><th>WR</th><th>Robust</th></tr>'
                for _, r in resistances.iterrows():
                    is_robust = r.get('wf_robust', False)
                    robust_tag = '<span class="robust-tag">✓</span>' if is_robust else ''
                    wr = r.get('wr_pct', 0)
                    ma_val = r.get('current_ma_value', 0)
                    val_str = f"{ma_val:.4f}" if price < 10 else f"{ma_val:.2f}"
                    html += f'<tr class="resistance-cell"><td>{r["ma_type"]}</td>'
                    html += f'<td>{int(r["period"])}</td><td>{val_str}</td>'
                    html += f'<td>{r.get("distance_pct", 0):+.2f}%</td>'
                    html += f'<td>{wr:.0f}%</td><td>{robust_tag}</td></tr>'
                html += '</table></div>'

            html += '</div>'

        # Cluster bölgeleri
        sc = p.get('support_clusters', [])
        rc = p.get('resistance_clusters', [])
        if sc or rc:
            html += '<h3>💎 MA Cluster Bölgeleri</h3>'
            for c in sc:
                strength_class = 'cluster-strong' if c['strength'] >= 3 else ''
                cv = c['center']
                cv_str = f"{cv:.4f}" if price < 10 else f"{cv:.2f}"
                ma_list = ', '.join([f"{r['ma_type']}{int(r['period'])}" for r in c['mas'][:5]])
                html += f'<div class="cluster-box {strength_class}" style="border-left:3px solid #7fc97f;">'
                html += f'<b style="color:#7fc97f">🟢 DESTEK Cluster</b> @ {cv_str} TL '
                html += f'<span style="color:#8b95a5">({c["strength"]} MA: {ma_list})</span>'
                html += '</div>'
            for c in rc:
                strength_class = 'cluster-strong' if c['strength'] >= 3 else ''
                cv = c['center']
                cv_str = f"{cv:.4f}" if price < 10 else f"{cv:.2f}"
                ma_list = ', '.join([f"{r['ma_type']}{int(r['period'])}" for r in c['mas'][:5]])
                html += f'<div class="cluster-box {strength_class}" style="border-left:3px solid #ff8c69;">'
                html += f'<b style="color:#ff8c69">🔴 DIRENC Cluster</b> @ {cv_str} TL '
                html += f'<span style="color:#8b95a5">({c["strength"]} MA: {ma_list})</span>'
                html += '</div>'

        html += '</div>'  # /stock-card

    html += "</body></html>"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path


def main():
    p = argparse.ArgumentParser(description='BIST Panorama Rapor')
    p.add_argument('--csv', required=True, help='Scanner CSV çıktısı')
    p.add_argument('--output', default='reports/panorama.html', help='Output HTML')
    p.add_argument('--min-confidence', type=float, default=0,
                   help='Minimum confidence puanı filtresi (0=hepsi)')
    p.add_argument('--top-stocks', type=int, default=None,
                   help='Sadece top N hisse göster (robust sayısına göre)')
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"✗ CSV bulunamadı: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"📂 CSV: {len(df):,} satır")

    # Confidence filter
    if args.min_confidence > 0 and 'confidence' in df.columns:
        df = df[df['confidence'] >= args.min_confidence]
        print(f"Min confidence {args.min_confidence}: {len(df)} satır")

    # Her hisse için panorama
    panoramas = []
    tickers = df['ticker'].unique()
    print(f"\n📊 Panorama oluşturuluyor: {len(tickers)} hisse...")
    for tk in tickers:
        ds = df[df['ticker'] == tk]
        p = stock_panorama(ds)
        if p:
            panoramas.append(p)

    # Top N filter
    if args.top_stocks:
        panoramas.sort(key=lambda x: -x.get('n_robust', 0))
        panoramas = panoramas[:args.top_stocks]
        print(f"Top {args.top_stocks}: {len(panoramas)} hisse")

    # HTML üret
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out = generate_panorama_html(panoramas, args.output)
    print(f"✓ Panorama: {out}")


if __name__ == '__main__':
    main()
