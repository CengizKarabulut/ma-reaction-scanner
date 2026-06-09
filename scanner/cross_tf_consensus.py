#!/usr/bin/env python3
"""
Cross-Timeframe Consensus — D + W + M tarama sonuçlarını birleştir.

Aynı (ticker, ma_type, period) kombinasyonu birden fazla timeframe'de
robust ise "süper robust" sayılır. Bunlar gerçek MA-saygısı gösteren,
trade etmeye en uygun pattern'ler.

Kullanim:
    python cross_tf_consensus.py \\
        --daily reports/scan_2026-06-07_1d.csv \\
        --weekly reports/scan_2026-06-07_1wk.csv \\
        --monthly reports/scan_2026-06-07_1mo.csv \\
        --output reports/consensus.html
"""

import argparse
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np


def load_robust(csv_path: str, label: str, fallback_score_thresh: float = 5.0) -> pd.DataFrame:
    """CSV'den robust olan kayitlari yukle.

    Eger wf_robust kolonu yoksa (walk-forward yapilmadiysa):
    composite_score >= fallback_score_thresh olanlari "robust" kabul et.
    Bu ozellikle monthly tarama icin gerekli (az veri = walk-forward zor).
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"  {label}: dosya yok, atlaniyor")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)

    if 'wf_robust' in df.columns:
        robust = df[df['wf_robust'] == True].copy()
        print(f"  {label}: {len(robust):,} robust MA (wf), {robust['ticker'].nunique()} hisse")
    elif 'composite_score' in df.columns:
        # Walk-forward yoksa, composite_score ile fallback
        robust = df[df['composite_score'] >= fallback_score_thresh].copy()
        print(f"  {label}: {len(robust):,} robust MA (composite >= {fallback_score_thresh}), "
              f"{robust['ticker'].nunique()} hisse [walk-forward yok, score-based]")
    else:
        print(f"  {label}: ne wf_robust ne composite_score kolonu var")
        return pd.DataFrame()
    return robust


def compute_consensus(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """3 timeframe'i merge et, hangisi hangisinde robust onu işaretle."""
    if d.empty and w.empty and m.empty:
        return pd.DataFrame()

    # Her timeframe icin (ticker, ma_type, period) anahtari ve istatistikler
    def prep(df, suffix):
        if df.empty:
            return pd.DataFrame()
        cols = ['ticker', 'ma_type', 'period']
        metrics = ['wr_pct', 'expectancy', 'composite_score', 'touches', 'adr']
        keep = cols + [c for c in metrics if c in df.columns]
        out = df[keep].copy()
        for c in metrics:
            if c in out.columns:
                out = out.rename(columns={c: f"{c}_{suffix}"})
        out[f'robust_{suffix}'] = True
        return out

    d_p = prep(d, 'D')
    w_p = prep(w, 'W')
    m_p = prep(m, 'M')

    # Outer join: hangi timeframe'de varsa
    merged = d_p
    for other in [w_p, m_p]:
        if not other.empty:
            if merged.empty:
                merged = other
            else:
                merged = merged.merge(other, on=['ticker', 'ma_type', 'period'], how='outer')

    # NaN'leri False olarak isaretle
    for tf in ['D', 'W', 'M']:
        col = f'robust_{tf}'
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)
        else:
            merged[col] = False

    # Consensus skoru: kac timeframe'de robust
    merged['tf_count'] = (
        merged['robust_D'].astype(int) +
        merged['robust_W'].astype(int) +
        merged['robust_M'].astype(int)
    )

    # Toplam composite skoru (mevcut tüm timeframe'lerin ortalaması)
    skor_cols = [c for c in ['composite_score_D', 'composite_score_W', 'composite_score_M']
                 if c in merged.columns]
    if skor_cols:
        merged['avg_composite'] = merged[skor_cols].mean(axis=1, skipna=True)
    else:
        merged['avg_composite'] = 0

    return merged.sort_values(['tf_count', 'avg_composite'], ascending=[False, False])


def generate_html(merged: pd.DataFrame, output: str):
    """HTML consensus raporu."""
    triple = merged[merged['tf_count'] == 3]
    double = merged[merged['tf_count'] == 2]
    single = merged[merged['tf_count'] == 1]

    html = [
        '<!DOCTYPE html>',
        '<html lang="tr"><head><meta charset="utf-8">',
        '<title>Cross-Timeframe Consensus</title>',
        '<style>',
        'body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
        'background:#0a0e14;color:#e6e6e6;padding:20px;max-width:1400px;margin:auto;}',
        'h1{color:#5fb3ff;border-bottom:2px solid #5fb3ff;padding-bottom:8px;}',
        'h2{color:#7be2ff;margin-top:30px;}',
        '.summary{background:#1a1f29;padding:15px;border-radius:8px;margin:15px 0;}',
        '.gold{background:linear-gradient(135deg,#3a2f00,#5a4500);border-left:4px solid #ffd700;'
        'padding:15px;border-radius:8px;margin:15px 0;}',
        '.silver{background:#1f2329;border-left:4px solid #b0b0b0;padding:15px;'
        'border-radius:8px;margin:15px 0;}',
        'table{border-collapse:collapse;width:100%;margin:10px 0;font-family:monospace;font-size:13px;}',
        'th,td{padding:6px 10px;text-align:left;border-bottom:1px solid #2a2f39;}',
        'th{background:#2a2f39;color:#5fb3ff;}',
        '.tf3{color:#ffd700;font-weight:bold;}',
        '.tf2{color:#b0b0b0;}',
        '.tf1{color:#888;}',
        '.true{color:#7fc97f;}.false{color:#444;}',
        '</style></head><body>',
        '<h1>Cross-Timeframe Consensus Raporu</h1>',
        f'<div class="summary">',
        f'<strong>Tarih:</strong> {datetime.now():%Y-%m-%d %H:%M}<br>',
        f'<strong>3 TF\'de robust (altın):</strong> {len(triple)} kombinasyon<br>',
        f'<strong>2 TF\'de robust (gümüş):</strong> {len(double)} kombinasyon<br>',
        f'<strong>1 TF\'de robust:</strong> {len(single)} kombinasyon<br>',
        '</div>'
    ]

    # === ALTIN: 3 TF'de de robust ===
    html.append('<h2>🥇 Altın Standart — Üç Timeframe\'de Robust</h2>')
    html.append('<div class="gold">')
    html.append('<p>Bu kombinasyonlar günlük, haftalık ve aylık zaman dilimlerinde '
                '<strong>tutarlı şekilde</strong> MA tepkisi gösteriyor. En güvenilir trade adayları.</p>')
    if len(triple) == 0:
        html.append('<p><em>Henüz üç timeframe consensus yok — daha çok run gerekli.</em></p>')
    else:
        html.append('<table><tr>')
        html.append('<th>Hisse</th><th>MA</th><th>Per</th>')
        html.append('<th>D-WR</th><th>D-Exp</th>')
        html.append('<th>W-WR</th><th>W-Exp</th>')
        html.append('<th>M-WR</th><th>M-Exp</th>')
        html.append('<th>Avg Skor</th></tr>')
        for _, r in triple.head(30).iterrows():
            html.append('<tr class="tf3">')
            html.append(f'<td>{r["ticker"]}</td>')
            html.append(f'<td>{r["ma_type"]}</td><td>{r["period"]}</td>')
            for tf in ['D', 'W', 'M']:
                wr = r.get(f'wr_pct_{tf}', np.nan)
                exp = r.get(f'expectancy_{tf}', np.nan)
                wr_str = f"{wr:.0f}%" if not pd.isna(wr) else "—"
                exp_str = f"{exp:+.2f}" if not pd.isna(exp) else "—"
                html.append(f'<td>{wr_str}</td><td>{exp_str}</td>')
            html.append(f'<td>{r["avg_composite"]:.2f}</td>')
            html.append('</tr>')
        html.append('</table>')
    html.append('</div>')

    # === GÜMÜŞ: 2 TF'de robust ===
    html.append('<h2>🥈 Gümüş — İki Timeframe\'de Robust</h2>')
    html.append('<div class="silver">')
    html.append('<p>İki timeframe\'de konfirme oluyor. Trade için kabul edilebilir, ama tek timeframe\'e göre daha güvenli.</p>')
    if len(double) > 0:
        html.append('<table><tr>')
        html.append('<th>Hisse</th><th>MA</th><th>Per</th>')
        html.append('<th>D</th><th>W</th><th>M</th>')
        html.append('<th>Skor</th></tr>')
        for _, r in double.head(40).iterrows():
            html.append('<tr class="tf2">')
            html.append(f'<td>{r["ticker"]}</td>')
            html.append(f'<td>{r["ma_type"]}</td><td>{r["period"]}</td>')
            for tf in ['D', 'W', 'M']:
                rob = r.get(f'robust_{tf}', False)
                cls = 'true' if rob else 'false'
                mark = '✓' if rob else '✗'
                html.append(f'<td class="{cls}">{mark}</td>')
            html.append(f'<td>{r["avg_composite"]:.2f}</td>')
            html.append('</tr>')
        html.append('</table>')
    html.append('</div>')

    # Hisse bazinda consensus skoru (en cok 3-TF robust olan hisseler)
    if len(triple) > 0:
        html.append('<h2>🏆 Hisse Bazlı Consensus Sıralaması</h2>')
        triple_per_stock = triple.groupby('ticker').size().sort_values(ascending=False).head(20)
        html.append('<table><tr><th>Hisse</th><th>Altın Kombo Sayısı</th></tr>')
        for tk, cnt in triple_per_stock.items():
            html.append(f'<tr><td>{tk}</td><td class="tf3">{cnt}</td></tr>')
        html.append('</table>')

    html.append('<div class="summary" style="margin-top:40px;">')
    html.append('<strong>Kullanım önerisi:</strong><br>')
    html.append('1. <strong>Altın listede</strong>ki kombinasyonlar öncelikli — bunlara TradingView\'da pozisyon kur<br>')
    html.append('2. Trend yönünde olanları seç (mevcut MA değeri vs fiyat)<br>')
    html.append('3. strategy_generator.py ile bu hisseler için somut entry/stop/TP hesabı yap<br>')
    html.append('4. Position sizing: portföy max %20 tek hissede, %1 risk per trade<br>')
    html.append('</div>')
    html.append('</body></html>')

    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"\n✓ Consensus raporu: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--daily', type=str, default='', help='Daily CSV path')
    p.add_argument('--weekly', type=str, default='', help='Weekly CSV path')
    p.add_argument('--monthly', type=str, default='', help='Monthly CSV path')
    p.add_argument('--output', type=str, default='consensus.html')
    p.add_argument('--csv-out', type=str, default='', help='Opsiyonel: consensus CSV ciktisi')
    args = p.parse_args()

    print("Cross-TF Consensus baslatildi...")
    d = load_robust(args.daily, 'Daily')
    w = load_robust(args.weekly, 'Weekly')
    m = load_robust(args.monthly, 'Monthly')

    merged = compute_consensus(d, w, m)
    if merged.empty:
        print("HATA: Hicbir robust veri bulunamadi")
        sys.exit(1)

    print(f"\nKombinasyon istatistikleri:")
    print(f"  3 TF robust: {(merged['tf_count'] == 3).sum()}")
    print(f"  2 TF robust: {(merged['tf_count'] == 2).sum()}")
    print(f"  1 TF robust: {(merged['tf_count'] == 1).sum()}")

    generate_html(merged, args.output)

    if args.csv_out:
        merged.to_csv(args.csv_out, index=False)
        print(f"✓ Consensus CSV: {args.csv_out}")


if __name__ == '__main__':
    main()
