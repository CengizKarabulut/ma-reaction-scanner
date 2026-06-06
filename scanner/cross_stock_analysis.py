#!/usr/bin/env python3
"""
Cross-Stock Pattern Analysis
============================
BIST genelinde hangi MA aileleri sürekli üst sıralarda çıkıyor analiz eder.
"ALMA 20-22 evrensel mi yoksa hisseye özel mi?" sorusunun cevabı burada.

Kullanım:
    python scanner/cross_stock_analysis.py \
        --input reports/weekly_scan.csv \
        --output reports/cross_stock.html
"""

import argparse
from datetime import datetime

import pandas as pd


def analyze(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Multi-stock pattern analizi"""
    result = {}
    
    # 1. Top N MA seçimi (her hisse için)
    top_per_stock = (
        df.groupby('ticker', group_keys=False)
        .apply(lambda g: g.nlargest(top_n, 'composite_score'))
    )
    
    # 2. MA aile popülerliği
    pop_by_pair = (
        top_per_stock.groupby(['ma_type', 'period']).size()
        .reset_index(name='stock_count')
        .sort_values('stock_count', ascending=False)
    )
    pop_by_pair['pct_stocks'] = pop_by_pair['stock_count'] / df['ticker'].nunique() * 100
    result['ma_popularity'] = pop_by_pair
    
    # 3. MA tip dağılımı
    pop_by_type = (
        top_per_stock.groupby('ma_type').size()
        .reset_index(name='top_count')
        .sort_values('top_count', ascending=False)
    )
    result['ma_type_dist'] = pop_by_type
    
    # 4. Periyot ailesi dağılımı (kısa/orta/uzun)
    def period_bucket(p):
        if p <= 13: return 'Kısa (3-13)'
        elif p <= 55: return 'Orta-Kısa (20-55)'
        elif p <= 200: return 'Orta-Uzun (89-200)'
        else: return 'Uzun (233+)'
    
    top_per_stock_ = top_per_stock.copy()
    top_per_stock_['bucket'] = top_per_stock_['period'].apply(period_bucket)
    pop_by_bucket = (
        top_per_stock_.groupby('bucket').size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    result['period_buckets'] = pop_by_bucket
    
    # 5. Walk-forward robust olanların oranı
    if 'wf_robust' in df.columns:
        total = len(df)
        robust = df['wf_robust'].sum()
        result['robust_pct'] = 100 * robust / total
        
        # Robust MA'ların aile dağılımı
        robust_df = df[df['wf_robust'] == True]
        robust_pop = (
            robust_df.groupby(['ma_type', 'period']).size()
            .reset_index(name='robust_count')
            .sort_values('robust_count', ascending=False)
            .head(20)
        )
        result['robust_top'] = robust_pop
    
    # 6. Hangi hisseler "kaliteli" (çok robust MA'ya sahip)
    if 'wf_robust' in df.columns:
        stock_quality = (
            df.groupby('ticker').agg(
                total_mas=('ma_type', 'count'),
                robust_mas=('wf_robust', 'sum'),
            ).reset_index()
        )
        stock_quality['robust_pct'] = stock_quality['robust_mas'] / stock_quality['total_mas'] * 100
        stock_quality = stock_quality.sort_values('robust_pct', ascending=False)
        result['stock_quality'] = stock_quality
    
    return result


def render_html(result: dict, total_stocks: int, total_mas: int) -> str:
    """HTML rapor üret"""
    style = """
    <style>
    body { font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee; padding: 30px; max-width: 1200px; margin: auto; }
    h1 { color: #4ec9b0; }
    h2 { color: #4ec9b0; border-bottom: 1px solid #444; padding-bottom: 8px; margin-top: 30px; }
    table { border-collapse: collapse; margin-bottom: 20px; font-size: 13px; }
    th { background: #2d2d2d; padding: 6px 10px; border: 1px solid #444; text-align: left; }
    td { padding: 5px 10px; border: 1px solid #333; }
    tr.universal td { background: #1a3d2e; font-weight: bold; }
    tr.common td { background: #2d3d2d; }
    .highlight { color: #4ade80; font-weight: bold; }
    .insight { background: #2a2a3a; padding: 15px; border-left: 3px solid #4ec9b0; margin: 15px 0; }
    </style>
    """
    
    html = [f"<html><head><meta charset='utf-8'>{style}</head><body>"]
    html.append(f"<h1>BIST MA Reaction — Cross-Stock Pattern Analizi</h1>")
    html.append(f"<p>Tarih: {datetime.now():%Y-%m-%d %H:%M}</p>")
    html.append(f"<p>Hisse sayısı: <b>{total_stocks}</b> | Toplam MA adayı: <b>{total_mas:,}</b></p>")
    
    if 'robust_pct' in result:
        html.append(f"<p>Walk-forward robust oranı: <b class='highlight'>{result['robust_pct']:.1f}%</b></p>")
    
    # En popüler MA aileleri
    html.append(f"<h2>🎯 En Yaygın MA Aileleri (her hissenin top 10'unda görünme)</h2>")
    html.append("<table><tr><th>MA Türü</th><th>Periyot</th><th>Hisse Sayısı</th><th>% (Tüm Hisseler)</th></tr>")
    pop = result['ma_popularity'].head(20)
    for _, r in pop.iterrows():
        cls = ''
        if r['pct_stocks'] >= 70:
            cls = 'universal'
        elif r['pct_stocks'] >= 40:
            cls = 'common'
        html.append(f"<tr class='{cls}'><td>{r['ma_type']}</td><td>{r['period']}</td><td>{r['stock_count']}</td><td>{r['pct_stocks']:.1f}%</td></tr>")
    html.append("</table>")
    html.append("<div class='insight'>")
    html.append("Yorumlama: <b>%70+ olan satırlar</b> BIST için <b>evrensel destek/direnç MA'ları</b> sayılabilir. ")
    html.append("<b>%40-70 arası</b> yaygın ama hisseye bağlı. <b>%40 altı</b> hisse-spesifik.")
    html.append("</div>")
    
    # MA tip dağılımı
    html.append(f"<h2>📊 MA Türü Dağılımı (Top 10'da)</h2>")
    html.append("<table><tr><th>MA Türü</th><th>Görünüm Sayısı</th></tr>")
    for _, r in result['ma_type_dist'].iterrows():
        html.append(f"<tr><td>{r['ma_type']}</td><td>{r['top_count']}</td></tr>")
    html.append("</table>")
    
    # Periyot kovaları
    html.append(f"<h2>📈 Periyot Aile Dağılımı</h2>")
    html.append("<table><tr><th>Periyot Kovaı</th><th>Görünüm Sayısı</th></tr>")
    for _, r in result['period_buckets'].iterrows():
        html.append(f"<tr><td>{r['bucket']}</td><td>{r['count']}</td></tr>")
    html.append("</table>")
    
    # Walk-forward robust top
    if 'robust_top' in result:
        html.append(f"<h2>💎 Walk-Forward Robust MA'lar (en güvenilir)</h2>")
        html.append("<table><tr><th>MA</th><th>Periyot</th><th>Robust Hisse Sayısı</th></tr>")
        for _, r in result['robust_top'].iterrows():
            html.append(f"<tr><td>{r['ma_type']}</td><td>{r['period']}</td><td>{r['robust_count']}</td></tr>")
        html.append("</table>")
    
    # Kaliteli hisseler
    if 'stock_quality' in result:
        html.append(f"<h2>⭐ En Kaliteli Hisseler (MA-saygısı yüksek)</h2>")
        html.append("<table><tr><th>Hisse</th><th>Toplam MA</th><th>Robust MA</th><th>Robust %</th></tr>")
        for _, r in result['stock_quality'].head(15).iterrows():
            html.append(f"<tr><td>{r['ticker']}</td><td>{r['total_mas']}</td><td>{r['robust_mas']}</td><td>{r['robust_pct']:.1f}%</td></tr>")
        html.append("</table>")
        html.append("<div class='insight'>")
        html.append("Yorumlama: Robust % yüksek hisseler MA stratejilerine en uygun olanlardır. ")
        html.append("Düşük olanlarda MA-based yaklaşım anlamsız, başka yöntemler düşünülmeli.")
        html.append("</div>")
    
    html.append("</body></html>")
    return '\n'.join(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='cross_stock_analysis.html')
    parser.add_argument('--top_n', type=int, default=10)
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    total_stocks = df['ticker'].nunique()
    total_mas = len(df)
    
    result = analyze(df, top_n=args.top_n)
    html = render_html(result, total_stocks, total_mas)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Rapor: {args.output}")
    print(f"\nÖzet:")
    print(f"  Hisse sayısı: {total_stocks}")
    print(f"  Toplam MA: {total_mas:,}")
    if 'robust_pct' in result:
        print(f"  Robust oranı: {result['robust_pct']:.1f}%")
    print(f"\nEn yaygın MA aileleri (top 5):")
    for _, r in result['ma_popularity'].head(5).iterrows():
        print(f"  {r['ma_type']} {r['period']}: {r['stock_count']} hisse ({r['pct_stocks']:.1f}%)")


if __name__ == '__main__':
    main()
