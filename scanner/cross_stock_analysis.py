"""
BIST Weekly Robustness Analysis — Cross-stock derin analiz.

Tarama CSV'sini alır, çok yönlü analiz yapar:
- En çok robust hisseler (hisse başı robust MA sayısı)
- En yaygın MA aileleri (cross-stock konsensüs)
- Sektör bazlı dağılım (sector_resolver entegre)
- Bullish/Bearish dengesi
- Top 20 setup'lar
- Önceki haftaya karşılaştırma (opsiyonel)
- HTML rapor + Markdown özet (Telegram için)

Kullanım:
    # Belirli CSV ile
    python cross_stock_analysis.py --input reports/weekly_2026-06-13.csv

    # AKILLI: input verilmezse reports/ klasöründe en yeni CSV'yi bulur
    python cross_stock_analysis.py

    # Önceki haftayla karşılaştır
    python cross_stock_analysis.py --input reports/weekly_2026-06-13.csv \
        --prev reports/weekly_2026-06-06.csv

Output:
    - reports/weekly_analysis_<tarih>.html (görsel rapor)
    - reports/weekly_summary_<tarih>.md (Telegram için kısa özet)
    - stdout: ana metrikler
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# sector_resolver entegrasyonu (opsiyonel)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from sector_resolver import get_sector_info, load_sectors_cache
    HAS_SECTOR_RESOLVER = True
except ImportError:
    HAS_SECTOR_RESOLVER = False


# ============================================================
# AKILLI CSV BULMA
# ============================================================

def find_latest_csv(reports_dir: str = 'reports', pattern: str = '*.csv') -> str:
    """reports/ klasöründe en yeni CSV'yi bul."""
    if not os.path.isdir(reports_dir):
        return None

    candidates = []
    # Öncelik sırası: weekly_*.csv > scan_*.csv > genel *.csv
    for prefix in ['weekly_', 'scan_', '']:
        files = glob.glob(os.path.join(reports_dir, f'{prefix}*.csv'))
        for f in files:
            # Geçici/temp dosyaları atla
            if 'summary' in f.lower() or 'analysis' in f.lower():
                continue
            mtime = os.path.getmtime(f)
            candidates.append((mtime, f))
        if candidates:
            break

    if not candidates:
        return None

    candidates.sort(reverse=True)  # En yeni başta
    return candidates[0][1]


def find_previous_csv(current_csv: str, days_ago: int = 7) -> str:
    """current_csv'den ~7 gün önceki benzer CSV'yi bul."""
    if not current_csv:
        return None
    reports_dir = os.path.dirname(current_csv) or '.'
    basename = os.path.basename(current_csv)
    # weekly_2026-06-13.csv → tarihi parse et
    import re
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', basename)
    if not m:
        return None
    current_date = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Benzer prefix'li tüm dosyalar
    prefix = basename.split(m.group(0))[0]  # 'weekly_'
    candidates = []
    for f in glob.glob(os.path.join(reports_dir, f'{prefix}*.csv')):
        if f == current_csv:
            continue
        m2 = re.search(r'(\d{4})-(\d{2})-(\d{2})', f)
        if not m2:
            continue
        d = datetime(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
        delta_days = abs((current_date - d).days)
        if delta_days >= days_ago - 2 and delta_days <= days_ago + 2:
            candidates.append((delta_days, f))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


# ============================================================
# ANALİZ FONKSİYONLARI
# ============================================================

def annotate_with_sectors(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame'e sector ve industry kolonları ekle."""
    if not HAS_SECTOR_RESOLVER:
        df['sector'] = 'Unknown'
        df['industry'] = 'Unknown'
        return df

    cache = load_sectors_cache()
    sectors = cache.get('sectors', {})
    if not sectors:
        df['sector'] = 'Unknown'
        df['industry'] = 'Unknown'
        return df

    def lookup(tk, field):
        info = sectors.get(tk.upper(), {})
        return info.get(field, 'Unknown')

    df['sector'] = df['ticker'].apply(lambda t: lookup(t, 'sector'))
    df['industry'] = df['ticker'].apply(lambda t: lookup(t, 'industry'))
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Genel istatistikleri hesapla."""
    metrics = {
        'total_records': len(df),
        'n_stocks': df['ticker'].nunique() if 'ticker' in df else 0,
    }
    if 'wf_robust' in df.columns:
        robust = df[df['wf_robust'] == True]
        metrics['robust_count'] = len(robust)
        metrics['robust_pct'] = 100 * len(robust) / max(len(df), 1)
        metrics['robust_stocks'] = robust['ticker'].nunique()
    return metrics


def top_robust_stocks(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Hisse başı robust MA sayısına göre top N."""
    if 'wf_robust' not in df.columns:
        return pd.DataFrame()
    robust = df[df['wf_robust'] == True]
    if len(robust) == 0:
        return pd.DataFrame()
    counts = robust.groupby('ticker').size().reset_index(name='robust_count')
    if 'sector' in robust.columns:
        sector_map = robust.drop_duplicates('ticker').set_index('ticker')['sector']
        counts['sector'] = counts['ticker'].map(sector_map)
    # En son fiyat (her hissenin)
    if 'current_close' in df.columns:
        price_map = df.drop_duplicates('ticker').set_index('ticker')['current_close']
        counts['price'] = counts['ticker'].map(price_map)
    return counts.nlargest(n, 'robust_count')


def ma_family_frequency(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """En yaygın MA türü+period kombinasyonları."""
    if 'wf_robust' not in df.columns:
        return pd.DataFrame()
    robust = df[df['wf_robust'] == True]
    if len(robust) == 0:
        return pd.DataFrame()
    freq = robust.groupby(['ma_type', 'period']).agg(
        n_stocks=('ticker', 'nunique'),
        avg_wr=('wr_pct', 'mean'),
        avg_exp=('expectancy', 'mean'),
    ).reset_index()
    return freq.nlargest(top_n, 'n_stocks')


def sector_trend_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Sektör bazlı robust dağılımı."""
    if 'wf_robust' not in df.columns or 'sector' not in df.columns:
        return pd.DataFrame()
    robust = df[df['wf_robust'] == True]
    if len(robust) == 0:
        return pd.DataFrame()

    by_sector = robust.groupby('sector').agg(
        n_stocks=('ticker', 'nunique'),
        n_setups=('ticker', 'count'),
        avg_wr=('wr_pct', 'mean'),
        avg_exp=('expectancy', 'mean'),
    ).reset_index()
    by_sector = by_sector.sort_values('n_setups', ascending=False)
    return by_sector


def bullish_bearish_split(df: pd.DataFrame) -> dict:
    """DESTEK (bullish) vs DIRENC (bearish) dağılımı."""
    if 'wf_robust' not in df.columns or 'current_close' not in df.columns:
        return {}
    robust = df[df['wf_robust'] == True].copy()
    if 'current_ma_value' not in robust.columns or len(robust) == 0:
        return {}
    # MA değeri < kapanış → DESTEK (bullish setup)
    robust['is_support'] = robust['current_ma_value'] < robust['current_close']
    n_support = int(robust['is_support'].sum())
    n_resistance = int((~robust['is_support']).sum())
    total = n_support + n_resistance
    return {
        'n_support': n_support,
        'n_resistance': n_resistance,
        'support_pct': 100 * n_support / max(total, 1),
        'resistance_pct': 100 * n_resistance / max(total, 1),
    }


def top_setups(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """En iyi composite score'lu robust setup'lar."""
    if 'wf_robust' not in df.columns:
        return df.nlargest(n, 'composite_score') if 'composite_score' in df.columns else df.head(n)
    robust = df[df['wf_robust'] == True]
    if 'composite_score' in robust.columns:
        return robust.nlargest(n, 'composite_score')
    return robust.head(n)


def week_over_week(curr: pd.DataFrame, prev: pd.DataFrame) -> dict:
    """Önceki haftaya karşı değişim."""
    if prev is None or prev.empty:
        return {}

    def robust_set(df):
        if 'wf_robust' not in df.columns:
            return set()
        return set(df[df['wf_robust'] == True]['ticker'].unique())

    curr_robust = robust_set(curr)
    prev_robust = robust_set(prev)

    new_robust = curr_robust - prev_robust
    lost_robust = prev_robust - curr_robust
    stable = curr_robust & prev_robust

    return {
        'curr_total': len(curr_robust),
        'prev_total': len(prev_robust),
        'new_robust': sorted(new_robust),
        'lost_robust': sorted(lost_robust),
        'stable': sorted(stable),
        'change': len(curr_robust) - len(prev_robust),
    }


# ============================================================
# RAPOR ÜRETİMİ
# ============================================================

def generate_markdown_summary(metrics, top_stocks, top_setups_df, ma_freq,
                                 sector_df, bb_split, wow=None) -> str:
    """Markdown özet (Telegram için)."""
    lines = []
    lines.append(f"📊 *Haftalık BIST Robustness Analizi*")
    lines.append(f"_{datetime.now():%Y-%m-%d %H:%M}_")
    lines.append("")

    lines.append(f"📈 Hisse: *{metrics.get('n_stocks', 0)}* | "
                 f"Toplam MA: *{metrics.get('total_records', 0):,}*")
    if 'robust_count' in metrics:
        lines.append(f"✅ Robust: *{metrics['robust_count']:,}* "
                     f"({metrics['robust_pct']:.1f}%) — "
                     f"{metrics.get('robust_stocks', 0)} hissede setup var")
    lines.append("")

    # Bullish/Bearish denge
    if bb_split:
        lines.append(f"⚖️ Denge: 🟢 Destek *{bb_split['n_support']}* ({bb_split['support_pct']:.0f}%)  |  "
                     f"🔴 Direnç *{bb_split['n_resistance']}* ({bb_split['resistance_pct']:.0f}%)")
        lines.append("")

    # Hafta üzeri değişim
    if wow:
        arrow = '📈' if wow.get('change', 0) >= 0 else '📉'
        lines.append(f"{arrow} *Önceki haftaya:* {wow.get('curr_total', 0)} "
                     f"(önceki {wow.get('prev_total', 0)}, fark {wow.get('change', 0):+})")
        if wow.get('new_robust'):
            lines.append(f"   🆕 Yeni: `{', '.join(wow['new_robust'][:8])}`")
        if wow.get('lost_robust'):
            lines.append(f"   ❌ Kaybedildi: `{', '.join(wow['lost_robust'][:8])}`")
        lines.append("")

    # Top robust hisseler - CODE BLOCK ile hizalama korunsun
    if not top_stocks.empty:
        lines.append("💎 *En Çok Robust 10 Hisse*")
        lines.append("```")
        lines.append(f"{'Hisse':<8} {'Robust':<7} {'Fiyat':<10} {'Sektör'}")
        lines.append("─" * 45)
        for _, r in top_stocks.head(10).iterrows():
            sector = str(r.get('sector', '?'))[:18]
            price = r.get('price', None)
            if price and not pd.isna(price):
                pf = f"{price:.2f}" if price >= 1 else f"{price:.4f}"
            else:
                pf = '-'
            lines.append(f"{r['ticker']:<8} {int(r['robust_count']):<7} {pf:<10} {sector}")
        lines.append("```")
        lines.append("")

    # Top MA aileleri
    if not ma_freq.empty:
        lines.append("🎯 *En Yaygın MA Aileleri (cross-stock)*")
        lines.append("```")
        lines.append(f"{'MA':<6} {'Per':<5} {'Hisse#':<7} {'WR':<5}")
        lines.append("─" * 28)
        for _, r in ma_freq.head(10).iterrows():
            lines.append(f"{r['ma_type']:<6} {int(r['period']):<5} {int(r['n_stocks']):<7} "
                         f"{r['avg_wr']:.0f}%")
        lines.append("```")
        lines.append("")

    # Sektör dağılımı - Unknown çok ise sektör cache yok demek, bölümü atla
    if not sector_df.empty:
        # Eğer sadece Unknown sektörü varsa (sector_resolver cache yok), bölümü atla
        non_unknown = sector_df[sector_df['sector'].str.lower() != 'unknown']
        if not non_unknown.empty:
            lines.append("🏭 *En Aktif Sektörler*")
            lines.append("```")
            lines.append(f"{'Sektör':<25} {'Hisse':<6} {'Setup':<7} {'WR'}")
            lines.append("─" * 48)
            for _, r in non_unknown.head(10).iterrows():
                sector = str(r['sector'])[:24]
                lines.append(f"{sector:<25} {int(r['n_stocks']):<6} {int(r['n_setups']):<7} "
                             f"{r['avg_wr']:.0f}%")
            lines.append("```")
        else:
            # Sektör cache yok - kullanıcıya bildir
            lines.append("ℹ️ _Sektör analizi atlandı (sektör cache yok)._")
            lines.append("_'Actions → Sektör Cache Oluştur' ile sektör bilgisini topla._")

    return '\n'.join(lines)


def generate_html_report(metrics, top_stocks, top_setups_df, ma_freq,
                          sector_df, bb_split, wow=None) -> str:
    """Detaylı HTML rapor."""
    css = """
    <style>
        body { font-family: -apple-system, sans-serif; background: #0a0e14; color: #e6e6e6;
               padding: 24px; max-width: 1400px; margin: auto; }
        h1, h2 { color: #5fb3ff; border-bottom: 1px solid #2a2f39; padding-bottom: 8px; }
        h2 { margin-top: 32px; }
        .metric-box { display: inline-block; background: #1a1f29; padding: 16px 24px;
                      margin: 8px; border-radius: 8px; min-width: 120px; }
        .metric-label { color: #8b95a5; font-size: 12px; text-transform: uppercase; }
        .metric-value { color: #fff; font-size: 24px; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 12px 0;
                background: #1a1f29; border-radius: 8px; overflow: hidden; }
        th { background: #2a2f39; color: #5fb3ff; padding: 10px; text-align: left; }
        td { padding: 8px 10px; border-bottom: 1px solid #2a2f39; }
        tr:hover { background: #232a36; }
        .support { color: #7fc97f; font-weight: bold; }
        .resistance { color: #ff8c69; font-weight: bold; }
        .new-tag { background: #1a3f2f; color: #7fc97f; padding: 2px 8px;
                   border-radius: 4px; font-size: 12px; margin: 2px; display: inline-block; }
        .lost-tag { background: #3f1a1a; color: #ff8c69; padding: 2px 8px;
                    border-radius: 4px; font-size: 12px; margin: 2px; display: inline-block; }
    </style>
    """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BIST Haftalık Robustness Analizi</title>
{css}</head><body>
<h1>📊 BIST Haftalık Robustness Analizi</h1>
<p>Oluşturuldu: {datetime.now():%Y-%m-%d %H:%M}</p>

<div>
    <div class="metric-box">
        <div class="metric-label">Hisse</div>
        <div class="metric-value">{metrics.get('n_stocks', 0)}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Toplam MA</div>
        <div class="metric-value">{metrics.get('total_records', 0):,}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Robust MA</div>
        <div class="metric-value">{metrics.get('robust_count', 0):,}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Robust Oran</div>
        <div class="metric-value">{metrics.get('robust_pct', 0):.1f}%</div>
    </div>
</div>
"""

    # Bullish/Bearish
    if bb_split:
        html += f"""
<h2>⚖️ Destek/Direnç Dengesi</h2>
<div>
    <div class="metric-box">
        <div class="metric-label">🟢 Destek</div>
        <div class="metric-value support">{bb_split['n_support']} ({bb_split['support_pct']:.0f}%)</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">🔴 Direnç</div>
        <div class="metric-value resistance">{bb_split['n_resistance']} ({bb_split['resistance_pct']:.0f}%)</div>
    </div>
</div>
"""

    # WoW karşılaştırma
    if wow:
        html += f"""
<h2>📈 Önceki Haftaya Karşı</h2>
<p>Bu hafta: <b>{wow['curr_total']}</b> robust hisse | Önceki: <b>{wow['prev_total']}</b> |
   Değişim: <b>{wow['change']:+}</b></p>
"""
        if wow.get('new_robust'):
            html += "<p><b>🆕 Yeni Robust:</b><br>"
            for tk in wow['new_robust']:
                html += f'<span class="new-tag">{tk}</span> '
            html += "</p>"
        if wow.get('lost_robust'):
            html += "<p><b>❌ Robust'tan Çıktı:</b><br>"
            for tk in wow['lost_robust']:
                html += f'<span class="lost-tag">{tk}</span> '
            html += "</p>"

    # Top robust hisseler
    if not top_stocks.empty:
        html += "<h2>💎 En Çok Robust 15 Hisse</h2><table><tr>"
        html += "<th>Hisse</th><th>Robust MA</th><th>Fiyat</th><th>Sektör</th></tr>"
        for _, r in top_stocks.iterrows():
            price = f"{r['price']:.2f}" if 'price' in r and pd.notna(r.get('price')) else '-'
            sector = str(r.get('sector', '?'))
            html += f"<tr><td><b>{r['ticker']}</b></td><td>{int(r['robust_count'])}</td>"
            html += f"<td>{price}</td><td>{sector}</td></tr>"
        html += "</table>"

    # MA aileleri
    if not ma_freq.empty:
        html += "<h2>🎯 En Yaygın MA Aileleri</h2><table><tr>"
        html += "<th>MA Türü</th><th>Period</th><th>Hisse Sayısı</th>"
        html += "<th>Ort. WR</th><th>Ort. Expectancy</th></tr>"
        for _, r in ma_freq.iterrows():
            html += f"<tr><td><b>{r['ma_type']}</b></td><td>{int(r['period'])}</td>"
            html += f"<td>{int(r['n_stocks'])}</td><td>{r['avg_wr']:.1f}%</td>"
            html += f"<td>{r['avg_exp']:+.2f}</td></tr>"
        html += "</table>"

    # Sektör dağılımı
    if not sector_df.empty:
        html += "<h2>🏭 Sektör Bazlı Dağılım</h2><table><tr>"
        html += "<th>Sektör</th><th>Hisse</th><th>Setup</th>"
        html += "<th>Ort. WR</th><th>Ort. Exp</th></tr>"
        for _, r in sector_df.iterrows():
            html += f"<tr><td>{r['sector']}</td><td>{int(r['n_stocks'])}</td>"
            html += f"<td>{int(r['n_setups'])}</td><td>{r['avg_wr']:.1f}%</td>"
            html += f"<td>{r['avg_exp']:+.2f}</td></tr>"
        html += "</table>"

    # Top setup'lar
    if not top_setups_df.empty:
        html += "<h2>🏆 Top 20 Setup (Composite Score)</h2><table><tr>"
        html += "<th>Hisse</th><th>MA</th><th>Per</th><th>Değer</th>"
        html += "<th>WR</th><th>Exp</th><th>Sektör</th></tr>"
        for _, r in top_setups_df.head(20).iterrows():
            ma_val = r.get('current_ma_value', None)
            curr = r.get('current_close', None)
            if ma_val and curr and not pd.isna(ma_val):
                tag_class = 'support' if ma_val < curr else 'resistance'
                tag = '🟢' if ma_val < curr else '🔴'
                val_str = f"{ma_val:.4f}" if curr < 10 else f"{ma_val:.2f}"
            else:
                tag_class = ''
                tag = ' '
                val_str = '-'
            sector = str(r.get('sector', '?'))[:20]
            html += f'<tr><td class="{tag_class}">{tag} <b>{r["ticker"]}</b></td>'
            html += f'<td>{r["ma_type"]}</td><td>{int(r["period"])}</td><td>{val_str}</td>'
            html += f'<td>{r["wr_pct"]:.0f}%</td><td>{r["expectancy"]:+.2f}</td>'
            html += f'<td>{sector}</td></tr>'
        html += "</table>"

    html += "</body></html>"
    return html


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='BIST Weekly Robustness Analysis')
    parser.add_argument('--input', type=str, default=None,
                        help='Tarama CSV (yoksa reports/ klasöründe en yenisini bulur)')
    parser.add_argument('--prev', type=str, default=None,
                        help='Önceki haftanın CSV (WoW karşılaştırma için)')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Output klasörü')
    parser.add_argument('--no-prev', action='store_true',
                        help='Önceki haftayla karşılaştırmayı atla')
    args = parser.parse_args()

    # === AKILLI CSV BULMA ===
    input_csv = args.input
    if input_csv and not os.path.exists(input_csv):
        print(f"⚠️ Verilen CSV bulunamadı: {input_csv}", file=sys.stderr)
        print(f"   reports/ klasöründe son CSV'ye düşülüyor...", file=sys.stderr)
        input_csv = None

    if not input_csv:
        input_csv = find_latest_csv(args.output_dir)
        if not input_csv:
            print(f"✗ '{args.output_dir}/' klasöründe CSV bulunamadı!", file=sys.stderr)
            print(f"  Önce scanner çalıştır: python scanner/bist_ma_reaction_scanner.py", file=sys.stderr)
            sys.exit(1)
        print(f"✓ Otomatik bulundu: {input_csv}")

    # CSV oku
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"✗ CSV okunamadı ({input_csv}): {e}", file=sys.stderr)
        sys.exit(1)

    if len(df) == 0:
        print(f"⚠️ CSV boş!", file=sys.stderr)
        sys.exit(1)

    print(f"📂 CSV: {input_csv} ({len(df):,} satır)")

    # === ÖNCEKİ HAFTA ===
    prev_df = None
    if not args.no_prev:
        prev_csv = args.prev or find_previous_csv(input_csv)
        if prev_csv and os.path.exists(prev_csv):
            try:
                prev_df = pd.read_csv(prev_csv)
                print(f"📂 Önceki: {prev_csv} ({len(prev_df):,} satır)")
            except Exception as e:
                print(f"⚠️ Önceki CSV okunamadı: {e}", file=sys.stderr)

    # === ANALİZLER ===
    print("\n=== Analiz başlıyor ===")
    df = annotate_with_sectors(df)
    if prev_df is not None:
        prev_df = annotate_with_sectors(prev_df)

    metrics = compute_metrics(df)
    top_stocks = top_robust_stocks(df, n=15)
    ma_freq = ma_family_frequency(df, top_n=15)
    sector_df = sector_trend_analysis(df)
    bb_split = bullish_bearish_split(df)
    top_setups_df = top_setups(df, n=25)
    wow = week_over_week(df, prev_df) if prev_df is not None else None

    # Konsola özet bas
    print(f"\n📊 Hisse: {metrics.get('n_stocks', 0)} | Toplam MA: {metrics.get('total_records', 0):,}")
    if 'robust_count' in metrics:
        print(f"✅ Robust: {metrics['robust_count']:,} ({metrics['robust_pct']:.1f}%)")
    if bb_split:
        print(f"⚖️ Destek: {bb_split['n_support']} | Direnç: {bb_split['n_resistance']}")
    if wow:
        print(f"📈 WoW: {wow['curr_total']} (önceki {wow['prev_total']}, fark {wow['change']:+})")

    # === ÇIKTI DOSYALARI ===
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d')

    # HTML rapor
    html_path = os.path.join(args.output_dir, f'weekly_analysis_{ts}.html')
    html = generate_html_report(metrics, top_stocks, top_setups_df, ma_freq,
                                  sector_df, bb_split, wow)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✓ HTML: {html_path}")

    # Markdown özet (Telegram için)
    md_path = os.path.join(args.output_dir, f'weekly_summary_{ts}.md')
    md = generate_markdown_summary(metrics, top_stocks, top_setups_df, ma_freq,
                                     sector_df, bb_split, wow)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"✓ Markdown: {md_path}")

    print("\n✓ Analiz tamamlandı.")


if __name__ == '__main__':
    main()
