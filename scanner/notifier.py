#!/usr/bin/env python3
"""
Telegram Notifier — BIST MA Reaction Scanner sonuçlarını özet halinde gönderir.

500+ hisse modunda (BIST_TUM) otomatik kısa formata geçer.

Kullanım:
    python scanner/notifier.py --csv reports/scan.csv --mode daily
    python scanner/notifier.py --csv reports/scan.csv --mode weekly
"""

import argparse
import io
import os
import sys
from datetime import datetime

import pandas as pd
import requests


def send_telegram(token: str, chat_id: str, text: str, parse_mode: str = "Markdown") -> bool:
    """Telegram'a mesaj gönder (uzunsa parçala — limit 4096).

    Markdown parse hatasi olursa otomatik plain text fallback yapar.
    Returns: True if all chunks sent successfully, False otherwise.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    MAX = 4000
    chunks = []
    if len(text) <= MAX:
        chunks = [text]
    else:
        lines = text.split('\n')
        current = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > MAX:
                chunks.append('\n'.join(current))
                current = [line]
                current_len = line_len
            else:
                current.append(line)
                current_len += line_len
        if current:
            chunks.append('\n'.join(current))

    import re
    all_ok = True

    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk = f"(devam {i+1}/{len(chunks)})\n" + chunk

        # 1. ONCE Markdown ile dene
        resp = requests.post(url, json={
            'chat_id': chat_id,
            'text': chunk,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True,
        }, timeout=20)

        if resp.ok:
            print(f"  ✓ Chunk {i+1}/{len(chunks)} Markdown OK ({len(chunk)} char)", file=sys.stderr)
            continue

        # 2. Markdown basarisizsa - markdown karakterlerini TEMIZLE ve plain dene
        print(f"  ⚠️ Markdown hatasi chunk {i+1}: {resp.text[:200]}", file=sys.stderr)
        plain = chunk.replace('*', '').replace('`', '')
        plain = re.sub(r'_([^_\n]+)_', r'\1', plain)  # _xxx_ → xxx

        resp2 = requests.post(url, json={
            'chat_id': chat_id,
            'text': plain,
            'disable_web_page_preview': True,
        }, timeout=20)
        if resp2.ok:
            print(f"  ✓ Chunk {i+1}/{len(chunks)} plain text OK", file=sys.stderr)
        else:
            print(f"  ✗ Chunk {i+1} plain text de basarisiz: {resp2.status_code} {resp2.text[:300]}", file=sys.stderr)
            all_ok = False

    return all_ok


def send_photo(token: str, chat_id: str, photo_bytes: bytes, caption: str = "") -> bool:
    """Telegram'a fotograf gonder. Tablo image'lerini gondermek icin."""
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {'photo': ('table.png', photo_bytes, 'image/png')}
    data = {'chat_id': chat_id, 'caption': caption}
    try:
        resp = requests.post(url, files=files, data=data, timeout=30)
        if resp.ok:
            print(f"  ✓ Photo gonderildi ({len(photo_bytes)/1024:.1f} KB)", file=sys.stderr)
            return True
        else:
            print(f"  ✗ Photo hata: {resp.status_code} {resp.text[:200]}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"  ✗ Photo exception: {e}", file=sys.stderr)
        return False


def render_table_image(headers: list, rows: list, title: str = "",
                        col_colors: dict = None) -> bytes:
    """Matplotlib ile tablo image uret. Sayfa = PIL bytes dondur.

    Args:
        headers: ['Hisse', 'MA', 'Per', 'Değer', 'WR', 'Exp']
        rows: [['ODAS', 'SMA', '144', '6.09', '93%', '+6.51'], ...]
        title: Üst başlık (emoji icermesin)
        col_colors: {0: ['#7fc97f', ...]} ticker satir renkleri (DESTEK=yesil, DIRENC=kirmizi)
    Returns:
        PNG bytes or None
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Headless
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # Emoji ve özel karakterleri temizle (matplotlib font sorunu)
    import re
    def clean_text(s):
        if not isinstance(s, str):
            return str(s)
        # Emoji (Unicode emojiler 0x1F000+) ve unicode kalp/etiketler temizle
        return re.sub(r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF]+', '', s).strip()

    title_clean = clean_text(title)
    clean_rows = [[clean_text(c) for c in r] for r in rows]
    clean_headers = [clean_text(h) for h in headers]

    n_rows = len(clean_rows)
    n_cols = len(clean_headers)
    if n_rows == 0:
        return None

    # Dinamik figür boyutu - row sayısına göre
    fig_w = max(8, n_cols * 1.4)
    fig_h = max(2, 0.40 * (n_rows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.axis('off')

    if title_clean:
        plt.title(title_clean, fontsize=13, fontweight='bold', loc='left',
                  color='#5fb3ff', pad=12)

    # Tablo
    table = ax.table(cellText=clean_rows, colLabels=clean_headers,
                      cellLoc='center', loc='center',
                      colColours=['#2a2f39'] * n_cols)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Header rengi
    for i in range(n_cols):
        cell = table[0, i]
        cell.set_text_props(color='#5fb3ff', fontweight='bold')
        cell.set_facecolor('#1a1f29')

    # Veri satır renkleri (her ikinci satır vurgu) + ticker sütununa DESTEK/DIRENC arka plan
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            bg = '#0f1419' if i % 2 == 0 else '#1a1f29'
            # Ticker sütunu için özel: yeşil/kırmızı arka plan
            if j == 0 and col_colors and 0 in col_colors and i - 1 < len(col_colors[0]):
                tag_color = col_colors[0][i - 1]
                if tag_color == '#7fc97f':  # Yeşil = DESTEK
                    bg = '#1a2f1f'
                    cell.set_text_props(color='#7fc97f', fontweight='bold')
                elif tag_color == '#ff8c69':  # Kırmızı = DIRENC
                    bg = '#2f1a1a'
                    cell.set_text_props(color='#ff8c69', fontweight='bold')
                else:
                    cell.set_text_props(color='#e6e6e6')
            else:
                cell.set_text_props(color='#e6e6e6')
            cell.set_facecolor(bg)

    fig.patch.set_facecolor('#0a0e14')
    plt.tight_layout()

    # PNG bytes'a yaz
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0a0e14', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _format_cross_stock_top(df: pd.DataFrame, n: int = 20) -> list:
    """Cross-stock top N (robust öncelikli) — fiyat + MA değer + 🟢🔴 etiket dahil.

    n=0 → robust olanların TÜMÜNÜ göster (endeks taraması için faydalı).
    """
    lines = []
    if 'wf_robust' in df.columns and df['wf_robust'].sum() > 0:
        robust_df = df[df['wf_robust'] == True]
        if n == 0 or n >= len(robust_df):
            # Tümünü göster (endeks taraması)
            top = robust_df.sort_values('composite_score', ascending=False)
            lines.append(f"🏆 *Robust TÜM ({len(top)} kayıt, cross-stock)*")
        else:
            top = robust_df.nlargest(n, 'composite_score')
            lines.append(f"🏆 *Robust Top {n} (cross-stock)*")
    else:
        if n == 0 or n >= len(df):
            top = df.sort_values('composite_score', ascending=False)
            lines.append(f"🏆 *TÜM ({len(top)} kayıt, composite skor)*")
        else:
            top = df.nlargest(n, 'composite_score')
            lines.append(f"🏆 *Top {n} (cross-stock, composite skor)*")

    lines.append("```")
    # YENI: Fiyat + MA Değer + Etiket sütunları
    lines.append(f"{'Hisse':<7} {'MA':<5} {'Per':<3} {'MA-Değ':<8} {'WR':<4} {'Exp'}")
    lines.append("-" * 40)
    for _, r in top.iterrows():
        ma_val = r.get('current_ma_value', None)
        curr_price = r.get('current_close', None)

        # MA değer + DESTEK/DIRENC etiketi
        if ma_val is not None and not pd.isna(ma_val) and curr_price is not None and not pd.isna(curr_price):
            if curr_price < 10:
                vf = f"{ma_val:.4f}"
            elif curr_price < 100:
                vf = f"{ma_val:.3f}"
            else:
                vf = f"{ma_val:.2f}"
            etiket = '🟢' if ma_val < curr_price else '🔴'
            lines.append(
                f"{etiket}{r['ticker']:<6} {r['ma_type']:<5} {int(r['period']):<3} "
                f"{vf:<8} {r['wr_pct']:<4.0f} {r['expectancy']:+.2f}"
            )
        else:
            # Fallback (eski format) — fiyat yoksa
            lines.append(
                f" {r['ticker']:<6} {r['ma_type']:<5} {int(r['period']):<3} "
                f"{'—':<8} {r['wr_pct']:<4.0f} {r['expectancy']:+.2f}"
            )
    lines.append("```")
    return lines


def _format_ma_family_stats(df: pd.DataFrame, n: int = 12) -> list:
    """En yaygın MA aileleri (kaç hissede top 5'e girmiş)"""
    lines = []
    lines.append(f"🎯 *En Yaygın MA Aileleri (hisse başı top 5'te görünme)*")
    top_per_stock = (
        df.groupby('ticker', group_keys=False)
        .apply(lambda g: g.nlargest(5, 'composite_score'), include_groups=False)
    )
    pop = (
        top_per_stock.groupby(['ma_type', 'period']).size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(n)
    )
    lines.append("```")
    lines.append(f"{'MA':<6} {'Per':<5} {'Hisse#':<7}")
    lines.append("-" * 22)
    for _, r in pop.iterrows():
        lines.append(f"{r['ma_type']:<6} {r['period']:<5} {r['count']:<7}")
    lines.append("```")
    return lines


def format_daily(df: pd.DataFrame) -> str:
    """Günlük özet — hisse sayısına göre adaptive format"""
    n_stocks = df['ticker'].nunique()
    n_robust = int(df['wf_robust'].sum()) if 'wf_robust' in df.columns else 0
    n_total = len(df)

    lines = []
    lines.append("📊 *BIST MA Reaction Scan — Günlük*")
    lines.append(f"_{datetime.now():%Y-%m-%d %H:%M}_")
    lines.append("")
    lines.append(f"*Hisse:* {n_stocks} | *Aday MA:* {n_total:,}")
    if 'wf_robust' in df.columns:
        lines.append(f"*Robust:* {n_robust:,} ({100*n_robust/max(n_total,1):.1f}%)")
    lines.append("")

    # === SMALL MODE (<50 hisse): her hisse için top 1 ===
    if n_stocks <= 50:
        lines.extend(_format_cross_stock_top(df, n=10))
        lines.append("")
        lines.append("📋 *Hisse Başı Top 3 MA*")
        lines.append("```")
        for ticker in sorted(df['ticker'].unique()):
            sub = df[df['ticker'] == ticker].nlargest(3, 'composite_score')
            if len(sub) == 0:
                continue
            lines.append(f"{ticker}:")
            for _, r in sub.iterrows():
                robust_mark = "✓" if r.get('wf_robust', False) else " "
                lines.append(f"  {r['ma_type']:<5} {r['period']:<4} "
                             f"WR={r['wr_pct']:.0f}% Exp={r['expectancy']:+.2f} {robust_mark}")
        lines.append("```")

    # === MEDIUM MODE (50-150 hisse): cross-stock + robust hisselerin özeti ===
    elif n_stocks <= 150:
        lines.extend(_format_cross_stock_top(df, n=20))
        lines.append("")
        lines.extend(_format_ma_family_stats(df, n=10))
        lines.append("")
        # Robust olan hisselerin sayısı
        if 'wf_robust' in df.columns:
            robust_per_stock = (
                df[df['wf_robust'] == True]
                .groupby('ticker').size()
                .sort_values(ascending=False)
                .head(15)
            )
            if len(robust_per_stock) > 0:
                lines.append(f"💎 *En Çok Robust MA'ya Sahip 12 Hisse — Her Birinin Top 5'i*")
                lines.append("```")
                for tk, cnt in robust_per_stock.head(top_n_stocks).items():
                    sub = (
                        df[(df['ticker'] == tk) & (df['wf_robust'] == True)]
                        .nlargest(5, 'composite_score')
                    )
                    curr_price = sub.iloc[0].get('current_close', None)
                    if curr_price and not pd.isna(curr_price):
                        if curr_price < 10: pf = f"{curr_price:.4f}"
                        elif curr_price < 100: pf = f"{curr_price:.3f}"
                        else: pf = f"{curr_price:.2f}"
                        lines.append(f"{tk} ({pf}) - {cnt} robust MA")
                    else:
                        lines.append(f"{tk} ({cnt} robust MA)")
                    for _, r in sub.iterrows():
                        ma_val = r.get('current_ma_value', None)
                        if ma_val and not pd.isna(ma_val) and curr_price:
                            if curr_price < 10: vf = f"{ma_val:.4f}"
                            elif curr_price < 100: vf = f"{ma_val:.3f}"
                            else: vf = f"{ma_val:.2f}"
                            etiket = '🟢' if ma_val < curr_price else '🔴'
                            lines.append(f"  {etiket} {r['ma_type']:<5} {r['period']:<4} "
                                         f"@{vf} "
                                         f"WR={r['wr_pct']:.0f}% Exp={r['expectancy']:+.2f}")
                        else:
                            lines.append(f"  {r['ma_type']:<5} {r['period']:<4} "
                                         f"WR={r['wr_pct']:.0f}% Exp={r['expectancy']:+.2f}")
                lines.append("```")

    # === LARGE MODE (>150 hisse, BIST_TUM): yüksek seviyeli özet ===
    else:
        lines.extend(_format_cross_stock_top(df, n=20))
        lines.append("")
        lines.extend(_format_ma_family_stats(df, n=12))
        lines.append("")
        # En iyi 12 hisse - HER BIRININ TOP 5 MA'si (Cengiz isteği)
        # composite_score'a göre en yüksek hisseleri seç, sonra her birinin top 5 MA'sını göster
        best_per_stock = (
            df.groupby('ticker')['composite_score']
            .max().sort_values(ascending=False)
            .head(12)
        )
        lines.append("🚀 *En Güçlü 12 Hissenin TOP 5 MA'sı*")
        lines.append("```")
        for tk, _ in best_per_stock.items():
            top5 = df[df['ticker'] == tk].nlargest(5, 'composite_score')
            curr_price = top5.iloc[0].get('current_close', None)
            if curr_price and not pd.isna(curr_price):
                if curr_price < 10: pf = f"{curr_price:.4f}"
                elif curr_price < 100: pf = f"{curr_price:.3f}"
                else: pf = f"{curr_price:.2f}"
                lines.append(f"{tk} ({pf}):")
            else:
                lines.append(f"{tk}:")
            for _, r in top5.iterrows():
                robust_mark = "✓" if r.get('wf_robust', False) else " "
                ma_val = r.get('current_ma_value', None)
                if ma_val and not pd.isna(ma_val) and curr_price:
                    if curr_price < 10: vf = f"{ma_val:.4f}"
                    elif curr_price < 100: vf = f"{ma_val:.3f}"
                    else: vf = f"{ma_val:.2f}"
                    # [D]/[R] yerine emoji - Markdown link sozdizimi sorunu olmaz
                    etiket = '🟢' if ma_val < curr_price else '🔴'
                    lines.append(f"  {etiket} {r['ma_type']:<5} {r['period']:<4} "
                                 f"@{vf} "
                                 f"WR={r['wr_pct']:.0f}% Skor={r['composite_score']:.1f} {robust_mark}")
                else:
                    lines.append(f"  {r['ma_type']:<5} {r['period']:<4} "
                                 f"WR={r['wr_pct']:.0f}% Skor={r['composite_score']:.1f} {robust_mark}")
        lines.append("```")
        lines.append("")
        lines.append("_Detaylı tüm hisseler ve setup'lar için GitHub Actions artifact'ında CSV/HTML dosyalarını indirin._")

    return '\n'.join(lines)


def format_weekly(df: pd.DataFrame) -> str:
    """Haftalık derinlemesine rapor — cross-stock pattern dahil"""
    lines = []
    lines.append("📊 *BIST MA Reaction Scan — Haftalık Derinlemesine*")
    lines.append(f"_{datetime.now():%Y-%m-%d}_")
    lines.append("")
    lines.append(f"*Hisse sayısı:* {df['ticker'].nunique()}")
    lines.append(f"*Toplam MA adayı:* {len(df):,}")

    if 'wf_robust' in df.columns:
        robust = df[df['wf_robust'] == True]
        lines.append(f"*Walk-forward robust:* {len(robust):,} ({100*len(robust)/max(len(df),1):.1f}%)")
    lines.append("")

    lines.extend(_format_ma_family_stats(df, n=15))
    lines.append("")

    if 'wf_test_exp' in df.columns and 'wf_robust' in df.columns:
        lines.append("💎 *Walk-Forward ROBUST — Top 15 (test set expectancy)*")
        robust_df = df[df['wf_robust'] == True].nlargest(15, 'wf_test_exp')
        lines.append("```")
        lines.append(f"{'Hisse':<7} {'MA':<6} {'Per':<4} {'Train':<6} {'Test':<6}")
        lines.append("-" * 35)
        for _, r in robust_df.iterrows():
            lines.append(
                f"{r['ticker']:<7} {r['ma_type']:<6} {r['period']:<4} "
                f"{r['wf_train_exp']:+5.2f} {r['wf_test_exp']:+5.2f}"
            )
        lines.append("```")

    return '\n'.join(lines)


def send_rich_daily(token: str, chat_id: str, df, label: str = 'Tarama',
                     top_n_setups: int = 20, top_n_stocks: int = 12):
    """Image-rich daily özet gönderir. Büyük tablolar image olarak,
    açıklamalar text olarak gider. Image üretilemezse fallback'le text gönderir.

    Args:
        top_n_setups: Cross-stock top tablosundaki satır sayısı (default 20).
                      Endeks tarama için 100 vermek mantıklı (79 endeks dahil).
        top_n_stocks: Hisse başı top 5 MA bölümünde gösterilecek hisse sayısı.
                      Endeks için 80, normal hisse için 12.
    """

    n_stocks = df['ticker'].nunique()
    n_total = len(df)
    n_robust = 0
    robust_pct = 0
    if 'wf_robust' in df.columns:
        n_robust = int(df['wf_robust'].sum())
        robust_pct = 100 * n_robust / max(n_total, 1)

    # 1. KISA TEXT ÖZET
    header_text = (
        f"📊 *{label}*\n"
        f"{pd.Timestamp.now():%Y-%m-%d %H:%M}\n\n"
        f"📈 Hisse/Endeks: *{n_stocks}* | Toplam MA: *{n_total:,}*\n"
        f"✅ Robust: *{n_robust:,}* ({robust_pct:.1f}%)\n"
    )
    send_telegram(token, chat_id, header_text, parse_mode='Markdown')

    # 2. CROSS-STOCK TOP — IMAGE TABLO (top_n_setups=0 → TÜMÜ göster)
    if 'wf_robust' in df.columns and df['wf_robust'].sum() > 0:
        robust_df = df[df['wf_robust'] == True]
        if top_n_setups == 0 or top_n_setups >= len(robust_df):
            top20 = robust_df.sort_values('composite_score', ascending=False)
            title = f"🏆 Robust TÜM ({len(top20)})"
        else:
            top20 = robust_df.nlargest(top_n_setups, 'composite_score')
            title = f"🏆 Robust Top {len(top20)} (Cross-Stock)"
    else:
        if top_n_setups == 0 or top_n_setups >= len(df):
            top20 = df.sort_values('composite_score', ascending=False)
            title = f"🏆 TÜM ({len(top20)}, Composite Score)"
        else:
            top20 = df.nlargest(top_n_setups, 'composite_score')
            title = f"🏆 Top {len(top20)} (Composite Score)"

    rows = []
    colors_col = {0: []}  # ticker sütunu için renk listesi
    for _, r in top20.iterrows():
        ma_val = r.get('current_ma_value', None)
        curr_price = r.get('current_close', None)

        if ma_val is not None and not pd.isna(ma_val) and curr_price is not None:
            if curr_price < 10:
                vf = f"{ma_val:.4f}"
            elif curr_price < 100:
                vf = f"{ma_val:.3f}"
            else:
                vf = f"{ma_val:.2f}"
            etiket = '🟢' if ma_val < curr_price else '🔴'
            tcolor = '#7fc97f' if ma_val < curr_price else '#ff8c69'
        else:
            vf = '—'
            etiket = ' '
            tcolor = None

        rows.append([
            f"{etiket} {r['ticker']}",
            r['ma_type'],
            str(int(r['period'])),
            vf,
            f"{r['wr_pct']:.0f}%",
            f"{r['expectancy']:+.2f}",
        ])
        colors_col[0].append(tcolor)

    headers = ['Hisse', 'MA', 'Per', 'MA Değer', 'WR', 'Exp']
    img_bytes = render_table_image(headers, rows, title=title, col_colors=colors_col)
    if img_bytes:
        send_photo(token, chat_id, img_bytes, caption=f"{title} — {n_stocks} hisse arasından")
    else:
        # Fallback - text gönder
        text_lines = [title, '```']
        text_lines.append(f"{'Hisse':<8} {'MA':<5} {'Per':<4} {'Değer':<10} {'WR':<5} {'Exp'}")
        text_lines.append('-' * 45)
        for row in rows:
            text_lines.append(f"{row[0]:<8} {row[1]:<5} {row[2]:<4} {row[3]:<10} {row[4]:<5} {row[5]}")
        text_lines.append('```')
        send_telegram(token, chat_id, '\n'.join(text_lines), parse_mode='Markdown')

    # 3. EN YAYGIN MA AİLELERİ (kısa text)
    fam_lines = _format_ma_family_stats(df, n=10)
    send_telegram(token, chat_id, '\n'.join(fam_lines), parse_mode='Markdown')

    # 4. EN ÇOK ROBUST HİSSE - HER BİRİ İÇİN AYRI IMAGE TABLO
    if 'wf_robust' in df.columns and n_robust > 0:
        # Hisse başı robust MA sayısına göre sırala (top_n_stocks=0 → tümü)
        robust_per_stock_full = (
            df[df['wf_robust'] == True]
            .groupby('ticker', group_keys=False)
            .size().sort_values(ascending=False)
        )
        if top_n_stocks == 0 or top_n_stocks >= len(robust_per_stock_full):
            robust_per_stock = robust_per_stock_full
        else:
            robust_per_stock = robust_per_stock_full.head(top_n_stocks)

        # Tüm hisselerin tek bir image'da kombinasyonu (daha sade)
        all_rows = []
        all_colors = {0: []}
        for tk, cnt in robust_per_stock.items():
            top5 = df[df['ticker'] == tk].nlargest(5, 'composite_score')
            curr_price = top5.iloc[0].get('current_close', None)
            for idx, (_, r) in enumerate(top5.iterrows()):
                ma_val = r.get('current_ma_value', None)
                if ma_val is not None and not pd.isna(ma_val) and curr_price is not None:
                    if curr_price < 10:
                        vf = f"{ma_val:.4f}"
                        pf = f"{curr_price:.4f}"
                    elif curr_price < 100:
                        vf = f"{ma_val:.3f}"
                        pf = f"{curr_price:.3f}"
                    else:
                        vf = f"{ma_val:.2f}"
                        pf = f"{curr_price:.2f}"
                    etiket = '🟢' if ma_val < curr_price else '🔴'
                    tcolor = '#7fc97f' if ma_val < curr_price else '#ff8c69'
                else:
                    vf = '—'; pf = '—'; etiket = ' '; tcolor = None

                # İlk satırda hisse adı + fiyat, diğer satırlar boş
                tk_label = f"{tk} ({pf})" if idx == 0 else ""
                all_rows.append([
                    tk_label,
                    f"{etiket} {r['ma_type']}",
                    str(int(r['period'])),
                    vf,
                    f"{r['wr_pct']:.0f}%",
                    f"{r['expectancy']:+.2f}",
                ])
                all_colors[0].append(tcolor)

        headers2 = ['Hisse (Fiyat)', 'MA', 'Per', 'MA Değer', 'WR', 'Exp']
        img2 = render_table_image(
            headers2, all_rows,
            title=f"💎 En Çok Robust MA'lı {len(robust_per_stock)} Hisse — Her Birinin Top 5'i",
            col_colors=all_colors
        )
        if img2:
            send_photo(token, chat_id, img2, caption="💎 Top 12 Hisse Detay")
        else:
            # Fallback - mevcut text format
            text2 = _build_large_mode_text(df, robust_per_stock)
            send_telegram(token, chat_id, text2, parse_mode='Markdown')


def _build_large_mode_text(df, robust_per_stock):
    """Image render edemezse fallback text format."""
    lines = ["💎 *En Çok Robust MA'lı 12 Hisse*", "```"]
    for tk, cnt in robust_per_stock.items():
        top5 = df[df['ticker'] == tk].nlargest(5, 'composite_score')
        curr_price = top5.iloc[0].get('current_close', None)
        if curr_price and not pd.isna(curr_price):
            if curr_price < 10: pf = f"{curr_price:.4f}"
            elif curr_price < 100: pf = f"{curr_price:.3f}"
            else: pf = f"{curr_price:.2f}"
            lines.append(f"{tk} ({pf}):")
        else:
            lines.append(f"{tk}:")
        for _, r in top5.iterrows():
            ma_val = r.get('current_ma_value', None)
            if ma_val and not pd.isna(ma_val) and curr_price:
                if curr_price < 10: vf = f"{ma_val:.4f}"
                elif curr_price < 100: vf = f"{ma_val:.3f}"
                else: vf = f"{ma_val:.2f}"
                etiket = '🟢' if ma_val < curr_price else '🔴'
                lines.append(f"  {etiket} {r['ma_type']:<5} {int(r['period']):<4} @{vf} WR={r['wr_pct']:.0f}%")
            else:
                lines.append(f"  {r['ma_type']:<5} {int(r['period']):<4} WR={r['wr_pct']:.0f}%")
    lines.append("```")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=False)
    parser.add_argument('--mode', choices=['daily', 'weekly'], default='daily')
    parser.add_argument('--label', type=str, default='BIST MA Reaction Scan',
                       help='Mesaj başlığı (ör: "BIST Endeks Tarama")')
    parser.add_argument('--top_setups', type=int, default=20,
                       help='Cross-stock tablo satır sayısı (endeks için 100 verebilirsin)')
    parser.add_argument('--top_stocks', type=int, default=12,
                       help='Hisse başı top 5 MA gösterilecek hisse sayısı')
    parser.add_argument('--test', action='store_true',
                       help='Sadece basit test mesaji gonder (CSV gerekmez)')
    parser.add_argument('--text-only', action='store_true',
                       help='Sadece text gonder (image yok)')
    args = parser.parse_args()

    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        print("✗ Hata: TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID env gerekli", file=sys.stderr)
        sys.exit(1)

    # Test modu
    if args.test:
        test_msg = "🧪 Test mesajı — Telegram bağlantısı çalışıyor!\n\n" \
                   "Eğer bu mesajı görüyorsan secrets doğru, bot aktif."
        success = send_telegram(token, chat_id, test_msg, parse_mode=None)
        if success:
            print("✓ Test mesajı gönderildi")
            sys.exit(0)
        else:
            print("✗ Test mesajı GİTMEDİ", file=sys.stderr)
            sys.exit(1)

    if not args.csv:
        print("✗ Hata: --csv gerekli (test için --test kullan)", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.csv):
        print(f"✗ Hata: CSV bulunamadı: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if len(df) == 0:
        send_telegram(token, chat_id, "⚠️ BIST tarama sonucu boş — veri çekme sorunu olabilir.",
                       parse_mode=None)
        return

    print(f"Mesaj hazirlaniyor: {df['ticker'].nunique()} hisse, {len(df)} kayit")

    if args.text_only:
        # ESKI YONTEM: sadece text
        text = format_daily(df) if args.mode == 'daily' else format_weekly(df)
        print(f"Text uzunlugu: {len(text)} char")
        success = send_telegram(token, chat_id, text)
        if success:
            print(f"✓ Telegram BAŞARILI (text-only, {args.mode})")
        else:
            print(f"✗ Telegram BAŞARISIZ", file=sys.stderr)
            sys.exit(1)
    else:
        # YENI YONTEM: image-rich daily ozet
        try:
            send_rich_daily(token, chat_id, df, label=args.label,
                            top_n_setups=args.top_setups,
                            top_n_stocks=args.top_stocks)
            print(f"✓ Telegram BAŞARILI (image-rich, {args.mode})")
        except Exception as e:
            print(f"⚠️ Rich mode hata: {e} - text fallback", file=sys.stderr)
            text = format_daily(df) if args.mode == 'daily' else format_weekly(df)
            success = send_telegram(token, chat_id, text)
            if not success:
                sys.exit(1)


if __name__ == '__main__':
    main()
