#!/usr/bin/env python3
"""
Telegram Notifier — BIST MA Reaction Scanner sonuçlarını özet halinde gönderir.

500+ hisse modunda (BIST_TUM) otomatik kısa formata geçer.

Kullanım:
    python scanner/notifier.py --csv reports/scan.csv --mode daily
    python scanner/notifier.py --csv reports/scan.csv --mode weekly
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import requests


def send_telegram(token: str, chat_id: str, text: str, parse_mode: str = "Markdown"):
    """Telegram'a mesaj gönder (uzunsa parçala — limit 4096)"""
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

    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk = f"_(devam {i+1}/{len(chunks)})_\n" + chunk
        resp = requests.post(url, json={
            'chat_id': chat_id,
            'text': chunk,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True,
        }, timeout=20)
        if not resp.ok:
            print(f"Telegram error: {resp.status_code} {resp.text}", file=sys.stderr)


def _format_cross_stock_top(df: pd.DataFrame, n: int = 20) -> list:
    """Cross-stock top N (robust öncelikli)"""
    lines = []
    if 'wf_robust' in df.columns and df['wf_robust'].sum() > 0:
        lines.append(f"🏆 *Robust Top {n} (cross-stock)*")
        top = df[df['wf_robust'] == True].nlargest(n, 'composite_score')
    else:
        lines.append(f"🏆 *Top {n} (cross-stock, composite skor)*")
        top = df.nlargest(n, 'composite_score')

    lines.append("```")
    lines.append(f"{'Hisse':<7} {'MA':<6} {'Per':<4} {'WR':<5} {'Exp':<6}")
    lines.append("-" * 35)
    for _, r in top.iterrows():
        lines.append(
            f"{r['ticker']:<7} {r['ma_type']:<6} {r['period']:<4} "
            f"{r['wr_pct']:<5.1f} {r['expectancy']:+.2f}"
        )
    lines.append("```")
    return lines


def _format_ma_family_stats(df: pd.DataFrame, n: int = 12) -> list:
    """En yaygın MA aileleri (kaç hissede top 5'e girmiş)"""
    lines = []
    lines.append(f"🎯 *En Yaygın MA Aileleri (hisse başı top 5'te görünme)*")
    top_per_stock = (
        df.groupby('ticker', group_keys=False)
        .apply(lambda g: g.nlargest(5, 'composite_score'))
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
                for tk, cnt in robust_per_stock.head(12).items():
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
                            etiket = 'D' if ma_val < curr_price else 'R'
                            lines.append(f"  {r['ma_type']:<5} {r['period']:<4} "
                                         f"@{vf} [{etiket}] "
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
                    etiket = 'D' if ma_val < curr_price else 'R'
                    lines.append(f"  {r['ma_type']:<5} {r['period']:<4} "
                                 f"@{vf} [{etiket}] "
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--mode', choices=['daily', 'weekly'], default='daily')
    args = parser.parse_args()

    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        print("Hata: TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID env gerekli", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.csv):
        print(f"Hata: CSV bulunamadı: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if len(df) == 0:
        send_telegram(token, chat_id, "⚠️ BIST tarama sonucu boş — veri çekme sorunu olabilir.")
        return

    text = format_daily(df) if args.mode == 'daily' else format_weekly(df)
    send_telegram(token, chat_id, text)
    print(f"Telegram gönderildi ({args.mode}, {df['ticker'].nunique()} hisse, {len(df)} kayıt)")


if __name__ == '__main__':
    main()
