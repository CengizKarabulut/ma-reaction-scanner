#!/usr/bin/env python3
"""
Telegram Notifier — BIST MA Reaction Scanner sonuçlarını özet halinde gönderir.

Kullanım:
    python scanner/notifier.py --csv reports/scan.csv --mode daily
    python scanner/notifier.py --csv reports/scan.csv --mode weekly

Gerekli env değişkenleri:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
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
    
    # Telegram mesaj limiti 4096 char
    MAX = 4000
    chunks = []
    if len(text) <= MAX:
        chunks = [text]
    else:
        # Satır bazında böl
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


def format_daily(df: pd.DataFrame) -> str:
    """Günlük özet: her hisse için top 3 + robust olanlar"""
    lines = []
    lines.append(f"📊 *BIST MA Reaction Scan — Günlük*")
    lines.append(f"_{datetime.now():%Y-%m-%d %H:%M}_")
    lines.append(f"")
    lines.append(f"*Toplam hisse:* {df['ticker'].nunique()}")
    lines.append(f"*Toplam aday MA:* {len(df):,}")
    
    if 'wf_robust' in df.columns:
        n_robust = df['wf_robust'].sum()
        lines.append(f"*Walk-forward robust:* {n_robust:,} ({100*n_robust/len(df):.1f}%)")
    lines.append(f"")
    
    # En İyi 10 (tüm hisseler arası composite skor sıralı, sadece robust olanlar varsa)
    if 'wf_robust' in df.columns and df['wf_robust'].sum() > 0:
        lines.append(f"🏆 *Robust Top 10 (cross-stock)*")
        top = df[df['wf_robust'] == True].nlargest(10, 'composite_score')
    else:
        lines.append(f"🏆 *Top 10 (cross-stock, composite skor)*")
        top = df.nlargest(10, 'composite_score')
    
    lines.append(f"```")
    lines.append(f"{'Hisse':<8} {'MA':<10} {'Per':<5} {'WR':<5} {'Exp':<6} {'Grade':<5}")
    lines.append(f"{'-'*45}")
    for _, r in top.iterrows():
        ma_per = f"{r['ma_type']} {r['period']}"
        lines.append(f"{r['ticker']:<8} {ma_per:<10} {'':<5} {r['wr_pct']:<5.1f} {r['expectancy']:+5.2f} {r['grade']:<5}")
    lines.append(f"```")
    lines.append(f"")
    
    # Her hissenin top 3'ü (kısa)
    lines.append(f"📋 *Hisse Başı Top 3 MA*")
    lines.append(f"")
    for ticker in sorted(df['ticker'].unique()):
        sub = df[df['ticker'] == ticker].nlargest(3, 'composite_score')
        if len(sub) == 0:
            continue
        top1 = sub.iloc[0]
        line = f"*{ticker}*: {top1['ma_type']} {top1['period']} (Exp {top1['expectancy']:+.2f}, {top1['grade']})"
        if 'wf_robust' in sub.columns and top1.get('wf_robust'):
            line += " ✓"
        lines.append(line)
    
    return '\n'.join(lines)


def format_weekly(df: pd.DataFrame) -> str:
    """Haftalık derinlemesine rapor — cross-stock pattern dahil"""
    lines = []
    lines.append(f"📊 *BIST MA Reaction Scan — Haftalık Derinlemesine*")
    lines.append(f"_{datetime.now():%Y-%m-%d}_")
    lines.append(f"")
    lines.append(f"*Hisse sayısı:* {df['ticker'].nunique()}")
    lines.append(f"*Toplam MA adayı:* {len(df):,}")
    
    if 'wf_robust' in df.columns:
        robust = df[df['wf_robust'] == True]
        lines.append(f"*Walk-forward robust:* {len(robust):,} ({100*len(robust)/len(df):.1f}%)")
    lines.append(f"")
    
    # En popüler MA aileleri (kaç hissede top 10'a girmiş)
    lines.append(f"🎯 *En Yaygın MA'lar (BIST genelinde top 10'da görünme sayısı)*")
    lines.append(f"")
    top_per_stock = (
        df.groupby('ticker', group_keys=False)
        .apply(lambda g: g.nlargest(10, 'composite_score'))
    )
    pop = (
        top_per_stock.groupby(['ma_type', 'period']).size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(15)
    )
    lines.append(f"```")
    lines.append(f"{'MA':<8} {'Per':<5} {'Hisse#':<8}")
    lines.append(f"{'-'*25}")
    for _, r in pop.iterrows():
        lines.append(f"{r['ma_type']:<8} {r['period']:<5} {r['count']:<8}")
    lines.append(f"```")
    lines.append(f"")
    
    # Walk-forward robust top picks (test set expectancy en yüksek)
    if 'wf_test_exp' in df.columns:
        lines.append(f"💎 *Walk-Forward ROBUST — Top 10 (test set expectancy)*")
        lines.append(f"")
        robust_df = df[df['wf_robust'] == True].nlargest(10, 'wf_test_exp')
        lines.append(f"```")
        lines.append(f"{'Hisse':<8} {'MA':<10} {'Train':<7} {'Test':<7}")
        lines.append(f"{'-'*35}")
        for _, r in robust_df.iterrows():
            ma_per = f"{r['ma_type']} {r['period']}"
            lines.append(f"{r['ticker']:<8} {ma_per:<10} {r['wf_train_exp']:+5.2f}  {r['wf_test_exp']:+5.2f}")
        lines.append(f"```")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Scanner çıktısı CSV dosya')
    parser.add_argument('--mode', choices=['daily', 'weekly'], default='daily')
    args = parser.parse_args()
    
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("Hata: TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID env değişkenleri gerekli", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.csv):
        print(f"Hata: CSV bulunamadı: {args.csv}", file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_csv(args.csv)
    if len(df) == 0:
        print("Uyarı: CSV boş", file=sys.stderr)
        send_telegram(token, chat_id, "⚠️ BIST tarama sonucu boş — veri çekme sorunu olabilir.")
        return
    
    if args.mode == 'daily':
        text = format_daily(df)
    else:
        text = format_weekly(df)
    
    send_telegram(token, chat_id, text)
    print(f"Telegram gönderildi ({args.mode} mode, {len(df)} kayıt)")


if __name__ == '__main__':
    main()
