# BIST MA Reaction Profile

Hangi hareketli ortalama (MA) periyot ve türlerine BIST hisseleri **gerçekten saygı duyuyor?** sorusunu istatistiksel olarak cevaplayan iki katmanlı analiz sistemi:

1. **Pine Script indikatörü** — TradingView'da gerçek zamanlı görselleştirme ve karar desteği
2. **Python tarama motoru** — BIST 100 geneli için walk-forward doğrulamalı batch analiz

## Özellikler

### Metodoloji
- **Pre-touch separation** filtresi (kısa MA "sahte saygı" bias'ı düzeltilmiş)
- **MAE + MFE** ölçümü her dokunmada (sadece WR değil)
- **Expectancy** = `WR × Avg_MFE − (1−WR) × Avg_MAE` (fund-manager metric)
- **Breakthrough sayımı** ve **Respect Ratio**
- **Composite skor** = `expectancy × √touches × respect_ratio`
- **Walk-forward** (Python tarafında): %70 training / %30 out-of-sample test
- **Rejim ayrımı** (ADX trend/yatay)
- **7 MA türü** × **20 periyot** = 140 kombinasyon

### Otomasyon
- **Günlük tarama** — Her BIST kapanışı sonrası (18:30 İstanbul) tüm BIST 100 tarama
- **Haftalık derinlemesine analiz** — Cumartesi sabah cross-stock pattern analizi
- **Telegram bildirim** — Sonuçlar Telegram'a markdown formatında özet
- **HTML rapor** — Renkli, sıralı görsel raporlar
- **CSV çıktı** — Excel'de işleme uygun ham veri

## Kurulum

### 1. Repo'yu klonla
```bash
git clone https://github.com/<kullanıcı>/ma-reaction-scanner.git
cd ma-reaction-scanner
```

### 2. Bağımlılıklar
```bash
pip install -r requirements.txt
```

### 3. Pine Script — TradingView
`pine/ma_reaction_profile_v6.pine` dosyasını TradingView Pine Editor'a kopyala, indikatör olarak ekle. Detaylar: [pine/USAGE.md](pine/USAGE.md)

### 4. Python Scanner — Lokal Test
```bash
# Hızlı test
python scanner/bist_ma_reaction_scanner.py --tickers ASELS,GARAN,THYAO

# BIST 100 tam tarama
python scanner/bist_ma_reaction_scanner.py --all-bist --period 5y

# Walk-forward atla (hızlı)
python scanner/bist_ma_reaction_scanner.py --all-bist --no_walk_forward
```

### 5. GitHub Actions — Otomasyon

#### Secret'ları ayarla:
Repo `Settings → Secrets and variables → Actions` altında:
- `TELEGRAM_BOT_TOKEN` — [BotFather](https://t.me/BotFather)'dan al
- `TELEGRAM_CHAT_ID` — Bot'a mesaj at, sonra `https://api.telegram.org/bot<TOKEN>/getUpdates` ile chat_id'i bul

#### Workflow'lar
- `.github/workflows/daily_scan.yml` — Her hafta içi 18:30 İstanbul (cron `30 15 * * 1-5`)
- `.github/workflows/weekly_robustness.yml` — Cumartesi 09:00 İstanbul

#### Manuel tetikleme
`Actions → Daily BIST MA Reaction Scan → Run workflow`

## Çıktı Yapısı

### Telegram (günlük)
```
📊 BIST MA Reaction Scan — Günlük
2026-06-06 18:35

Toplam hisse: 47
Toplam aday MA: 4,832
Walk-forward robust: 412 (8.5%)

🏆 Robust Top 10 (cross-stock)
Hisse    MA          WR    Exp    Grade
ASELS    HMA 21      72.5  +4.20  A+
ASELS    ALMA 20     75.5  +4.18  A+
GARAN    EMA 50      68.2  +3.85  A+
...
```

### CSV
Her satır = (hisse × MA türü × periyot) için tam metrik seti:
```
ticker, ma_type, period, touches, wr_pct, avg_mfe, avg_mae, expectancy,
trend_wr_pct, range_wr_pct, grade, composite_score,
current_ma_value, current_close, distance_pct,
wf_train_exp, wf_test_exp, wf_train_wr, wf_test_wr, wf_robust
```

### HTML Rapor
Her hisse için top 10 MA, renkli expectancy (yeşil/kırmızı), robust olanlar vurgulu.

### Cross-Stock Analiz (Haftalık)
"ALMA 20-22 hangi hisseler için top 10'da?" — BIST geneli pattern bulma:
- Evrensel MA aileleri (≥%70 hissede top 10'da)
- Hisse-spesifik MA'lar
- MA türü dağılımı
- Kaliteli hisseler (yüksek robust oranı)

## Mimari Notu

**Pine** → reactive görselleştirme. Gerçek zamanlı seviye takibi.
**Python** → static analysis. Walk-forward, multi-stock pattern, robustluk.

Walk-forward Pine'da yapılamaz (140 MA × 2 dönem = 1960 ek var değişkeni, Pine limiti ~1000). Doğru iş bölümü budur.

## Parametre Ayarları

`scanner/bist_ma_reaction_scanner.py` argümanları:
- `--react_bars` (varsayılan 5) — Dokunma sonrası inceleme penceresi
- `--react_pct` (varsayılan 1.5) — Min tepki büyüklüğü %
- `--atr_mult` (varsayılan 0.2) — Touch zone genişliği (ATR çarpanı)
- `--adx_threshold` (varsayılan 25) — Trend/yatay eşik
- `--min_touches` (varsayılan 10) — Min dokunma sayısı

Volatil hisseler için `atr_mult=0.3`, scalping için `react_bars=3 react_pct=0.5` deneyebilirsin.

## Lisans

MIT
