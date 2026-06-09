# BIST MA Reaction Scanner — Tam Trade Platformu

BIST hisselerinin **hareketli ortalamalara saygısını** istatistiksel olarak analiz eden ve trade önerisi üreten kapsamlı sistem.

## Pipeline Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Daily Scan (BIST_TUM / BIST_100 / sektör endeksleri)│
│         → reports/scan_DATE_1d.csv                          │
└────────────────────┬────────────────────────────────────────┘
                     │ daily-robust hisseler
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Weekly + Monthly Scan (sadece daily-robust hisseler)│
│         → reports/scan_DATE_1wk.csv, _1mo.csv               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Cross-TF Consensus                                  │
│         🥇 Altın: D+W+M üçünde de robust olan kombinasyonlar│
│         🥈 Gümüş: İki TF'de robust olanlar                  │
│         → reports/consensus_DATE.html                       │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
┌──────────┐  ┌──────────┐  ┌────────────┐
│ Strategy │  │ Backtest │  │ Multi-Ind  │
│Generator │  │Simulator │  │Confirmation│
└──────────┘  └──────────┘  └────────────┘
   somut         tarihi         RSI+MACD+BB+
   entry/exit    PnL/DD         İchimoku+OBV+SMI
   her hisse     her MA         confirmation score
```

## Modüller

### 1. Scanner (`scanner/bist_ma_reaction_scanner.py`)
- 7 MA türü × 20 periyot = 140 MA per hisse
- v6.1 metodoloji: pre-touch separation, breakthrough, respect ratio, ADR
- Walk-forward analizi (sıkı kriter: train×0.5 ≤ test, WR %20 düşmez, ≥5 test touch)
- Multi-timeframe: `--interval 1d|1wk|1mo`

### 2. Tickers (`scanner/tickers.py`)
- Dinamik endeks bileşenleri (borsapy ile)
- `BIST_30`, `BIST_50`, `BIST_100`, **`BIST_TUM` (500+ hisse)**
- Sektör endeksleri: `XBANK`, `XUSIN`, `XUMAL`, `XK030` (Katılım), vs

### 3. Strategy Generator (`scanner/strategy_generator.py`)
- Her robust hisse için somut entry/stop/TP1/TP2/trailing önerisi
- Position sizing (portföy %1 risk per trade)
- Setup durumu: WATCH / WAIT_FOR_TOUCH / NO_TRADE
- HTML rapor

### 4. Cross-TF Consensus (`scanner/cross_tf_consensus.py`)
- D + W + M tarama sonuçlarını birleştir
- "Süper-robust" = üç timeframe'de de geçerli olan kombinasyonlar
- HTML + CSV çıktı

### 5. Backtest Simulator (`scanner/backtest_simulator.py`)
- Robust MA'lar için tarihi trade simülasyonu
- TP1 → TP2 → Trailing → Stop logic, komisyon + slippage dahil
- Win rate, Profit Factor, Max Drawdown, Total Return

### 6. Multi-Indicator Confirmation (`scanner/multi_indicator_confirm.py`)
- RSI, MACD, Bollinger Bands, İchimoku, OBV, SMI
- Her indikatör vote: -1 / 0 / +1
- Confirmation score: -6..+6
- 🥇 Altın setup'lar: score +4 veya üstü

### 7. Telegram Notifier (`scanner/notifier.py`)
- Hisse sayısına göre adaptive format
- BIST_30: her hisse top 3 MA
- BIST_100: en çok robust MA'lı 15 hisse + detayları
- BIST_TUM: cross-stock top 30 + MA family stats

### 8. Pine v6.2 (`pine/ma_trade_signals.pine`)
- TradingView'da görsel entry/exit ok'ları
- Stop, TP1, TP2, Trailing çizgileri
- Setup durum tablosu (state machine: SEARCHING / READY / IN POSITION)
- LONG/SHORT alert conditions

## Tipik Kullanım

### A. Otomatik (Workflow)
```bash
# Multi-TF Cascade (önerilen, her hafta sonu)
Actions → Multi-Timeframe Cascade Scan → Run workflow
  Tickers: BIST_100
  Period: 3y
  Portfolio: 100000
```

90 dakika içinde elinizde olur:
- 3 ayrı CSV (D, W, M)
- Consensus raporu (altın/gümüş kombinasyonlar)
- Strategy raporu (somut trade önerileri)
- Backtest sonuçları (tarihi performans)
- Confirmation raporu (RSI/MACD/BB doğrulama)
- Telegram özeti

### B. Manuel (CLI)
```bash
# 1. Daily tarama
python scanner/bist_ma_reaction_scanner.py \
    --all-bist --bist-list BIST_100 \
    --source borsapy --interval 1d --period 3y \
    --output reports/scan_today_1d.csv

# 2. Strategy raporu
python scanner/strategy_generator.py \
    --csv reports/scan_today_1d.csv \
    --tickers ASELS,BRSAN,GARAN \
    --portfolio 100000 --risk_pct 1.0 \
    --output reports/strategies.html

# 3. Backtest
python scanner/backtest_simulator.py \
    --csv reports/scan_today_1d.csv \
    --tickers ASELS \
    --top_ma 3 --output reports/backtest_asels.html

# 4. Confirmation
python scanner/multi_indicator_confirm.py \
    --csv reports/scan_today_1d.csv \
    --all-robust --n_tickers 15 \
    --output reports/confirm.html
```

### C. TradingView (Pine)
1. Pine Editor'a `pine/ma_trade_signals.pine`'i yapıştır
2. Save → Add to chart
3. Hisse aç (örn. BRSAN)
4. İndikatör ayarlarında scanner'dan bulduğun MA'yı gir (örn. HMA 21)
5. Avg MFE %'ini ayarla (CSV'den)
6. Portföy + risk %'ini gir
7. Tablo + ok'lar + Stop/TP çizgileri otomatik

## Trade Akışı (Önerilen)

1. **Hafta sonu:** Cascade workflow çalıştır
2. **Consensus raporunda altın listeyi gör** — D+W+M'de tutarlı kombinasyonlar
3. **Confirmation raporunda score +4 olanları seç** — multi-indicator doğrulamalı
4. **Backtest raporunda PnL > 0 ve Profit Factor > 1.5 olanları öne çıkar**
5. **TradingView'da Pine indikatörünü çalıştır** — canlı setup takibi
6. **Setup geldiğinde** strategy_generator.py'nin önerdiği lot ile pozisyon aç
7. **TP1/TP2/Trailing/Stop**'u takip et — duygusal karar vermeden çık

## Risk Notları

- Algoritma tarihi veriye dayanır, gelecek garanti değil
- Walk-forward yapsak da overfitting riski var
- Slippage + komisyon backtest'te dahil, ama gerçekte daha yüksek olabilir
- WR %85 olsa bile 3-4 üst üste kayıp serisi olabilir (psikolojik dayanma şart)
- Önce **paper trade** ile 20-30 setup'ı dene, sonra canlı paraya geç
- Bir hisseye birden fazla MA için aynı anda pozisyon açma (korelasyon)

## Dosya Yapısı

```
ma-reaction-scanner/
├── pine/
│   ├── ma_reaction_profile_v6.pine    # Ana tarayıcı (140 MA matrix tablo)
│   └── ma_trade_signals.pine          # Trade sinyalleri (entry/exit)
├── scanner/
│   ├── bist_ma_reaction_scanner.py    # Ana tarayıcı
│   ├── tickers.py                     # Endeks bileşenleri (borsapy dinamik)
│   ├── strategy_generator.py          # Trade önerisi raporu
│   ├── cross_tf_consensus.py          # D+W+M consensus
│   ├── backtest_simulator.py          # Tarihi PnL simülasyonu
│   ├── multi_indicator_confirm.py     # RSI/MACD/BB/İchimoku/OBV/SMI
│   └── notifier.py                    # Telegram bildirimi
├── .github/workflows/
│   ├── daily_scan.yml                 # Günlük tarama
│   ├── multi_tf_cascade.yml           # Multi-TF + tüm modüller
│   └── weekly_robustness.yml          # Haftalık robust analizi
└── requirements.txt
```

## Bağımlılıklar

```
borsapy>=0.5.0     # TradingView WebSocket (BIST için tavsiye)
yfinance>=0.2.30   # Yahoo Finance (fallback)
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
```

## License

MIT
