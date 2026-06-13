# BIST MA Reaction Scanner — v6.3 (Borsapy v0.10+ Native)

Türkiye Borsası (BIST) hisseleri için **hareketli ortalama saygısını** tarayan profesyonel otomasyon sistemi. Borsapy v0.10+'nın TradingView WebSocket backend'i sayesinde **tüm BIST verilerine** tam erişim.

## Yenilikler (v6.3 — Borsapy v0.10 Native)

- ✅ **Intraday tam destek:** 1m, 5m, 15m, 30m, 1h, 4h (borsapy native + resample)
- ✅ **79 BIST endeksi:** Tümü artık erişilebilir (önceki 4 sınırı yok)
- ✅ **Endeks bileşenleri otomatik:** `BIST_BILES:XBANK` → tüm banka hisseleri
- ✅ **Dinamik sektör:** `BIST_SEKTOR:Bankacılık` → borsapy'den otomatik
- ✅ **Image-rich Telegram:** PNG tablo, DESTEK/DIRENC renkli arka plan
- ✅ Bug fix: 1h/4h/1d aynı sonuç hatası çözüldü

## Workflow Tablosu

| Workflow | Ne Yapar | Süre |
|---|---|---|
| **Daily Scan** | Tek TF tarama, hızlı pulse | 5-30 dk |
| **Multi-TF Cascade** | 3 TF konsensüsü (D+W+M veya 1h+4h+1d) | 45-90 dk |
| **Index Scan** | BIST endeksleri (4/36/79 seç) | 15-30 dk |
| **Single Ticker** | Tek hisse derin analiz | 30 sn |
| **Build Sector Cache** | Sektör cache oluştur (ayda 1) | 5 dk |

## YENİ — Sektör & Endeks Bileşenleri (En Güçlü Özellik)

Borsapy v0.10 ile artık BIST endekslerinin bileşen hisselerini direkt çekebiliyoruz. **Hardcoded liste tutmak gereksiz**.

### Endeks Bileşenleri (`BIST_BILES:`)

XBANK endeksinin tüm bileşen hisselerini topluca tara:

```
Daily Scan → Run
  Tickers: BIST_BILES:XBANK    ← Otomatik tüm banka hisseleri
  Interval: 1d
```

Çıktı:
```
Endeks bileşenleri: XBANK → 12 hisse (borsapy)
Tarama başlıyor: 12 hisse × 140 MA = 1,680 aday
```

**Tüm sektör endeksleri için çalışır:**
- `BIST_BILES:XBANK` → Bankalar
- `BIST_BILES:XKMYA` → Kimya
- `BIST_BILES:XGIDA` → Gıda
- `BIST_BILES:XTEKS` → Tekstil
- `BIST_BILES:XHOLD` → Holding
- `BIST_BILES:XU100` → BIST 100 bileşenleri
- ... 79 endeksin **hepsi**

### Sektör Adı ile (`BIST_SEKTOR:`)

```
Daily Scan → Run
  Tickers: BIST_SEKTOR:Bankacılık
  Interval: 1d
```

Bunun için önce **build_sector_cache** workflow'unu bir kez çalıştır.

### Tüm 79 BIST Endeksi

```
Index Scan → Run
  Tickers: BIST_TUM_ENDEKSLER    ← borsapy'den dinamik 79 endeks
  Period: 5y
  Interval: 1d
```

## Daily vs Multi-TF Cascade

### Daily Scan — Tek Timeframe
```
Tickers: BIST_BILES:XBANK
Interval: 1d   ← TEK timeframe
```
Hızlı pulse, günlük takip.

### Multi-TF Cascade — 3 TF + Konsensüs
```
Tickers: BIST_BILES:XBANK
Cascade tipi: daily       (1d+1wk+1mo) veya
              intraday    (1h+4h+1d)
```

**Multi-TF'nin avantajları:**
1. 3 TF birden tarar (Daily ile 3 ayrı run gerekir)
2. Filtreleme yapar (TF1 robust olmayanı atlar — hızlı)
3. **Cross-TF Consensus** = 3 TF'de de robust = 🥇 ALTIN setup
4. Backtest + Multi-Indicator Confirm raporu otomatik

## Tüm Interval Desteği

| Interval | Borsapy | Yöntem |
|---|---|---|
| 1m, 3m, 5m, 15m, 30m, 45m | Native | bp.Ticker.history(interval='1m', ...) |
| **1h** | Native | bp.Ticker.history(interval='1h', period='1ay') |
| **4h** | Türetilen | 1h çekilir → 4h resample |
| 1d | Native | bp.Ticker.history(interval='1d', period='3y') |
| 1wk | Türetilen | 1d çekilir → W resample |
| 1mo | Türetilen | 1d çekilir → ME resample |

Önceki versiyonlardaki "hepsinde aynı sonuç" bug'ı kalktı. Her interval gerçekten farklı veri çekiyor.

## Telegram Image-Rich Tablo

`notifier.py` artık PNG tablo image üretiyor:
- DESTEK hisseleri: **yeşil arka plan + yeşil yazı**
- DIRENC hisseleri: **kırmızı arka plan + kırmızı yazı**
- Tüm sütunlar mükemmel hizalı
- Karanlık tema, profesyonel görünüm

Text fallback otomatik: image üretilemezse text gönderir.

## Hızlı Komut Referansı

### Endeks Bileşenleri
```bash
python scanner/sector_resolver.py --index XBANK
# → AKBNK,GARAN,ISCTR,...

python scanner/sector_resolver.py --list-indices
# → 79 endeks listesi
```

### Sektör
```bash
python scanner/sector_resolver.py --list-sectors
# → borsapy sektörleri (53 adet, Türkçe)

python scanner/sector_resolver.py --sector "Bankacılık"
# → cache'ten banka hisseleri

python scanner/sector_resolver.py --build-cache
# → 500+ hisse için sektör cache (5 dk)
```

### Tek Hisse Sorgu
```bash
python scanner/single_ticker_query.py --ticker ASELS --interval 4h
```

### Lokal Test
```bash
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... \
  python scanner/notifier.py --test
```

## Repo Yapısı

```
ma-reaction-scanner/
├── .github/workflows/
│   ├── daily_scan.yml               # Tek TF tarama
│   ├── multi_tf_cascade.yml         # 3 TF cascade (daily/intraday)
│   ├── index_scan.yml               # 4/36/79 endeks
│   ├── single_ticker.yml            # Tek hisse
│   └── build_sector_cache.yml       # Sektör cache (ayda 1)
├── scanner/
│   ├── bist_ma_reaction_scanner.py  # Ana scanner (borsapy v0.10)
│   ├── tickers.py                   # Dinamik liste + BIST_BILES/BIST_SEKTOR
│   ├── sector_resolver.py           # Sektör/endeks çözücü (borsapy)
│   ├── strategy_generator.py
│   ├── backtest_simulator.py
│   ├── multi_indicator_confirm.py
│   ├── cross_tf_consensus.py
│   ├── single_ticker_query.py
│   └── notifier.py                  # Image-rich Telegram
├── pine/
│   ├── ma_reaction_profile_v6.pine
│   └── ma_trade_signals.pine
├── requirements.txt
└── README.md
```

## Sorun Giderme

**"Telegram mesajı gelmiyor"**
1. Secrets kontrol: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
2. `python scanner/notifier.py --test` ile test
3. Log'da `✓ Telegram BAŞARILI` ara

**"BIST_SEKTOR:xxx çalışmıyor"**
- Önce `Actions → Sektör Cache Oluştur → Run` ile cache oluştur
- Sonra `BIST_SEKTOR:Bankacılık` çalışır

**"borsapy intraday hata veriyor"**
- borsapy versiyonu eski olabilir
- `requirements.txt`'te `borsapy>=0.10.0` zorla
- yfinance fallback otomatik devreye girer

**"Image üretilmiyor"**
- `pip install matplotlib`
- requirements.txt'te `matplotlib` var mı kontrol

## Önemli Notlar

### TradingView Gecikme
Borsapy default ~15 dakika gecikmeli veri kullanır. **Real-time veri** için:
1. TradingView Pro/Pro+/Premium abonelik
2. BIST Real-time Market Data paketi (ek ücretli)
3. `bp.set_tradingview_auth(session=..., session_sign=...)` ile auth

Şu an gecikmeli veri scanner için **yeterli** — günlük tarama EOD verisi kullanıyor.

### Veri Period Limitleri (Borsapy)
- **1m:** Son 1 gün
- **5m, 15m, 30m:** Son 5 gün
- **1h:** Son 1 ay
- **1d+:** 3y, 5y, max destekli

Scanner bunları otomatik ayarlıyor (interval'a göre period seçiyor).
