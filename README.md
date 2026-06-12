# BIST MA Reaction Scanner — Kullanım Kılavuzu

Bu repo BIST hisselerinin **hareketli ortalama saygısını** tarayıp trade fırsatlarına çeviren bir otomasyon sistemidir. **4 farklı workflow** var, hangisini ne zaman kullanacağın net olsun:

## Workflow'lar Arası Fark

| Workflow | Ne Zaman | Ne Yapar | Süre |
|---|---|---|---|
| **Daily Scan** | Hızlı pulse — tek TF | 1 timeframe seçtiğin (1h/4h/1d/1wk/1mo) | 5-30 dk |
| **Multi-TF Cascade** | Derin analiz — 3 TF konsensüsü | 3 timeframe cascade, tümünde robust = altın | 45-90 dk |
| **Index Scan** | Sektör/makro trend | 4 ana endeks taraması | 15-30 dk |
| **Single Ticker** | Tek bir hisseyi her açıdan | Tek hisse derin analiz + mobile rapor | 30 sn |

## "Daily" vs "Multi-TF" — Cengiz'in Sorduğu Soru

**"Daily'de farklı zaman aralıkları seçebiliyorsam Multi-TF'ye gerek var mı?"**

**Cevap:** Evet, ÇOK farklı işler. İşte ayrım:

### Daily Scan — Tek TF
"BIST'te BU bir zaman diliminde ne var?"

- Sadece seçtiğin TF'de tarar
- 1d seçersen → sadece günlük
- 4h seçersen → sadece 4 saatlik
- **Multi-TF konsensüsü YOK**
- Hızlı, günlük takip

### Multi-TF Cascade — 3 TF + Konsensüs
"BIST'te 3 zaman diliminde DE robust olan setup'lar hangileri?"

- **3 TF'yi birden** tarar (cascade ile filtreli)
- TF1 → robust olanları al → TF2'yi sadece onlarda yap → TF3 aynı şekilde
- **Cross-TF Consensus** raporu = D+W+M üçünde de tutarlı → 🥇 ALTIN setup
- Backtest + Multi-Indicator Confirm raporu da otomatik
- 1h+4h+1d (intraday) veya 1d+1wk+1mo (daily) seçeneği var

**Sonuç:**
- Daily = "1 TF kontrol" (hızlı pulse)
- Multi-TF = "3 TF konsensüs" (yatırım kararı)

## Multi-TF Intraday Modu — Nasıl Çalıştırılır?

Sırasıyla:
1. GitHub repo → **Actions** tab
2. Sol menüden **Multi-Timeframe Cascade Scan** seç
3. Sağ üstte **Run workflow** mavi butonuna tıkla
4. Açılan formda:
   - **Tickers:** `BIST_100` veya istediğin
   - **Cascade tipi:** dropdown'dan **`intraday`** seç
   - **no_walk_forward:** `true` (intraday hızlı için)
   - Diğer alanlar default
5. **Run workflow** yeşil butona bas

~30-45 dakika sonra:
- Telegram'a **image-rich tablo** düşer (1h cascade sonuçları)
- Artifact'ta 4 HTML rapor (strategies, backtest, confirm, consensus)
- HTML'lerde her hisse için cluster + DESTEK/DIRENC + RSI/MACD

## Single Ticker (Tek Hisse)

```
Single Ticker → Run
  Ticker: ASELS
  Interval: 4h (1h scalp / 1d klasik / 1wk vs)
  use_latest_csv: false
  strict_mode: false
```

Telegram'a zengin rapor: fiyat + MA değerleri + cluster + RSI/MACD + MA tepki bölgeleri.

## Telegram Format — Yeni Image Modu

`notifier.py` artık iki modda:
- **Image-rich (default):** PNG tablo olarak gönderir. DESTEK/DIRENC renkli arka plan. Mükemmel hizalama.
- **Text-only:** `--text-only` flag ile eski yöntem.

## BIST Endeks Tarama — Neden 4 Endeks?

borsapy kütüphanesi sadece XU030, XU050, XU100, XUTUM destekliyor. Sektör endeksleri (XBANK, XKMYA, XGIDA vs.) `bp.Index()`'te yok. Bu **kütüphane sınırlaması**, kodumuzun değil.

**Sektörel analiz için:** Multi-TF Cascade'le `BIST_100` tara, ardından strategies.html'den sektör hisselerini incele.

## Sorun Giderme

**"Telegram mesajı gelmiyor"**
1. Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` doğru mu
2. Test: `python scanner/notifier.py --test`
3. Log'da `✓ Telegram BAŞARILI` ara

**"Multi-TF cascade çalışıyor ama Telegram boş"**
- `multi_tf_cascade.yml` repo'da güncel mi?
- Step 9 (Telegram Özet) log'una bak

**"Image üretilmiyor, sadece text"**
- `requirements.txt`'te `matplotlib` var mı?
- `pip install matplotlib`

## Repo Yapısı

```
ma-reaction-scanner/
├── .github/workflows/
│   ├── daily_scan.yml
│   ├── multi_tf_cascade.yml
│   ├── index_scan.yml
│   └── single_ticker.yml
├── scanner/
│   ├── bist_ma_reaction_scanner.py
│   ├── tickers.py
│   ├── strategy_generator.py
│   ├── backtest_simulator.py
│   ├── multi_indicator_confirm.py
│   ├── cross_tf_consensus.py
│   ├── single_ticker_query.py
│   └── notifier.py
├── pine/
│   ├── ma_reaction_profile_v6.pine
│   └── ma_trade_signals.pine
├── requirements.txt
└── README.md
```
