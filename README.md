# MA Reaction Scanner — Guarded Research Pipeline

Bu depo, fiyatın hareketli ortalamalara bağımsız dokunuşlardan sonra verdiği tepkinin
rastgele ve alternatif seviyelerden daha güçlü ve tekrarlanabilir olup olmadığını
araştırır. Ana çalışma hattı `scanner/ma_core.py` ve `scanner/ma_research_cli.py`dır.

`CERTIFIED`, bir seviyenin discovery, eşleştirilmiş kontrol/FDR, validation ve
dokunulmamış holdout kapılarını geçtiğini belirtir. Gelecekte kesin tepki veya kâr
garantisi değildir.

Ayrıntılar:

- [Araştırma metodolojisi](docs/METHODOLOGY.md)
- [Varlık evrenleri ve tek satırlık özet](docs/ASSET_UNIVERSES.md)
- [Eski sistemden geçiş](docs/MIGRATION.md)

## GitHub Actions

Actions ekranı bilinçli olarak yedi workflow ile sınırlandırılmıştır:

| Workflow | Kullanım |
|---|---|
| `Guarded MA Research Panel (Manual)` | Esnek evren, sektör, zaman dilimi ve MA araştırması |
| `Daily Guarded MA Scan` | Açılır menüden seçilen evrende tek zaman dilimi taraması |
| `Guarded Multi-Timeframe Scan` | Daily: 1d+1wk+1mo veya intraday: 1h+4h+1d |
| `Guarded Single Instrument Analysis` | Açık varlık sınıfıyla tek hisse/endeks/kripto/emtia |
| `Betimsel MA Saygı Taraması` | Tek/çoklu varlıkta ham MA temas ve tepki karnesi |
| `BIST Veri Güncelleyici (Endeks Listeleri)` | BIST endeks bileşenlerini haftalık yeniler |
| `Guarded MA Core Tests` | PR ve main değişikliklerinde otomatik test |

Eski Endeks Tarama, Weekly, Sektör Cache ve Keepalive workflow'ları kaldırılmıştır.
Daily, Multi-Timeframe ve Tek Varlık artık eski v6 tarayıcısını değil guarded
çekirdeği çağırır.

Daily, Multi-Timeframe, Manual ve Tek Varlık aynı kanıt profiliyle çalışır:
varsayılan 499 null iterasyonu, shift/horizontal kontroller ve tam operasyonel
periyot listesi kullanılır. `--fast` geriye uyumluluk için kabul edilir ama artık
iterasyon, kontrol veya periyot kısıtlamaz. Uzak MA seviyeleri de kanıt testinden
kaçırılmaz; uzaklık yalnız `actionable=False` / `certified_but_far` gibi çıktı
yorumunu etkiler.

## Betimsel MA saygı taraması

Asıl “bu hisse hangi ortalamalara saygı gösteriyor?” sorusu için **Betimsel MA
Saygı Taraması** workflow'unu kullanın. Bu yol guarded araştırma hattından ayrıdır:
null/FDR/holdout kapılarıyla seviye elemez; 7 MA türünü varsayılan 16 periyotla
karşılaştırıp ham ziyaret, tepki, sarkma ve geri dönüş karnesi üretir.

Varsayılan havuz: `5,8,10,13,20,21,22,34,50,55,89,100,144,200,233,377` ve
`SMA,EMA,WMA,VWMA,KAMA,ALMA,HMA`. Bu alanlar workflow'da serbestçe değiştirilebilir;
istersen 50/200 çıkarabilir, başka periyot ekleyebilirsin. Ana sıralama varsayılan
olarak en çok temas/ziyaret alan MA'lardan başlar (`sort_by=visits`). Varsayılan
`min_visits=5` kullanılır; çünkü 1-2 temas pratikte güçlü ortalama sayılmaz ve
yakın olsa bile ana raporda öne çıkarılmamalıdır. `top=0` tüm eşik üstü satırları
gösterir. Ham denetim için `min_visits=1` veya `0` yapılabilir; tam ham liste zaten
CSV artifact içinde saklanır. `side=auto` artık bugünkü fiyata göre tek taraf seçmez; destek ve direnç davranışını aynı MA havuzu için ayrı ayrı tarar. İstersen yalnız destek için `support`, yalnız direnç için `resistance` seçebilirsin.

CSV artifact ayrımı bilinçlidir: `ma_respect_scorecard.csv` ham ziyaret/tepki karnesi,
`ma_respect_current.csv` bugünkü yakın MA listesi, `ma_dna_profile.csv` ise hissenin
asıl “MA DNA” profilidir. `dna_skoru` geçmiş karakteri ölçer; `guncel_aksiyon_skoru`
bu karaktere bugünkü fiyat yakınlığını ekler. Çoklu evrenlerde varlık başına kısa DNA
listesi `ma_dna_top_per_symbol.csv` olarak saklanır. `universe=bist_all_stocks` gibi
evrenlerle çoklu tarama yapılabilir; hızlı tek sembol için `universe=custom` ve
`ticker=ASELS` yeterlidir.

## Kod yazmadan evren seçimi

Daily, Multi-Timeframe ve ana Research Panel aynı açılır evrenleri kullanır:

- `bist30_stocks`, `bist50_stocks`, `bist100_stocks`, `bist_all_stocks`
- `bist_sector_stocks` ve Türkçe sektör menüsü
- `bist_bank_stocks`, `bist_technology_stocks`, `bist_food_stocks`, `bist_chemistry_stocks`
- `bist_main_indices`, `bist_sector_indices`, `bist_all_indices`
- `crypto_majors`, `commodities_majors`
- `custom`

BIST 100 içindeki bütün sektörleri taramak için `bist100_stocks` seçin ve
sektör alanını `Tümü / uygulanmaz` bırakın. Sektör alanı BIST 30/50/100
ve tüm BIST evrenlerinde filtre uygulamaz; bu evrenler zaten içlerindeki bütün
sektörleri kapsar.

Yalnız belirli bir sektörün hisselerini taramak için `bist_sector_stocks`
seçin ve Türkçe sektör adını menüden belirleyin. Endeks kodu yazmak gerekmez.
`custom` dışında sembol alanı kullanılmaz.

## Önerilen başlangıç ayarları

```text
universe: bist30_stocks veya bist_sector_stocks
timeframes/interval: 1d
periods: 5,8,10,13,20,21,22,34,50,55,89,100,144,200,233,377
source: auto
top: 5
null_iterations: 499
```

Tüm BIST ve üç zaman dilimi birlikte hâlâ maliyetlidir; ancak null sonucu
önhesaplama sayesinde 499 iterasyon geniş evrenlerde de ana profil haline
gelmiştir. Süre yine uzarsa evreni küçültmek veya GitHub Actions matrix ile
parçalamak tercih edilmelidir; istatistiksel kanıt kapıları gevşetilmez.

## Tek varlık analizi

Tek hisse taramak için Actions ekranında **Guarded Single Instrument Analysis**
workflow'unu kullanın:

1. `asset_class`: BIST hissesi için `stock`
2. `market`: `BIST`
3. `symbol`: yalnızca hisse kodu, örneğin `THYAO`
4. `timeframes`: ilk denemede `1d`
5. `periods`: `5,8,10,13,20,21,22,34,50,55,89,100,144,200,233,377`
6. `source`: `auto`
7. `top`: `5`
8. `null_iterations`: gerçek taramada `499`

Bu workflow `--universe custom` ile çalışır ve `symbol` alanındaki tek koddan
başka piyasa varlığı taramaz. Logda `Etkin seçim: ... varlık_sayısı=1,
semboller=THYAO` satırı bunu doğrular.

Daily, Multi-Timeframe veya ana Research Panel'de tek sembol taramak isterseniz
önce `universe=custom` seçmelisiniz. Başka bir evren (örneğin
`bist100_stocks`) seçiliyken custom sembol alanı bilinçli olarak yok sayılır ve
seçilen evrenin tamamı taranır.

Diğer tek varlık örnekleri:

```text
Hisse:          asset_class=stock,        market=BIST,   symbol=GARAN
Endeks:         asset_class=index,        market=BIST,   symbol=XU100
Sektör endeksi: asset_class=sector_index, market=BIST,   symbol=XBANK
Kripto:         asset_class=crypto,       market=GLOBAL, symbol=BTC-USD
Emtia:          asset_class=commodity,    market=GLOBAL, symbol=GC=F
```

Bu ayrım endekslerin “Hisse”, kriptonun BIST sembolü gibi etiketlenmesini engeller.

## Tekilleştirilmiş sonuçlar

Her koşu iki görünüm üretir:

- `instrument_summary.csv` ve `panel.csv`: her varlık tam olarak bir satır
- `panel_detail.csv` ve `all_candidates.csv`: denetlenebilir tüm MA ayrıntıları

Ana özette şu toplulaştırılmış alanlar bulunur:

- test edilen aktif seviye sayısı
- discovery kapısını geçen seviye sayısı
- sertifikali, dusuk-guven ve aksiyon alinabilir seviye sayisi
- sertifikasyon oranı
- sertifikalı seviyelerin ortalama holdout isabet oranı
- sertifikalı seviyelerin ortalama holdout ATR getirisi
- en iyi destek ve direnç seviyesi
- hisse için sektör ve endeks üyelikleri

Telegram “Top 20” tablosu da bu özeti kullanır. Aynı hissenin farklı MA'ları artık
ayrı satırlara dağılmaz; her varlık bir kez görünür.

Ek olarak her kosu ilk amaca donen uc davranis tablosu uretir:

- ma_behavior_most_visited.csv: evren icinde en cok temas alan MA seviyeleri
- ma_behavior_best_reactions.csv: temas sonrasi en iyi tepki istatistigi veren MA seviyeleri
- ma_behavior_near_price.csv: bugunku fiyata en yakin temasli destek/direnc adaylari

Bu tablolardaki Temas sayisi sertifikasyon olay sayisi degil, tum gecmis veri
uzerinde MA bandina pratik ziyaret sayisidir. Varsayilan olarak Temas < 10 olan
seviyeler davranis tablolarina girmez. Genis evren taramalarinda Telegram
tablolari evren genelinden top 20 varligi gosterir; ayni varlik tabloyu
doldurmaz. Tek sembol taramasinda ise ayni sembolun minimum temas esigini gecen
en iyi MA satirlari gosterilir. Tablolarda fiyat, MA seviyesi, uzaklik, temas
sayisi, tepki orani, MedATR ve skor vardir; arastirma kanit etiketleri bu pratik
davranis tablolarinda gosterilmez.

`0/28`, veri bulunamadı demek değildir: 28 aktif MA seviyesi test edilmiş fakat
hiçbiri discovery, kontrol, validation ve holdout kapılarının tamamını
geçememiştir. Bu durumda Telegram tablosu boş oranlar yerine en yakın güncel
adayı; MA, taraf, seviye, ATR/yüzde uzaklık, olay sayısı ve elenme durumuyla
gosterir. `CANDIDATE_ONLY` ve `LOW_CONFIDENCE` islem sinyali degildir.

## Telegram bildirimi

Aşağıdaki repository secret'ları tanımlı olmalıdır:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Guarded Research Panel, Daily, Multi-Timeframe ve Tek Varlık başarılı taramadan
sonra `scanner.guarded_notifier` çağırır. Bildirim gönderilemese bile araştırma
çıktıları artifact olarak korunur; hata workflow logundaki `Telegram` adımında
görülür.

## BIST bileşen verisi

`BIST Veri Güncelleyici (Endeks Listeleri)` her pazartesi 06.00 İstanbul saatinde
otomatik çalışır. Endeks dönemi değiştiğinde elle de çalıştırılabilir.

```text
probe_only=false  → cache güncellenir ve değişiklik varsa commit edilir
probe_only=true   → yalnız veri kaynağı test edilir
```

Yeni sektör seçimi doğrulanmış `scanner/data/bist_indices.json` bileşen cache'ini
kullanır; ayrı bir sektör-cache workflow'una ihtiyaç yoktur.

## Lokal kullanım

Evren ve sektörleri listeleyin:

```bash
python -m scanner.ma_research_cli --list-universes
python -m scanner.ma_research_cli --list-sectors
```

BIST 30 günlük tarama:

```bash
python -m scanner.ma_research_cli \
  --universe bist30_stocks \
  --timeframes 1d \
  --periods 5,8,10,13,20,21,22,34,50,55,89,100,144,200,233,377 \
  --source auto \
  --null-iterations 499
```

Bankacılık sektörü:

```bash
python -m scanner.ma_research_cli \
  --universe bist_sector_stocks \
  --sector "Bankacılık" \
  --timeframes 1d
```

Testler:

```bash
python -m unittest discover -s tests -v
```

## Paper-test artifactleri

`scanner.paper_tracker` artik her create/update kosusunda `summary.csv`,
`equity_curve.csv` ve `open_watches.csv` artifactleri uretebilir. Bu dosyalar
sertifikalarin ileriye donuk expectancy'sini ve rejim curumesini izlemek icin
kullanilmalidir; gecmis backtest kaniti yerine gecmez.

## Araştırma sınırı

Kod içi testler gelecekteki kârlılığı kanıtlamaz. Sağlayıcı fiyat ayarlamalarının,
survivorship bias'ın, gerçek işlem maliyetlerinin ve ileriye dönük paper-test
sonuclarinin ayrica izlenmesi gerekir. Guncel BIST evrenleri tarihsel uyelik
snapshot'i degildir; bu nedenle ozet ve metadata ciktilari acik survivorship
uyarisi basar. Zaman dilimi confluence'i baglamsaldir; korelasyonlu zaman
dilimleri bagimsiz oy sayilmaz.
