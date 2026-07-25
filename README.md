# MA Trend ve Tepki Tarayıcısı

Bu proje, seçilen hareketli ortalamalara göre iki ayrı soruyu cevaplar:

1. Hissenin güncel trend yönü nedir?
2. Bağımsız MA temaslarından sonra maliyet ve stop içeren işlem modeli nasıl sonuçlandı?

Trend, geçmiş tepki kalitesi ve bugünkü yakınlık birbirine karıştırılmaz. Ana piyasa
özetinde her varlık tam olarak bir satırdır; tüm MA ve zaman dilimi ayrıntıları ayrı
CSV dosyasında korunur.

## Seçilebilir alanlar

- MA türleri: `SMA, EMA, WMA, VWMA, KAMA, ALMA, HMA`
- İstenen bütün pozitif MA periyotları
- Zaman dilimleri: `5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo`
- Trend eğimi bar sayısı ve ATR eşiği
- Temas bölgesi ve temas öncesi uzaklaşma
- Minimum bağımsız temas
- İlk stop tamponu, takip eden stop ve azami pozisyon süresi
- İşlem maliyeti, minimum baz avantajı ve güncel yakınlık

## Trend tanımı

Her MA dört durumdan birine girer:

- **Yükselen:** fiyat MA üzerinde ve seçilen bar aralığında MA eğimi pozitif.
- **Alçalan:** fiyat MA altında ve MA eğimi negatif.
- **Yatay:** ATR-normalize eğim seçilen eşikten küçük.
- **Geçiş:** fiyat konumu ile MA eğimi birbirini doğrulamıyor.

Bir zaman dilimindeki seçili MA'ların en az %70'i aynı yöndeyse güçlü trend,
en az %50'si aynı yöndeyse normal trend, diğer durumlarda karışık/geçiş etiketi
üretilir.

## Tepki ve işlem modeli

- Fiyat önce MA'dan seçilen ATR kadar uzaklaşır.
- MA bandına ardışık girişler tek bağımsız temas sayılır.
- Giriş temasın ertesi bar açılışındadır.
- İlk stop temas mumunun ve MA'nın ters tarafına ATR tamponuyla yerleştirilir.
- Sabit kâr hedefi pozisyonu kapatmaz; 1R ve 2R görülme oranları ayrıca raporlanır.
- Pozisyon ATR tabanlı takip eden stopla veya azami süre sonunda kapanır.
- Sonuçlar maliyet sonrası `R` birimindedir.
- Aynı yön/zaman dilimindeki koşulsuz girişler baz sonuç olarak gösterilir.
- Olaylar üç kronolojik parçaya ayrılır; pozitif dönem sayısı istikrar alanıdır.

Ana sınıflar: `Güçlü uyum`, `Uyumlu`, `İzleme`, `Uyumsuz`, `Yetersiz veri`.
Bunlar gelecek performans garantisi veya otomatik al/sat emri değildir.


## Raporu nasil okuyacagim?

Ana tabloda her hisse tek satirdir. En iyi uygun kombinasyon su alanlarla aciklanir:

- `current_price`: Secilen en iyi zaman diliminin son kapanis fiyati.
- `price_time`: Fiyat ve MA degerinin ait oldugu son mum zamani.
- `best_ma` / `best_ma_value`: MA adi ve ayni mumdaki sayisal MA degeri.
- `best_difference`: `MA - fiyat` farki, fiyat para birimindedir.
- `best_distance_pct`: `(MA - fiyat) / fiyat * 100`; gercek yuzde uzaklik.
- `best_distance_atr`: Ayni farkin ATR'ye bolunmus hali. Yuzde degildir.
- `filter_status`: `Uygun` veya `Filtre disi`.
- `filter_reasons`: Fiyat, likidite, sifir hacim, gap veya aykiri Edge nedeni.
- `best_edge_r`: MA temaslarinin kosulsuz baz girislere gore R avantaji.
- `best_median_net_r`: Maliyet sonrasi medyan islem sonucu.
- `best_side_adherence_pct`: Destek icin MA ustunde, direnc icin MA altinda gecirilen sure.
- `best_compatibility_score`: 0-100 tarihsel Uyum Skoru. Temas %20, taraf koruma %15,
  kazanma %15, Medyan R %15, Edge %15, üç dönem istikrarı %10 ve düşük normalize
  kesişim/gürültü %10 ağırlığındadır. Negatif veya hesaplanamayan Medyan R/Edge puan
  kazandırmaz. Kanıt sınıfı puana tavan koyar: `Yetersiz veri` en fazla 39,
  `Uyumsuz` 49, `İzleme` 59, `Uyumlu` 79 ve `Güçlü uyum` 100.

Ornek: fiyat `52.85`, SMA233 `53.42` ise fark `0.57`, yuzde uzaklik
yaklasik `%1.08` olur. ATR uzakligi ayri bir sayidir ve `UZK%` gibi okunmamalidir.

## Karar ve teyit katmanı

`Uyum` geçmişte ne olduğunu, `Karar` ise bugünkü durumun işlem modelindeki yerini
anlatır. Sıralama şu sırayla yapılır: geçmiş kalite, MA'ya yakınlık, fiyat tetiği ve
son olarak hacim/trend gücü teyidi.

- `Güçlü Aday`: geçmiş kalite, aktif destek/direnç rolü, MA temas mumu, RVOL ve ADX
  eşikleri birlikte geçmiştir.
- `Tetik Bekliyor`: fiyat MA bölgesindedir fakat dönüş mumu, RVOL veya ADX teyitlerinden
  en az biri eksiktir.
- `Yaklaşıyor`: kaliteli MA izleme mesafesindedir, henüz temas bölgesinde değildir.
- `Uzak`: kaliteli MA vardır fakat güncel fiyat seçilen izleme ATR mesafesinin dışındadır.
- `İşlem Yok`: MA'nın geçmiş kalitesi geçse bile destek/direnç rolü bugün aktif değildir.
- `Uyumsuz`: Medyan R, Edge, taraf koruma, dönem istikrarı, trend yönü veya normalize
  kesişim şartlarından biri geçmemiştir.
- `Yetersiz Veri` ve `Filtre Dışı`: karar üretmek için kanıt yoktur veya piyasa kalitesi
  filtresi geçilmemiştir.

Teyit panelindeki alanlar:

- `RVOL`: son hacim / önceki seçili barların medyan hacmi. Varsayılan teyit `>= 1.20`.
- `ADX`: trendin yönünü değil gücünü ölçer. Varsayılan güç eşiği `20`.
- `RSI` ve `RSI Yumuşatma`: ham RSI ile seçilen EMA yumuşatmasının ilişkisini ve
  aşırı alım/aşırı satım bölgesini gösterir.
- `MACD`, `MACD Signal`, `MACD Açılma / ATR`: çizgi ilişkisi, kesişim ve farklı
  fiyat seviyelerinde karşılaştırılabilir açılmayı gösterir. `MACD Açılma Yüzdeliği`,
  mutlak açılmanın son seçili penceredeki yüzdelik sırasıdır. Örneğin 94, mevcut farkın
  pencerenin yaklaşık %94'ünden büyük olduğunu söyler; düzeltme garantisi değildir.
- `SMI` ve `SMI Signal`: kesişim ile +40/-40 aşırılık bölgelerini birlikte gösterir.
- `Ichimoku Durumu`: fiyatın bulut üstü/içi/altı konumu ve Tenkan-Kijun ilişkisi.
- `Bollinger %B`: fiyatın bant içindeki göreli yeri; `Bollinger Genişlik Yüzdeliği`
  ise güncel bant genişliğinin yakın tarihe göre sıkışık mı geniş mi olduğunu gösterir.

RSI, MACD, SMI, Ichimoku ve Bollinger ham değerleri görünür bağlamdır; aynı bilgiyi
tekrar tekrar sayıp sahte güven üretmemeleri için varsayılan olarak tarihsel MA Uyum
Skoruna ek puan vermezler. Giriş teyidinde yalnızca fiyat tetiği, RVOL ve ADX kullanılır.
Tüm periyotlar ve eşikler CLI'dan; GitHub ekranında üç ayar paketinden değiştirilebilir.
## Piyasa kalitesi filtreleri

Filtreler hisseleri rapordan silmez. Tum hisseler gorunur; uygun kombinasyonlar ustte
siralanir, uygun kombinasyonu olmayan hisselerde neden yazilir. Tum esikler hem GitHub
Actions ekranindan hem CLI'dan degistirilebilir:

- `quality_lookback`: Son kalite penceresi (varsayilan 60).
- `min_price`: Minimum fiyat (varsayilan 1 TRY).
- `min_daily_turnover_try`: Medyan gunluk islem degeri (varsayilan 1 milyon TRY).
- `max_zero_volume_pct`: Sifir hacimli bar ust siniri (varsayilan %20).
- `max_gap_pct`: Yakin donem maksimum gap (varsayilan %15).
- `max_abs_edge_r`: Siralamayi bozabilecek aykiri mutlak Edge siniri (varsayilan 5R).

## GitHub Actions

Kullanıcıya açık iki analiz vardır:

- **Tüm BIST MA Trend ve Tepki Taraması:** BIST'teki tüm hisseleri 20 paralel
  parçaya böler, sonuçları birleştirir ve her hisseyi tek satırda sıralar.
- **Tek Hisse MA Trend ve Tepki Analizi:** bir sembolü seçilen tüm MA ve zaman
  dilimlerinde ayrıntılı inceler.

Altyapı iş akışları:

- **MA Trend ve Tepki Testleri**
- **BIST Veri Güncelleyici (Endeks Listeleri)**

Telegram en iyi satırları gönderir ve tam tekilleştirilmiş CSV'yi belge olarak ekler.
GitHub artifact içinde:

- `market_summary.csv`: her varlık bir satır, tum teknik alanlarla
- `market_table.csv`: her varlık bir satır; MA, zaman dilimi, temas, taraf koruma,
  kazanma, Medyan R, Edge, uzaklik ve Uyum Skoru iceren sade tablo
- `single_stock_table.csv`: secilen tum tek-hisse kombinasyonlari; hesaplanamayanlar
  `Yetersiz veri` olarak korunur
- `ma_detail.csv`: tüm zaman dilimi/MA/yön ayrıntıları
- `market_report.html`: tam okunabilir piyasa raporu
- `errors.csv`: veri sağlayıcı hataları
- `run_config.json`: koşunun bütün ayarları

bulunur.

## Yerel kullanım

Tek hisse ve tüm zaman dilimleri:

```bash
python -m scanner.ma_scan \
  --universe custom \
  --asset-class stock \
  --market BIST \
  --symbol ASELS \
  --timeframes all \
  --ma-types SMA,EMA,WMA,VWMA,KAMA,ALMA,HMA \
  --periods 5,8,10,13,20,21,22,34,50,55,89,100,144,200,233,377
```

Özel seçim:

```bash
python -m scanner.ma_scan \
  --universe custom \
  --symbol THYAO \
  --timeframes 15m,1h,4h,1d \
  --ma-types EMA,KAMA \
  --periods 20,50,100,200 \
  --trend-slope-bars 15 \
  --trailing-stop-atr 2.5
```

Testler:

```bash
python -m unittest discover -s tests -v
```

## Veri sınırları

Intraday veri geçmişi sağlayıcıya bağlıdır. Mevcut yfinance adaptörü 5/15/30 dakika
için en fazla 60 günlük veri ister; borsapy geçmişi daha kısa olabilir. Uzun periyotlu
intraday MA'larda yeterli bar yoksa sonuç açıkça `Yetersiz veri` olur.

Güncel BIST listeleri tarihsel üyelik snapshot'ı değildir. Sonuçlar yatırım tavsiyesi
değildir; gerçek emir uygulaması, likidite, kayma ve aracı kurum maliyetleri ayrıca
değerlendirilmelidir.
