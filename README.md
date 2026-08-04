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


## MA DNA / gozlemsel seviye skoru

Bu repo artik iki farkli katmani ayri raporlar:

- **Seviye Skoru (`level_score`)**: hissenin hangi ortalamaya gercekten saygi duydugunu
  okumak icin ana skor. Ham temas sayisi, temas sonrasi seviyeyi tutma ve medyan sicrama
  once puanlanir; fiyat MA'yi cok sik kesiyorsa skor carpimla sonumlenir. Boylece cok
  temas alan ama gurultunun ortasinda duran kisa MA'lar yapay olarak sismez.
- **Uyum Skoru (`compatibility_score`)**: eski islem-simulasyon katmani. Stop, maliyet,
  trailing stop ve baz giris varsayimlarina baglidir; karsilastirma icin korunur.

Varsayilan siralama `rank_by=level` oldugu icin 1-2 temasli ama fiyata cok yakin
ortalamalar ana raporu isgal etmez. Bir ortalamanin guclu sayilmasi icin once yeterli
ham ziyaret/temas, sonra tutma ve tepki aranir; yakinlik tek basina sinyal degildir.

Yeni alanlar:

- `level_touches` / `best_level_touches`: ham bagimsiz seviye temasi.
- `hold_rate_pct`: temas sonrasi seviyenin kirilmadan kaldigi oran.
- `median_bounce_atr`: temas sonrasi lehe medyan hareket, ATR cinsinden.
- `touch_density_per_100`: farkli uzunluktaki ortalamalari karsilastirmak icin 100 bar basina temas.
- `cross_per_100`: fiyat ortalamayi ne kadar sik kesiyor; yuksekse seviye gurultulu olabilir.
- `plateau_ratio`: yakin periyotlar da benzer skor veriyor mu? Dusukse tek periyot sansi olabilir.
- `adherence_excess_pct`: hissenin kendi ortalama taraf-koruma seviyesine gore fark.

Istege bagli `--relative-to XU100` kullanilirsa fiyat serisi once endekse gore normalize edilir.
Bu, nominal TRY yukselisinden gelen sahte "her ortalama destek calisiyor" etkisini azaltmak icindir.

GitHub Actions ekraninda ana ayarlar sade tutulur. MA DNA'nin nadir degisen ileri ayarlari
`level_config_json` alanindan tek JSON olarak verilebilir; ornek:
`{"evidence_target_touches":30,"reaction_bars":12,"cross_damping":0.5}`.
Komut satirinda eski tek tek parametreler de geriye donuk uyumluluk icin calismaya devam eder.

### Izleme seti

Tam MA DNA tablosu arastirma icindir; tek hissede yuzlerce satir uretebilir.
`ma_watchlist.csv` ve HTML'deki **Izleme seti** bu tabloyu pratik bolgelere indirir:

- Sadece bugun aktif olan taraf kullanilir: fiyat MA'nin ustundeyse destek, altindaysa direnc.
- En az temas, seviye skoru, plato ve uzaklik esiklerini gecmeyen satirlar elenir.
- Birbirine yakin MA'lar ATR mesafesine gore tek bolgeye kumelenir; temaslar toplanmaz, ayni olayi iki kez saymamak icin en yuksek temas sayisi kullanilir.
- Sira skor sirasi degil, fiyata yakinlik sirasidir: izleme icin once fiyat nereye degecek sorusu onemlidir.

Bu katman otomatik al-sat sinyali degil; grafikte hangi MA bolgelerinin izlenmeye deger oldugunu hizli gosterir.
Komut satirinda `--watch-min-touches`, `--watch-min-score`, `--watch-cluster-atr`,
`--watch-max-zones` gibi ayarlar degistirilebilir. Actions ekraninda yeni input eklenmez;
varsayilan izleme seti otomatik uretilir.


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
- `best_compatibility_score`: 0-100 Uyum Skoru. Temas %25, taraf koruma %20,
  kazanma %15, Medyan R %15, Edge %15 ve uc donem istikrari %10 agirliklidir.
  Negatif veya hesaplanamayan Medyan R/Edge puan kazandirmiyor.

Ornek: fiyat `52.85`, SMA233 `53.42` ise fark `0.57`, yuzde uzaklik
yaklasik `%1.08` olur. ATR uzakligi ayri bir sayidir ve `UZK%` gibi okunmamalidir.

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

Telegram once okunabilir PNG tablo gorseli gonderir; gorsel uretilemezse eski metin tablosuna duser. Tam CSV/HTML dosyalari yine belge olarak eklenir.
GitHub artifact içinde:

- `market_summary.csv`: her varlık bir satır, tum teknik alanlarla
- `market_table.csv`: her varlik bir satir; MA, zaman dilimi, ham temas, Seviye Skoru,
  tutma, sicrama, uzaklik ve eski Uyum Skoru iceren sade tablo
- `ma_watchlist.csv`: izlemeye deger aktif MA bolgeleri; ham teknik kolonlarla
- `watchlist.csv`: ayni izleme setinin okunabilir tablo hali
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
  --periods 5,8,10,13,20,21,22,34,50,55,89,100,144,200,233,377 \
  --rank-by level
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
