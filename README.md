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

- `market_summary.csv`: her varlık bir satır
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
