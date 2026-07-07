# Varlık evrenleri ve tekilleştirilmiş raporlar

Yeni araştırma hattı her sembolü açık bir varlık sınıfıyla taşır:

| Sınıf | Rapor etiketi | Örnek evren |
|---|---|---|
| `stock` | Hisse | `bist100_stocks` |
| `index` | Endeks | `bist_main_indices` |
| `sector_index` | Sektör Endeksi | `bist_sector_indices` |
| `crypto` | Kripto | `crypto_majors` |
| `commodity` | Emtia | `commodities_majors` |

Bir koşu tek varlık sınıfı içerir. Böylece endeksler “Hisse”, kripto veya emtia
olarak gösterilemez ve sağlayıcı sembolleri birbirine karışmaz.

## Kod ezberlemeden seçim

Kullanılabilir hazır evrenleri görmek için:

```bash
python -m scanner.ma_research_cli --list-universes
```

Sektör menüsünü görmek için:

```bash
python -m scanner.ma_research_cli --list-sectors
```


Örnekler:

```bash
python -m scanner.ma_research_cli --universe bist100_stocks --timeframes 1d
python -m scanner.ma_research_cli \
  --universe bist_sector_stocks \
  --sector "Bankacılık" \
  --timeframes 1d
python -m scanner.ma_research_cli --universe bist_sector_indices --timeframes 1d
python -m scanner.ma_research_cli --universe crypto_majors --timeframes 4h,1d
python -m scanner.ma_research_cli --universe commodities_majors --timeframes 1d
```

GitHub Actions içindeki **Guarded MA Research Panel (Manual)** ekranında aynı
evrenler açılır menü olarak bulunur. `custom` seçilmedikçe sembol yazılmaz.

## Sektör endeksi ile sektör hisseleri farklıdır

- `bist_sector_indices`: XBANK veya XUTEK gibi endekslerin kendisini tarar;
  raporda her sektör endeksi bir varlıktır.
- `bist_sector_stocks --sector "Bankacılık"`: sektör kodu yazdırmadan seçilen
  sektör endeksinin bileşen hisselerini ayrı hisseler olarak tarar.
- `bist_bank_stocks`, `bist_technology_stocks` ve benzerleri: seçilen sektör
  endeksinin bileşen hisselerini ayrı hisseler olarak tarar.

Bu iki araştırma aynı tabloda karıştırılmaz.

BIST hisselerinin özetinde `sector`, `industry` ve `index_memberships` alanları
bulunur. Endeks üyelikleri borsapy `Index.component_symbols` verisinden üretilen
doğrulanmış yerel cache ile eşlenir; canlı kaynak geçici olarak çalışmazsa tarama
sembol kodu istemeden devam eder.


## Tek satırlık özet ve teknik ayrıntı

Her koşu iki ayrı görünüm üretir:

- `panel.csv/md/txt`: her varlık tam olarak bir satır; sektör/endeks üyeliği,
  en iyi destek ve en iyi direnç aynı satırdadır.
- `instrument_summary.csv/md/txt`: aynı tekilleştirilmiş özetin açık adlı kopyasıdır.
- `panel_detail.csv/md` ve `all_candidates.csv`: tüm MA adaylarının denetlenebilir
  teknik ayrıntısıdır. Aynı varlığın burada birden çok MA satırı olması beklenir.

Özet ekranında aynı endeksin veya hissenin defalarca görünmesi engellenmiştir.
Teknik ayrıntı dosyaları istatistiksel iz bırakmak için korunur.

## Özel semboller

Yalnızca hazır katalog dışında bir varlık gerektiğinde `custom` kullanılır ve
varlık sınıfı açıkça belirtilir:

```bash
python -m scanner.ma_research_cli \
  --universe custom \
  --asset-class crypto \
  --market GLOBAL \
  --tickers BTC-USD,ETH-USD \
  --timeframes 1d
```

Bir özel listede farklı varlık sınıfları karıştırılmaz; ayrı koşular açılır.
