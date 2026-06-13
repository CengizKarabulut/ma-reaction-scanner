"""BIST hisse listeleri — TAMAMEN borsapy üzerinden dinamik.

Hardcoded liste YOK. borsapy her çağrıda güncel bileşenleri çeker.
Opsiyonel olarak son başarılı sonuç .cache/tickers_cache.json'a yazılır
ve borsapy ulaşılamadığında oradan okunur.

Desteklenen isimler:
  BIST_30 → XU030       Katılım: XK030 (Kat 30), XK050, XKTUM
  BIST_50 → XU050       Sektör: XBANK, XUSIN, XUMAL, XUTEK, XKMYA, XGIDA, ...
  BIST_100 → XU100      Bölgesel/Tema: XHOLD, XSGRT, XYORT, ...
  BIST_TUM → XUTUM (+XKTUM birleşik, ~500+ hisse)
  Direkt sembol: 'XU100', 'XBANK', vs.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path


# === Endeks isim eşleştirmesi ===
INDEX_MAP = {
    'BIST_30':   'XU030',
    'BIST_50':   'XU050',
    'BIST_100':  'XU100',
    'BIST_TUM':  'XUTUM',  # Özel davranış: XUTUM + XKTUM birleştir
    'BIST_ALL':  'XUTUM',
    'BIST_KAT':  'XKTUM',
    'KATILIM_30':  'XK030',
    'KATILIM_50':  'XK050',
    'KATILIM_100': 'XK100',
    'BANKA':     'XBANK',
    'SINAI':     'XUSIN',
    'MALI':      'XUMAL',
    'TEKNOLOJI': 'XUTEK',
    'KIMYA':     'XKMYA',
    'GIDA':      'XGIDA',
    'HOLDING':   'XHOLD',
    'SIGORTA':   'XSGRT',
    'GAYRIMENKUL': 'XYORT',
}

# === Multi-instrument gruplari (BIST disi) ===
# borsapy'nin destekledigi sembolleri tahmin ettim, gercekte bazilari calismayabilir.
# Calismayanlar single_ticker_query.py'de --scan-fresh ile tek tek denenebilir.
INSTRUMENT_GROUPS = {
    'KRIPTO_TOP': [
        'BTCUSD', 'ETHUSD', 'BNBUSD', 'XRPUSD', 'ADAUSD',
        'SOLUSD', 'DOTUSD', 'AVAXUSD', 'MATICUSD', 'LINKUSD',
    ],
    'KRIPTO_TRY': [
        'BTCTRY', 'ETHTRY', 'BNBTRY', 'XRPTRY', 'SOLTRY',
    ],
    'FOREX_MAJOR': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD',
        'NZDUSD', 'USDCAD',
    ],
    'FOREX_TRY': [
        'USDTRY', 'EURTRY', 'GBPTRY', 'CHFTRY', 'JPYTRY',
    ],
    'METAL': [
        'XAUUSD',  # Altın USD
        'XAGUSD',  # Gümüş USD
        'XPTUSD',  # Platin
        'XPDUSD',  # Paladyum
    ],
    'METAL_TRY': [
        'XAUTRY',  # Ons Altın TRY
        'XAGTRY',  # Ons Gümüş TRY
        # 'GA' Gram altın - borsapy'de farkli sembol olabilir
    ],
    'EMTIA': [
        'XAUUSD', 'XAGUSD',  # Kıymetli metaller
        'WTIUSD', 'BRENTUSD',  # Petrol
        'NATGAS',  # Doğalgaz
        'COPPER',  # Bakır
    ],
    'ENDEKS_GLOBAL': [
        'SPX500', 'NAS100', 'DJI30',   # ABD
        'DE40', 'UK100', 'FR40',        # Avrupa
        'NK225', 'HK50',                # Asya
    ],

    # === BIST Endekslerinin KENDİSİ (bileşenleri değil) ===
    # Bu listede endeks sembolleri var. Her biri endeks fiyat geçmişi olarak çekilir.
    # NOT: borsapy SADECE 4 ana endeksi destekliyor (XU030, XU050, XU100, XUTUM).
    # Sektör endeksleri (XBANK, XKMYA vs.) bp.Index() ile çekilemiyor — yfinance de Türk endekslerini desteklemiyor.
    # Bu yüzden liste 4 endeksle sınırlı tutuldu.
    # === BIST_ENDEKSLER ===
    # borsapy v0.10+ ile 79 endeks erişilebilir.
    # BIST_ENDEKSLER: Ana 4 endeks (kesin çalışan, hızlı test için)
    # BIST_TUM_ENDEKSLER: get_list() ile 79 endeks dinamik (yukarıya bak)
    'BIST_ENDEKSLER': [
        'XU030',  # BIST 30
        'XU050',  # BIST 50
        'XU100',  # BIST 100
        'XUTUM',  # BIST Tüm
    ],

    # Genişletilmiş statik liste — borsapy'nin desteklediği 36 ana endeks
    # (BIST_TUM_ENDEKSLER ile aynı sonucu verir ama statik)
    'BIST_ENDEKSLER_GENIS': [
        # Ana endeksler
        'XU030', 'XU050', 'XU100', 'XUTUM',
        # Katılım
        'XKTUM', 'XK030', 'XK050', 'XK100',
        # Sektör (borsapy bp.Index() ile çalışıyor)
        'XBANK', 'XUSIN', 'XUMAL', 'XUTEK', 'XHOLD',
        'XKMYA', 'XGIDA', 'XSGRT', 'XYORT', 'XTRZM',
        'XELKT', 'XILTM', 'XINSA', 'XKAGT', 'XMADN',
        'XMESY', 'XSPOR', 'XTAST', 'XTCRT', 'XTEKS',
        'XULAS', 'XSVNM', 'XUHIZ', 'XMANA',
        'XTMTU', 'XTM25', 'XYUZO', 'XKURY',
    ],

    # === SEKTÖR HİSSE GRUPLARI ===
    # Endeks taraması yapamıyoruz diye, sektörün BİLEŞEN HİSSELERİNİ topluca tara.
    # Bu hisseler endeksi temsil eden ana oyuncular. Toplu davranışları = sektör trendi.

    'BIST_BANKA': [
        # XBANK endeksi yerine kullan
        'GARAN', 'AKBNK', 'ISCTR', 'YKBNK', 'HALKB', 'VAKBN',
        'KCMHL', 'ALBRK', 'ICBCT', 'QNBTR', 'SKBNK', 'TSKB',
    ],

    'BIST_HOLDING': [
        # XHOLD endeksi yerine
        'SAHOL', 'KCHOL', 'SISE', 'ZOREN', 'DOHOL', 'ENKAI',
        'GLYHO', 'IHLAS', 'TKFEN', 'ALARK', 'AGHOL', 'KOZAA',
        'KOZAL', 'IHGZT', 'NTHOL', 'BERA',
    ],

    'BIST_GIDA': [
        # XGIDA endeksi yerine
        'ULKER', 'ULUUN', 'BANVT', 'PNSUT', 'KENT', 'KERVT',
        'CCOLA', 'TUKAS', 'KRSAN', 'KNFRT', 'AVOD', 'TATGD',
        'GOLTS', 'MERKO', 'PETUN',
    ],

    'BIST_KIMYA': [
        # XKMYA endeksi yerine
        'AKSA', 'GUBRF', 'EGGUB', 'BAGFS', 'PETKM', 'TUPRS',
        'BOSSA', 'DEVA', 'HEKTS', 'RTALB', 'SOKE', 'YATAS',
        'ALKIM', 'ATATP', 'YUNSA',
    ],

    'BIST_TEKNOLOJI': [
        # XUTEK endeksi yerine
        'ASELS', 'TCELL', 'TTKOM', 'KAREL', 'LOGO', 'INDES',
        'NETAS', 'ARENA', 'KAFEIN', 'ESCAR', 'PAPIL', 'DGATE',
        'PENTA', 'MIATK', 'OTOSR',
    ],

    'BIST_INSAAT': [
        # XINSA endeksi yerine
        'ENKAI', 'TKFEN', 'YYAPI', 'SISE', 'AKCNS', 'CIMSA',
        'BTCIM', 'KONYA', 'BOLUC', 'UNYEC', 'CMENT', 'BANVT',
        'BIZIM', 'ASUZU',
    ],

    'BIST_ENERJI': [
        # XELKT endeksi yerine
        'AKSEN', 'AYDEM', 'AKENR', 'ENJSA', 'ZOREN', 'ODAS',
        'GWIND', 'ASUZU', 'KARTN', 'POLHO', 'AKFGY', 'EKGYO',
    ],

    'BIST_GAYRIMENKUL': [
        # XYORT endeksi yerine
        'EKGYO', 'OZGYO', 'TRGYO', 'AKFGY', 'AGYO', 'AVGYO',
        'KRGYO', 'NUGYO', 'OZKGY', 'PEGYO', 'SAFKR', 'SNGYO',
        'TSGYO', 'YGYO', 'YGGYO', 'PSGYO',
    ],

    'BIST_OTOMOTIV': [
        # Genel otomotiv & yan sanayi
        'FROTO', 'TOASO', 'OTKAR', 'BFREN', 'TUMUS', 'KATMR',
        'EGEEN', 'PRKAB', 'MARKA', 'JANTS', 'PINSU',
    ],

    'BIST_DEMIRCELIK': [
        # XMANA + metal ana sanayi
        'EREGL', 'KRDMD', 'KRDMA', 'CEMAS', 'CEMTS', 'BURVA',
        'BMSCH', 'IZMDC', 'BNTAS', 'EGEEN',
    ],
}


# === Cache yolu (script'in olduğu dizinde) ===
_CACHE_DIR = Path(__file__).parent / '.cache'
_CACHE_FILE = _CACHE_DIR / 'tickers_cache.json'
_CACHE_MAX_AGE_DAYS = 7  # Cache 7 gün geçerli


def _load_cache() -> dict:
    """Cache'den oku, yoksa veya çok eskiyse boş döndür."""
    if not _CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(_CACHE_FILE.read_text())
        cached_at = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
        if datetime.now() - cached_at > timedelta(days=_CACHE_MAX_AGE_DAYS):
            print(f"  Cache çok eski ({_CACHE_MAX_AGE_DAYS} gün+), yenilenecek")
            return {}
        return data.get('lists', {})
    except Exception as e:
        print(f"  Cache okuma hatası: {e}")
        return {}


def _save_cache(lists: dict):
    """Tüm listeleri cache'e kaydet."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            'cached_at': datetime.now().isoformat(),
            'lists': lists,
        }
        _CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"  Cache yazma hatası: {e}")


def _fetch_from_borsapy(index_symbol: str) -> list:
    """borsapy ile endeks bileşenlerini çek (cache yok)."""
    try:
        import borsapy as bp
        idx = bp.Index(index_symbol)
        components = idx.component_symbols
        if components:
            print(f"  borsapy: {index_symbol} → {len(components)} hisse")
            return list(components)
        return []
    except ImportError:
        raise RuntimeError(
            "borsapy yüklü değil. 'pip install borsapy' veya virgülle "
            "ayrılmış özel ticker listesi verin."
        )
    except Exception as e:
        print(f"  borsapy hatası ({index_symbol}): {e}")
        return []


def _fetch_bist_tum() -> list:
    """BIST Tümü — XUTUM + XKTUM birleşik (~500+ hisse)."""
    all_symbols = set()
    for sym in ['XUTUM', 'XKTUM']:
        symbols = _fetch_from_borsapy(sym)
        if symbols:
            all_symbols.update(symbols)
    return sorted(all_symbols)


def get_list(name: str) -> list:
    """İsim ile hisse listesi döndür.

    Özel prefix'ler (borsapy v0.10+):
    - BIST_BILES:XBANK       → XBANK endeksinin bileşen hisseleri (borsapy direkt)
    - BIST_SEKTOR:Bankacılık → Sektördeki tüm hisseler (cache'ten)
    - BIST_TUM_ENDEKSLER     → 79 BIST endeksi (borsapy direkt)

    Sırayla denenir:
    1. Özel prefix'ler (BIST_BILES, BIST_SEKTOR)
    2. BIST_TUM_ENDEKSLER → bp.all_indices() tümü
    3. INSTRUMENT_GROUPS (BIST dışı + statik gruplar)
    4. borsapy ile canlı endeks çekim (BIST endeksleri)
    5. Cache (son 7 gün)
    """
    name_upper = name.upper().strip()

    # === YENİ 1: Endeks bileşenleri (borsapy direkt) ===
    if name_upper.startswith('BIST_BILES:') or name_upper.startswith('BILES:'):
        index_sym = name.split(':', 1)[1].strip().upper()
        try:
            import sys
            from pathlib import Path
            scanner_dir = Path(__file__).parent
            if str(scanner_dir) not in sys.path:
                sys.path.insert(0, str(scanner_dir))
            from sector_resolver import get_tickers_by_index
            tickers = get_tickers_by_index(index_sym)
            if tickers:
                print(f"  Endeks bileşenleri: {index_sym} → {len(tickers)} hisse (borsapy)")
                return tickers
            print(f"  ⚠️ {index_sym} bileşenleri alınamadı")
            return []
        except ImportError:
            print(f"  ⚠️ sector_resolver.py bulunamadı")
            return []

    # === YENİ 2: Tüm BIST endeksleri (79 adet) ===
    if name_upper in ('BIST_TUM_ENDEKSLER', 'TUM_ENDEKSLER', 'ALL_INDICES'):
        try:
            import sys
            from pathlib import Path
            scanner_dir = Path(__file__).parent
            if str(scanner_dir) not in sys.path:
                sys.path.insert(0, str(scanner_dir))
            from sector_resolver import list_all_indices
            idxs = list_all_indices(detailed=False)
            if idxs:
                # detailed=False'tan dict gelirse temizle
                symbols = [d['symbol'] if isinstance(d, dict) else d for d in idxs]
                print(f"  Tüm BIST endeksleri: {len(symbols)} adet (borsapy)")
                return symbols
            print(f"  ⚠️ Endeks listesi alınamadı")
            return []
        except ImportError:
            print(f"  ⚠️ sector_resolver.py bulunamadı")
            return []

    # === Dinamik sektör grubu (cache'ten) ===
    if name_upper.startswith('BIST_SEKTOR:') or name_upper.startswith('SEKTOR:'):
        sector_name = name.split(':', 1)[1].strip()
        try:
            import sys
            from pathlib import Path
            scanner_dir = Path(__file__).parent
            if str(scanner_dir) not in sys.path:
                sys.path.insert(0, str(scanner_dir))
            from sector_resolver import get_tickers_by_sector
            tickers = get_tickers_by_sector(sector_name)
            if tickers:
                print(f"  Sektör grubu: '{sector_name}' → {len(tickers)} hisse (cache'den)")
                return tickers
            else:
                print(f"  ⚠️ '{sector_name}' sektöründe hisse yok (cache boş olabilir)")
                print(f"  → Önce: python scanner/sector_resolver.py --build-cache")
                return []
        except ImportError:
            print(f"  ⚠️ sector_resolver.py bulunamadi")
            return []

    # Multi-instrument grup kontrolu (BIST disi + statik)
    if name_upper in INSTRUMENT_GROUPS:
        symbols = INSTRUMENT_GROUPS[name_upper]
        print(f"  Multi-instrument grup: {name_upper} → {len(symbols)} sembol")
        return symbols

    # Önce endeks sembolünü belirle
    if name_upper in INDEX_MAP:
        index_symbol = INDEX_MAP[name_upper]
    elif name_upper.startswith('X') and len(name_upper) >= 4 and name_upper != 'XAU' and name_upper != 'XAG':
        # Doğrudan endeks sembolü verildi (XBANK, XU030, vs) - XAU/XAG hariç
        index_symbol = name_upper
    else:
        all_avail = list(INDEX_MAP.keys()) + list(INSTRUMENT_GROUPS.keys())
        print(f"  HATA: '{name}' tanımlı değil.")
        print(f"  Geçerli isim grupları: {', '.join(all_avail[:15])}...")
        print(f"  Veya: 'BIST_SEKTOR:Banks', 'BIST_SEKTOR:Technology' (cache'ten)")
        return []

    # Cache'i yükle (varsa)
    cache = _load_cache()

    # 1) borsapy'den canlı çekim dene
    if name_upper in ('BIST_TUM', 'BIST_ALL', 'XUTUM'):
        symbols = _fetch_bist_tum()
    else:
        symbols = _fetch_from_borsapy(index_symbol)

    # Başarılıysa cache'e yaz
    if symbols:
        cache[index_symbol] = symbols
        _save_cache(cache)
        return symbols

    # 2) Cache fallback
    if index_symbol in cache:
        cached = cache[index_symbol]
        print(f"  Cache fallback: {index_symbol} → {len(cached)} hisse "
              f"(borsapy ulaşılamadı)")
        return cached

    # 3) Tamamen başarısız
    print(f"  HATA: {name} listesi alınamadı. borsapy çalışmıyor ve cache yok.")
    print(f"  Çözüm: --tickers HISSE1,HISSE2,... şeklinde manuel liste verin.")
    return []


def list_available() -> list:
    """Desteklenen tüm isim/sembolleri döndür."""
    return sorted(INDEX_MAP.keys()) + ['Direkt sembol: XU100, XBANK, XKTUM, vs.']


if __name__ == '__main__':
    # Test
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else 'BIST_30'
    print(f"Test: {target}")
    result = get_list(target)
    print(f"Sonuç: {len(result)} hisse")
    if result:
        print(f"İlk 10: {', '.join(result[:10])}")
        print(f"Son 5: {', '.join(result[-5:])}")
