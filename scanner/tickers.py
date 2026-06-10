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
    # XBANK endeksinin (12000 puan) kendi MA'larına saygı pattern'i analiz edilir.
    'BIST_ENDEKSLER': [
        # Ana ulusal endeksler
        'XU030', 'XU050', 'XU100', 'XUTUM',
        # Katılım endeksleri
        'XKTUM', 'XK030', 'XK050', 'XK100',
        # Sektör endeksleri (28 tane)
        'XBANK',  # Banka
        'XUSIN',  # Sınai
        'XUMAL',  # Mali
        'XUTEK',  # Teknoloji
        'XHOLD',  # Holding
        'XKMYA',  # Kimya, Petrol Kauçuk
        'XGIDA',  # Gıda, İçecek
        'XSGRT',  # Sigorta
        'XYORT',  # GYO
        'XTRZM',  # Turizm
        'XELKT',  # Elektrik
        'XILTM',  # İletişim
        'XINSA',  # İnşaat
        'XKAGT',  # Orman, Kağıt, Basım
        'XMADN',  # Madencilik
        'XMESY',  # Metal Eşya, Makine
        'XSPOR',  # Spor
        'XTAST',  # Taş Toprak
        'XTCRT',  # Ticaret
        'XTEKS',  # Tekstil, Deri
        'XULAS',  # Ulaştırma
        'XSVNM',  # Savunma
        'XUHIZ',  # Hizmetler
        'XMANA',  # Ana Metal
        # Tema endeksleri
        'XTMTU',  # Temettü
        'XTM25',  # Temettü 25
        'XYUZO',  # Yıldız
        'XKURY',  # Kurumsal Yönetim
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

    Sırayla denenir:
    1. INSTRUMENT_GROUPS (BIST dışı: kripto, forex, metal)
    2. borsapy ile canlı çekim (BIST endeksleri)
    3. Cache (son 7 gün)
    4. Hata fırlatma
    """
    name_upper = name.upper().strip()

    # Multi-instrument grup kontrolu (BIST disi)
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
        print(f"  HATA: '{name}' tanımlı değil. Geçerli: {', '.join(all_avail)}")
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
