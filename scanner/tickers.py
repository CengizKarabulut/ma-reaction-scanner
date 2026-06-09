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
    1. borsapy ile canlı çekim (en güncel)
    2. Cache (son 7 gün)
    3. Hata fırlatma (kullanıcı manuel liste vermeli)
    """
    name_upper = name.upper().strip()

    # Önce endeks sembolünü belirle
    if name_upper in INDEX_MAP:
        index_symbol = INDEX_MAP[name_upper]
    elif name_upper.startswith('X') and len(name_upper) >= 4:
        # Doğrudan endeks sembolü verildi (XBANK, XU030, vs)
        index_symbol = name_upper
    else:
        print(f"  HATA: '{name}' tanımlı değil. Geçerli: {', '.join(INDEX_MAP.keys())}")
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
