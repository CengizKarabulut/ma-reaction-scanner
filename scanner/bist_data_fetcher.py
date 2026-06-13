"""
BIST Veri Kaynağı — çoklu fetcher ile güncel veri.

Bu modül BIST endekslerinin bileşen listesini birden fazla kaynaktan toplar:
1. borsapy (TradingView WebSocket backend) — birinci tercih
2. ISYatırım web scraping — yedek (resmi BIST aracı kurumu)
3. mynet.com.tr — son çare scraping

Her kaynak çalıştığında veri "doğrulama" geçer (min sanity check),
sonra cache JSON'a yazılır. tickers.py bu JSON'dan okur.

Kullanım:
    # Tek endeks güncelle
    python bist_data_fetcher.py --update XU100

    # Tüm ana endeksleri güncelle
    python bist_data_fetcher.py --update-all

    # Mevcut cache'in yaşını göster
    python bist_data_fetcher.py --status

    # Kaynaklardan canlı test
    python bist_data_fetcher.py --probe XU100
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx  # borsapy ile geliyor

try:
    import borsapy as bp
    HAS_BORSAPY = True
except ImportError:
    HAS_BORSAPY = False

DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDICES_FILE = DATA_DIR / 'bist_indices.json'

# Sanity check — bir endeks bileşeninin minimum hisse sayısı
MIN_COMPONENTS = {
    'XU030': 25, 'XU050': 40, 'XU100': 80,
    'XU500': 400, 'XUTUM': 400, 'XKTUM': 200,
    'XBANK': 8, 'XHOLD': 8, 'XKMYA': 8, 'XGIDA': 8,
    'XUTEK': 6, 'XINSA': 6, 'XUSIN': 30, 'XUMAL': 30,
}


# ============================================================
# FETCHER 1: BORSAPY
# ============================================================

def fetch_borsapy(index_symbol: str) -> tuple:
    """borsapy ile endeks bileşenleri. Çoklu API yöntemi dener."""
    if not HAS_BORSAPY:
        return [], 'borsapy: yüklü değil'

    try:
        idx = bp.Index(index_symbol)

        # v0.10 native API
        if hasattr(idx, 'component_symbols'):
            try:
                syms = idx.component_symbols
                if syms:
                    return list(syms), 'borsapy.component_symbols'
            except Exception:
                pass

        # components - dict liste olabilir
        if hasattr(idx, 'components'):
            try:
                comps = idx.components
                if isinstance(comps, list) and len(comps) > 0:
                    if isinstance(comps[0], dict):
                        syms = [c.get('symbol', '') for c in comps if c.get('symbol')]
                    else:
                        syms = list(comps)
                    if syms:
                        return syms, 'borsapy.components'
            except Exception:
                pass

        return [], 'borsapy: API metodu bulunamadı'

    except Exception as e:
        return [], f'borsapy hata: {e}'


# ============================================================
# FETCHER 2: ISYATIRIM (resmi BIST aracı kurumu)
# ============================================================

def fetch_isyatirim(index_symbol: str) -> tuple:
    """ISYatırım endeks bileşen sayfasından scrape.

    URL: https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx?endeks=XU100
    """
    url = (f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/"
           f"Temel-Degerler-Ve-Oranlar.aspx?endeks={index_symbol}")

    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/131.0.0.0 Safari/537.36'),
        'Accept': ('text/html,application/xhtml+xml,application/xml;'
                   'q=0.9,image/avif,image/webp,*/*;q=0.8'),
        'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.isyatirim.com.tr/tr-tr/Sayfalar/default.aspx',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1',
    }

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return [], f'ISYatırım HTTP {resp.status_code}'

            html = resp.text
            # Hisse kodları: tablo içinde <a>...</a> veya data-symbol
            # ISYatırım pattern: hisse kodları büyük harfli, 3-5 karakter
            # /Sayfalar/sirket-karti.aspx?hisse=GARAN şeklinde linkler var
            symbols = set()

            # Pattern 1: hisse= parametresi
            for m in re.finditer(r'hisse=([A-Z][A-Z0-9]{2,5})(?:["&\s])', html):
                sym = m.group(1)
                if sym not in ('NEW', 'WEB'):  # filter
                    symbols.add(sym)

            # Pattern 2: data attributes
            for m in re.finditer(r'data-symbol=["\']([A-Z]{3,5})["\']', html):
                symbols.add(m.group(1))

            symbols = sorted(symbols)
            if symbols:
                return symbols, f'ISYatırım scrape ({len(symbols)} sembol)'
            return [], 'ISYatırım: hisse bulunamadı'

    except Exception as e:
        return [], f'ISYatırım hata: {e}'


# ============================================================
# FETCHER 3: MYNET FINANS
# ============================================================

def fetch_mynet(index_symbol: str) -> tuple:
    """mynet finans endeks sayfasından scrape (yedek)."""
    # mynet sadece bazı endeksleri destekliyor
    mynet_endeks_map = {
        'XU100': 'bist-100',
        'XU030': 'bist-30',
        'XU050': 'bist-50',
        'XBANK': 'bist-banka',
    }
    slug = mynet_endeks_map.get(index_symbol)
    if not slug:
        return [], 'mynet: endeks desteklenmiyor'

    url = f"https://finans.mynet.com/borsa/endeks/{slug}/"
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/131.0.0.0 Safari/537.36'),
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'tr-TR,tr;q=0.9',
        'Referer': 'https://finans.mynet.com/borsa/',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'DNT': '1',
    }
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return [], f'mynet HTTP {resp.status_code}'

            symbols = set()
            # mynet hisse linkleri: /borsa/hisseler/GARAN-garanti-bbva/
            for m in re.finditer(r'/borsa/hisseler/([A-Z][A-Z0-9]{2,5})-', resp.text):
                symbols.add(m.group(1))

            symbols = sorted(symbols)
            if symbols:
                return symbols, f'mynet scrape ({len(symbols)} sembol)'
            return [], 'mynet: hisse bulunamadı'
    except Exception as e:
        return [], f'mynet hata: {e}'


# ============================================================
# ANA ÇOKLU-KAYNAK FETCHER
# ============================================================

def fetch_index_components(index_symbol: str, verbose: bool = True) -> dict:
    """Birden fazla kaynak deneyip ilk başarılı olanı dön.

    Returns:
        {
            'index': 'XU100',
            'symbols': [...],
            'source': 'borsapy.component_symbols',
            'fetched_at': '2026-06-13T...',
            'verified': True,  # min sanity check geçti
            'failures': ['ISYatırım: HTTP 503', ...]
        }
    """
    if verbose:
        print(f"\n📥 {index_symbol} bileşenleri çekiliyor...")

    min_expected = MIN_COMPONENTS.get(index_symbol, 5)
    failures = []

    fetchers = [
        ('borsapy', fetch_borsapy),
        ('ISYatırım', fetch_isyatirim),
        ('mynet', fetch_mynet),
    ]

    for name, func in fetchers:
        if verbose:
            print(f"  → {name} deneniyor...")
        symbols, info = func(index_symbol)

        if symbols and len(symbols) >= min_expected:
            if verbose:
                print(f"  ✓ {name} başarılı: {info}")
            return {
                'index': index_symbol,
                'symbols': symbols,
                'source': info,
                'fetched_at': datetime.now().isoformat(),
                'verified': True,
                'count': len(symbols),
                'failures': failures,
            }
        elif symbols:
            # Çok az sembol — şüpheli
            failures.append(f"{name}: az veri ({len(symbols)} < {min_expected})")
            if verbose:
                print(f"  ⚠️ {name} az veri verdi: {len(symbols)} (beklenen min {min_expected})")
        else:
            failures.append(f"{name}: {info}")
            if verbose:
                print(f"  ✗ {name}: {info}")

    return {
        'index': index_symbol,
        'symbols': [],
        'source': None,
        'fetched_at': datetime.now().isoformat(),
        'verified': False,
        'count': 0,
        'failures': failures,
    }


# ============================================================
# JSON CACHE OKUMA/YAZMA + YAŞ KONTROLÜ
# ============================================================

def load_cache() -> dict:
    if not INDICES_FILE.exists():
        return {'updated_at': None, 'indices': {}}
    try:
        return json.loads(INDICES_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"⚠️ Cache okunamadı: {e}", file=sys.stderr)
        return {'updated_at': None, 'indices': {}}


def save_cache(cache: dict):
    cache['updated_at'] = datetime.now().isoformat()
    INDICES_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8'
    )


def get_index_age_days(index_symbol: str) -> float:
    """Cache'teki bu endeksin son güncelleme tarihinden bu yana geçen gün."""
    cache = load_cache()
    entry = cache.get('indices', {}).get(index_symbol)
    if not entry or not entry.get('fetched_at'):
        return float('inf')
    try:
        fetched = datetime.fromisoformat(entry['fetched_at'])
        return (datetime.now() - fetched).total_seconds() / 86400.0
    except Exception:
        return float('inf')


def get_cached_components(index_symbol: str, max_age_days: float = 7.0) -> tuple:
    """Cache'ten endeks bileşeni — yaş limiti ile.

    Returns:
        (symbols_list, status):
            status: 'fresh' | 'stale' | 'missing'
    """
    cache = load_cache()
    entry = cache.get('indices', {}).get(index_symbol)
    if not entry or not entry.get('symbols'):
        return [], 'missing'

    age = get_index_age_days(index_symbol)
    if age > max_age_days:
        return entry['symbols'], 'stale'
    return entry['symbols'], 'fresh'


def update_index(index_symbol: str, verbose: bool = True) -> bool:
    """Tek endeksi canlı kaynaktan güncelle, cache'e yaz."""
    result = fetch_index_components(index_symbol, verbose=verbose)
    if not result['verified']:
        if verbose:
            print(f"  ✗ Tüm kaynaklar fail. Cache güncellenmedi.")
            for f in result['failures']:
                print(f"     - {f}")
        return False

    cache = load_cache()
    cache.setdefault('indices', {})
    cache['indices'][index_symbol] = result
    save_cache(cache)

    if verbose:
        age_before = get_index_age_days(index_symbol)
        print(f"  ✓ {index_symbol} cache güncellendi: {len(result['symbols'])} hisse")
    return True


def update_all(verbose: bool = True) -> dict:
    """Tüm ana endeksleri güncelle."""
    primary = ['XU030', 'XU050', 'XU100', 'XUTUM']
    sectoral = ['XBANK', 'XHOLD', 'XKMYA', 'XGIDA', 'XUTEK',
                'XINSA', 'XUSIN', 'XUMAL', 'XSGRT', 'XYORT']

    results = {'success': [], 'failed': []}
    for sym in primary + sectoral:
        ok = update_index(sym, verbose=verbose)
        if ok:
            results['success'].append(sym)
        else:
            results['failed'].append(sym)
        time.sleep(0.5)  # Rate limit

    if verbose:
        print(f"\n=== ÖZET ===")
        print(f"Başarılı: {len(results['success'])} ({', '.join(results['success'])})")
        print(f"Başarısız: {len(results['failed'])} ({', '.join(results['failed'])})")
    return results


def status_report():
    """Cache durumu — hangi endeksler güncel, hangileri eski?"""
    cache = load_cache()
    indices = cache.get('indices', {})
    if not indices:
        print("⚠️ Cache boş. Önce: python bist_data_fetcher.py --update-all")
        return

    print(f"\n📊 BIST Veri Cache Durumu")
    print(f"Son güncelleme: {cache.get('updated_at', 'bilinmiyor')}")
    print(f"Toplam endeks: {len(indices)}\n")

    print(f"{'Endeks':<10} {'Hisse':<6} {'Yaş':<10} {'Durum':<8} {'Kaynak'}")
    print('-' * 80)
    for sym in sorted(indices.keys()):
        entry = indices[sym]
        age = get_index_age_days(sym)
        if age < 1:
            age_str = f"{age*24:.0f}h"
        elif age == float('inf'):
            age_str = "∞"
        else:
            age_str = f"{age:.1f}g"

        if age <= 7:
            status = "✓ taze"
        elif age <= 30:
            status = "⚠ eski"
        else:
            status = "✗ ÇOK ESKİ"

        source = entry.get('source', '?')[:30]
        print(f"{sym:<10} {entry.get('count', 0):<6} {age_str:<10} {status:<8} {source}")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(description='BIST Veri Kaynağı')
    p.add_argument('--update', type=str, help='Tek endeksi canlı güncelle')
    p.add_argument('--update-all', action='store_true',
                   help='Tüm ana endeksleri güncelle')
    p.add_argument('--status', action='store_true',
                   help='Cache durumunu göster (yaş, kaynak)')
    p.add_argument('--probe', type=str,
                   help='Endeksi tüm kaynaklardan dene (debug)')
    args = p.parse_args()

    if args.update:
        ok = update_index(args.update)
        sys.exit(0 if ok else 1)

    if args.update_all:
        r = update_all()
        sys.exit(0 if r['success'] and not r['failed'] else 1)

    if args.status:
        status_report()
        return

    if args.probe:
        sym = args.probe
        print(f"\n=== {sym} probe — tüm kaynaklar ===\n")
        for name, func in [('borsapy', fetch_borsapy),
                           ('ISYatırım', fetch_isyatirim),
                           ('mynet', fetch_mynet)]:
            print(f"\n--- {name} ---")
            symbols, info = func(sym)
            print(f"Info: {info}")
            print(f"Sembol sayısı: {len(symbols)}")
            if symbols:
                print(f"İlk 10: {', '.join(symbols[:10])}")
        return

    p.print_help()


if __name__ == '__main__':
    main()
