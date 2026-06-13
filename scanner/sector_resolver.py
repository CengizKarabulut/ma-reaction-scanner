"""
BIST Sektör & Endeks Çözücü — borsapy v0.10+ TradingView backend.

Üç farklı sorgu desteği:
1. Hisse → Sektör/Industry bilgisi (bp.Ticker.info)
2. Endeks → Bileşen hisseler (bp.Index().component_symbols)
3. Sektör adı → BIST'teki tüm hisseleri

Kullanım:
    # Cache oluştur (~3-5 dk, 500+ hisse)
    python sector_resolver.py --build-cache

    # Endeks bileşenleri (anlık, cache gereksiz)
    python sector_resolver.py --index XBANK

    # Tüm BIST endekslerini listele
    python sector_resolver.py --list-indices

    # Sektördeki hisseleri listele
    python sector_resolver.py --sector "Bankacılık"

    # Borsapy sektör listesi
    python sector_resolver.py --list-sectors

Modül kullanımı:
    from sector_resolver import (
        get_tickers_by_index,    # Endeks bileşenleri (borsapy direkt)
        get_tickers_by_sector,   # Sektör hisseleri (cache'ten)
        list_all_indices,        # 79 BIST endeksi
        list_all_sectors,        # 53 sektör
    )
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import borsapy as bp
    HAS_BORSAPY = True
except ImportError:
    HAS_BORSAPY = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


_CACHE_DIR = Path(__file__).parent / '.cache'
_SECTORS_CACHE = _CACHE_DIR / 'sectors_cache.json'
_INDICES_CACHE = _CACHE_DIR / 'indices_cache.json'


def _ensure_cache_dir():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. ENDEKS BİLEŞENLERİ — Borsapy direkt (cache gereksiz)
# ============================================================

def get_tickers_by_index(index_symbol: str) -> list:
    """Endeksin bileşen hisselerini döndür - borsapy ile direkt.

    Örnek:
        get_tickers_by_index('XBANK')
        → ['AKBNK', 'GARAN', 'ISCTR', 'YKBNK', ...]
    """
    if not HAS_BORSAPY:
        print("⚠️ borsapy yok", file=sys.stderr)
        return []
    try:
        idx = bp.Index(index_symbol.upper())
        symbols = idx.component_symbols
        if symbols:
            return sorted(symbols)
    except Exception as e:
        print(f"⚠️ Endeks {index_symbol} bileşenleri alınamadı: {e}", file=sys.stderr)
    return []


def list_all_indices(detailed: bool = False) -> list:
    """Tüm BIST endekslerini listele - 79 endeks.

    Args:
        detailed: True ise [{'symbol', 'name', 'count'}] döner, False ise sadece sembol listesi
    """
    if not HAS_BORSAPY:
        return []
    try:
        if hasattr(bp, 'all_indices'):
            return bp.all_indices() if detailed else [
                d['symbol'] if isinstance(d, dict) else d
                for d in bp.all_indices()
            ]
        # Fallback - 33 popüler endeks
        if hasattr(bp, 'indices'):
            return bp.indices(detailed=detailed)
    except Exception as e:
        print(f"⚠️ Endeks listesi alınamadı: {e}", file=sys.stderr)
    return []


def list_all_sectors() -> list:
    """borsapy'nin tanıdığı 53 sektör listesi (Türkçe)."""
    if not HAS_BORSAPY:
        return []
    try:
        if hasattr(bp, 'sectors'):
            return bp.sectors()
    except Exception as e:
        print(f"⚠️ Sektör listesi alınamadı: {e}", file=sys.stderr)
    return []


# ============================================================
# 2. HİSSE → SEKTÖR (cache + borsapy backend)
# ============================================================

def load_sectors_cache() -> dict:
    if not _SECTORS_CACHE.exists():
        return {}
    try:
        return json.loads(_SECTORS_CACHE.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_sectors_cache(data: dict):
    _ensure_cache_dir()
    payload = {
        'updated_at': datetime.now().isoformat(),
        'count': len(data.get('sectors', {})),
        'backend': data.get('backend', 'borsapy'),
        'sectors': data.get('sectors', {}),
    }
    _SECTORS_CACHE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


def fetch_sector_for_ticker(ticker: str) -> dict:
    """Tek hissenin sektör bilgisi — borsapy → yfinance fallback."""
    base = ticker.replace('.IS', '').upper()

    # 1. Borsapy (öncelikli)
    if HAS_BORSAPY:
        try:
            t = bp.Ticker(base)
            info = t.info or {}
            sector = (info.get('sector') or '').strip()
            industry = (info.get('industry') or '').strip()
            name = (info.get('longName') or info.get('shortName') or info.get('name') or base).strip()
            if sector or industry:
                return {
                    'sector': sector or 'Unknown',
                    'industry': industry or 'Unknown',
                    'name': name,
                    'source': 'borsapy',
                }
        except Exception:
            pass

    # 2. yfinance fallback
    if HAS_YFINANCE:
        try:
            t = yf.Ticker(f"{base}.IS")
            info = t.info or {}
            sector = (info.get('sector') or '').strip()
            industry = (info.get('industry') or '').strip()
            name = (info.get('longName') or info.get('shortName') or base).strip()
            if sector or industry:
                return {
                    'sector': sector or 'Unknown',
                    'industry': industry or 'Unknown',
                    'name': name,
                    'source': 'yfinance',
                }
        except Exception:
            pass

    return None


def build_sectors_cache(tickers: list, verbose: bool = True) -> dict:
    """Tüm ticker listesi için sektör bilgisi topla."""
    if not (HAS_BORSAPY or HAS_YFINANCE):
        print("✗ Ne borsapy ne yfinance var", file=sys.stderr)
        return {}

    backend = 'borsapy' if HAS_BORSAPY else 'yfinance'
    print(f"Sektör cache oluşturuluyor: {len(tickers)} hisse (backend={backend})")

    sectors = {}
    failed = []
    for i, tk in enumerate(tickers, 1):
        if verbose and i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {tk}... ({len(sectors)} ok, {len(failed)} fail)")

        info = fetch_sector_for_ticker(tk)
        if info:
            sectors[tk] = info
        else:
            failed.append(tk)

        # Rate limit (borsapy daha hızlı ama yine de)
        if i % 20 == 0:
            time.sleep(0.3)

    payload = {'sectors': sectors, 'backend': backend}
    save_sectors_cache(payload)
    print(f"\n✓ {len(sectors)} hisse cache'lendi, {len(failed)} fail")
    if failed:
        print(f"  Fail: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
    return payload


def get_tickers_by_sector(sector_name: str, exact: bool = False) -> list:
    """Belirli sektördeki BIST hisselerini cache'ten döndür."""
    cache = load_sectors_cache()
    sectors = cache.get('sectors', {})
    if not sectors:
        print("⚠️ Sektör cache boş. Önce: python sector_resolver.py --build-cache",
              file=sys.stderr)
        return []

    sector_lower = sector_name.lower()
    matched = []
    for tk, info in sectors.items():
        s = info.get('sector', '').lower()
        i = info.get('industry', '').lower()
        if exact:
            if s == sector_lower:
                matched.append(tk)
        else:
            if sector_lower in s or sector_lower in i:
                matched.append(tk)
    return sorted(matched)


def get_all_sectors_from_cache() -> dict:
    """Cache'ten sektör dağılımı: {sector: [tickers]}"""
    cache = load_sectors_cache()
    sectors = cache.get('sectors', {})
    by_sector = {}
    for tk, info in sectors.items():
        s = info.get('sector', 'Unknown')
        by_sector.setdefault(s, []).append(tk)
    return {k: sorted(v) for k, v in sorted(by_sector.items())}


def get_sector_info(ticker: str) -> dict:
    cache = load_sectors_cache()
    return cache.get('sectors', {}).get(ticker.upper())


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(description='BIST Sektör & Endeks Çözücü (borsapy)')
    p.add_argument('--build-cache', action='store_true',
                   help='Tüm BIST hisseleri için sektör cache oluştur')
    p.add_argument('--stats', action='store_true',
                   help='Cache istatistikleri')
    p.add_argument('--sector', type=str,
                   help='Sektördeki hisseleri listele')
    p.add_argument('--ticker', type=str,
                   help='Tek hisse sektör bilgisi')
    p.add_argument('--index', type=str,
                   help='Endeksin bileşen hisselerini listele (borsapy direkt)')
    p.add_argument('--list-indices', action='store_true',
                   help='Tüm BIST endekslerini listele (79 adet)')
    p.add_argument('--list-sectors', action='store_true',
                   help='borsapy sektör listesi (53 adet)')
    p.add_argument('--tickers-file', type=str, default=None,
                   help='Build-cache için ticker listesi dosyası')
    args = p.parse_args()

    # Endeks bileşenleri
    if args.index:
        tickers = get_tickers_by_index(args.index)
        if not tickers:
            print(f"'{args.index}' bileşenleri bulunamadı")
            print("Mevcut endeksler için: --list-indices")
            return
        print(f"\n📊 {args.index} → {len(tickers)} hisse:")
        print(','.join(tickers))
        return

    # Endeks listesi
    if args.list_indices:
        idxs = list_all_indices(detailed=True)
        if not idxs:
            print("Endeks alınamadı (borsapy yok?)")
            return
        print(f"\n📋 BIST Endeksleri ({len(idxs)} adet):\n")
        for d in idxs:
            if isinstance(d, dict):
                print(f"  {d.get('symbol', '?'):<10} {d.get('name', ''):<35} ({d.get('count', '?')} hisse)")
            else:
                print(f"  {d}")
        return

    # Borsapy sektör listesi
    if args.list_sectors:
        secs = list_all_sectors()
        if not secs:
            print("Sektör listesi alınamadı")
            return
        print(f"\n📋 borsapy sektörleri ({len(secs)} adet):\n")
        for s in secs:
            print(f"  - {s}")
        return

    # Cache build
    if args.build_cache:
        if args.tickers_file:
            with open(args.tickers_file) as f:
                tickers = [l.strip().upper() for l in f if l.strip()]
        else:
            try:
                # tickers.py'dan al
                sys.path.insert(0, str(Path(__file__).parent))
                from tickers import get_tickers
                tickers = get_tickers('BIST_TUM')
            except Exception as e:
                print(f"tickers.py'dan liste alınamadı: {e}", file=sys.stderr)
                sys.exit(1)
        build_sectors_cache(tickers)
        return

    # Stats
    if args.stats:
        by_sector = get_all_sectors_from_cache()
        if not by_sector:
            print("Cache boş. Önce: --build-cache")
            return
        total = sum(len(v) for v in by_sector.values())
        print(f"\n📊 Sektör Dağılımı (toplam {total} hisse)\n")
        for s, tks in sorted(by_sector.items(), key=lambda x: -len(x[1])):
            print(f"  {s:<40} {len(tks):>4} hisse")
        return

    # Sektördeki hisseleri
    if args.sector:
        tickers = get_tickers_by_sector(args.sector)
        if not tickers:
            print(f"'{args.sector}' sektöründe hisse yok")
            print("Mevcut sektörler için: --stats")
            return
        print(f"\n📋 '{args.sector}' → {len(tickers)} hisse:")
        print(','.join(tickers))
        return

    # Tek hisse
    if args.ticker:
        info = get_sector_info(args.ticker)
        if info:
            print(f"\n{args.ticker}:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        else:
            print(f"{args.ticker} cache'te yok, anlık çekiliyor...")
            info = fetch_sector_for_ticker(args.ticker)
            if info:
                for k, v in info.items():
                    print(f"  {k}: {v}")
        return

    p.print_help()


if __name__ == '__main__':
    main()
