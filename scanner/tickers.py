"""BIST hisse listeleri — borsapy varsa dinamik endeks bileşenleri, yoksa hardcoded fallback."""

# === Fallback listeler (borsapy yoksa kullanılır) ===

BIST_30_FALLBACK = [
    "AKBNK","AKSEN","ARCLK","ASELS","BIMAS","DOAS","DOHOL","EKGYO","EREGL","FROTO",
    "GARAN","GUBRF","HEKTS","ISCTR","KCHOL","KOZAA","KOZAL","KRDMD","ODAS","PETKM",
    "PGSUS","SAHOL","SASA","SISE","TAVHL","TCELL","THYAO","TOASO","TUPRS","YKBNK",
]

BIST_50_FALLBACK = BIST_30_FALLBACK + [
    "ALARK","ASUZU","BRSAN","CCOLA","ENJSA","ENKAI","MGROS","OYAKC","SOKM",
    "TKFEN","TSKB","TTKOM","ULKER","VAKBN","VESBE","VESTL","ZOREN","ALBRK","ALCTL",
]

BIST_100_FALLBACK = sorted(set(BIST_50_FALLBACK + [
    "AGHOL","AKCNS","AKFGY","AKSA","ALCAR","ALKIM","ASTOR","AYDEM","AYGAZ","BAGFS",
    "BANVT","BERA","BIENY","BIOEN","BIZIM","BJKAS","BRYAT","CANTE","CIMSA","CWENE",
    "DEVA","DGGYO","DOCO","ECILC","ECZYT","EGEEN","EGGUB","ENERY","ESEN","EUPWR",
    "EUREN","FENER","GENIL","GESAN","GOLTS","GOZDE","GSDHO","GSRAY","HALKB","INDES",
    "ISMEN","IZINV","JANTS","KAREL","KAYSE","KFEIN","KLMSN","KLNMA","KMPUR","KONTR",
]))


def _get_borsapy_components(index_symbol: str) -> list:
    """borsapy ile endeks bileşenlerini çek. Hata durumunda boş liste döner."""
    try:
        import borsapy as bp
        idx = bp.Index(index_symbol)
        components = idx.component_symbols
        if components:
            print(f"  borsapy: {index_symbol} icin {len(components)} hisse cekildi")
        return components or []
    except Exception as e:
        print(f"  Uyari: borsapy'den {index_symbol} bilesenleri alinamadi ({e}), fallback kullaniliyor")
        return []


def _get_all_bist_stocks() -> list:
    """BIST'in tum hisselerini cek (XUTUM endeksi, 500+ hisse).
    Birden fazla endeksi birlestirerek daha kapsamli liste olusturur."""
    try:
        import borsapy as bp
        all_symbols = set()

        # XUTUM: BIST Tum (~500+ hisse, ana liste)
        try:
            xutum = bp.Index("XUTUM")
            symbols = xutum.component_symbols
            if symbols:
                all_symbols.update(symbols)
                print(f"  borsapy: XUTUM -> {len(symbols)} hisse")
        except Exception as e:
            print(f"  XUTUM cekilemedi: {e}")

        # XKTUM: BIST Katilim Tum (218 hisse, ek hisseler icin)
        try:
            xktum = bp.Index("XKTUM")
            symbols = xktum.component_symbols
            if symbols:
                all_symbols.update(symbols)
        except Exception:
            pass

        return sorted(all_symbols)
    except ImportError:
        print("  HATA: BIST_TUM icin borsapy gerekli")
        return []


def get_list(name: str) -> list:
    """Isim ile liste dondur. borsapy varsa dinamik, yoksa hardcoded fallback.

    Desteklenen isimler:
    - BIST_30, BIST_50, BIST_100  -> Ana endeksler
    - BIST_TUM (veya BIST_ALL)    -> Tum BIST (~500+ hisse, borsapy gerekli)
    - XU030, XU050, XU100, XUTUM  -> Endeks sembolleri direkt
    - XK030, XK050, XK100, XKTUM  -> Katilim endeksleri
    - XBANK, XUSIN, XUMAL, ...    -> Sektor endeksleri
    """
    name_upper = name.upper()

    # Tum BIST ozel durum
    if name_upper in ('BIST_TUM', 'BIST_ALL', 'XUTUM'):
        components = _get_all_bist_stocks()
        if components:
            return components
        # Fallback: BIST 100
        print("  Tum BIST icin borsapy gerekli, BIST_100 fallback'ine donuluyor")
        return BIST_100_FALLBACK

    # Endeks sembolu direkt verildiyse
    if name_upper.startswith('X') and len(name_upper) >= 4:
        components = _get_borsapy_components(name_upper)
        if components:
            return components
        print(f"  HATA: {name_upper} icin borsapy gerekli")
        return []

    # Adlandirilmis kisayollar
    mapping = {
        'BIST_30':  ('XU030', BIST_30_FALLBACK),
        'BIST_50':  ('XU050', BIST_50_FALLBACK),
        'BIST_100': ('XU100', BIST_100_FALLBACK),
    }

    if name_upper not in mapping:
        return []

    index_symbol, fallback = mapping[name_upper]
    components = _get_borsapy_components(index_symbol)
    if components:
        return components
    return fallback


# Backward compatibility
BIST_30 = BIST_30_FALLBACK
BIST_50 = BIST_50_FALLBACK
BIST_100 = BIST_100_FALLBACK
BIST_ALL_APPROX = BIST_100_FALLBACK
