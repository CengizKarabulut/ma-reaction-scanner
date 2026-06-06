"""BIST hisse listeleri — endeks bazlı."""

# BIST 30
BIST_30 = [
    "AKBNK","AKSEN","ARCLK","ASELS","BIMAS","DOAS","DOHOL","EKGYO","EREGL","FROTO",
    "GARAN","GUBRF","HEKTS","ISCTR","KCHOL","KOZAA","KOZAL","KRDMD","ODAS","PETKM",
    "PGSUS","SAHOL","SASA","SISE","TAVHL","TCELL","THYAO","TOASO","TUPRS","YKBNK",
]

# BIST 50 (BIST 30 + 20)
BIST_50 = BIST_30 + [
    "ALARK","ASUZU","BRSAN","CCOLA","ENJSA","ENKAI","KOZAL","MGROS","OYAKC","SOKM",
    "TKFEN","TSKB","TTKOM","ULKER","VAKBN","VESBE","VESTL","ZOREN","ALBRK","ALCTL",
]

# BIST 100 (BIST 50 + 50)
BIST_100 = list(set(BIST_50 + [
    "AGHOL","AKCNS","AKFGY","AKSA","ALCAR","ALKIM","ASTOR","AYDEM","AYGAZ","BAGFS",
    "BANVT","BERA","BIENY","BIOEN","BIZIM","BJKAS","BRYAT","CANTE","CIMSA","CWENE",
    "DEVA","DGGYO","DOCO","ECILC","ECZYT","EGEEN","EGGUB","ENERY","ESEN","EUPWR",
    "EUREN","FENER","GENIL","GESAN","GOLTS","GOZDE","GSDHO","GSRAY","HALKB","INDES",
    "ISMEN","IZINV","JANTS","KAREL","KAYSE","KFEIN","KLMSN","KLNMA","KMPUR","KONTR",
]))

# Tüm BIST (yaklaşık)
BIST_ALL_APPROX = sorted(set(BIST_100))


def get_list(name: str) -> list:
    """İsim ile liste döndür"""
    mapping = {
        'BIST_30': BIST_30,
        'BIST_50': BIST_50,
        'BIST_100': BIST_100,
        'BIST_ALL': BIST_ALL_APPROX,
    }
    return mapping.get(name.upper(), [])
