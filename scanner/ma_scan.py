#!/usr/bin/env python3
"""Configurable MA trend and reaction scanner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

from .asset_universe import ASSET_CLASSES, build_custom_instruments, list_universes, resolve_universe
from .ma_data import MarketDataProvider
from .ma_engine import DEFAULT_PERIODS, MA_TYPES, TIMEFRAMES, ScanConfig, build_market_summary, scan_frame
from .ma_levels import LevelConfig, finalize_level_frame
from .ma_relative import BenchmarkCache
from .stock_metadata import enrich_stock_instruments, format_index_memberships


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_periods(value: str) -> tuple[int, ...]:
    periods = tuple(dict.fromkeys(int(item) for item in parse_csv(value)))
    if not periods:
        raise ValueError("En az bir periyot gereklidir")
    if any(period < 1 or period > 5_000 for period in periods):
        raise ValueError(
            "MA periods must be between 1 and 5000; separate CSV values with commas"
        )
    return periods


def parse_ma_types(value: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(item.upper() for item in parse_csv(value)))
    unknown = sorted(set(values) - set(MA_TYPES))
    if unknown:
        raise ValueError(f"Bilinmeyen MA türleri: {', '.join(unknown)}")
    return values


def parse_timeframes(value: str) -> tuple[str, ...]:
    presets = {
        "all": TIMEFRAMES,
        "intraday": ("5m", "15m", "30m", "1h", "4h"),
        "swing": ("1d", "1wk", "1mo"),
        "daily": ("1d",),
    }
    key = value.strip().lower()
    values = presets.get(key, tuple(dict.fromkeys(item.lower() for item in parse_csv(value))))
    unknown = sorted(set(values) - set(TIMEFRAMES))
    if unknown:
        raise ValueError(f"Bilinmeyen zaman dilimleri: {', '.join(unknown)}")
    if not values:
        raise ValueError("En az bir zaman dilimi gereklidir")
    return tuple(values)


_LEVEL_CONFIG_JSON_KEYS = {
    "reaction_bars",
    "hold_bars",
    "break_atr",
    "bounce_cap_atr",
    "cross_cap_per_100",
    "evidence_target_touches",
    "neighbor_ratio",
}


def parse_level_config_json(value: str | None) -> dict[str, object]:
    """Parse optional LevelConfig overrides from a compact JSON workflow input."""

    text = (value or "").strip()
    if not text or text == "{}":
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"level_config_json gecersiz JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("level_config_json bir JSON obje olmali")
    unknown = sorted(set(payload) - _LEVEL_CONFIG_JSON_KEYS)
    if unknown:
        raise ValueError(f"level_config_json bilinmeyen alan: {', '.join(unknown)}")
    return payload


def resolve_instruments(args: argparse.Namespace):
    if args.universe == "custom":
        instruments = build_custom_instruments(
            parse_csv(args.symbols or args.symbol),
            asset_class=args.asset_class,
            market=args.market,
        )
    else:
        instruments = resolve_universe(args.universe, sector=args.sector)
    instruments = enrich_stock_instruments(instruments)
    if args.shard_count > 1:
        instruments = [
            item for index, item in enumerate(instruments)
            if index % args.shard_count == args.shard_index
        ]
    return instruments[: args.max_symbols] if args.max_symbols > 0 else instruments


def instrument_metadata(instrument) -> dict[str, object]:
    return {
        "asset_class": instrument.asset_class,
        "asset_label": instrument.asset_label,
        "display_name": instrument.display_name,
        "market": instrument.market,
        "sector": instrument.sector,
        "industry": instrument.industry,
        "index_memberships": format_index_memberships(instrument.index_memberships),
    }


_SINGLE_TABLE_COLUMNS = {
    "timeframe": "Zaman Dilimi",
    "ma_type": "MA Türü",
    "period": "Periyot",
    "ma": "MA",
    "side": "Taraf",
    "active_side": "Güncel Rol",
    "trend_state": "Trend",
    "current_price": "Fiyat",
    "current_ma": "MA Değeri",
    "level_touches": "Temas",
    "touch_density_per_100": "Temas/100",
    "cross_count": "Kesişim",
    "cross_per_100": "Kesişim/100",
    "hold_rate_pct": "Tutma %",
    "break_rate_pct": "Kırılma %",
    "median_bounce_atr": "Sıçrama ATR",
    "median_penetration_atr": "Sarkma ATR",
    "side_adherence_pct": "Taraf Koruma %",
    "wrong_side_pct": "Yanlış Taraf %",
    "level_score": "Seviye Skoru",
    "level_class": "Seviye",
    "plateau_ratio": "Plato",
    "adherence_excess_pct": "Taraf Farkı",
    "win_rate_pct": "Kazanma %",
    "median_net_r": "Medyan R",
    "edge_r": "Edge R",
    "positive_periods": "Pozitif Dönem",
    "compatibility": "Uyum",
    "compatibility_score": "Uyum Skoru",
    "distance_pct": "Uzaklık %",
    "distance_atr": "ATR Uzaklık",
    "analysis_basis": "Analiz Bazı",
    "filter_status": "Filtre",
    "filter_reasons": "Filtre Nedeni",
}


def build_single_stock_table(
    detail: pd.DataFrame,
    timeframes: list[str] | tuple[str, ...] | None = None,
    ma_types: list[str] | tuple[str, ...] | None = None,
    periods: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    table = detail.copy() if detail is not None else pd.DataFrame()
    requested = {
        (str(timeframe), str(ma_type), int(period))
        for timeframe in (timeframes or [])
        for ma_type in (ma_types or [])
        for period in (periods or [])
    }
    present = (
        set(
            zip(
                table.get("timeframe", pd.Series(dtype=str)).astype(str),
                table.get("ma_type", pd.Series(dtype=str)).astype(str),
                pd.to_numeric(
                    table.get("period", pd.Series(dtype=float)), errors="coerce"
                ).fillna(-1).astype(int),
            )
        )
        if not table.empty
        else set()
    )
    missing_rows = []
    for timeframe, ma_type, period in sorted(requested - present):
        missing_rows.append({
            "timeframe": timeframe, "ma_type": ma_type, "period": period,
            "ma": f"{ma_type}{period}", "side": "Veri yok", "active_side": False,
            "trend_state": "Yetersiz veri", "touches": 0, "level_touches": 0,
            "touch_density_per_100": 0.0, "cross_count": 0, "cross_per_100": 0.0,
            "hold_rate_pct": float("nan"), "break_rate_pct": float("nan"),
            "median_bounce_atr": float("nan"), "median_penetration_atr": float("nan"),
            "level_score": 0.0, "level_class": "Yetersiz temas",
            "plateau_ratio": float("nan"), "adherence_excess_pct": float("nan"),
            "positive_periods": 0, "compatibility": "Yetersiz veri",
            "compatibility_score": 0.0, "analysis_basis": "nominal",
            "filter_pass": False, "filter_status": "Yetersiz veri",
            "filter_reasons": "Seçilen MA için yeterli geçmiş mum yok",
        })
    if missing_rows:
        table = pd.concat([table, pd.DataFrame(missing_rows)], ignore_index=True)
    if table.empty:
        return pd.DataFrame(columns=list(_SINGLE_TABLE_COLUMNS.values()))
    if "level_touches" not in table:
        table["level_touches"] = table.get("touches", 0)
    else:
        table["level_touches"] = pd.to_numeric(table["level_touches"], errors="coerce").fillna(
            pd.to_numeric(table.get("touches", 0), errors="coerce")
        )
    for column in _SINGLE_TABLE_COLUMNS:
        if column not in table:
            table[column] = float("nan")
    quality_rank = {
        "Güçlü uyum": 4, "Guclu uyum": 4, "Uyumlu": 3,
        "İzleme": 2, "Izleme": 2, "Uyumsuz": 1, "Yetersiz veri": 0,
    }
    level_rank = {
        "Guclu seviye": 4, "Güçlü seviye": 4, "Seviye": 3,
        "Zayif seviye": 2, "Zayıf seviye": 2,
        "Seviye degil": 1, "Seviye değil": 1, "Yetersiz temas": 0,
    }
    filter_pass = table["filter_pass"] if "filter_pass" in table else pd.Series(True, index=table.index)
    active_side = table["active_side"] if "active_side" in table else pd.Series(False, index=table.index)
    table["_filter_rank"] = filter_pass.fillna(False).astype(int)
    table["_active_rank"] = active_side.fillna(False).astype(int)
    table["_quality_rank"] = table["compatibility"].map(quality_rank).fillna(0)
    table["_level_rank"] = table["level_class"].map(level_rank).fillna(0)
    table["compatibility_score"] = pd.to_numeric(
        table["compatibility_score"], errors="coerce"
    ).fillna(table["_quality_rank"] * 20.0)
    table["level_score"] = pd.to_numeric(table["level_score"], errors="coerce").fillna(
        table["compatibility_score"]
    )
    table["level_touches"] = pd.to_numeric(table["level_touches"], errors="coerce").fillna(0)
    table["_abs_distance"] = pd.to_numeric(table["distance_atr"], errors="coerce").abs()
    table = table.sort_values(
        [
            "_filter_rank", "_active_rank", "level_score", "level_touches",
            "_level_rank", "hold_rate_pct", "median_bounce_atr",
            "compatibility_score", "_quality_rank", "positive_periods",
            "edge_r", "median_net_r", "_abs_distance",
        ],
        ascending=[False, False, False, False, False, False, False, False, False, False, False, False, True],
        na_position="last",
    )
    available = [column for column in _SINGLE_TABLE_COLUMNS if column in table]
    result = table[available].rename(columns=_SINGLE_TABLE_COLUMNS).reset_index(drop=True)
    if "Güncel Rol" in result:
        result["Güncel Rol"] = result["Güncel Rol"].map(
            {True: "Aktif", False: "Diğer taraf"}
        ).fillna(result["Güncel Rol"])
    return result

def write_outputs(
    output_dir: Path,
    detail: pd.DataFrame,
    errors: pd.DataFrame,
    config_payload: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail = detail.copy() if detail is not None else pd.DataFrame()
    level_payload = config_payload.get("level", {})
    level_config = LevelConfig(**level_payload) if isinstance(level_payload, dict) else LevelConfig()
    if not detail.empty:
        detail = finalize_level_frame(detail, level_config)
    scan_payload = config_payload.get("scan", {}) if isinstance(config_payload.get("scan", {}), dict) else {}
    summary = build_market_summary(
        detail,
        near_distance_atr=float(scan_payload.get("near_distance_atr", ScanConfig().near_distance_atr)),
        rank_by=str(config_payload.get("rank_by", "level")),
    )
    detail.to_csv(output_dir / "ma_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "market_summary.csv", index=False, encoding="utf-8-sig")
    market_columns = {
        "symbol": "Varl?k", "best_ma": "MA", "best_timeframe": "Zaman Dilimi",
        "best_level_touches": "Temas", "best_level_score": "Seviye Skoru",
        "best_level_class": "Seviye", "best_hold_rate_pct": "Tutma %",
        "best_median_bounce_atr": "S??rama ATR", "best_touch_density_per_100": "Temas/100",
        "best_cross_count": "Kesi?im", "best_cross_per_100": "Kesi?im/100",
        "best_plateau_ratio": "Plato", "best_adherence_excess_pct": "Taraf Fark?",
        "best_side_adherence_pct": "Taraf Koruma %",
        "best_win_rate_pct": "Kazanma %", "best_median_net_r": "Medyan R",
        "best_edge_r": "Edge R", "best_distance_pct": "Uzakl?k %",
        "best_compatibility_score": "Uyum Skoru", "best_compatibility": "Uyum",
        "best_side": "Taraf", "current_price": "Fiyat", "best_ma_value": "MA De?eri",
        "analysis_basis": "Analiz Baz?", "filter_status": "Filtre", "filter_reasons": "Filtre Nedeni",
    }
    market_table = summary[[column for column in market_columns if column in summary]].rename(
        columns=market_columns
    )
    market_table.to_csv(output_dir / "market_table.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")
    (output_dir / "run_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    single_table = None
    if int(config_payload.get("instrument_count", 0)) == 1:
        scan_settings = config_payload.get("scan", {})
        single_table = build_single_stock_table(
            detail,
            timeframes=config_payload.get("timeframes", []),
            ma_types=scan_settings.get("ma_types", []),
            periods=scan_settings.get("periods", []),
        )
        single_table.to_csv(
            output_dir / "single_stock_table.csv",
            index=False,
            encoding="utf-8-sig",
        )
    html = (
        "<!doctype html><html lang='tr'><head><meta charset='utf-8'>"
        "<title>MA Trend ve Tepki Raporu</title>"
        "<style>body{font:14px system-ui;margin:24px;background:#f7f8fa;color:#18202a}"
        "table{border-collapse:collapse;width:100%;background:white}th,td{padding:7px;"
        "border:1px solid #dfe3e8;text-align:left;white-space:nowrap}th{position:sticky;"
        "top:0;background:#172b4d;color:white}tr:nth-child(even){background:#f2f5f8}"
        "h1,h2{color:#172b4d}.meta{color:#566}.guide{background:white;padding:16px;"
        "border:1px solid #dfe3e8;margin:14px 0}.guide code{font-weight:bold}</style>"
        "</head><body><h1>MA Trend ve Tepki - Piyasa Ozeti</h1>"
        f"<p class='meta'>{len(summary)} varlik - her varlik tek satir</p>"
        "<div class='guide'><h2>Tablo nasil okunur?</h2><ul>"
        "<li><code>current_price</code> ve <code>best_ma_value</code> ayni zaman "
        "dilimindeki <code>price_time</code> anina aittir.</li>"
        "<li><code>best_difference</code> = MA - fiyat (TL).</li>"
        "<li><code>best_distance_pct</code> gercek yuzde uzakliktir. "
        "<code>best_distance_atr</code> ATR uzakligidir; yuzde degildir.</li>"
        "<li><code>best_compatibility_score</code> 0-100 arası Uyum Skorudur; temas, taraf koruma, kazanma, Medyan R, Edge ve istikrarı birleştirir.</li>"
        "<li><code>filter_status</code> Uygun satirlar once gelir. Filtre disi "
        "hisseler silinmez; neden <code>filter_reasons</code> alanindadir.</li>"
        "<li>Seviye/uyum s?n?flar? ge?mi? davran?? ?zetidir; otomatik al-sat emri de?ildir.</li>"
        "</ul></div>"
        + market_table.to_html(
            index=False,
            border=0,
            escape=True,
            float_format=lambda value: f"{value:,.2f}",
        )
        + "</body></html>"
    )
    (output_dir / "market_report.html").write_text(html, encoding="utf-8")
    if single_table is not None:
        single_html = (
            "<!doctype html><html lang='tr'><head><meta charset='utf-8'>"
            "<title>Tek Hisse MA Detay Tablosu</title>"
            "<style>body{font:14px system-ui;margin:24px;background:#f7f8fa;color:#18202a}"
            "table{border-collapse:collapse;width:100%;background:white}th,td{padding:7px;"
            "border:1px solid #dfe3e8;text-align:left;white-space:nowrap}th{position:sticky;"
            "top:0;background:#172b4d;color:white}tr:nth-child(even){background:#f2f5f8}"
            "h1,h2{color:#172b4d}.guide{background:white;padding:16px;border:1px solid "
            "#dfe3e8;margin:14px 0}</style></head><body>"
            "<h1>Tek Hisse - Seçilen MA Türleri ve Periyotları</h1>"
            f"<p>{len(single_table)} analiz satırı</p>"
            "<div class='guide'><h2>Tablo nasıl okunur?</h2><ul>"
            "<li><b>Temas</b>: birbirinden ayrıştırılmış tarihsel MA temas sayısıdır.</li>"
            "<li><b>Taraf Koruma %</b>: destek için MA üstünde, direnç için MA altında "
            "kapanan mumların oranıdır.</li>"
            "<li><b>Yanlış Taraf %</b>: fiyatın beklenen tarafın tersinde kaldığı orandır.</li>"
            "<li><b>Kesişim</b>: fiyatın MA tarafını kaç kez değiştirdiğini gösterir; "
            "yüksek değer kararsızlığa işaret edebilir.</li>"
            "<li><b>Uyum Skoru</b>: temas, taraf koruma, kazanma, Medyan R, Edge ve istikrarın 0-100 birleşimidir.</li>"
            "<li><b>Medyan R / Edge R</b>: temas sonrası maliyet düzeltilmiş tepki ve "
            "rastgele giriş bazına göre avantajdır.</li></ul></div>"
            + single_table.to_html(
                index=False,
                border=0,
                escape=True,
                float_format=lambda value: f"{value:,.2f}",
            )
            + "</body></html>"
        )
        (output_dir / "single_stock_report.html").write_text(
            single_html,
            encoding="utf-8",
        )

def merge_outputs(
    merge_dir: Path,
    output_dir: Path,
    expected_shards: int = 0,
) -> int:
    detail_paths = list(merge_dir.rglob("ma_detail.csv"))
    error_paths = list(merge_dir.rglob("errors.csv"))
    config_paths = list(merge_dir.rglob("run_config.json"))
    if not detail_paths:
        raise RuntimeError(f"Birleştirilecek ma_detail.csv bulunamadı: {merge_dir}")
    if expected_shards and len(detail_paths) != expected_shards:
        raise RuntimeError(
            f"Missing shards: expected={expected_shards}, found={len(detail_paths)}"
        )
    if expected_shards and len(config_paths) != expected_shards:
        raise RuntimeError(
            f"Missing shard configs: expected={expected_shards}, found={len(config_paths)}"
        )
    configs = [
        json.loads(path.read_text(encoding="utf-8-sig"))
        for path in config_paths
    ]
    if expected_shards:
        indexes = {int(payload.get("shard_index", -1)) for payload in configs}
        expected_indexes = set(range(expected_shards))
        if indexes != expected_indexes:
            missing = sorted(expected_indexes - indexes)
            raise RuntimeError(f"Missing shard indexes: {missing}")
    details = []
    for path in detail_paths:
        if path.stat().st_size <= 3:
            continue
        try:
            details.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    if not detail.empty:
        detail = detail.drop_duplicates(
            ["asset_class", "symbol", "timeframe", "ma_type", "period", "side"]
        )
    error_frames = [pd.read_csv(path) for path in error_paths if path.stat().st_size > 3]
    errors = (
        pd.concat(error_frames, ignore_index=True)
        if error_frames
        else pd.DataFrame(columns=["symbol", "timeframe", "error"])
    )
    config_payload = (
        json.loads(config_paths[0].read_text(encoding="utf-8-sig"))
        if config_paths
        else {"scan": ScanConfig().to_dict()}
    )
    config_payload["merged_shards"] = len(detail_paths)
    write_outputs(output_dir, detail, errors, config_payload)
    print(f"Birleştirildi: {len(detail_paths)} parça")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="custom")
    parser.add_argument("--symbol", default="ASELS")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--asset-class", choices=list(ASSET_CLASSES), default="stock")
    parser.add_argument("--market", choices=["BIST", "GLOBAL"], default="BIST")
    parser.add_argument("--sector", default="Tümü / uygulanmaz")
    parser.add_argument("--list-universes", action="store_true")
    parser.add_argument("--timeframes", default="1d", help="CSV veya all/intraday/swing/daily")
    parser.add_argument("--ma-types", default=",".join(MA_TYPES))
    parser.add_argument("--periods", default=",".join(map(str, DEFAULT_PERIODS)))
    parser.add_argument("--trend-slope-bars", type=int, default=10)
    parser.add_argument("--trend-slope-threshold-atr", type=float, default=0.10)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--touch-zone-atr", type=float, default=0.20)
    parser.add_argument("--separation-atr", type=float, default=2.0)
    parser.add_argument("--min-touches", type=int, default=12)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.20)
    parser.add_argument("--trailing-stop-atr", type=float, default=2.0)
    parser.add_argument("--max-holding-bars", type=int, default=20)
    parser.add_argument("--roundtrip-cost-bps", type=float, default=25.0)
    parser.add_argument("--min-edge-r", type=float, default=0.10)
    parser.add_argument("--near-distance-atr", type=float, default=1.0)
    parser.add_argument("--quality-lookback", type=int, default=60)
    parser.add_argument("--min-price", type=float, default=1.0)
    parser.add_argument("--min-daily-turnover-try", type=float, default=1_000_000.0)
    parser.add_argument("--max-zero-volume-pct", type=float, default=20.0)
    parser.add_argument("--max-gap-pct", type=float, default=15.0)
    parser.add_argument("--max-abs-edge-r", type=float, default=5.0)
    parser.add_argument("--reaction-bars", type=int, default=10)
    parser.add_argument("--hold-bars", type=int, default=5)
    parser.add_argument("--break-atr", type=float, default=0.50)
    parser.add_argument("--bounce-cap-atr", type=float, default=2.0)
    parser.add_argument("--cross-cap-per-100", type=float, default=4.0)
    parser.add_argument("--evidence-target-touches", type=int, default=20)
    parser.add_argument("--neighbor-ratio", type=float, default=0.25)
    parser.add_argument(
        "--level-config-json",
        default="",
        help=(
            "Optional JSON overrides for MA-DNA level settings, e.g. "
            '{"evidence_target_touches": 30}'
        ),
    )
    parser.add_argument("--relative-to", default="", help="Benchmark symbol, e.g. XU100, for relative MA-DNA analysis")
    parser.add_argument("--rank-by", choices=["level", "compat"], default="level")
    parser.add_argument("--lookback", type=int, default=0)
    parser.add_argument("--source", choices=["auto", "borsapy", "yfinance"], default="auto")
    parser.add_argument("--prefer-cache", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-dir", default="reports/ma_scan")
    parser.add_argument("--merge-dir", default="")
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=0,
        help="Required shard count during merge; zero disables validation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_universes:
        print(pd.DataFrame(list_universes()).to_string(index=False))
        return 0
    output_dir = Path(args.output_dir)
    if args.merge_dir:
        return merge_outputs(
            Path(args.merge_dir),
            output_dir,
            expected_shards=args.expected_shards,
        )
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Shard ayarları geçersiz")
    timeframes = parse_timeframes(args.timeframes)
    config = ScanConfig(
        ma_types=parse_ma_types(args.ma_types),
        periods=parse_periods(args.periods),
        trend_slope_bars=args.trend_slope_bars,
        trend_slope_threshold_atr=args.trend_slope_threshold_atr,
        atr_period=args.atr_period,
        touch_zone_atr=args.touch_zone_atr,
        separation_atr=args.separation_atr,
        min_touches=args.min_touches,
        stop_buffer_atr=args.stop_buffer_atr,
        trailing_stop_atr=args.trailing_stop_atr,
        max_holding_bars=args.max_holding_bars,
        roundtrip_cost_bps=args.roundtrip_cost_bps,
        min_edge_r=args.min_edge_r,
        near_distance_atr=args.near_distance_atr,
        quality_lookback=args.quality_lookback,
        min_price=args.min_price,
        min_daily_turnover_try=args.min_daily_turnover_try,
        max_zero_volume_pct=args.max_zero_volume_pct,
        max_gap_pct=args.max_gap_pct,
        max_abs_edge_r=args.max_abs_edge_r,
    )
    level_config_values: dict[str, object] = {
        "reaction_bars": args.reaction_bars,
        "hold_bars": args.hold_bars,
        "break_atr": args.break_atr,
        "bounce_cap_atr": args.bounce_cap_atr,
        "cross_cap_per_100": args.cross_cap_per_100,
        "evidence_target_touches": args.evidence_target_touches,
        "neighbor_ratio": args.neighbor_ratio,
    }
    level_config_values.update(parse_level_config_json(args.level_config_json))
    level_config = LevelConfig(**level_config_values)
    instruments = resolve_instruments(args)
    attempted_requests = len(instruments) * len(timeframes)
    provider = MarketDataProvider(source=args.source)
    relative_to = args.relative_to.strip().upper()
    benchmark_cache = (
        BenchmarkCache(provider, relative_to, prefer_cache=args.prefer_cache)
        if relative_to
        else None
    )
    details: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    for instrument in instruments:
        for timeframe in timeframes:
            try:
                benchmark = benchmark_cache.get(timeframe) if benchmark_cache is not None else None
                fetched = provider.fetch(
                    instrument.symbol,
                    timeframe,
                    prefer_cache=args.prefer_cache,
                    asset_class=instrument.asset_class,
                    market=instrument.market,
                )
                frame = fetched.frame.tail(args.lookback) if args.lookback > 0 else fetched.frame
                result = scan_frame(
                    frame,
                    symbol=instrument.symbol,
                    timeframe=timeframe,
                    config=config,
                    level_config=level_config,
                    benchmark=benchmark,
                    relative_to=relative_to or None,
                    metadata={
                        **instrument_metadata(instrument),
                        "data_source": fetched.source,
                        "data_fingerprint": fetched.fingerprint,
                    },
                )
                if not result.empty:
                    details.append(result)
                print(f"OK {instrument.symbol} {timeframe}: {len(result)} satır")
            except Exception as exc:
                errors.append({"symbol": instrument.symbol, "timeframe": timeframe, "error": str(exc)})
                print(f"ERROR {instrument.symbol} {timeframe}: {exc}", file=sys.stderr)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    error_frame = pd.DataFrame(errors, columns=["symbol", "timeframe", "error"])
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe": args.universe,
        "timeframes": list(timeframes),
        "scan": config.to_dict(),
        "level": level_config.to_dict(),
        "relative_to": relative_to,
        "rank_by": args.rank_by,
        "lookback": args.lookback,
        "source": args.source,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "instrument_count": len(instruments),
    }
    write_outputs(output_dir, detail, error_frame, payload)
    print(f"Çıktılar: {output_dir.resolve()}")
    return 1 if attempted_requests > 0 and len(errors) == attempted_requests else 0


if __name__ == "__main__":
    raise SystemExit(main())
