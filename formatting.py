from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from telegram import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton

from config import FEELS_LIKE_DIFF_C, RAIN_POP_THRESHOLD, UMBRELLA_POP_THRESHOLD, WIND_STRONG_KPH
from forecast import compute_rain_periods_threshold, extract_min_max, pick_point_for_hour
from providers import ProviderResult
from storage import Storage
from utils import (
    clamp, condition_group_from_icon, format_date_italian, format_temp, format_time_italian, format_wind,
    get_rain_risk_icon, get_weather_icon, hour_band, humanize_description, md_escape, rain_risk_label,
    season_from_date, zone_bucket
)

def format_meteo_message(
    fused: Dict[str, Any],
    city: str,
    country: str,
    min_t: Optional[float],
    max_t: Optional[float],
    rain_msg: str,
    sources_label: str,
    rain_pop: Optional[float] = None,
    cache_age_min: Optional[int] = None,
    local_dt: Optional[datetime] = None,
    reliability_percent: Optional[int] = None,
    fallback_note: Optional[str] = None,
    details: Optional[str] = None,
    prefs: Optional[Dict[str, Any]] = None,
    cache_stale: bool = False,
) -> str:
    city_md = md_escape(city)
    country_md = md_escape(country)
    if cache_age_min is None:
        cache_label = "Live"
    else:
        cache_label = f"Cache {cache_age_min}m"
        if cache_stale:
            cache_label += " (stale)"
    if local_dt is None:
        local_dt = datetime.now()
    is_day = 6 <= local_dt.hour <= 19
    icon = get_weather_icon(fused.get("icon", "01d"), is_day=is_day)
    rain_icon = get_rain_risk_icon(
        rain_pop,
        icon_code=fused.get("icon"),
        description=fused.get("description"),
    )
    rain_risk = rain_risk_label(rain_pop)
    if rain_icon and icon in {"\U0001F327\ufe0f", "\U0001F326\ufe0f", "\u26c8\ufe0f"}:
        rain_icon = ""
    date_label = format_date_italian(local_dt)
    time_label = format_time_italian(local_dt)
    desc = humanize_description(
        fused.get("description", ""),
        float(fused.get("temp", 0)),
        float(fused.get("feels_like", 0)),
        fused.get("humidity"),
    )
    desc_md = md_escape(desc)
    reliability_txt = f"Affidabilita: {int(reliability_percent)}%" if reliability_percent is not None else ""
    reliability_txt = md_escape(reliability_txt) if reliability_txt else ""
    fallback_note = md_escape(fallback_note) if fallback_note else None
    cond_line = f"{icon} Condizioni: {desc_md}"
    if rain_icon:
        cond_line += f" {rain_icon}"
    lines = [
        f"{icon} METEO {city_md}, {country_md}",
        f"{date_label} {time_label}",
        "",
        f"\U0001F321 Ora: {format_temp(fused.get('temp'), prefs)} (Percepita {format_temp(fused.get('feels_like'), prefs)})",
        cond_line,
    ]
    if rain_risk:
        lines.append(f"Rischio pioggia: {rain_risk}")
    if rain_msg:
        if rain_icon:
            lines.append(f"{rain_icon} Pioggia: {rain_msg}")
        else:
            lines.append(f"Pioggia: {rain_msg}")
    lines.append("")
    lines.append(f"Fonti: {sources_label}")
    lines.append(f"Dato: {cache_label}")
    if fallback_note:
        lines.append(f"Nota: {fallback_note}")
    if details:
        lines.append(details)
    if reliability_txt:
        lines.append(reliability_txt)
    return "\n".join(lines)



def build_advice(fused: Dict[str, Any], rain_current: bool, rain_max_prob: float, prefs: Optional[Dict[str, float]] = None) -> str:
    tips = []
    pop_th = float(prefs.get("pop_threshold", UMBRELLA_POP_THRESHOLD)) if prefs else UMBRELLA_POP_THRESHOLD
    wind_th = float(prefs.get("wind_threshold", WIND_STRONG_KPH)) if prefs else WIND_STRONG_KPH
    diff_th = float(prefs.get("feels_diff", FEELS_LIKE_DIFF_C)) if prefs else FEELS_LIKE_DIFF_C
    if rain_current or (rain_max_prob is not None and rain_max_prob >= pop_th):
        tips.append("ombrello")
    wind = fused.get("wind_kph")
    if wind is not None and wind > wind_th:
        tips.append("vento forte")
    if abs(fused.get("feels_like", 0) - fused.get("temp", 0)) > diff_th:
        tips.append("percepita diversa")
    return ", ".join(tips) if tips else "Nessun consiglio particolare"


def format_sources_label(sources: List[str], corrected: bool, missing: Optional[List[str]] = None) -> str:
    used = len({s for s in sources if s})
    label = f"{used}"
    if missing:
        miss_txt = ", ".join(missing)
        if len(missing) == 1:
            label += f" (manca {miss_txt})"
        else:
            label += f" (mancano {miss_txt})"
    if corrected:
        label += " (correzione errori)"
    return label


def compute_confidence(ow: ProviderResult, wa: ProviderResult, acc: Dict[str, Dict[str, Any]]) -> str:
    if ow.success and wa.success:
        diff = abs(ow.temp - wa.temp)
    else:
        diff = 0.0
    acc_vals = []
    if acc.get("OpenWeather"):
        acc_vals.append(acc["OpenWeather"]["accuracy"])
    if acc.get("WeatherAPI"):
        acc_vals.append(acc["WeatherAPI"]["accuracy"])
    acc_avg = sum(acc_vals) / len(acc_vals) if acc_vals else 70.0
    if diff <= 1.5 and acc_avg >= 70:
        return "alta"
    if diff <= 3 and acc_avg >= 50:
        return "media"
    return "bassa"

def confidence_from_accuracy(acc: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not acc:
        return None
    acc_vals = [v["accuracy"] for v in acc.values() if v.get("accuracy") is not None]
    if not acc_vals:
        return None
    acc_avg = sum(acc_vals) / len(acc_vals)
    if acc_avg >= 70:
        return "alta"
    if acc_avg >= 50:
        return "media"
    return "bassa"


def format_details_line(fused: Dict[str, Any], acc: Dict[str, Dict[str, Any]], prefs: Optional[Dict[str, Any]] = None) -> str:
    wind = format_wind(fused.get("wind_kph"), prefs)
    humidity = f"{fused.get('humidity'):.0f}%" if fused.get("humidity") is not None else "n/d"
    pressure = f"{fused.get('pressure'):.0f} hPa" if fused.get("pressure") is not None else "n/d"
    line1 = f"Dettagli: vento {wind}, umidita {humidity}, pressione {pressure}"
    provider_details = fused.get("provider_details") or {}
    parts = []
    for name, info in provider_details.items():
        short = "OW" if name == "OpenWeather" else "WA" if name == "WeatherAPI" else "MB" if name == "Meteoblue" else name
        temp = info.get("temp")
        weight = info.get("weight")
        acc_val = acc.get(name, {}).get("accuracy") if acc else None
        seg = f"{short} {format_temp(temp, prefs)}" if temp is not None else f"{short} n/d"
        if weight is not None:
            seg += f" peso {weight:.2f}"
        if acc_val is not None:
            seg += f" acc {acc_val:.0f}%"
        parts.append(seg)
    line2 = "Provider: " + (" | ".join(parts) if parts else "n/d")
    return line1 + "\n" + line2
def format_forecast_message(
    points: List[Dict[str, Any]],
    city: str,
    country: str,
    days: int,
    sources_label: str,
    fallback_note: Optional[str] = None,
    target_date: Optional[date] = None,
    points_ow: Optional[List[Dict[str, Any]]] = None,
    points_wa: Optional[List[Dict[str, Any]]] = None,
    points_mb: Optional[List[Dict[str, Any]]] = None,
    acc: Optional[Dict[str, Dict[str, Any]]] = None,
    rain_threshold: Optional[float] = None,
    storage: Optional[Storage] = None,
    season: Optional[str] = None,
    zone: Optional[str] = None,
    kind_group: Optional[str] = None,
    prefs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not points:
        return None
    if target_date is None:
        target_date = (datetime.now().date() + timedelta(days=days))
    title = "OGGI" if days == 0 else "DOMANI"

    min_t, max_t = extract_min_max(points)
    acc = acc or {}

    if days == 1:
        chosen = sorted(
            [p for p in points if p["local_time"].date() == target_date],
            key=lambda x: x["dt_utc"]
        )
    else:
        blocks = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
        chosen = []
        used = set()
        for b in blocks:
            bh = int(b.split(":")[0])
            best = pick_point_for_hour(points, bh, 1)
            if best and best["ora"] not in used:
                used.add(best["ora"])
                chosen.append(best)

    th = float(rain_threshold) if rain_threshold is not None else RAIN_POP_THRESHOLD
    rain_periods = compute_rain_periods_threshold(points, th)

    lines = []
    lines.append(f"PREVISIONI {title}")
    lines.append(f"{md_escape(city)}, {md_escape(country)}")
    lines.append(f"{format_date_italian(datetime.combine(target_date, datetime.min.time()))}")
    lines.append("")
    if min_t is not None and max_t is not None:
        lines.append(f"Min/Max: {format_temp(min_t, prefs)} / {format_temp(max_t, prefs)}")
    max_pop = max((float(p.get("pop", 0)) for p in points), default=None)
    rain_risk = rain_risk_label(max_pop)
    if rain_risk:
        lines.append(f"Rischio pioggia: {rain_risk}")
    lines.append("Orari (1h):" if days == 1 else "Orari (3h):")
    for p in chosen:
        hour = int(p["ora"].split(":")[0])
        ow_p = pick_point_for_hour(points_ow, hour, 2) if points_ow else None
        wa_p = pick_point_for_hour(points_wa, hour, 1) if points_wa else None
        mb_p = pick_point_for_hour(points_mb, hour, 1) if points_mb else None
        if storage:
            cond = condition_group_from_icon(str(p.get("icon", "01d")), float(p.get("pop", 0)))
            conf = hourly_confidence_percent_context(
                storage, ow_p, wa_p, mb_p, hour_band(hour), cond,
                kind_group=kind_group, season=season, zone=zone
            )
        else:
            conf = hourly_confidence_percent(ow_p, wa_p, mb_p, acc)
        is_day = 6 <= hour <= 19
        icon = get_weather_icon(str(p.get("icon", "01d")), pop=p.get("pop"), is_day=is_day)
        rain_icon = get_rain_risk_icon(p.get("pop"), icon_code=p.get("icon"))
        if rain_icon and icon in {"\U0001F327\ufe0f", "\U0001F326\ufe0f", "\u26c8\ufe0f"}:
            rain_icon = ""
        rain = f" (POP {p['pop']:.0f}%)" if p["pop"] > 20 else ""
        risk = f" {rain_icon}" if rain_icon else ""
        lines.append(f"- {p['ora']}: {format_temp(p.get('temp'), prefs)}{rain} {icon}{risk} {conf}%")
    lines.append("Pioggia:")
    if rain_periods:
        for r in rain_periods[:3]:
            s = r["start"].strftime("%H:%M")
            e = r["end"].strftime("%H:%M")
            lines.append(f"- {s}-{e}: {r['max_prob']:.0f}%")
    else:
        lines.append("- Nessuna pioggia significativa")
    lines.append(f"Fonti: {sources_label}")
    if fallback_note:
        lines.append(f"Nota: {fallback_note}")
    if chosen:
        confs = []
        for p in chosen:
            hour = p["local_time"].hour
            ow_p = pick_point_for_hour(points_ow, hour, 2) if points_ow else None
            wa_p = pick_point_for_hour(points_wa, hour, 1) if points_wa else None
            mb_p = pick_point_for_hour(points_mb, hour, 1) if points_mb else None
            if storage:
                cond = condition_group_from_icon(str(p.get("icon", "01d")), float(p.get("pop", 0)))
                confs.append(hourly_confidence_percent_context(
                    storage, ow_p, wa_p, mb_p, hour_band(hour), cond,
                    kind_group=kind_group, season=season, zone=zone
                ))
            else:
                confs.append(hourly_confidence_percent(ow_p, wa_p, mb_p, acc))
        if confs:
            lines.append(f"Affidabilita: {int(sum(confs) / len(confs))}%")
    return "\n".join(lines)


def _trend_label(delta: float) -> str:
    if delta > 1.0:
        return "in aumento"
    if delta < -1.0:
        return "in diminuzione"
    return "stabile"


def _avg(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def format_today_summary(
    points: List[Dict[str, Any]],
    city: str,
    country: str,
    target_date: datetime.date,
    now_local: Optional[datetime] = None,
    reliability_percent: Optional[int] = None,
    rain_threshold: Optional[float] = None,
    prefs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not points:
        return None
    min_t, max_t = extract_min_max(points)
    lines = []
    lines.append("PREVISIONI OGGI")
    lines.append(f"{md_escape(city)}, {md_escape(country)}")
    lines.append(f"{format_date_italian(datetime.combine(target_date, datetime.min.time()))}")
    lines.append("")
    if min_t is not None and max_t is not None:
        lines.append(f"\U0001F321 Min/Max: {format_temp(min_t, prefs)} / {format_temp(max_t, prefs)}")

    # Temperature trend (morning vs afternoon)
    early = [p["temp"] for p in points if 6 <= p["local_time"].hour <= 11]
    late = [p["temp"] for p in points if 12 <= p["local_time"].hour <= 18]
    early_avg = _avg(early) if early else _avg([p["temp"] for p in points[:6]])
    late_avg = _avg(late) if late else _avg([p["temp"] for p in points[-6:]])
    if early_avg is not None and late_avg is not None:
        lines.append(f"Andamento temperatura: {_trend_label(late_avg - early_avg)}")

    # Cloud trend (first half vs second half)
    def cloud_count(hour_min: int, hour_max: int) -> int:
        c = 0
        for p in points:
            h = p["local_time"].hour
            if hour_min <= h <= hour_max:
                grp = condition_group_from_icon(str(p["icon"]), float(p.get("pop", 0)))
                if grp in {"nuvoloso", "nebbia"}:
                    c += 1
        return c

    c1 = cloud_count(6, 12)
    c2 = cloud_count(13, 19)
    if c1 or c2:
        if c2 > c1 + 1:
            cloud_trend = "in aumento"
        elif c1 > c2 + 1:
            cloud_trend = "in diminuzione"
        else:
            cloud_trend = "stabile"
        lines.append(f"Nuvolosita: {cloud_trend}")

    # Rain summary (solo futuro)
    if now_local is None:
        now_local = datetime.now().astimezone()
    future_points = [p for p in points if p["local_time"] >= now_local]
    th = float(rain_threshold) if rain_threshold is not None else RAIN_POP_THRESHOLD
    periods = compute_rain_periods_threshold(future_points, th)
    if periods:
        p0 = periods[0]
        s = p0["start"].strftime("%H:%M")
        e = p0["end"].strftime("%H:%M")
        lines.append(f"\U0001F327\ufe0f Pioggia possibile: {s}-{e} (max {p0['max_prob']:.0f}%)")

    if reliability_percent is not None:
        lines.append(f"Affidabilita: {int(reliability_percent)}%")

    return "\n".join(lines)


def hourly_confidence_percent(
    ow_p: Optional[Dict[str, Any]],
    wa_p: Optional[Dict[str, Any]],
    mb_p: Optional[Dict[str, Any]],
    acc: Dict[str, Dict[str, Any]],
) -> int:
    acc_vals = []
    if acc.get("OpenWeather"):
        acc_vals.append(acc["OpenWeather"]["accuracy"])
    if acc.get("WeatherAPI"):
        acc_vals.append(acc["WeatherAPI"]["accuracy"])
    if acc.get("Meteoblue"):
        acc_vals.append(acc["Meteoblue"]["accuracy"])
    acc_avg = sum(acc_vals) / len(acc_vals) if acc_vals else 70.0
    temps = []
    for p in (ow_p, wa_p, mb_p):
        if p and p.get("temp") is not None:
            temps.append(float(p["temp"]))
    if len(temps) >= 2:
        diff = max(temps) - min(temps)
        base = acc_avg - diff * 6
    else:
        base = acc_avg - 8
    return int(clamp(base, 35, 95))


def hourly_confidence_percent_context(
    storage: Storage,
    ow_p: Optional[Dict[str, Any]],
    wa_p: Optional[Dict[str, Any]],
    mb_p: Optional[Dict[str, Any]],
    band: str,
    cond: str,
    kind_group: Optional[str] = None,
    season: Optional[str] = None,
    zone: Optional[str] = None,
) -> int:
    acc_global = storage.get_provider_accuracy()
    acc_vals = []
    for name, p in [("OpenWeather", ow_p), ("WeatherAPI", wa_p), ("Meteoblue", mb_p)]:
        if not p:
            continue
        err = storage.get_provider_accuracy_context(name, band, cond, kind_group=kind_group, season=season, zone=zone)
        if err is None:
            acc = acc_global.get(name, {}).get("accuracy")
        else:
            acc = clamp(100 - (float(err) * 15), 0, 100)
        if acc is not None:
            acc_vals.append(float(acc))
    acc_avg = sum(acc_vals) / len(acc_vals) if acc_vals else 70.0
    temps = []
    for p in (ow_p, wa_p, mb_p):
        if p and p.get("temp") is not None:
            temps.append(float(p["temp"]))
    if len(temps) >= 2:
        diff = max(temps) - min(temps)
        base = acc_avg - diff * 6
    else:
        base = acc_avg - 8
    return int(clamp(base, 35, 95))


def overall_confidence_percent(
    points: List[Dict[str, Any]],
    points_ow: Optional[List[Dict[str, Any]]],
    points_wa: Optional[List[Dict[str, Any]]],
    points_mb: Optional[List[Dict[str, Any]]],
    acc: Dict[str, Dict[str, Any]],
    target_date: date,
    from_hour: Optional[int] = None,
    storage: Optional[Storage] = None,
    kind_group: Optional[str] = None,
    season: Optional[str] = None,
    zone: Optional[str] = None,
) -> Optional[int]:
    if not points:
        return None
    chosen = [
        p for p in points
        if p["local_time"].date() == target_date and (from_hour is None or p["local_time"].hour >= from_hour)
    ]
    if not chosen:
        return None
    confs = []
    for p in chosen:
        hour = p["local_time"].hour
        ow_p = pick_point_for_hour(points_ow, hour, 2) if points_ow else None
        wa_p = pick_point_for_hour(points_wa, hour, 1) if points_wa else None
        mb_p = pick_point_for_hour(points_mb, hour, 1) if points_mb else None
        if storage:
            cond = condition_group_from_icon(str(p.get("icon", "01d")), float(p.get("pop", 0)))
            confs.append(hourly_confidence_percent_context(
                storage, ow_p, wa_p, mb_p, hour_band(hour), cond,
                kind_group=kind_group, season=season, zone=zone
            ))
        else:
            confs.append(hourly_confidence_percent(ow_p, wa_p, mb_p, acc))
    return int(sum(confs) / len(confs))


def save_predictions_from_points(
    storage: Storage,
    user_id: int,
    city: str,
    kind: str,
    fused_points: List[Dict[str, Any]],
    points_ow: Optional[List[Dict[str, Any]]],
    points_wa: Optional[List[Dict[str, Any]]],
    points_mb: Optional[List[Dict[str, Any]]],
    target_date: date,
    tz_offset: int,
    lat: float,
    lon: float,
    rain_threshold: float = RAIN_POP_THRESHOLD,
):
    season = season_from_date(target_date)
    zone = zone_bucket(lat, lon)
    for p in fused_points:
        if p["local_time"].date() != target_date:
            continue
        hour = p["local_time"].hour
        ow_p = pick_point_for_hour(points_ow, hour, 2) if points_ow else None
        wa_p = pick_point_for_hour(points_wa, hour, 1) if points_wa else None
        mb_p = pick_point_for_hour(points_mb, hour, 1) if points_mb else None
        pred_ow = float(ow_p["temp"]) if ow_p and ow_p.get("temp") is not None else None
        pred_wa = float(wa_p["temp"]) if wa_p and wa_p.get("temp") is not None else None
        pred_mb = float(mb_p["temp"]) if mb_p and mb_p.get("temp") is not None else None
        pred_pop = float(p.get("pop")) if p.get("pop") is not None else None
        cond_group = condition_group_from_icon(str(p.get("icon", "01d")), float(pred_pop or 0))
        storage.save_prediction(
            user_id, city, kind,
            target_dt_utc=p["dt_utc"],
            predicted_fused=float(p["temp"]),
            predicted_ow=pred_ow,
            predicted_wa=pred_wa,
            predicted_mb=pred_mb,
            predicted_wind_kph=p.get("wind_kph"),
            predicted_humidity=p.get("humidity"),
            predicted_pressure=p.get("pressure"),
            predicted_rain=(pred_pop >= rain_threshold) if pred_pop is not None else None,
            predicted_pop=pred_pop,
            target_hour_local=hour,
            condition_group=cond_group,
            season=season,
            zone=zone,
        )


def build_reply_keyboard() -> ReplyKeyboardMarkup:
    keyboard = [
        [KeyboardButton("Meteo"), KeyboardButton("Oggi"), KeyboardButton("Domani")],
        [KeyboardButton("Dettagli"), KeyboardButton("Aggiorna"), KeyboardButton("Cambia citta")],
        [KeyboardButton("Comandi")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def build_inline_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton("Dettagli", callback_data="DETTAGLI"),
            InlineKeyboardButton("Aggiorna", callback_data="AGGIORNA"),
        ],
        [InlineKeyboardButton("Cambia citta", callback_data="CAMBIA")],
    ]
    return InlineKeyboardMarkup(buttons)
