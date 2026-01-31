from __future__ import annotations

import asyncio
import json
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from telegram import Update
from telegram.ext import ContextTypes

from app_state import AppState
from config import (
    ADMIN_SECRET, ADMIN_USER_IDS, ALERT_CHECK_MIN, ALERT_COOLDOWN_MIN, ALERT_POP_THRESHOLD,
    ALERT_TEMP_COLD, ALERT_TEMP_HOT, ALERT_WIND_KPH, BOT_RELEASES, BOT_VERSION,
    CACHE_TTL_CURRENT_MIN, CACHE_TTL_FORECAST_MIN, DAILY_CHECK_HOUR, DAILY_CHECK_MIN,
    DB_FILE,
    DEFAULT_ALERT_END_HOUR, DEFAULT_ALERT_START_HOUR, DEFAULT_TEMP_UNIT, DEFAULT_WIND_UNIT,
    FEELS_LIKE_DIFF_C, HEALTH_PATH, HEALTH_PATH_EFFECTIVE, METEOBLUE_API_KEY, MORNING_METEO_HOUR,
    MSG_CITY_NOT_FOUND, MSG_NEED_CITY, MSG_SERVICE_UNAVAILABLE, OFFLINE_TEST_DAYS, OPENWEATHER_API_KEY,
    PORT, PROVIDERS, RAIN_POP_THRESHOLD, TOMORROW_RAIN_HOUR, TOMORROW_RAIN_WINDOW_MIN,
    UMBRELLA_POP_THRESHOLD, WEATHERAPI_KEY, WEBHOOK_PATH, WEBHOOK_URL, WIND_STRONG_KPH,
)
from forecast import (
    apply_pop_calibration, build_fused_points, build_pop_calibrator, compute_rain_periods_threshold,
    extract_day_points, extract_day_points_meteoblue, extract_day_points_weatherapi,
    extract_day_points_weatherapi_history, extract_min_max, get_dynamic_thresholds,
    interpolate_hourly_points,
    key_current, key_forecast, pick_point_for_hour, summarize_rain_forecast
)
from formatting import (
    build_advice, build_reply_keyboard, confidence_from_accuracy, format_details_line,
    format_forecast_message, format_meteo_message, format_sources_label, format_today_summary,
    hourly_confidence_percent, hourly_confidence_percent_context, overall_confidence_percent,
    save_predictions_from_points
)
from geo import get_coords
from providers import ProviderResult
from storage import Storage
from utils import (
    ADMIN_IDS_SET, BOT_STARTED_AT_UTC, adaptive_weights, build_provider_note, cache_age_min_from_payload, clamp,
    condition_group_from_description, condition_group_from_icon, format_alert_window,
    format_date_italian, format_temp, format_temp_delta, format_time_italian, format_uptime,
    format_wind, forecast_ttl_minutes, from_iso, get_rain_risk_icon, get_tz_offset_sec, get_weather_icon,
    hour_band, is_admin, is_rain_description, iso, local_now, log_event, md_escape, normalize_temp_unit,
    normalize_wind_unit, now_utc, parse_alert_window, rain_risk_label, season_from_date, tzinfo_from_offset,
    within_alert_window, zone_bucket, logger
)
async def _call_with_args(fn, update: Update, context: ContextTypes.DEFAULT_TYPE, args: List[str]):
    old_args = getattr(context, "args", None)
    context.args = args
    try:
        await fn(update, context)
    finally:
        context.args = old_args

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    user = update.effective_user
    name_md = md_escape(user.first_name)

    msg = (
        f"Ciao *{name_md}*!\n\n"
        "Sono un bot meteo che combina *3 fonti* per darti previsioni piu affidabili.\n"
        "Imparo dagli errori passati e correggo i dati quando serve.\n\n"
        "Per iniziare:\n"
        "- usa /meteo e scrivi la tua citta\n"
        "- oppure imposta la citta con /setcitta\n\n"
        "Comandi principali:\n"
        "- /meteo\n"
        "- /oggi\n"
        "- /domani\n"
        "- /prev\n"
        "- /trend\n"
        "- /setcitta\n"
        "- /citta\n"
        "- /aggiorna\n\n"
        "Per vedere tutti i comandi disponibili: /comandi"
    )

    await update.effective_message.reply_text(msg, parse_mode="Markdown", reply_markup=build_reply_keyboard())


async def aiuto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "GUIDA BOT METEO\n\n"
        "Cosa fa:\n"
        "- combina piu fonti meteo per una previsione piu affidabile\n"
        "- salva le previsioni e le confronta con i dati reali\n"
        "- migliora i pesi dei provider nel tempo\n\n"
        "Comandi principali:\n"
        "- /meteo [citta] - meteo attuale\n"
        "- /oggi - riepilogo oggi\n"
        "- /domani - previsione domani\n"
        "- /prev - ora per ora fino a mezzanotte\n"
        "- /trend - riassunto rapido giornata\n"
        "- /aggiorna - forza aggiornamento\n\n"
        "Citta e preferenze:\n"
        "- /setcitta Roma - imposta la citta\n"
        "- /citta - lista o cambia citta\n"
        "- /pref - mostra preferenze\n"
        "- /setpref pop=60 vento=40 diff=3 unit=c windunit=kmh alert60=on alertdomani=on notifiche=07-22\n\n"
        "Diagnostica:\n"
        "- /info - stato sistema\n"
        "- /stat - errori recenti\n"
        "- /controlla - verifica previsioni passate\n"
        "- /testoffline [giorni] - test su storico\n\n"
        "Tecnologie:\n"
        "- Python + python-telegram-bot\n"
        "- API meteo: OpenWeather, WeatherAPI, Meteoblue\n"
        "- Cache: RAM + SQLite\n"
        "- Webhook (Railway) + job schedulati per alert\n\n"
        "Affidabilita:\n"
        "- la percentuale indica quanto le fonti concordano e quanto sono accurate storicamente\n"
        "- cresce nel tempo con piu dati reali\n"
    )
    await update.effective_message.reply_text(msg, parse_mode="Markdown")


async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = []
    lines.append("NEWS DEL BOT")
    lines.append("")
    rel = BOT_RELEASES[0] if BOT_RELEASES else {"version": BOT_VERSION, "changes": []}
    lines.append(f"Versione {md_escape(rel['version'])}")
    for ch in rel.get("changes", []):
        lines.append(f"- {md_escape(ch)}")
    await update.effective_message.reply_text("\n".join(lines), parse_mode="Markdown")


async def comandi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Comandi disponibili\n"
        "- /meteo - meteo attuale\n"
        "- /aggiorna - forza aggiornamento meteo\n"
        "- /oggi - riepilogo giornata\n"
        "- /prev - ora per ora fino a mezzanotte\n"
        "- /domani - previsioni domani\n"
        "- /trend - riassunto rapido giornata\n"
        "- /setcitta - salva e imposta citta\n"
        "- /citta - lista o cambia citta\n"
        "- /delcitta - rimuove citta\n"
        "- /controlla - verifica previsioni vicine all'orario\n"
        "- /pref - mostra preferenze\n"
        "- /setpref pop=60 vento=40 diff=3 unit=c windunit=kmh alert60=on alertdomani=on notifiche=07-22\n"
        "- /stat - errori recenti\n"
        "- /testoffline [giorni] - test su storico\n"
        "- /info - stato sistema\n"
        "- /versione - info versione bot\n"
        "- /admin <token> - stato admin (se autorizzato)\n"
        "- /aiuto - guida completa\n"
        "- /news - ultime novita\n"
    )
    await update.effective_message.reply_text(msg)


async def aggiorna(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _call_with_args(meteo, update, context, ["force"])


async def setcitta(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    if not context.args:
        await update.effective_message.reply_text("Errore: usa `/setcitta Roma`", parse_mode="Markdown")
        return

    city = " ".join(context.args)
    coords = await get_coords(state, city)
    if not coords:
        await update.effective_message.reply_text(MSG_CITY_NOT_FOUND)
        return

    state.storage.set_user_city(update.effective_user.id, city)
    name_md = md_escape(coords.get("name", ""))
    country_md = md_escape(coords.get("country", ""))
    await update.effective_message.reply_text(f"Salvata: {name_md}, {country_md}", parse_mode="Markdown")


async def mycitta(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    cities = state.storage.get_user_cities(update.effective_user.id)
    if not cities:
        await update.effective_message.reply_text("Nessuna citta salvata. Usa `/setcitta Roma`", parse_mode="Markdown")
        return
    default_city = state.storage.get_user_city(update.effective_user.id)
    lines = []
    if default_city:
        coords = await get_coords(state, default_city)
        if coords:
            lines.append(f"Citta predefinita: {md_escape(coords.get('name',''))}, {md_escape(coords.get('country',''))}")
        else:
            lines.append(f"Citta predefinita: {md_escape(default_city)}")
    lines.append("")
    lines.append("Citta salvate:")
    for c in cities:
        mark = " *" if c == default_city else ""
        lines.append(f"- {md_escape(c)}{mark}")
    lines.append("")
    lines.append("Usa /citta <nome> per cambiare rapidamente.")
    await update.effective_message.reply_text("\n".join(lines), parse_mode="Markdown")


async def citta(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    if not context.args:
        await mycitta(update, context)
        return
    city = " ".join(context.args)
    coords = await get_coords(state, city)
    if not coords:
        await update.effective_message.reply_text(MSG_CITY_NOT_FOUND)
        return
    state.storage.set_user_city(update.effective_user.id, city)
    name_md = md_escape(coords.get("name", ""))
    country_md = md_escape(coords.get("country", ""))
    await update.effective_message.reply_text(f"Citta impostata: {name_md}, {country_md}", parse_mode="Markdown")


async def delcitta(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    if not context.args:
        await update.effective_message.reply_text("Usa `/delcitta Roma`", parse_mode="Markdown")
        return
    city = " ".join(context.args)
    ok = state.storage.remove_user_city(update.effective_user.id, city)
    if ok:
        await update.effective_message.reply_text(f"Citta rimossa: {md_escape(city)}", parse_mode="Markdown")
    else:
        await update.effective_message.reply_text(f"Citta non trovata: {md_escape(city)}", parse_mode="Markdown")


async def meteo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]

    force = False
    details = False
    args = list(context.args) if context.args else []
    while args and args[-1].lower() in {"force", "dettagli", "details"}:
        flag = args.pop().lower()
        if flag == "force":
            force = True
        else:
            details = True

    if args:
        city = " ".join(args)
    else:
        city = state.storage.get_user_city(update.effective_user.id)
        if not city:
            await update.effective_message.reply_text(MSG_NEED_CITY)
            return

    coords = await get_coords(state, city)
    if not coords:
        await update.effective_message.reply_text(MSG_CITY_NOT_FOUND)
        return

    prefs = state.storage.get_user_prefs(update.effective_user.id)
    lat, lon = coords["lat"], coords["lon"]
    ck_cur = key_current(lat, lon)
    ck_fc = key_forecast(lat, lon, 0)
    ths = get_dynamic_thresholds(state.storage)

    forecast_ow = None
    forecast_wa = None
    forecast_mb = None
    if not force:
        cached_fc = state.ram_cache.get(ck_fc) or state.storage.cache_get(ck_fc)
        if cached_fc:
            forecast_ow = cached_fc.get("forecast_ow") or cached_fc.get("forecast")
            forecast_wa = cached_fc.get("forecast_wa")
            forecast_mb = cached_fc.get("forecast_mb")

    tz_offset = get_tz_offset_sec(forecast_ow, forecast_wa, forecast_mb, coords=coords)
    local_dt = local_now(tz_offset)
    band = hour_band(local_dt.hour)
    bias_overall = state.storage.get_provider_bias()
    season = season_from_date(local_dt.date())
    zone = zone_bucket(lat, lon)

    if force or (forecast_ow is None and forecast_wa is None and forecast_mb is None):
        forecast_ow, forecast_wa, forecast_mb = await asyncio.gather(
            state.ow.forecast(lat, lon),
            state.wa.forecast(lat, lon, days=2),
            state.mb.forecast(lat, lon),
        )
        if forecast_ow or forecast_wa or forecast_mb:
            payload_fc = {"forecast_ow": forecast_ow, "forecast_wa": forecast_wa, "forecast_mb": forecast_mb, "saved_at": iso(now_utc())}
            tz_offset_fc = get_tz_offset_sec(forecast_ow, forecast_wa, forecast_mb, coords=coords)
            ttl_fc = forecast_ttl_minutes(local_now(tz_offset_fc))
            state.storage.cache_set(ck_fc, payload_fc, ttl=timedelta(minutes=ttl_fc))
            state.ram_cache.set(ck_fc, payload_fc, ttl_seconds=30)

    if not force:
        cached = state.ram_cache.get(ck_cur) or state.storage.cache_get(ck_cur)
        if cached:
            saved_at = from_iso(cached["saved_at"])
            age_min = int((now_utc() - saved_at).total_seconds() // 60)
            fused = cached["fused"]
            min_t = cached.get("min_t")
            max_t = cached.get("max_t")
            rain_msg = cached.get("rain_msg")
            rain_max_prob = cached.get("rain_max_prob", 0)
            rain_current = cached.get("rain_current")
            correction_used = cached.get("correction_used")
            if rain_current is None:
                rain_current = is_rain_description(str(fused.get("description", "")))

            if forecast_ow or forecast_wa or forecast_mb:
                target_date = datetime.now().date()
                tz_offset = get_tz_offset_sec(forecast_ow, forecast_wa, forecast_mb, coords=coords)
                local_dt = local_now(tz_offset)
                target_date = local_dt.date()
                season = season_from_date(target_date)
                zone = zone_bucket(lat, lon)
                points_ow = interpolate_hourly_points(extract_day_points(forecast_ow, target_date, tz_offset)) if forecast_ow else []
                points_wa = extract_day_points_weatherapi(forecast_wa, target_date, tz_offset) if forecast_wa else []
                points_mb = extract_day_points_meteoblue(forecast_mb, target_date, tz_offset) if forecast_mb else []
                weights = state.weather.get_dynamic_weights()
                bias_func = lambda prov, hr, cond: state.storage.get_provider_bias_context(
                    prov, hour_band(hr), cond, kind_group="now", season=season, zone=zone
                )
                fused_points = build_fused_points(points_ow, points_wa, points_mb, weights, bias_overall, target_date, tz_offset, bias_func=bias_func)
                pop_calibrator = build_pop_calibrator(state.storage)
                fused_points = apply_pop_calibration(fused_points, pop_calibrator)
                if min_t is None or max_t is None:
                    min_t, max_t = extract_min_max(fused_points)
                future_points = [p for p in fused_points if p["local_time"] >= local_dt]
                periods = compute_rain_periods_threshold(future_points, ths.get("pop", RAIN_POP_THRESHOLD))
                max_pop_future = max((float(p.get("pop", 0)) for p in future_points), default=0)
                rain_msg = summarize_rain_forecast(periods, max_pop_future, local_dt, rain_current)
                rain_max_prob = max_pop_future

            if not rain_msg:
                rain_msg = summarize_rain_forecast([], 0, local_dt, rain_current)

            if correction_used is None:
                cond = condition_group_from_description(str(fused.get("description", "")))
                bias_ctx = {
                    "OpenWeather": state.storage.get_provider_bias_context("OpenWeather", band, cond, kind_group="now", season=season, zone=zone),
                    "WeatherAPI": state.storage.get_provider_bias_context("WeatherAPI", band, cond, kind_group="now", season=season, zone=zone),
                    "Meteoblue": state.storage.get_provider_bias_context("Meteoblue", band, cond, kind_group="now", season=season, zone=zone),
                }
                available = fused.get("sources", [])
                correction_used = any(abs(bias_ctx.get(p, 0.0)) >= 0.3 for p in available)

            acc = state.storage.get_provider_accuracy()
            details_line = format_details_line(fused, acc, prefs) if details else None
            enabled = [p for p, k in [("OpenWeather", OPENWEATHER_API_KEY), ("WeatherAPI", WEATHERAPI_KEY), ("Meteoblue", METEOBLUE_API_KEY)] if k]
            available = fused.get("sources", [])
            missing = [p for p in enabled if p not in available]
            sources_label = format_sources_label(available, correction_used, missing)
            prov_details = fused.get("provider_details") or {}
            ow_t = prov_details.get("OpenWeather", {}).get("temp")
            wa_t = prov_details.get("WeatherAPI", {}).get("temp")
            mb_t = prov_details.get("Meteoblue", {}).get("temp")
            ow_p = {"temp": ow_t} if ow_t is not None else None
            wa_p = {"temp": wa_t} if wa_t is not None else None
            mb_p = {"temp": mb_t} if mb_t is not None else None
            season = season_from_date(local_dt.date())
            zone = zone_bucket(lat, lon)
            cond = condition_group_from_description(str(fused.get("description", "")))
            reliability = hourly_confidence_percent_context(
                state.storage, ow_p, wa_p, mb_p, band, cond,
                kind_group="now", season=season, zone=zone
            )
            fallback_note = build_provider_note(enabled, available)
            msg = format_meteo_message(
                fused, coords["name"], coords["country"], min_t, max_t, rain_msg, sources_label, rain_max_prob, age_min,
                local_dt, reliability, fallback_note, details_line, prefs=prefs
            )
            await update.effective_message.reply_text(msg, parse_mode="Markdown")
            return

    ow_cur, wa_cur, mb_cur = await asyncio.gather(
        state.ow.current(lat, lon),
        state.wa.current(lat, lon),
        state.mb.current(lat, lon),
    )

    tz_offset = get_tz_offset_sec(forecast_ow, forecast_wa, forecast_mb, ow_cur, wa_cur, mb_cur, coords=coords)
    local_dt = local_now(tz_offset)
    target_date = local_dt.date()
    band = hour_band(local_dt.hour)

    weights = state.weather.get_dynamic_weights()
    w_map = {
        "OpenWeather": float(weights.get("OpenWeather", 0.0)),
        "WeatherAPI": float(weights.get("WeatherAPI", 0.0)),
        "Meteoblue": float(weights.get("Meteoblue", 0.0)),
    }
    available = {}
    if ow_cur.success:
        available["OpenWeather"] = ow_cur.description
    if wa_cur.success:
        available["WeatherAPI"] = wa_cur.description
    if mb_cur.success:
        available["Meteoblue"] = mb_cur.description
    if available:
        top = max(available.keys(), key=lambda k: w_map.get(k, 0.0))
        cond_desc = available.get(top, "")
    else:
        cond_desc = ""
    cond_group = condition_group_from_description(cond_desc)
    bias_ctx = {
        "OpenWeather": state.storage.get_provider_bias_context("OpenWeather", band, cond_group, kind_group="now", season=season, zone=zone),
        "WeatherAPI": state.storage.get_provider_bias_context("WeatherAPI", band, cond_group, kind_group="now", season=season, zone=zone),
        "Meteoblue": state.storage.get_provider_bias_context("Meteoblue", band, cond_group, kind_group="now", season=season, zone=zone),
    }
    correction_used = any(abs(bias_ctx.get(p, 0.0)) >= 0.3 for p in available.keys())

    temps = [t for t in [ow_cur.temp if ow_cur.success else None, wa_cur.temp if wa_cur.success else None, mb_cur.temp if mb_cur.success else None] if t is not None]
    diff = (max(temps) - min(temps)) if len(temps) >= 2 else 0.0
    err_ow = state.storage.get_provider_accuracy_context("OpenWeather", band, cond_group, kind_group="now", season=season, zone=zone)
    err_wa = state.storage.get_provider_accuracy_context("WeatherAPI", band, cond_group, kind_group="now", season=season, zone=zone)
    weights_ctx = adaptive_weights(weights, diff, err_ow, err_wa)
    weights_ctx["Meteoblue"] = float(weights.get("Meteoblue", 0.0))

    fused = state.weather.fuse(ow_cur, wa_cur, mb_cur, bias=bias_ctx, weights_override=weights_ctx)
    if not fused:
        stale_payload, created_at, _, expired = state.storage.cache_get_with_meta(ck_cur, allow_expired=True)
        if stale_payload and stale_payload.get("fused"):
            age_min = cache_age_min_from_payload(stale_payload, created_at)
            fused = stale_payload.get("fused")
            min_t = stale_payload.get("min_t")
            max_t = stale_payload.get("max_t")
            rain_msg = stale_payload.get("rain_msg") or ""
            rain_current = stale_payload.get("rain_current")
            correction_used = stale_payload.get("correction_used")
            if rain_current is None:
                rain_current = is_rain_description(str(fused.get("description", "")))
            acc = state.storage.get_provider_accuracy()
            details_line = format_details_line(fused, acc, prefs) if details else None
            enabled = [p for p, k in [("OpenWeather", OPENWEATHER_API_KEY), ("WeatherAPI", WEATHERAPI_KEY), ("Meteoblue", METEOBLUE_API_KEY)] if k]
            available = fused.get("sources", [])
            missing = [p for p in enabled if p not in available]
            sources_label = format_sources_label(available, correction_used, missing)
            fallback_note = build_provider_note(enabled, available, cache_stale=expired, cache_age_min=age_min)
            msg = format_meteo_message(
                fused,
                coords["name"],
                coords["country"],
                min_t,
                max_t,
                rain_msg,
                sources_label,
                None,
                age_min,
                local_dt,
                None,
                fallback_note,
                details_line,
                prefs=prefs,
                cache_stale=expired,
            )
            await update.effective_message.reply_text(msg, parse_mode="Markdown")
            return
        await update.effective_message.reply_text(MSG_SERVICE_UNAVAILABLE)
        return

    points_ow = interpolate_hourly_points(extract_day_points(forecast_ow, target_date, tz_offset)) if forecast_ow else []
    points_wa = extract_day_points_weatherapi(forecast_wa, target_date, tz_offset) if forecast_wa else []
    points_mb = extract_day_points_meteoblue(forecast_mb, target_date, tz_offset) if forecast_mb else []
    weights = state.weather.get_dynamic_weights()
    bias_func = lambda prov, hr, cond: state.storage.get_provider_bias_context(
        prov, hour_band(hr), cond, kind_group="now", season=season, zone=zone
    )
    fused_points = build_fused_points(points_ow, points_wa, points_mb, weights, bias_overall, target_date, tz_offset, bias_func=bias_func)
    pop_calibrator = build_pop_calibrator(state.storage)
    fused_points = apply_pop_calibration(fused_points, pop_calibrator)
    min_t, max_t = extract_min_max(fused_points)

    rain_current = False
    if ow_cur.success and is_rain_description(ow_cur.description):
        rain_current = True
    if wa_cur.success and is_rain_description(wa_cur.description):
        rain_current = True
    if mb_cur.success and is_rain_description(mb_cur.description):
        rain_current = True
    future_points = [p for p in fused_points if p["local_time"] >= local_dt]
    periods = compute_rain_periods_threshold(future_points, ths.get("pop", RAIN_POP_THRESHOLD))
    max_pop_all = max((float(p.get("pop", 0)) for p in future_points), default=0)
    rain_info = {
        "currently_raining": rain_current,
        "periods": periods,
        "max_prob": max_pop_all,
    }
    rain_msg = summarize_rain_forecast(periods, max_pop_all, local_dt, rain_current)

    acc = state.storage.get_provider_accuracy()
    details_line = format_details_line(fused, acc, prefs) if details else None
    enabled = [p for p, k in [("OpenWeather", OPENWEATHER_API_KEY), ("WeatherAPI", WEATHERAPI_KEY), ("Meteoblue", METEOBLUE_API_KEY)] if k]
    available = fused.get("sources", [])
    missing = [p for p in enabled if p not in available]
    sources_label = format_sources_label(available, correction_used, missing)
    ow_p = {"temp": ow_cur.temp} if ow_cur.success else None
    wa_p = {"temp": wa_cur.temp} if wa_cur.success else None
    mb_p = {"temp": mb_cur.temp} if mb_cur.success else None
    season = season_from_date(local_dt.date())
    zone = zone_bucket(lat, lon)
    reliability = hourly_confidence_percent_context(
        state.storage, ow_p, wa_p, mb_p, band, cond_group,
        kind_group="now", season=season, zone=zone
    )
    fallback_note = build_provider_note(enabled, available)

    payload = {
        "fused": fused,
        "rain_msg": rain_msg,
        "rain_max_prob": rain_info.get("max_prob", 0),
        "rain_current": rain_info.get("currently_raining", False),
        "min_t": min_t,
        "max_t": max_t,
        "correction_used": correction_used,
        "saved_at": iso(now_utc()),
    }
    ttl_min = CACHE_TTL_CURRENT_MIN
    if rain_info.get("currently_raining") or (fused.get("wind_kph") and fused.get("wind_kph") > WIND_STRONG_KPH):
        ttl_min = max(5, int(CACHE_TTL_CURRENT_MIN / 2))
    state.storage.cache_set(ck_cur, payload, ttl=timedelta(minutes=ttl_min))
    state.ram_cache.set(ck_cur, payload, ttl_seconds=30)

    msg = format_meteo_message(
        fused, coords["name"], coords["country"], min_t, max_t, rain_msg, sources_label, rain_info.get("max_prob", 0),
        None, local_dt, reliability, fallback_note, details_line, prefs=prefs
    )
    if force:
        msg += "\nAggiornamento forzato"
    await update.effective_message.reply_text(msg, parse_mode="Markdown")

    target = now_utc()
    pred_ow = ow_cur.temp if ow_cur.success else None
    pred_wa = wa_cur.temp if wa_cur.success else None
    pred_mb = mb_cur.temp if mb_cur.success else None
    season = season_from_date(local_dt.date())
    zone = zone_bucket(lat, lon)
    state.storage.save_prediction(
        update.effective_user.id, city, "meteo",
        target_dt_utc=target,
        predicted_fused=float(fused["temp"]),
        predicted_ow=pred_ow,
        predicted_wa=pred_wa,
        predicted_mb=pred_mb,
        predicted_wind_kph=fused.get("wind_kph"),
        predicted_humidity=fused.get("humidity"),
        predicted_pressure=fused.get("pressure"),
        predicted_rain=rain_info.get("currently_raining", False),
        predicted_pop=rain_info.get("max_prob", 0),
        target_hour_local=local_dt.hour,
        condition_group=condition_group_from_description(str(fused.get("description", ""))),
        season=season,
        zone=zone,
    )
async def _load_forecast_points(
    state: AppState,
    coords: Dict[str, Any],
    days: int,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    datetime.date,
    int,
    str,
    bool,
    bool,
    Optional[int],
]:
    lat, lon = coords["lat"], coords["lon"]
    ck = key_forecast(lat, lon, 0)

    cached_used = False
    cache_stale = False
    cache_age_min = None
    stale_payload = None
    stale_created_at = None

    cached = state.ram_cache.get(ck)
    if cached:
        forecast_ow = cached.get("forecast_ow")
        forecast_wa = cached.get("forecast_wa")
        forecast_mb = cached.get("forecast_mb")
        cached_used = True
        cache_age_min = cache_age_min_from_payload(cached, None)
    else:
        cached_meta, created_at, _, expired = state.storage.cache_get_with_meta(ck, allow_expired=True)
        if cached_meta and not expired:
            forecast_ow = cached_meta.get("forecast_ow")
            forecast_wa = cached_meta.get("forecast_wa")
            forecast_mb = cached_meta.get("forecast_mb")
            cached_used = True
            state.ram_cache.set(ck, cached_meta, ttl_seconds=30)
            cache_age_min = cache_age_min_from_payload(cached_meta, created_at)
        elif cached_meta and expired:
            stale_payload = cached_meta
            stale_created_at = created_at

    if not cached_used:
        forecast_ow, forecast_wa, forecast_mb = await asyncio.gather(
            state.ow.forecast(lat, lon),
            state.wa.forecast(lat, lon, days=2),
            state.mb.forecast(lat, lon),
        )
        if forecast_ow or forecast_wa or forecast_mb:
            payload = {"forecast_ow": forecast_ow, "forecast_wa": forecast_wa, "forecast_mb": forecast_mb, "saved_at": iso(now_utc())}
            tz_offset_fc = get_tz_offset_sec(forecast_ow, forecast_wa, forecast_mb, coords=coords)
            ttl_fc = forecast_ttl_minutes(local_now(tz_offset_fc))
            state.storage.cache_set(ck, payload, ttl=timedelta(minutes=ttl_fc))
            state.ram_cache.set(ck, payload, ttl_seconds=30)
        elif stale_payload:
            forecast_ow = stale_payload.get("forecast_ow")
            forecast_wa = stale_payload.get("forecast_wa")
            forecast_mb = stale_payload.get("forecast_mb")
            cached_used = True
            cache_stale = True
            cache_age_min = cache_age_min_from_payload(stale_payload, stale_created_at)

    if not forecast_ow and not forecast_wa and not forecast_mb:
        return None, None, None, [], [], [], [], datetime.now().date(), 0, "", cached_used, cache_stale, cache_age_min

    tz_offset = get_tz_offset_sec(forecast_ow, forecast_wa, forecast_mb, coords=coords)
    target_date = (local_now(tz_offset).date() + timedelta(days=days))
    season = season_from_date(target_date)
    zone = zone_bucket(lat, lon)
    points_ow = interpolate_hourly_points(extract_day_points(forecast_ow, target_date, tz_offset)) if forecast_ow else []
    points_wa = extract_day_points_weatherapi(forecast_wa, target_date, tz_offset) if forecast_wa else []
    points_mb = extract_day_points_meteoblue(forecast_mb, target_date, tz_offset) if forecast_mb else []
    weights = state.weather.get_dynamic_weights()
    bias = state.storage.get_provider_bias()
    bias_func = lambda prov, hr, cond: state.storage.get_provider_bias_context(
        prov, hour_band(hr), cond, kind_group="forecast", season=season, zone=zone
    )

    def weight_func(hr, ow_p, wa_p, mb_p):
        if not ow_p or not wa_p:
            return weights.get("OpenWeather", 0.33), weights.get("WeatherAPI", 0.33), weights.get("Meteoblue", 0.34)
        cond = condition_group_from_icon(str(ow_p["icon"]), float(ow_p["pop"]))
        err_ow = state.storage.get_provider_accuracy_context("OpenWeather", hour_band(hr), cond, kind_group="forecast", season=season, zone=zone)
        err_wa = state.storage.get_provider_accuracy_context("WeatherAPI", hour_band(hr), cond, kind_group="forecast", season=season, zone=zone)
        diff = abs(float(ow_p["temp"]) - float(wa_p["temp"]))
        w = adaptive_weights(weights, diff, err_ow, err_wa)
        w["Meteoblue"] = float(weights.get("Meteoblue", 0.34))
        return w.get("OpenWeather", 0.33), w.get("WeatherAPI", 0.33), w.get("Meteoblue", 0.34)

    fused_points = build_fused_points(points_ow, points_wa, points_mb, weights, bias, target_date, tz_offset, bias_func=bias_func, weight_func=weight_func)
    pop_calibrator = build_pop_calibrator(state.storage)
    fused_points = apply_pop_calibration(fused_points, pop_calibrator)
    sources = []
    if points_ow:
        sources.append("OpenWeather")
    if points_wa:
        sources.append("WeatherAPI")
    if points_mb:
        sources.append("Meteoblue")
    correction_used = any(abs(bias.get(p, 0.0)) >= 0.3 for p in sources) if bias else False
    enabled = [p for p, k in [("OpenWeather", OPENWEATHER_API_KEY), ("WeatherAPI", WEATHERAPI_KEY), ("Meteoblue", METEOBLUE_API_KEY)] if k]
    missing = [p for p in enabled if p not in sources]
    sources_label = format_sources_label(sources, correction_used, missing)

    return forecast_ow, forecast_wa, forecast_mb, points_ow, points_wa, points_mb, fused_points, target_date, tz_offset, sources_label, cached_used, cache_stale, cache_age_min


async def _forecast_common(update: Update, context: ContextTypes.DEFAULT_TYPE, days: int):
    state: AppState = context.application.bot_data["state"]

    if context.args:
        city = " ".join(context.args)
    else:
        city = state.storage.get_user_city(update.effective_user.id)
        if not city:
            await update.effective_message.reply_text(MSG_NEED_CITY)
            return

    coords = await get_coords(state, city)
    if not coords:
        await update.effective_message.reply_text(MSG_CITY_NOT_FOUND)
        return

    forecast_ow, forecast_wa, forecast_mb, points_ow, points_wa, points_mb, fused_points, target_date, tz_offset, sources_label, cached_used, cache_stale, cache_age_min = await _load_forecast_points(
        state, coords, days
    )
    if not forecast_ow and not forecast_wa and not forecast_mb:
        await update.effective_message.reply_text(MSG_SERVICE_UNAVAILABLE)
        return
    acc = state.storage.get_provider_accuracy()
    ths = get_dynamic_thresholds(state.storage)
    season = season_from_date(target_date)
    zone = zone_bucket(coords["lat"], coords["lon"])
    prefs = state.storage.get_user_prefs(update.effective_user.id)
    enabled = [p for p, k in [("OpenWeather", OPENWEATHER_API_KEY), ("WeatherAPI", WEATHERAPI_KEY), ("Meteoblue", METEOBLUE_API_KEY)] if k]
    sources = []
    if points_ow:
        sources.append("OpenWeather")
    if points_wa:
        sources.append("WeatherAPI")
    if points_mb:
        sources.append("Meteoblue")
    fallback_note = build_provider_note(enabled, sources, cache_stale=cache_stale, cache_age_min=cache_age_min)
    msg = format_forecast_message(
        fused_points,
        coords["name"],
        coords["country"],
        days,
        sources_label,
        fallback_note=fallback_note,
        target_date=target_date,
        points_ow=points_ow,
        points_wa=points_wa,
        points_mb=points_mb,
        acc=acc,
        rain_threshold=ths.get("pop"),
        storage=state.storage,
        season=season,
        zone=zone,
        kind_group="forecast",
        prefs=prefs,
    )
    if not msg:
        await update.effective_message.reply_text("Nessuna previsione disponibile")
        return
    if cached_used:
        if cache_age_min is not None:
            cache_line = f"\nDato: Cache {cache_age_min}m"
            if cache_stale:
                cache_line += " (stale)"
        else:
            cache_line = "\nDato: Cache (stale)" if cache_stale else "\nDato: Cache"
        await update.effective_message.reply_text(msg + cache_line, parse_mode="Markdown")
    else:
        await update.effective_message.reply_text(msg, parse_mode="Markdown")

    # ACCURACY: salva ogni slot previsto
    save_predictions_from_points(
        state.storage,
        update.effective_user.id,
        city,
        "oggi" if days == 0 else "domani",
        fused_points,
        points_ow,
        points_wa,
        points_mb,
        target_date,
        tz_offset,
        coords["lat"],
        coords["lon"],
        rain_threshold=ths.get("pop", RAIN_POP_THRESHOLD),
    )


async def oggi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]

    if context.args:
        city = " ".join(context.args)
    else:
        city = state.storage.get_user_city(update.effective_user.id)
        if not city:
            await update.effective_message.reply_text(MSG_NEED_CITY)
            return

    coords = await get_coords(state, city)
    if not coords:
        await update.effective_message.reply_text(MSG_CITY_NOT_FOUND)
        return

    forecast_ow, forecast_wa, forecast_mb, points_ow, points_wa, points_mb, fused_points, target_date, tz_offset, sources_label, cached_used, cache_stale, cache_age_min = await _load_forecast_points(
        state, coords, 0
    )
    if not forecast_ow and not forecast_wa and not forecast_mb:
        await update.effective_message.reply_text(MSG_SERVICE_UNAVAILABLE)
        return

    now_local = local_now(tz_offset)
    acc = state.storage.get_provider_accuracy()
    ths = get_dynamic_thresholds(state.storage)
    season = season_from_date(target_date)
    zone = zone_bucket(coords["lat"], coords["lon"])
    reliability = overall_confidence_percent(
        fused_points, points_ow, points_wa, points_mb, acc, target_date, from_hour=now_local.hour,
        storage=state.storage, kind_group="forecast", season=season, zone=zone
    )
    prefs = state.storage.get_user_prefs(update.effective_user.id)
    msg = format_today_summary(
        fused_points,
        coords["name"],
        coords["country"],
        target_date,
        now_local=now_local,
        reliability_percent=reliability,
        rain_threshold=ths.get("pop"),
        prefs=prefs,
    )
    if not msg:
        await update.effective_message.reply_text("Nessuna previsione disponibile")
        return
    if cached_used:
        if cache_age_min is not None:
            msg += f"\nDato: Cache {cache_age_min}m"
            if cache_stale:
                msg += " (stale)"
        else:
            msg += "\nDato: Cache (stale)" if cache_stale else "\nDato: Cache"
    await update.effective_message.reply_text(msg, parse_mode="Markdown")

    # ACCURACY: salva ogni slot previsto
    save_predictions_from_points(
        state.storage,
        update.effective_user.id,
        city,
        "oggi",
        fused_points,
        points_ow,
        points_wa,
        points_mb,
        target_date,
        tz_offset,
        coords["lat"],
        coords["lon"],
        rain_threshold=ths.get("pop", RAIN_POP_THRESHOLD),
    )


async def trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]

    if context.args:
        city = " ".join(context.args)
    else:
        city = state.storage.get_user_city(update.effective_user.id)
        if not city:
            await update.effective_message.reply_text(MSG_NEED_CITY)
            return

    coords = await get_coords(state, city)
    if not coords:
        await update.effective_message.reply_text(MSG_CITY_NOT_FOUND)
        return

    forecast_ow, forecast_wa, forecast_mb, _, _, _, fused_points, target_date, tz_offset, _, _, _, _ = await _load_forecast_points(
        state, coords, 0
    )
    if not forecast_ow and not forecast_wa and not forecast_mb:
        await update.effective_message.reply_text(MSG_SERVICE_UNAVAILABLE)
        return
    if not fused_points:
        await update.effective_message.reply_text("Nessuna previsione disponibile")
        return

    def _avg(vals: List[float]) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    points_today = [p for p in fused_points if p["local_time"].date() == target_date]
    early = [p["temp"] for p in points_today if 6 <= p["local_time"].hour <= 11]
    late = [p["temp"] for p in points_today if 12 <= p["local_time"].hour <= 18]
    if not early:
        early = [p["temp"] for p in points_today[:6]]
    if not late:
        late = [p["temp"] for p in points_today[-6:]]
    early_avg = _avg(early)
    late_avg = _avg(late)
    if early_avg is None or late_avg is None:
        temp_trend = "n/d"
    else:
        delta = late_avg - early_avg
        if delta > 1.0:
            temp_trend = "in aumento"
        elif delta < -1.0:
            temp_trend = "in diminuzione"
        else:
            temp_trend = "stabile"

    ths = get_dynamic_thresholds(state.storage)
    max_pop = max((float(p.get("pop", 0)) for p in points_today), default=0)
    rain_yes = "si" if max_pop >= float(ths.get("pop", RAIN_POP_THRESHOLD)) else "no"
    risk = rain_risk_label(max_pop) or "n/d"

    name_md = md_escape(coords.get("name", ""))
    line = (
        f"Trend oggi a {name_md}: temperatura {temp_trend}, "
        f"pioggia {rain_yes} (max {max_pop:.0f}%, rischio {risk})."
    )
    await update.effective_message.reply_text(line, parse_mode="Markdown")


async def prev(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]

    if context.args:
        city = " ".join(context.args)
    else:
        city = state.storage.get_user_city(update.effective_user.id)
        if not city:
            await update.effective_message.reply_text(MSG_NEED_CITY)
            return

    coords = await get_coords(state, city)
    if not coords:
        await update.effective_message.reply_text(MSG_CITY_NOT_FOUND)
        return

    forecast_ow, forecast_wa, forecast_mb, points_ow, points_wa, points_mb, fused_points, target_date, tz_offset, sources_label, cached_used, cache_stale, cache_age_min = await _load_forecast_points(
        state, coords, 0
    )
    if not forecast_ow and not forecast_wa and not forecast_mb:
        await update.effective_message.reply_text(MSG_SERVICE_UNAVAILABLE)
        return

    now_local = local_now(tz_offset)
    start_hour = now_local.hour
    by_hour = {p["local_time"].hour: p for p in fused_points}
    season = season_from_date(target_date)
    zone = zone_bucket(coords["lat"], coords["lon"])

    name_md = md_escape(coords.get("name", ""))
    country_md = md_escape(coords.get("country", ""))

    lines = []
    lines.append("PREVISIONE ORA PER ORA")
    lines.append(f"{name_md}, {country_md}")
    lines.append(f"{format_date_italian(datetime.combine(target_date, datetime.min.time()))}")
    lines.append(f"Da {start_hour:02d}:00 a 23:00")
    max_pop = max(
        (float(p.get("pop", 0)) for p in fused_points if p["local_time"].hour >= start_hour),
        default=None,
    )
    rain_risk = rain_risk_label(max_pop)
    if rain_risk:
        lines.append(f"Rischio pioggia: {rain_risk}")
    lines.append("")

    prefs = state.storage.get_user_prefs(update.effective_user.id)
    for h in range(start_hour, 24):
        p = by_hour.get(h)
        if not p:
            continue
        ow_p = pick_point_for_hour(points_ow, h, 2) if points_ow else None
        wa_p = pick_point_for_hour(points_wa, h, 1) if points_wa else None
        mb_p = pick_point_for_hour(points_mb, h, 1) if points_mb else None
        cond = condition_group_from_icon(str(p.get("icon", "01d")), float(p.get("pop", 0)))
        conf = hourly_confidence_percent_context(
            state.storage, ow_p, wa_p, mb_p, hour_band(h), cond,
            kind_group="forecast", season=season, zone=zone
        )
        is_day = 6 <= h <= 19
        icon = get_weather_icon(str(p.get("icon", "01d")), pop=p.get("pop"), is_day=is_day)
        rain_icon = get_rain_risk_icon(p.get("pop"), icon_code=p.get("icon"))
        if rain_icon and icon in {"\U0001F327\ufe0f", "\U0001F326\ufe0f", "\u26c8\ufe0f"}:
            rain_icon = ""
        risk = f" {rain_icon}" if rain_icon else ""
        lines.append(f"{h:02d}:00 {format_temp(p.get('temp'), prefs)} {conf}% {icon}{risk}")

    lines.append("")
    lines.append(f"Fonti: {sources_label}")
    enabled = [p for p, k in [("OpenWeather", OPENWEATHER_API_KEY), ("WeatherAPI", WEATHERAPI_KEY), ("Meteoblue", METEOBLUE_API_KEY)] if k]
    sources = []
    if points_ow:
        sources.append("OpenWeather")
    if points_wa:
        sources.append("WeatherAPI")
    if points_mb:
        sources.append("Meteoblue")
    fallback_note = build_provider_note(enabled, sources, cache_stale=cache_stale, cache_age_min=cache_age_min)
    if fallback_note:
        lines.append(f"Nota: {fallback_note}")
    if cached_used:
        if cache_age_min is not None:
            cache_line = f"Dato: Cache {cache_age_min}m"
            if cache_stale:
                cache_line += " (stale)"
            lines.append(cache_line)
        else:
            lines.append("Dato: Cache (stale)" if cache_stale else "Dato: Cache")

    await update.effective_message.reply_text("\n".join(lines), parse_mode="Markdown")

async def domani(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _forecast_common(update, context, days=1)

async def pref(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    prefs = state.storage.get_user_prefs(update.effective_user.id)
    temp_unit = normalize_temp_unit(prefs.get("temp_unit"))
    wind_unit = normalize_wind_unit(prefs.get("wind_unit"))
    wind_unit_label = "mph" if wind_unit == "mph" else "km/h"
    msg = (
        "Preferenze utente:\n"
        f"- POP ombrello: {prefs['pop_threshold']:.0f}%\n"
        f"- Vento forte: {format_wind(prefs['wind_threshold'], prefs)}\n"
        f"- Percepita diversa: {format_temp_delta(prefs['feels_diff'], prefs)}\n"
        f"- Avviso pioggia 60 min: {'ON' if prefs.get('alert_rain_60') else 'OFF'}\n"
        f"- Avviso pioggia domani: {'ON' if prefs.get('alert_tomorrow_rain', 1) else 'OFF'}\n"
        f"- Unita temperatura: {temp_unit}\n"
        f"- Unita vento: {wind_unit_label}\n"
        f"- Fascia notifiche: {format_alert_window(prefs)}"
    )
    await update.effective_message.reply_text(msg, parse_mode="Markdown")

async def setpref(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    if not context.args:
        await update.effective_message.reply_text(
            "Usa: /setpref pop=60 vento=40 diff=3 unit=c windunit=kmh alert60=on alertdomani=on notifiche=07-22",
            parse_mode="Markdown"
        )
        return
    pop = wind = diff = None
    alert60 = alertdomani = None
    temp_unit = wind_unit = None
    alert_window = None
    for arg in context.args:
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        try:
            val = float(v)
        except Exception:
            val = None
        if k in {"pop", "pioggia"}:
            pop = val
        elif k in {"vento", "wind"}:
            wind = val
        elif k in {"diff", "percepita"}:
            diff = val
        elif k in {"unit", "temp", "tempunit"}:
            temp_unit = v
        elif k in {"windunit", "unita_vento"}:
            wind_unit = v
        elif k in {"alert60", "pioggia60"}:
            alert60 = 1 if v.lower() in {"on", "si", "true", "1"} else 0
        elif k in {"alertdomani", "domani"}:
            alertdomani = 1 if v.lower() in {"on", "si", "true", "1"} else 0
        elif k in {"notifiche", "alertwindow", "finestra"}:
            alert_window = v

    prefs = state.storage.get_user_prefs(update.effective_user.id)
    final_temp_unit = normalize_temp_unit(temp_unit or prefs.get("temp_unit"))
    final_wind_unit = normalize_wind_unit(wind_unit or prefs.get("wind_unit"))

    if diff is not None and final_temp_unit == "F":
        diff = float(diff) / 1.8
    if wind is not None and final_wind_unit == "mph":
        wind = float(wind) * 1.60934

    alert_start = alert_end = None
    if alert_window:
        parsed = parse_alert_window(alert_window)
        if parsed:
            alert_start, alert_end = parsed

    state.storage.set_user_prefs(
        update.effective_user.id,
        pop,
        wind,
        diff,
        alert_rain_60=alert60,
        alert_tomorrow_rain=alertdomani,
        temp_unit=final_temp_unit if temp_unit is not None else None,
        wind_unit=final_wind_unit if wind_unit is not None else None,
        alert_start_hour=alert_start,
        alert_end_hour=alert_end,
    )
    await update.effective_message.reply_text("Preferenze aggiornate.", parse_mode="Markdown")

async def stat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    rows = state.storage.get_recent_errors(5)
    if not rows:
        await update.effective_message.reply_text("Nessun errore recente.", parse_mode="Markdown")
        return
    lines = ["Errori recenti:"]
    for r in rows:
        created = md_escape(r["created_at"])
        status = md_escape(str(r["status"]))
        url = md_escape(r["url"])
        detail = md_escape(r["detail"])
        lines.append(f"- {created} | {status} | {url} | {detail}")
    await update.effective_message.reply_text("\n".join(lines), parse_mode="Markdown")


async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    token = " ".join(context.args) if context.args else None
    uid = update.effective_user.id
    if not ADMIN_SECRET and not ADMIN_IDS_SET:
        await update.effective_message.reply_text("Admin non configurato. Imposta ADMIN_SECRET o ADMIN_USER_IDS.")
        return
    if not is_admin(uid, token):
        await update.effective_message.reply_text("Accesso negato.")
        return

    state: AppState = context.application.bot_data["state"]
    lines = []
    lines.append("\U0001F6E1 STATUS ADMIN")
    lines.append(f"{format_date_italian()} {format_time_italian()}")
    lines.append("")
    try:
        with state.storage._connect() as conn:
            cache_count = conn.execute("SELECT COUNT(*) AS n FROM cache").fetchone()["n"]
            pred_total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
            pred_ok = conn.execute("SELECT COUNT(*) AS n FROM predictions WHERE verified=1").fetchone()["n"]
            err_total = conn.execute("SELECT COUNT(*) AS n FROM errors_log").fetchone()["n"]
            users_count = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        db_size_kb = int(os.path.getsize(DB_FILE) / 1024) if os.path.exists(DB_FILE) else 0
        lines.append("DB")
        lines.append(f"- Cache: {cache_count}")
        lines.append(f"- Predizioni: {pred_total} (verificate {pred_ok})")
        lines.append(f"- Errori totali: {err_total}")
        lines.append(f"- Utenti: {users_count}")
        lines.append(f"- DB size: {db_size_kb} KB")
    except Exception:
        pass

    try:
        lines.append("")
        lines.append("Cache RAM")
        lines.append(f"- Items: {len(state.ram_cache.data)} (max {state.ram_cache.max_items})")
    except Exception:
        pass

    try:
        ths = get_dynamic_thresholds(state.storage)
        lines.append("")
        lines.append("Parametri")
        lines.append(f"- POP base/dinamico: {RAIN_POP_THRESHOLD}% / {ths.get('pop', RAIN_POP_THRESHOLD):.0f}%")
        lines.append(f"- Vento forte base/dinamico: {WIND_STRONG_KPH} / {ths.get('wind', WIND_STRONG_KPH):.0f} km/h")
        lines.append(f"- Ombrello: {UMBRELLA_POP_THRESHOLD}%")
        lines.append(f"- Percepita diff: {FEELS_LIKE_DIFF_C}C")
        lines.append(f"- TTL cache current/forecast: {CACHE_TTL_CURRENT_MIN}m / {CACHE_TTL_FORECAST_MIN}m")
    except Exception:
        pass

    try:
        weights = state.weather.get_dynamic_weights()
        lines.append("")
        lines.append("Pesi provider")
        for k, v in weights.items():
            lines.append(f"- {k}: {float(v):.2f}")
    except Exception:
        pass

    try:
        bias = state.storage.get_provider_bias()
        lines.append("")
        lines.append("Bias provider")
        if not bias:
            lines.append("- nessun dato")
        else:
            for k, v in bias.items():
                lines.append(f"- {k}: {float(v):+.2f}C")
    except Exception:
        pass

    rows = state.storage.get_recent_errors(10)
    lines.append("")
    lines.append("Errori recenti")
    if not rows:
        lines.append("- nessuno")
    else:
        for r in rows:
            created = md_escape(r["created_at"])
            status = md_escape(str(r["status"]))
            detail = md_escape(r["detail"])
            lines.append(f"- {created} | {status} | {detail}")

    try:
        backoff_rows = state.storage.get_provider_backoff_status()
        lines.append("")
        lines.append("Backoff provider")
        now = now_utc()
        if not backoff_rows:
            lines.append("- nessun dato")
        else:
            for r in backoff_rows:
                until = r["backoff_until"]
                if not until:
                    status = "OK"
                else:
                    until_dt = from_iso(until)
                    if now >= until_dt:
                        status = "OK"
                    else:
                        mins = int((until_dt - now).total_seconds() // 60)
                        status = f"PAUSA {mins}m"
                fail_count = int(r["fail_count"]) if r["fail_count"] is not None else 0
                last_err = r["last_error"] or "-"
                lines.append(f"- {r['provider']}: {status} (fail {fail_count}, last {last_err})")
    except Exception:
        pass

    lines.append("")
    lines.append("Provider accuracy")
    acc = state.storage.get_provider_accuracy()
    if not acc:
        lines.append("- nessun dato")
    else:
        for k, v in acc.items():
            lines.append(f"- {k}: {v['accuracy']:.1f}% (avg err {format_temp_delta(v['avg_error'], None)})")

    await update.effective_message.reply_text("\n".join(lines), parse_mode="Markdown")


async def testoffline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    days = OFFLINE_TEST_DAYS
    if context.args:
        try:
            days = int(context.args[0])
        except Exception:
            pass
    stats = state.storage.get_offline_stats(update.effective_user.id, days=days)
    if not stats or stats.get("count", 0) == 0:
        await update.effective_message.reply_text("Nessun dato storico verificato.", parse_mode="Markdown")
        return
    prefs = state.storage.get_user_prefs(update.effective_user.id)
    lines = []
    lines.append("TEST OFFLINE")
    lines.append(f"Periodo: ultimi {stats['days']} giorni")
    lines.append(f"Campioni: {stats['count']}")
    if stats.get("fused_mae") is not None:
        lines.append(f"Errore medio (fuso): {format_temp_delta(stats['fused_mae'], prefs)}")
    if stats.get("ow_mae") is not None:
        lines.append(f"Errore medio OW: {format_temp_delta(stats['ow_mae'], prefs)}")
    if stats.get("wa_mae") is not None:
        lines.append(f"Errore medio WA: {format_temp_delta(stats['wa_mae'], prefs)}")
    if stats.get("mb_mae") is not None:
        lines.append(f"Errore medio MB: {format_temp_delta(stats['mb_mae'], prefs)}")
    if stats.get("rain_total", 0):
        lines.append(f"Pioggia corretta: {stats['rain_hits']}/{stats['rain_total']}")
    if stats.get("cond_total", 0):
        lines.append(f"Condizioni corrette: {stats['cond_hits']}/{stats['cond_total']}")
    await update.effective_message.reply_text("\n".join(lines), parse_mode="Markdown")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = (query.data or "").upper()
    if data == "DETTAGLI":
        await _call_with_args(meteo, update, context, ["dettagli"])
    elif data == "AGGIORNA":
        await _call_with_args(meteo, update, context, ["force"])
    elif data == "CAMBIA":
        await update.effective_message.reply_text("Usa /setcitta <citta> per cambiare la citta.")

async def handle_text_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.effective_message.text or "").strip().lower()
    if text == "meteo":
        await _call_with_args(meteo, update, context, [])
    elif text == "oggi":
        await _call_with_args(oggi, update, context, [])
    elif text == "domani":
        await _call_with_args(domani, update, context, [])
    elif text == "trend":
        await _call_with_args(trend, update, context, [])
    elif text == "dettagli":
        await _call_with_args(meteo, update, context, ["dettagli"])
    elif text == "aggiorna":
        await _call_with_args(meteo, update, context, ["force"])
    elif text == "cambia citta":
        await update.effective_message.reply_text("Usa /citta per cambiare rapidamente o /setcitta <citta>.")
    elif text == "comandi":
        await _call_with_args(comandi, update, context, [])


def _alert_ready(storage: Storage, user_id: int, alert_type: str) -> bool:
    last = storage.get_last_alert(user_id, alert_type)
    if not last:
        return True
    return (now_utc() - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)


async def _send_rain_alert(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    if not job:
        return
    data = job.data or {}
    user_id = data.get("user_id")
    text = data.get("text")
    alert_type = data.get("alert_type")
    if not user_id or not text or not alert_type:
        return
    state: AppState = context.application.bot_data["state"]
    if not _alert_ready(state.storage, int(user_id), alert_type):
        return
    await context.bot.send_message(chat_id=int(user_id), text=text)
    state.storage.set_last_alert(int(user_id), alert_type, now_utc())


async def _get_forecast_cached(state: AppState, lat: float, lon: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    ck = key_forecast(lat, lon, 0)
    cached = state.ram_cache.get(ck) or state.storage.cache_get(ck)
    if cached:
        return cached.get("forecast_ow"), cached.get("forecast_wa"), cached.get("forecast_mb")
    forecast_ow, forecast_wa, forecast_mb = await asyncio.gather(
        state.ow.forecast(lat, lon),
        state.wa.forecast(lat, lon, days=2),
        state.mb.forecast(lat, lon),
    )
    if forecast_ow or forecast_wa or forecast_mb:
        payload = {"forecast_ow": forecast_ow, "forecast_wa": forecast_wa, "forecast_mb": forecast_mb, "saved_at": iso(now_utc())}
        tz_offset_fc = get_tz_offset_sec(forecast_ow, forecast_wa, forecast_mb, coords=None)
        ttl_fc = forecast_ttl_minutes(local_now(tz_offset_fc))
        state.storage.cache_set(ck, payload, ttl=timedelta(minutes=ttl_fc))
        state.ram_cache.set(ck, payload, ttl_seconds=30)
    return forecast_ow, forecast_wa, forecast_mb


async def _get_current_fused_cached(state: AppState, lat: float, lon: float, coords: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ck = key_current(lat, lon)
    cached = state.ram_cache.get(ck) or state.storage.cache_get(ck)
    if cached and cached.get("fused"):
        return cached.get("fused")
    ow_cur, wa_cur, mb_cur = await asyncio.gather(
        state.ow.current(lat, lon),
        state.wa.current(lat, lon),
        state.mb.current(lat, lon),
    )
    if not ow_cur.success and not wa_cur.success and not mb_cur.success:
        return None
    tz_offset = get_tz_offset_sec(None, None, None, ow_cur, wa_cur, mb_cur, coords=coords)
    local_dt = local_now(tz_offset)
    band = hour_band(local_dt.hour)
    season = season_from_date(local_dt.date())
    zone = zone_bucket(lat, lon)
    weights = state.weather.get_dynamic_weights()
    w_ow = float(weights.get("OpenWeather", 0.5))
    w_wa = float(weights.get("WeatherAPI", 0.5))
    if ow_cur.success and (not wa_cur.success or w_ow >= w_wa):
        cond_desc = ow_cur.description
    elif wa_cur.success:
        cond_desc = wa_cur.description
    else:
        cond_desc = ""
    cond_group = condition_group_from_description(cond_desc)
    bias_ctx = {
        "OpenWeather": state.storage.get_provider_bias_context("OpenWeather", band, cond_group, kind_group="now", season=season, zone=zone),
        "WeatherAPI": state.storage.get_provider_bias_context("WeatherAPI", band, cond_group, kind_group="now", season=season, zone=zone),
        "Meteoblue": state.storage.get_provider_bias_context("Meteoblue", band, cond_group, kind_group="now", season=season, zone=zone),
    }
    temps = [t for t in [ow_cur.temp if ow_cur.success else None, wa_cur.temp if wa_cur.success else None, mb_cur.temp if mb_cur.success else None] if t is not None]
    diff = (max(temps) - min(temps)) if len(temps) >= 2 else 0.0
    err_ow = state.storage.get_provider_accuracy_context("OpenWeather", band, cond_group, kind_group="now", season=season, zone=zone)
    err_wa = state.storage.get_provider_accuracy_context("WeatherAPI", band, cond_group, kind_group="now", season=season, zone=zone)
    weights_ctx = adaptive_weights(weights, diff, err_ow, err_wa)
    weights_ctx["Meteoblue"] = float(weights.get("Meteoblue", 0.0))
    return state.weather.fuse(ow_cur, wa_cur, mb_cur, bias=bias_ctx, weights_override=weights_ctx)


def _schedule_rain_alerts(
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    city_name: str,
    rain_period: Dict[str, Any],
    tz_offset: int,
):
    job_queue = context.application.job_queue if context and context.application else None
    if not job_queue:
        return
    start_local = rain_period["start"]
    start_utc = start_local.astimezone(timezone.utc)
    now = now_utc()
    offsets = [("rain_3h", 3), ("rain_1h", 1)]
    for key, hours in offsets:
        when = start_utc - timedelta(hours=hours)
        if when <= now:
            continue
        alert_type = f"{key}_{start_local.date().isoformat()}"
        state: AppState = context.application.bot_data["state"]
        if not _alert_ready(state.storage, user_id, alert_type):
            continue
        text = f"Pioggia prevista a {city_name} tra {rain_period['start'].strftime('%H:%M')}-{rain_period['end'].strftime('%H:%M')}: avviso {hours}h prima"
        job_queue.run_once(
            _send_rain_alert,
            when=when,
            data={"user_id": user_id, "text": text, "alert_type": alert_type},
            name=f"{alert_type}_{user_id}",
        )


async def _check_alerts_for_user(state: AppState, user_id: int, context: ContextTypes.DEFAULT_TYPE):
    city = state.storage.get_user_city(user_id)
    if not city:
        return
    prefs = state.storage.get_user_prefs(user_id)
    coords = await get_coords(state, city)
    if not coords:
        return
    lat, lon = coords["lat"], coords["lon"]
    forecast_ow, forecast_wa, forecast_mb = await _get_forecast_cached(state, lat, lon)
    if not forecast_ow and not forecast_wa and not forecast_mb:
        return
    tz_offset = get_tz_offset_sec(forecast_ow, forecast_wa, forecast_mb, coords=coords)
    now_local = local_now(tz_offset)
    if not within_alert_window(now_local, prefs):
        return
    target_date = now_local.date()
    season = season_from_date(target_date)
    zone = zone_bucket(lat, lon)
    points_ow = interpolate_hourly_points(extract_day_points(forecast_ow, target_date, tz_offset)) if forecast_ow else []
    points_wa = extract_day_points_weatherapi(forecast_wa, target_date, tz_offset) if forecast_wa else []
    points_mb = extract_day_points_meteoblue(forecast_mb, target_date, tz_offset) if forecast_mb else []
    if not points_ow and not points_wa and not points_mb:
        return
    weights = state.weather.get_dynamic_weights()
    bias = state.storage.get_provider_bias()
    bias_func = lambda prov, hr, cond: state.storage.get_provider_bias_context(
        prov, hour_band(hr), cond, kind_group="forecast", season=season, zone=zone
    )
    def weight_func(hr, ow_p, wa_p, mb_p):
        if not ow_p or not wa_p:
            return weights.get("OpenWeather", 0.33), weights.get("WeatherAPI", 0.33), weights.get("Meteoblue", 0.34)
        cond = condition_group_from_icon(str(ow_p["icon"]), float(ow_p["pop"]))
        err_ow = state.storage.get_provider_accuracy_context("OpenWeather", hour_band(hr), cond, kind_group="forecast", season=season, zone=zone)
        err_wa = state.storage.get_provider_accuracy_context("WeatherAPI", hour_band(hr), cond, kind_group="forecast", season=season, zone=zone)
        diff = abs(float(ow_p["temp"]) - float(wa_p["temp"]))
        w = adaptive_weights(weights, diff, err_ow, err_wa)
        w["Meteoblue"] = float(weights.get("Meteoblue", 0.34))
        return w.get("OpenWeather", 0.33), w.get("WeatherAPI", 0.33), w.get("Meteoblue", 0.34)
    fused_points = build_fused_points(points_ow, points_wa, points_mb, weights, bias, target_date, tz_offset, bias_func=bias_func, weight_func=weight_func)
    pop_calibrator = build_pop_calibrator(state.storage)
    fused_points = apply_pop_calibration(fused_points, pop_calibrator)

    # Rain alert
    ths = get_dynamic_thresholds(state.storage)
    periods = compute_rain_periods_threshold(fused_points, ths.get("pop", RAIN_POP_THRESHOLD))
    window_end = now_local + timedelta(minutes=90)
    rain_period = None
    for p in periods:
        if p["end"] >= now_local and p["start"] <= window_end:
            if rain_period is None or p["start"] < rain_period["start"]:
                rain_period = p
    if prefs.get("alert_rain_60") and rain_period and float(rain_period.get("max_prob", 0)) >= ALERT_POP_THRESHOLD:
        if _alert_ready(state.storage, user_id, "rain"):
            s = rain_period["start"].strftime("%H:%M")
            e = rain_period["end"].strftime("%H:%M")
            text = f"Pioggia entro 90 min a {coords['name']}: {s}-{e} (max {rain_period['max_prob']:.0f}%)"
            await context.bot.send_message(chat_id=user_id, text=text)
            state.storage.set_last_alert(user_id, "rain", now_utc())

    # Wind alert
    fused_cur = await _get_current_fused_cached(state, lat, lon, coords)
    wind_kph = fused_cur.get("wind_kph") if fused_cur else None
    if wind_kph is not None and wind_kph >= ALERT_WIND_KPH:
        if _alert_ready(state.storage, user_id, "wind"):
            text = f"Vento forte a {coords['name']}: {format_wind(wind_kph, prefs)}"
            await context.bot.send_message(chat_id=user_id, text=text)
            state.storage.set_last_alert(user_id, "wind", now_utc())

    # Temperature alerts (today)
    min_t, max_t = extract_min_max(fused_points)
    if max_t is not None and max_t >= ALERT_TEMP_HOT:
        if _alert_ready(state.storage, user_id, "hot"):
            text = f"Caldo estremo oggi a {coords['name']}: max {format_temp(max_t, prefs, decimals=0)}"
            await context.bot.send_message(chat_id=user_id, text=text)
            state.storage.set_last_alert(user_id, "hot", now_utc())
    if min_t is not None and min_t <= ALERT_TEMP_COLD:
        if _alert_ready(state.storage, user_id, "cold"):
            text = f"Freddo estremo oggi a {coords['name']}: min {format_temp(min_t, prefs, decimals=0)}"
            await context.bot.send_message(chat_id=user_id, text=text)
            state.storage.set_last_alert(user_id, "cold", now_utc())


async def check_alerts_job(context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    users = state.storage.get_all_users()
    for uid in users:
        try:
            await _check_alerts_for_user(state, int(uid), context)
        except Exception as exc:
            logger.error(f"Alert job error for user {uid}: {exc}")


async def _build_controlla_report(
    state: AppState,
    user_id: int,
    city: str,
    coords: Dict[str, Any],
) -> Optional[str]:
    prefs = state.storage.get_user_prefs(user_id)
    if not WEATHERAPI_KEY and not METEOBLUE_API_KEY:
        return "Storico non disponibile (manca WeatherAPI e Meteoblue)."

    tz_offset = get_tz_offset_sec(None, None, None, coords=coords)
    tz = tzinfo_from_offset(tz_offset)
    now_local = local_now(tz_offset)
    today_local = now_local.date()
    day_start_local = datetime.combine(today_local, datetime.min.time()).replace(tzinfo=tz)
    day_end_local = day_start_local + timedelta(days=1)
    with state.storage._connect() as conn:
        due = conn.execute("""
          SELECT * FROM predictions
          WHERE user_id=? AND verified=0
            AND target_dt_utc BETWEEN ? AND ?
          ORDER BY target_dt_utc ASC
        """, (str(user_id), iso(day_start_local.astimezone(timezone.utc)), iso(day_end_local.astimezone(timezone.utc)))).fetchall()
    due = [row for row in due if from_iso(row["target_dt_utc"]).astimezone(tz) <= now_local + timedelta(minutes=10)]
    if not due:
        return None

    lat, lon = coords["lat"], coords["lon"]
    history_cache: Dict[str, Any] = {}
    mb_history_cache: Dict[str, Any] = {}
    verified_count = 0
    errors = []
    details = []
    rain_checks = 0
    rain_hits = 0
    cond_checks = 0
    cond_hits = 0
    wind_errs = []
    humidity_errs = []
    pressure_errs = []
    pop_briers = []
    actual_temps = []
    skipped = 0
    mb_used_slots = 0
    mb_used_dates: Dict[str, int] = {}

    for row in due:
        pid = row["id"]
        target_dt = from_iso(row["target_dt_utc"]).astimezone(tz)
        pred_fused = float(row["predicted_fused"])

        actual_temp = None
        actual_rain = None
        actual_cond = None
        actual_wind = None
        actual_humidity = None
        actual_pressure = None

        date_key = target_dt.date().isoformat()
        p = None
        hist = history_cache.get(date_key)
        if hist is None and WEATHERAPI_KEY:
            hist = await state.wa.history(lat, lon, date_key)
            history_cache[date_key] = hist
        points_hist = extract_day_points_weatherapi_history(hist, target_dt.date(), tz_offset) if hist else []
        p = pick_point_for_hour(points_hist, target_dt.hour, 1) if points_hist else None
        if not p and METEOBLUE_API_KEY:
            mb_hist = mb_history_cache.get(date_key)
            if mb_hist is None:
                mb_hist = await state.mb.history(lat, lon, history_days=1)
                mb_history_cache[date_key] = mb_hist
                try:
                    mb_points = extract_day_points_meteoblue(mb_hist, target_dt.date(), tz_offset) if mb_hist else []
                    logger.info("Meteoblue history %s: %s punti", date_key, len(mb_points))
                except Exception:
                    logger.info("Meteoblue history %s: errore parsing punti", date_key)
            points_mb_hist = extract_day_points_meteoblue(mb_hist, target_dt.date(), tz_offset) if mb_hist else []
            p = pick_point_for_hour(points_mb_hist, target_dt.hour, 1)
            if p:
                mb_used_slots += 1
                mb_used_dates[date_key] = mb_used_dates.get(date_key, 0) + 1
        if p:
            actual_temp = float(p["temp"])
            desc = str(p.get("desc", ""))
            actual_cond = condition_group_from_description(desc) if desc else None
            precip_mm = p.get("precip_mm")
            if precip_mm is None:
                precip_mm = 0.0
            actual_rain = bool(float(precip_mm) > 0.1) or is_rain_description(desc)
            if p.get("wind_kph") is not None:
                actual_wind = float(p["wind_kph"])
            if p.get("humidity") is not None:
                actual_humidity = float(p["humidity"])
            press_val = p.get("pressure_mb", p.get("pressure"))
            if press_val is not None:
                actual_pressure = float(press_val)
        else:
            skipped += 1
            continue

        actual_temps.append(float(actual_temp))
        err = abs(pred_fused - float(actual_temp))

        pred_wind = row["predicted_wind_kph"]
        pred_hum = row["predicted_humidity"]
        pred_pres = row["predicted_pressure"]
        err_wind = abs(float(pred_wind) - float(actual_wind)) if pred_wind is not None and actual_wind is not None else None
        err_hum = abs(float(pred_hum) - float(actual_humidity)) if pred_hum is not None and actual_humidity is not None else None
        err_pres = abs(float(pred_pres) - float(actual_pressure)) if pred_pres is not None and actual_pressure is not None else None
        if err_wind is not None:
            wind_errs.append(err_wind)
        if err_hum is not None:
            humidity_errs.append(err_hum)
        if err_pres is not None:
            pressure_errs.append(err_pres)

        pred_rain = row["predicted_rain"]
        pred_pop = row["predicted_pop"]
        brier = None
        if pred_pop is not None and actual_rain is not None:
            p_val = float(pred_pop) / 100.0
            a_val = 1.0 if bool(actual_rain) else 0.0
            brier = (p_val - a_val) ** 2
            pop_briers.append(brier)

        state.storage.mark_prediction_verified(
            pid,
            actual_temp=float(actual_temp),
            error_fused=err,
            actual_rain=bool(actual_rain) if actual_rain is not None else None,
            actual_condition_group=actual_cond,
            actual_wind_kph=actual_wind,
            actual_humidity=actual_humidity,
            actual_pressure=actual_pressure,
            error_wind=err_wind,
            error_humidity=err_hum,
            error_pressure=err_pres,
            error_pop=brier,
        )
        verified_count += 1
        errors.append(err)

        if row["predicted_ow"] is not None:
            state.weather.update_provider_accuracy("OpenWeather", float(row["predicted_ow"]), float(actual_temp))
        if row["predicted_wa"] is not None:
            state.weather.update_provider_accuracy("WeatherAPI", float(row["predicted_wa"]), float(actual_temp))
        if row["predicted_mb"] is not None:
            state.weather.update_provider_accuracy("Meteoblue", float(row["predicted_mb"]), float(actual_temp))

        rain_note = ""
        if pred_rain is not None and actual_rain is not None:
            rain_checks += 1
            hit = (bool(pred_rain) == bool(actual_rain))
            if hit:
                rain_hits += 1
            pop_txt = f"{float(pred_pop):.0f}%" if pred_pop is not None else "n/d"
            rain_note = f" | pioggia: {pop_txt} {'si' if pred_rain else 'no'} / {'si' if actual_rain else 'no'}"

        cond_note = ""
        pred_cond = row["condition_group"]
        if pred_cond and actual_cond:
            cond_checks += 1
            if str(pred_cond) == str(actual_cond):
                cond_hits += 1
            cond_note = f" | meteo: {pred_cond} / {actual_cond}"

        status_icon = "\u2705" if err <= 1.5 else "\u26a0\ufe0f" if err <= 3 else "\u274c"
        details.append(
            f"{status_icon} {row['kind']} {target_dt.strftime('%H:%M')} "
            f"{format_temp(pred_fused, prefs)} -> {format_temp(float(actual_temp), prefs)} (err {format_temp_delta(err, prefs)}){rain_note}{cond_note}"
        )

    avg_err = sum(errors) / len(errors) if errors else 0.0
    accuracy = clamp(100 - (avg_err * 15), 0, 100)

    acc = state.storage.get_provider_accuracy()
    ref_temp = sum(actual_temps) / len(actual_temps) if actual_temps else None
    name_md = md_escape(coords.get("name", ""))
    country_md = md_escape(coords.get("country", ""))

    lines = []
    lines.append("\U0001F4CA VERIFICA PREVISIONI")
    lines.append(f"{name_md}, {country_md}")
    lines.append(f"{format_date_italian()} {format_time_italian()}")
    lines.append("")
    lines.append("Riepilogo")
    lines.append(f"- Verificate: {verified_count}")
    if skipped:
        lines.append(f"- Saltate (mancano dati storici): {skipped}")
    if ref_temp is not None:
        lines.append(f"- Temp riferimento: {format_temp(ref_temp, prefs)}")
    lines.append(f"- Errore medio: {format_temp_delta(avg_err, prefs)}")
    lines.append(f"- Accuratezza: {accuracy:.1f}%")
    if rain_checks:
        lines.append(f"- Pioggia corretta: {rain_hits}/{rain_checks}")
    else:
        lines.append("- Pioggia: nessun dato")
    if cond_checks:
        lines.append(f"- Meteo corretto: {cond_hits}/{cond_checks}")
    if wind_errs:
        lines.append(f"- Errore vento medio: {format_wind(sum(wind_errs)/len(wind_errs), prefs)}")
    if humidity_errs:
        lines.append(f"- Errore umidita medio: {sum(humidity_errs)/len(humidity_errs):.1f}%")
    if pressure_errs:
        lines.append(f"- Errore pressione medio: {sum(pressure_errs)/len(pressure_errs):.1f} hPa")
    if pop_briers:
        lines.append(f"- Brier score pioggia: {sum(pop_briers)/len(pop_briers):.3f}")
    if mb_used_slots:
        days = ", ".join(sorted(mb_used_dates.keys()))
        lines.append(f"- Storico Meteoblue usato: {mb_used_slots} slot (date: {days})")

    lines.append("")
    lines.append("Dettaglio (ultime 10)")
    lines.extend(details[:10])

    lines.append("")
    lines.append("Provider")
    if acc:
        for k, v in acc.items():
            lines.append(f"- {k}: {v['accuracy']:.1f}% (avg err {format_temp_delta(v['avg_error'], prefs)})")
    else:
        lines.append("- nessun dato")

    return "\n".join(lines)


async def controlla(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Verifica SOLO previsioni in orario e SOLO del giorno corrente.
    Aggiorna accuratezza provider e mostra anche esito pioggia.
    """
    state: AppState = context.application.bot_data["state"]

    city = state.storage.get_user_city(update.effective_user.id)
    if context.args:
        city = " ".join(context.args)
    if not city:
        await update.effective_message.reply_text(MSG_NEED_CITY)
        return

    coords = await get_coords(state, city)
    if not coords:
        await update.effective_message.reply_text(MSG_CITY_NOT_FOUND)
        return

    report = await _build_controlla_report(state, update.effective_user.id, city, coords)
    if not report:
        await update.effective_message.reply_text(
            "Niente da verificare ora.\n"
            "Le verifiche controllano tutte le ore previste di oggi fino all'ora attuale.\n"
            "Esempio: fai /oggi e poi /controlla nel pomeriggio per verificare gli slot gia passati.",
            parse_mode="Markdown"
        )
        return
    await update.effective_message.reply_text(report, parse_mode="Markdown")
    # Invalida cache forecast così /domani (e /oggi) pesca dati aggiornati
    ck_fc = key_forecast(coords["lat"], coords["lon"], 0)
    state.ram_cache.delete(ck_fc)
    state.storage.cache_delete(ck_fc)


async def daily_check_job(context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    if not WEATHERAPI_KEY and not METEOBLUE_API_KEY:
        return
    users = state.storage.get_all_users()
    for uid in users:
        try:
            user_id = int(uid)
            city = state.storage.get_user_city(user_id)
            if not city:
                continue
            coords = await get_coords(state, city)
            if not coords:
                continue
            tz_offset = get_tz_offset_sec(None, None, None, coords=coords)
            now_local = local_now(tz_offset)
            if now_local.hour != DAILY_CHECK_HOUR or now_local.minute < DAILY_CHECK_MIN:
                continue
            tz = tzinfo_from_offset(tz_offset)
            last = state.storage.get_last_alert(user_id, "daily_check")
            if last and last.astimezone(tz).date() == now_local.date():
                continue
            report = await _build_controlla_report(state, user_id, city, coords)
            if not report:
                continue
            msg = "Verifica automatica di fine giornata\n\n" + report
            await context.bot.send_message(chat_id=user_id, text=msg, parse_mode="Markdown")
            state.storage.set_last_alert(user_id, "daily_check", now_utc())
        except Exception as exc:
            logger.error(f"Daily check error for user {uid}: {exc}")


async def tomorrow_rain_job(context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    users = state.storage.get_all_users()
    for uid in users:
        try:
            user_id = int(uid)
            prefs = state.storage.get_user_prefs(user_id)
            if not prefs.get("alert_tomorrow_rain", 1):
                continue
            city = state.storage.get_user_city(user_id)
            if not city:
                continue
            coords = await get_coords(state, city)
            if not coords:
                continue
            tz_offset = get_tz_offset_sec(None, None, None, coords=coords)
            now_local = local_now(tz_offset)
            if now_local.hour != TOMORROW_RAIN_HOUR or now_local.minute > TOMORROW_RAIN_WINDOW_MIN:
                continue
            last = state.storage.get_last_alert(user_id, "tomorrow_rain")
            tz = tzinfo_from_offset(tz_offset)
            if last and last.astimezone(tz).date() == now_local.date():
                continue
            _, _, _, points_ow, points_wa, points_mb, fused_points, target_date, _, _, _, _, _ = await _load_forecast_points(
                state, coords, 1
            )
            if not fused_points:
                continue
            max_pop = max((float(p.get("pop", 0)) for p in fused_points), default=0)
            if max_pop < prefs.get("pop_threshold", UMBRELLA_POP_THRESHOLD):
                continue
            periods = compute_rain_periods_threshold(fused_points, prefs.get("pop_threshold", UMBRELLA_POP_THRESHOLD))
            window = ""
            if periods:
                s = periods[0]["start"].strftime("%H:%M")
                e = periods[0]["end"].strftime("%H:%M")
                window = f" ({s}-{e})"
            text = f"Domani possibile pioggia a {coords['name']}: max {max_pop:.0f}%{window}"
            await context.bot.send_message(chat_id=user_id, text=text)
            state.storage.set_last_alert(user_id, "tomorrow_rain", now_utc())
        except Exception as exc:
            logger.error(f"Tomorrow rain error for user {uid}: {exc}")


async def morning_meteo_job(context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    users = state.storage.get_all_users()
    for uid in users:
        try:
            user_id = int(uid)
            city = state.storage.get_user_city(user_id)
            if not city:
                continue
            coords = await get_coords(state, city)
            if not coords:
                continue
            tz_offset = get_tz_offset_sec(None, None, None, coords=coords)
            now_local = local_now(tz_offset)
            if now_local.hour != MORNING_METEO_HOUR:
                continue
            tz = tzinfo_from_offset(tz_offset)
            last = state.storage.get_last_alert(user_id, "morning_meteo")
            if last and last.astimezone(tz).date() == now_local.date():
                continue
            forecast_ow, forecast_wa, forecast_mb, points_ow, points_wa, points_mb, fused_points, target_date, _, sources_label, cached_used, cache_stale, cache_age_min = await _load_forecast_points(
                state, coords, 0
            )
            if not fused_points:
                continue
            ths = get_dynamic_thresholds(state.storage)
            prefs = state.storage.get_user_prefs(user_id)
            rain_periods = compute_rain_periods_threshold(fused_points, ths.get("pop", RAIN_POP_THRESHOLD))
            max_pop = max((float(p.get("pop", 0)) for p in fused_points), default=0)
            rain_msg = summarize_rain_forecast(rain_periods, max_pop, now_local, False)
            min_t, max_t = extract_min_max(fused_points)
            enabled = [p for p, k in [("OpenWeather", OPENWEATHER_API_KEY), ("WeatherAPI", WEATHERAPI_KEY), ("Meteoblue", METEOBLUE_API_KEY)] if k]
            sources = []
            if points_ow:
                sources.append("OpenWeather")
            if points_wa:
                sources.append("WeatherAPI")
            if points_mb:
                sources.append("Meteoblue")
            correction_used = False
            sources_label = format_sources_label(sources, correction_used, [p for p in enabled if p not in sources])
            fallback_note = build_provider_note(enabled, sources, cache_stale=cache_stale, cache_age_min=cache_age_min)
            fused = await _get_current_fused_cached(state, coords["lat"], coords["lon"], coords)
            if not fused:
                continue
            msg = format_meteo_message(
                fused,
                coords["name"],
                coords["country"],
                min_t,
                max_t,
                rain_msg,
                sources_label,
                max_pop,
                cache_age_min,
                now_local,
                None,
                fallback_note,
                None,
                prefs=prefs,
                cache_stale=cache_stale,
            )
            await context.bot.send_message(chat_id=user_id, text=msg, parse_mode="Markdown")
            state.storage.set_last_alert(user_id, "morning_meteo", now_utc())

            if rain_periods:
                _schedule_rain_alerts(context, user_id, coords["name"], rain_periods[0], tz_offset)
        except Exception as exc:
            logger.error(f"Morning meteo error for user {uid}: {exc}")


async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: AppState = context.application.bot_data["state"]
    acc = state.storage.get_provider_accuracy()

    lines = []
    lines.append("\U0001F527 STATO SISTEMA")
    lines.append(f"{format_date_italian()} {format_time_italian()}")
    lines.append("-" * 40)
    uptime = format_uptime(now_utc() - BOT_STARTED_AT_UTC)
    lines.append(f"Versione bot: {BOT_VERSION}")
    lines.append(f"Uptime: {uptime}")
    try:
        users_count = len(state.storage.get_all_users())
        lines.append(f"Utenti registrati: {users_count}")
    except Exception:
        pass
    try:
        with state.storage._connect() as conn:
            cache_count = conn.execute("SELECT COUNT(*) AS n FROM cache").fetchone()["n"]
            pred_total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
            pred_ok = conn.execute("SELECT COUNT(*) AS n FROM predictions WHERE verified=1").fetchone()["n"]
            err_total = conn.execute("SELECT COUNT(*) AS n FROM errors_log").fetchone()["n"]
        db_size_kb = int(os.path.getsize(DB_FILE) / 1024) if os.path.exists(DB_FILE) else 0
        lines.append(f"Cache voci: {cache_count}")
        lines.append(f"Predizioni totali: {pred_total} (verificate {pred_ok})")
        lines.append(f"Errori registrati: {err_total}")
        lines.append(f"DB size: {db_size_kb} KB")
    except Exception:
        pass
    lines.append("")
    lines.append("\U0001F310 API")
    lines.append(f"- OpenWeather: {'OK' if OPENWEATHER_API_KEY else 'NO'}")
    lines.append(f"- WeatherAPI: {'OK' if WEATHERAPI_KEY else 'NO'}")
    lines.append(f"- Meteoblue: {'OK' if METEOBLUE_API_KEY else 'NO'}")
    try:
        rows = state.storage.get_provider_backoff_status()
        if rows:
            lines.append("")
            lines.append("\U0001F6A6 BACKOFF PROVIDER")
            now = now_utc()
            for r in rows:
                until = r["backoff_until"]
                if not until:
                    status = "OK"
                else:
                    until_dt = from_iso(until)
                    if now >= until_dt:
                        status = "OK"
                    else:
                        mins = int((until_dt - now).total_seconds() // 60)
                        status = f"PAUSA {mins}m"
                fail_count = int(r["fail_count"]) if r["fail_count"] is not None else 0
                last_err = r["last_error"] or "-"
                lines.append(f"- {r['provider']}: {status} (fail {fail_count}, last {last_err})")
    except Exception:
        pass
    lines.append("")
    lines.append("\U0001F4CA ACCURATEZZA PROVIDER")
    if not acc:
        lines.append("- Nessun dato ancora. Usa /meteo e poi /controlla.")
    else:
        for k, v in acc.items():
            lines.append(f"- {k}: {v['accuracy']:.1f}% (avg err {format_temp_delta(v['avg_error'], None)})")
    await update.effective_message.reply_text("\n".join(lines), parse_mode="Markdown")

async def versione(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = [
        f"Versione bot: {md_escape(BOT_VERSION)}",
        "Changelog (ultime versioni):",
        "",
    ]
    for rel in BOT_RELEASES[:3]:
        lines.append(f"Versione {md_escape(rel['version'])}:")
        for ch in rel["changes"]:
            lines.append(f"- {md_escape(ch)}")
        lines.append("")
    await update.effective_message.reply_text("\n".join(lines), parse_mode="Markdown")
