from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from config import (
    OUTLIER_TEMP_C,
    POP_CALIBRATION_DAYS,
    RAIN_POP_THRESHOLD,
    WIND_STRONG_KPH,
)
from providers import map_weatherapi_icon
from storage import Storage
from utils import (
    clamp,
    condition_group_from_icon,
    local_dt_from_ts,
    md5,
    meteoblue_description,
    meteoblue_icon_from_desc,
    now_utc,
    iso,
    tzinfo_from_offset,
)


def key_geo(city: str) -> str:
    return md5(f"geo:{city.strip().lower()}")


def key_current(lat: float, lon: float) -> str:
    bucket = datetime.now().hour // 2
    return md5(f"cur:{lat:.4f}:{lon:.4f}:{bucket}")


def key_forecast(lat: float, lon: float, days: int) -> str:
    bucket = datetime.now().hour // 2
    return md5(f"fc:{days}:{lat:.4f}:{lon:.4f}:{bucket}")


def extract_day_points(forecast: Optional[Dict[str, Any]], target_date, tz_offset_sec: int = 0) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    if not forecast or "list" not in forecast:
        return points
    for it in forecast["list"]:
        t_local = local_dt_from_ts(int(it["dt"]), tz_offset_sec)
        if t_local.date() == target_date:
            main = it.get("main", {})
            points.append({
                "dt_utc": datetime.fromtimestamp(it["dt"], tz=timezone.utc),
                "local_time": t_local,
                "ora": t_local.strftime("%H:%M"),
                "temp": float(main.get("temp", 0.0)),
                "temp_min": float(main.get("temp_min", main.get("temp", 0.0))),
                "temp_max": float(main.get("temp_max", main.get("temp", 0.0))),
                "icon": it.get("weather", [{}])[0].get("icon", "01d"),
                "pop": float(it.get("pop", 0)) * 100,
                "wind_kph": float(it.get("wind", {}).get("speed", 0.0)) * 3.6 if it.get("wind", {}).get("speed") is not None else None,
                "humidity": float(main.get("humidity")) if main.get("humidity") is not None else None,
                "pressure": float(main.get("pressure")) if main.get("pressure") is not None else None,
                "step_hours": 3,
            })
    return points


def extract_min_max(points: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not points:
        return None, None
    min_t = min(p["temp_min"] for p in points)
    max_t = max(p["temp_max"] for p in points)
    return min_t, max_t


def extract_day_points_weatherapi(forecast: Optional[Dict[str, Any]], target_date, tz_offset_sec: int = 0) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    if not forecast:
        return points
    forecast_days = forecast.get("forecast", {}).get("forecastday", [])
    for day in forecast_days:
        day_date = local_dt_from_ts(int(day.get("date_epoch", 0)), tz_offset_sec).date()
        if day_date != target_date:
            continue
        for hour in day.get("hour", []):
            t_local = local_dt_from_ts(int(hour.get("time_epoch", 0)), tz_offset_sec)
            code = int(hour.get("condition", {}).get("code", 1000))
            pop = hour.get("chance_of_rain", 0)
            try:
                pop_val = float(pop)
            except Exception:
                pop_val = 0.0
            points.append({
                "dt_utc": datetime.fromtimestamp(hour.get("time_epoch", 0), tz=timezone.utc),
                "local_time": t_local,
                "ora": t_local.strftime("%H:%M"),
                "temp": float(hour.get("temp_c", 0.0)),
                "temp_min": float(hour.get("temp_c", 0.0)),
                "temp_max": float(hour.get("temp_c", 0.0)),
                "icon": map_weatherapi_icon(code),
                "pop": pop_val,
                "wind_kph": float(hour.get("wind_kph")) if hour.get("wind_kph") is not None else None,
                "humidity": float(hour.get("humidity")) if hour.get("humidity") is not None else None,
                "pressure": float(hour.get("pressure_mb")) if hour.get("pressure_mb") is not None else None,
                "step_hours": 1,
            })
    return points


def extract_day_points_weatherapi_history(history: Optional[Dict[str, Any]], target_date, tz_offset_sec: int = 0) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    if not history:
        return points
    forecast_days = history.get("forecast", {}).get("forecastday", [])
    for day in forecast_days:
        day_date = local_dt_from_ts(int(day.get("date_epoch", 0)), tz_offset_sec).date()
        if day_date != target_date:
            continue
        for hour in day.get("hour", []):
            t_local = local_dt_from_ts(int(hour.get("time_epoch", 0)), tz_offset_sec)
            code = int(hour.get("condition", {}).get("code", 1000))
            desc = str(hour.get("condition", {}).get("text", ""))
            precip = float(hour.get("precip_mm", 0.0)) if hour.get("precip_mm") is not None else 0.0
            points.append({
                "dt_utc": datetime.fromtimestamp(hour.get("time_epoch", 0), tz=timezone.utc),
                "local_time": t_local,
                "ora": t_local.strftime("%H:%M"),
                "temp": float(hour.get("temp_c", 0.0)),
                "temp_min": float(hour.get("temp_c", 0.0)),
                "temp_max": float(hour.get("temp_c", 0.0)),
                "icon": map_weatherapi_icon(code),
                "pop": float(hour.get("chance_of_rain", 0)) if hour.get("chance_of_rain") is not None else 0.0,
                "step_hours": 1,
                "desc": desc,
                "precip_mm": precip,
                "wind_kph": float(hour.get("wind_kph")) if hour.get("wind_kph") is not None else None,
                "humidity": float(hour.get("humidity")) if hour.get("humidity") is not None else None,
                "pressure_mb": float(hour.get("pressure_mb")) if hour.get("pressure_mb") is not None else None,
            })
    return points


def _parse_meteoblue_time(ts: str, tz_offset_sec: int) -> datetime:
    dt = None
    try:
        if isinstance(ts, str) and ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
    except Exception:
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
        except Exception:
            dt = datetime.utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tzinfo_from_offset(tz_offset_sec))
    return dt.astimezone(tzinfo_from_offset(tz_offset_sec))


def extract_day_points_meteoblue(forecast: Optional[Dict[str, Any]], target_date, tz_offset_sec: int = 0) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    if not forecast:
        return points
    data = forecast.get("data_1h") or forecast
    times = data.get("time") or []
    temps = data.get("temperature") or []
    feels = data.get("felttemperature") or []
    pops = data.get("precipitation_probability") or []
    precs = data.get("precipitation") or []
    pictos = data.get("pictocode") or data.get("pictocode_detailed") or []
    winds = data.get("windspeed") or []
    hums = data.get("relativehumidity") or []
    press = data.get("sealevelpressure") or []
    for i, t in enumerate(times):
        t_local = _parse_meteoblue_time(str(t), tz_offset_sec)
        if t_local.date() != target_date:
            continue
        temp = float(temps[i]) if i < len(temps) and temps[i] is not None else 0.0
        pop_val = 0.0
        if i < len(pops) and pops[i] is not None:
            try:
                pop_val = float(pops[i])
            except Exception:
                pop_val = 0.0
        elif i < len(precs) and precs[i] is not None:
            pop_val = 60.0 if float(precs[i]) > 0 else 0.0
        precip_mm = float(precs[i]) if i < len(precs) and precs[i] is not None else None
        code = int(pictos[i]) if i < len(pictos) and pictos[i] is not None else None
        desc = meteoblue_description(code)
        is_day = 6 <= t_local.hour <= 19
        icon = meteoblue_icon_from_desc(desc, is_day=is_day)
        points.append({
            "dt_utc": t_local.astimezone(timezone.utc),
            "local_time": t_local,
            "ora": t_local.strftime("%H:%M"),
            "temp": temp,
            "temp_min": temp,
            "temp_max": temp,
            "icon": icon,
            "pop": float(pop_val),
            "desc": desc,
            "precip_mm": precip_mm,
            "wind_kph": float(winds[i]) if i < len(winds) and winds[i] is not None else None,
            "humidity": float(hums[i]) if i < len(hums) and hums[i] is not None else None,
            "pressure": float(press[i]) if i < len(press) and press[i] is not None else None,
            "step_hours": 1,
        })
    return points


def interpolate_hourly_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not points:
        return points
    points_sorted = sorted(points, key=lambda x: x["dt_utc"])
    if all(int(p.get("step_hours", 1)) == 1 for p in points_sorted):
        return points_sorted
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(points_sorted):
        base = dict(p)
        base["step_hours"] = 1
        out.append(base)
        if i + 1 >= len(points_sorted):
            continue
        n = points_sorted[i + 1]
        gap = int((n["dt_utc"] - p["dt_utc"]).total_seconds() // 3600)
        if gap <= 1:
            continue
        for h in range(1, gap):
            frac = h / gap
            new_dt = p["dt_utc"] + timedelta(hours=h)
            new_local = p["local_time"] + timedelta(hours=h)
            temp = p["temp"] + (n["temp"] - p["temp"]) * frac
            tmin = p["temp_min"] + (n["temp_min"] - p["temp_min"]) * frac
            tmax = p["temp_max"] + (n["temp_max"] - p["temp_max"]) * frac
            pop = p["pop"] + (n["pop"] - p["pop"]) * frac
            icon = p["icon"] if frac < 0.5 else n["icon"]
            out.append({
                "dt_utc": new_dt,
                "local_time": new_local,
                "ora": new_local.strftime("%H:%M"),
                "temp": float(temp),
                "temp_min": float(tmin),
                "temp_max": float(tmax),
                "icon": icon,
                "pop": float(pop),
                "step_hours": 1,
            })
    return sorted(out, key=lambda x: x["dt_utc"])


def compute_rain_periods(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return compute_rain_periods_threshold(points, RAIN_POP_THRESHOLD)


def compute_rain_periods_threshold(points: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    periods: List[Dict[str, Any]] = []
    cur_s = cur_e = None
    cur_m = 0.0
    for p in sorted(points, key=lambda x: x["dt_utc"]):
        if p["pop"] > threshold:
            if cur_s is None:
                cur_s = p["local_time"]
                cur_m = p["pop"]
            step = int(p.get("step_hours", 1))
            cur_e = p["local_time"] + timedelta(hours=step)
            cur_m = max(cur_m, p["pop"])
        else:
            if cur_s is not None:
                periods.append({"start": cur_s, "end": cur_e, "max_prob": cur_m})
                cur_s = cur_e = None
                cur_m = 0.0
    if cur_s is not None:
        periods.append({"start": cur_s, "end": cur_e, "max_prob": cur_m})
    return periods


def summarize_rain_forecast(
    periods: List[Dict[str, Any]],
    max_pop: float,
    now_local: datetime,
    raining_now: bool = False
) -> str:
    chosen = None
    for p in periods:
        if p["start"] <= now_local <= p["end"]:
            chosen = p
            break
    if not chosen:
        future = [p for p in periods if p["start"] > now_local]
        chosen = future[0] if future else None
    if chosen:
        start_dt = max(chosen["start"], now_local)
        s = start_dt.strftime("%H:%M")
        e = chosen["end"].strftime("%H:%M")
        return f"Prevista pioggia da {s} a {e}"
    if max_pop > 0 or raining_now:
        return "Possibile pioggia"
    return "Nessuna pioggia prevista"


def get_dynamic_thresholds(storage: Storage, days: int = 45) -> Dict[str, float]:
    pop_th = float(RAIN_POP_THRESHOLD)
    wind_th = float(WIND_STRONG_KPH)
    cutoff = now_utc() - timedelta(days=int(days))
    try:
        with storage._connect() as conn:
            rows = conn.execute(
                "SELECT predicted_pop, actual_rain, verified_at_utc FROM predictions "
                "WHERE verified=1 AND predicted_pop IS NOT NULL AND actual_rain IS NOT NULL "
                "AND verified_at_utc >= ?",
                (iso(cutoff),)
            ).fetchall()
        if rows:
            briers = []
            hits = 0
            for r in rows:
                p = float(r["predicted_pop"]) / 100.0
                a = 1.0 if int(r["actual_rain"]) else 0.0
                briers.append((p - a) ** 2)
                if (p >= 0.5) == (a == 1.0):
                    hits += 1
            brier = sum(briers) / len(briers) if briers else None
            hit_rate = hits / len(rows) if rows else None
            if brier is not None and brier > 0.25:
                pop_th += 10
            elif brier is not None and brier < 0.15:
                pop_th -= 5
            if hit_rate is not None and hit_rate < 0.6:
                pop_th += 5
        with storage._connect() as conn:
            rows = conn.execute(
                "SELECT error_wind FROM predictions "
                "WHERE verified=1 AND error_wind IS NOT NULL AND verified_at_utc >= ?",
                (iso(cutoff),)
            ).fetchall()
        if rows:
            mae = sum(abs(float(r["error_wind"])) for r in rows) / len(rows)
            if mae > 10:
                wind_th += 5
            elif mae < 5:
                wind_th -= 5
    except Exception:
        pass
    pop_th = clamp(pop_th, 20, 70)
    wind_th = clamp(wind_th, 20, 60)
    return {"pop": pop_th, "wind": wind_th}


def build_pop_calibrator(storage: Storage, days: int = POP_CALIBRATION_DAYS, min_samples: int = 80):
    cutoff = now_utc() - timedelta(days=int(days))
    with storage._connect() as conn:
        rows = conn.execute(
            "SELECT predicted_pop, actual_rain FROM predictions "
            "WHERE verified=1 AND predicted_pop IS NOT NULL AND actual_rain IS NOT NULL "
            "AND verified_at_utc >= ? "
            "ORDER BY predicted_pop ASC",
            (iso(cutoff),)
        ).fetchall()
    if not rows or len(rows) < min_samples:
        return None
    pairs = []
    for r in rows:
        try:
            x = clamp(float(r["predicted_pop"]) / 100.0, 0.0, 1.0)
            y = 1.0 if int(r["actual_rain"]) else 0.0
            pairs.append((x, y))
        except Exception:
            continue
    if len(pairs) < min_samples:
        return None
    pairs.sort(key=lambda t: t[0])
    blocks: List[Dict[str, float]] = []
    for x, y in pairs:
        blocks.append({"min_x": x, "max_x": x, "sum_y": y, "n": 1, "avg": y})
        while len(blocks) >= 2 and blocks[-2]["avg"] > blocks[-1]["avg"]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            n = b1["n"] + b2["n"]
            s = b1["sum_y"] + b2["sum_y"]
            merged = {
                "min_x": min(b1["min_x"], b2["min_x"]),
                "max_x": max(b1["max_x"], b2["max_x"]),
                "sum_y": s,
                "n": n,
                "avg": s / n if n else 0.0,
            }
            blocks.append(merged)

    def _calibrate(pct: float) -> float:
        if pct is None:
            return pct
        x = clamp(float(pct) / 100.0, 0.0, 1.0)
        for b in blocks:
            if x <= b["max_x"]:
                return clamp(b["avg"] * 100.0, 0.0, 100.0)
        return clamp(blocks[-1]["avg"] * 100.0, 0.0, 100.0)

    return _calibrate


def apply_pop_calibration(points: List[Dict[str, Any]], calibrator) -> List[Dict[str, Any]]:
    if not points or calibrator is None:
        return points
    out: List[Dict[str, Any]] = []
    for p in points:
        np = dict(p)
        if np.get("pop") is not None:
            np["pop_raw"] = np.get("pop")
            np["pop"] = float(calibrator(np["pop"]))
        out.append(np)
    return out


def pick_point_for_hour(points: List[Dict[str, Any]], target_hour: int, max_diff: int) -> Optional[Dict[str, Any]]:
    best = None
    best_diff = 999
    for p in points:
        ph = int(p["ora"].split(":")[0])
        diff = abs(ph - target_hour)
        if diff < best_diff and diff <= max_diff:
            best_diff = diff
            best = p
    return best


def build_fused_points(
    points_ow: List[Dict[str, Any]],
    points_wa: List[Dict[str, Any]],
    points_mb: Optional[List[Dict[str, Any]]],
    weights: Dict[str, float],
    bias: Dict[str, float],
    target_date,
    tz_offset_sec: int = 0,
    bias_func: Optional[Any] = None,
    weight_func: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    fused: List[Dict[str, Any]] = []
    w_ow = float(weights.get("OpenWeather", 0.33))
    w_wa = float(weights.get("WeatherAPI", 0.33))
    w_mb = float(weights.get("Meteoblue", 0.34))
    ow_bias = float(bias.get("OpenWeather", 0.0))
    wa_bias = float(bias.get("WeatherAPI", 0.0))
    mb_bias = float(bias.get("Meteoblue", 0.0))
    for hour in range(0, 24):
        ow_p = pick_point_for_hour(points_ow, hour, 2)
        wa_p = pick_point_for_hour(points_wa, hour, 1)
        mb_p = pick_point_for_hour(points_mb, hour, 1) if points_mb else None
        if not ow_p and not wa_p and not mb_p:
            continue
        cond = None
        w_ow_h, w_wa_h, w_mb_h = w_ow, w_wa, w_mb
        if weight_func:
            w = weight_func(hour, ow_p, wa_p, mb_p)
            if isinstance(w, dict):
                w_ow_h = float(w.get("OpenWeather", w_ow_h))
                w_wa_h = float(w.get("WeatherAPI", w_wa_h))
                w_mb_h = float(w.get("Meteoblue", w_mb_h))
            else:
                try:
                    if len(w) == 3:
                        w_ow_h, w_wa_h, w_mb_h = w
                    elif len(w) == 2:
                        w_ow_h, w_wa_h = w
                except Exception:
                    pass
        if ow_p:
            cond = condition_group_from_icon(str(ow_p["icon"]), float(ow_p["pop"]))
        elif wa_p:
            cond = condition_group_from_icon(str(wa_p["icon"]), float(wa_p["pop"]))
        elif mb_p:
            cond = condition_group_from_icon(str(mb_p["icon"]), float(mb_p["pop"]))
        if bias_func and cond:
            if ow_p:
                ow_bias = float(bias_func("OpenWeather", hour, cond))
            if wa_p:
                wa_bias = float(bias_func("WeatherAPI", hour, cond))
            if mb_p:
                mb_bias = float(bias_func("Meteoblue", hour, cond))
        weights_map = {}
        if ow_p:
            weights_map["OpenWeather"] = w_ow_h
        if wa_p:
            weights_map["WeatherAPI"] = w_wa_h
        if mb_p:
            weights_map["Meteoblue"] = w_mb_h
        tot = sum(weights_map.values())
        if tot <= 0:
            n = len(weights_map)
            weights_map = {k: 1 / n for k in weights_map}
        else:
            weights_map = {k: v / tot for k, v in weights_map.items()}
        temps_map = {}
        if ow_p:
            temps_map["OpenWeather"] = ow_p["temp"]
        if wa_p:
            temps_map["WeatherAPI"] = wa_p["temp"]
        if mb_p:
            temps_map["Meteoblue"] = mb_p["temp"]
        if len(temps_map) >= 2:
            vals = sorted(temps_map.values())
            if len(vals) % 2 == 1:
                median = vals[len(vals) // 2]
            else:
                median = sum(vals[len(vals) // 2 - 1:len(vals) // 2 + 1]) / 2
            for k, v in list(weights_map.items()):
                if k in temps_map and abs(float(temps_map[k]) - float(median)) >= OUTLIER_TEMP_C:
                    weights_map[k] = v * 0.2
            tot = sum(weights_map.values())
            if tot > 0:
                weights_map = {k: v / tot for k, v in weights_map.items()}
        temp = 0.0
        pop = 0.0
        wind = 0.0
        humidity = 0.0
        pressure = 0.0
        wind_w = hum_w = pres_w = 0.0
        for name, wgt in weights_map.items():
            if name == "OpenWeather" and ow_p:
                temp += (ow_p["temp"] - ow_bias) * wgt
                pop += float(ow_p["pop"]) * wgt
                if ow_p.get("wind_kph") is not None:
                    wind += float(ow_p["wind_kph"]) * wgt
                    wind_w += wgt
                if ow_p.get("humidity") is not None:
                    humidity += float(ow_p["humidity"]) * wgt
                    hum_w += wgt
                if ow_p.get("pressure") is not None:
                    pressure += float(ow_p["pressure"]) * wgt
                    pres_w += wgt
            elif name == "WeatherAPI" and wa_p:
                temp += (wa_p["temp"] - wa_bias) * wgt
                pop += float(wa_p["pop"]) * wgt
                if wa_p.get("wind_kph") is not None:
                    wind += float(wa_p["wind_kph"]) * wgt
                    wind_w += wgt
                if wa_p.get("humidity") is not None:
                    humidity += float(wa_p["humidity"]) * wgt
                    hum_w += wgt
                if wa_p.get("pressure") is not None:
                    pressure += float(wa_p["pressure"]) * wgt
                    pres_w += wgt
            elif name == "Meteoblue" and mb_p:
                temp += (mb_p["temp"] - mb_bias) * wgt
                pop += float(mb_p["pop"]) * wgt
                if mb_p.get("wind_kph") is not None:
                    wind += float(mb_p["wind_kph"]) * wgt
                    wind_w += wgt
                if mb_p.get("humidity") is not None:
                    humidity += float(mb_p["humidity"]) * wgt
                    hum_w += wgt
                if mb_p.get("pressure") is not None:
                    pressure += float(mb_p["pressure"]) * wgt
                    pres_w += wgt
        top_provider = max(weights_map.items(), key=lambda x: x[1])[0]
        if top_provider == "OpenWeather":
            icon = ow_p["icon"]
        elif top_provider == "WeatherAPI":
            icon = wa_p["icon"]
        else:
            icon = mb_p["icon"]
        sources = list(weights_map.keys())
        tz = tzinfo_from_offset(tz_offset_sec)
        local_time = datetime.combine(target_date, datetime.min.time()).replace(hour=hour).replace(tzinfo=tz)
        fused.append({
            "dt_utc": local_time.astimezone(timezone.utc),
            "local_time": local_time,
            "ora": f"{hour:02d}:00",
            "temp": float(temp),
            "temp_min": float(temp),
            "temp_max": float(temp),
            "icon": icon,
            "pop": float(pop),
            "wind_kph": (wind / wind_w) if wind_w > 0 else None,
            "humidity": (humidity / hum_w) if hum_w > 0 else None,
            "pressure": (pressure / pres_w) if pres_w > 0 else None,
            "step_hours": 1,
            "sources": sources,
        })
    return fused
