from __future__ import annotations

import json
import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, Optional, List, Tuple, TYPE_CHECKING
from zoneinfo import ZoneInfo

try:
    import tornado.web
    from telegram.ext._utils.webhookhandler import TelegramHandler
except Exception:
    tornado = None
    TelegramHandler = None

import config
from config import (
    ADMIN_SECRET,
    ADMIN_USER_IDS,
    BOT_VERSION,
    CACHE_TTL_FORECAST_MIN,
    DEFAULT_ALERT_END_HOUR,
    DEFAULT_ALERT_START_HOUR,
    DEFAULT_TEMP_UNIT,
    DEFAULT_WIND_UNIT,
    NIGHT_FORECAST_TTL_MULT,
    PROVIDERS,
    RAIN_KEYWORDS,
    RAIN_POP_THRESHOLD,
    UMBRELLA_POP_THRESHOLD,
    OUTLIER_TEMP_C,
)

if TYPE_CHECKING:
    from providers import ProviderResult


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("meteo-bot")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


BOT_STARTED_AT_UTC = now_utc()


def iso(dt: datetime) -> str:
    return dt.isoformat()


def from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def md_escape(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text)
    for ch in ("\\", "_", "*", "`", "[", "]"):
        s = s.replace(ch, "\\" + ch)
    return s


def _parse_admin_ids(raw: str) -> set:
    out = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out


ADMIN_IDS_SET = _parse_admin_ids(ADMIN_USER_IDS)


def is_admin(user_id: int, token: Optional[str] = None) -> bool:
    if ADMIN_IDS_SET and user_id in ADMIN_IDS_SET:
        return True
    if ADMIN_SECRET and token:
        if token.strip().lower() == ADMIN_SECRET.strip().lower():
            return True
    return False


class HealthHandler(tornado.web.RequestHandler if tornado else object):
    def initialize(self, payload: Dict[str, Any]) -> None:
        self.payload = payload

    def get(self) -> None:
        self.set_header("Content-Type", 'application/json; charset="utf-8"')
        self.write(json.dumps(self.payload))


class CustomWebhookApp(tornado.web.Application if tornado else object):
    def __init__(
        self,
        webhook_path: str,
        bot: Any,
        update_queue: Any,
        secret_token: str | None = None,
    ):
        if not webhook_path.startswith("/"):
            webhook_path = f"/{webhook_path}"
        health_path = config.HEALTH_PATH_EFFECTIVE
        if not health_path.startswith("/"):
            health_path = f"/{health_path}"
        payload = {"status": "ok", "version": BOT_VERSION, "ts": iso(now_utc())}
        shared = {"bot": bot, "update_queue": update_queue, "secret_token": secret_token}
        handlers = [
            (rf"{webhook_path}/?", TelegramHandler, shared),
            (rf"{health_path}/?", HealthHandler, {"payload": payload}),
        ]
        tornado.web.Application.__init__(self, handlers)  # type: ignore


def normalize_temp_unit(unit: Optional[str]) -> str:
    if not unit:
        return DEFAULT_TEMP_UNIT
    u = str(unit).strip().lower()
    return "F" if u in {"f", "fahrenheit"} else "C"


def normalize_wind_unit(unit: Optional[str]) -> str:
    if not unit:
        return DEFAULT_WIND_UNIT
    u = str(unit).strip().lower()
    return "mph" if u in {"mph", "mi/h"} else "kmh"


def temp_to_unit(temp_c: float, unit: str) -> float:
    return (temp_c * 9 / 5) + 32 if unit == "F" else temp_c


def temp_delta_to_unit(delta_c: float, unit: str) -> float:
    return delta_c * 9 / 5 if unit == "F" else delta_c


def wind_to_unit(wind_kph: float, unit: str) -> float:
    return wind_kph * 0.621371 if unit == "mph" else wind_kph


def format_temp(value_c: Optional[float], prefs: Optional[Dict[str, Any]] = None, decimals: int = 1) -> str:
    if value_c is None:
        return "n/d"
    unit = normalize_temp_unit(prefs.get("temp_unit") if prefs else None)
    val = temp_to_unit(float(value_c), unit)
    suffix = "F°" if unit == "F" else "C°"
    return f"{val:.{decimals}f} {suffix}"


def format_temp_delta(value_c: Optional[float], prefs: Optional[Dict[str, Any]] = None, decimals: int = 1) -> str:
    if value_c is None:
        return "n/d"
    unit = normalize_temp_unit(prefs.get("temp_unit") if prefs else None)
    val = temp_delta_to_unit(float(value_c), unit)
    suffix = "F°" if unit == "F" else "C°"
    return f"{val:.{decimals}f} {suffix}"


def format_wind(value_kph: Optional[float], prefs: Optional[Dict[str, Any]] = None, decimals: int = 0) -> str:
    if value_kph is None:
        return "n/d"
    unit = normalize_wind_unit(prefs.get("wind_unit") if prefs else None)
    val = wind_to_unit(float(value_kph), unit)
    suffix = "mph" if unit == "mph" else "km/h"
    return f"{val:.{decimals}f} {suffix}"


def format_alert_window(prefs: Dict[str, Any]) -> str:
    start = prefs.get("alert_start_hour", DEFAULT_ALERT_START_HOUR)
    end = prefs.get("alert_end_hour", DEFAULT_ALERT_END_HOUR)
    if start == -1 and end == -1:
        return "OFF"
    if start == end:
        return "24h"
    return f"{int(start):02d}-{int(end):02d}"


def parse_alert_window(value: str) -> Optional[Tuple[int, int]]:
    v = value.strip().lower()
    if v in {"off", "no", "false", "0"}:
        return -1, -1
    if v in {"all", "24", "24h"}:
        return 0, 0
    if "-" not in v:
        return None
    left, right = v.split("-", 1)
    try:
        start = int(left.strip())
        end = int(right.strip())
    except Exception:
        return None
    if not (0 <= start <= 23 and 0 <= end <= 23):
        return None
    return start, end


def within_alert_window(now_local: datetime, prefs: Dict[str, Any]) -> bool:
    start = int(prefs.get("alert_start_hour", DEFAULT_ALERT_START_HOUR))
    end = int(prefs.get("alert_end_hour", DEFAULT_ALERT_END_HOUR))
    if start == -1 and end == -1:
        return False
    if start == end:
        return True
    hour = now_local.hour
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def format_date_italian(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = datetime.now()
    giorni = ['Lunedi', 'Martedi', 'Mercoledi', 'Giovedi', 'Venerdi', 'Sabato', 'Domenica']
    mesi = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno', 'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']
    return f"{giorni[dt.weekday()]} {dt.day} {mesi[dt.month - 1]}"


def format_time_italian(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%H:%M")


def format_uptime(td: timedelta) -> str:
    total = int(td.total_seconds())
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{mins}m")
    return " ".join(parts)


def forecast_ttl_minutes(local_dt: Optional[datetime]) -> int:
    if not local_dt:
        return CACHE_TTL_FORECAST_MIN
    hour = local_dt.hour
    if 0 <= hour < 6:
        return int(CACHE_TTL_FORECAST_MIN * NIGHT_FORECAST_TTL_MULT)
    return CACHE_TTL_FORECAST_MIN


def cache_age_min_from_payload(payload: Optional[Dict[str, Any]], created_at: Optional[datetime] = None) -> Optional[int]:
    if not payload:
        return None
    saved_at = payload.get("saved_at")
    if saved_at:
        try:
            dt = from_iso(saved_at)
        except Exception:
            dt = created_at
    else:
        dt = created_at
    if not dt:
        return None
    return int((now_utc() - dt).total_seconds() // 60)


def build_provider_note(
    enabled: List[str],
    available: List[str],
    cache_stale: bool = False,
    cache_age_min: Optional[int] = None,
) -> Optional[str]:
    missing = [p for p in enabled if p not in available]
    parts = []
    if missing:
        parts.append(f"Provider non disponibili: {', '.join(missing)}")
    if cache_stale:
        age_txt = f" ({cache_age_min}m)" if cache_age_min is not None else ""
        parts.append(f"Dati cache non aggiornati{age_txt}")
    return ". ".join(parts) if parts else None


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": iso(now_utc())}
    payload.update({k: v for k, v in fields.items() if v is not None})
    logger.info(json.dumps(payload, separators=(",", ":")))


def season_from_date(d: date) -> str:
    m = d.month
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


def zone_bucket(lat: float, lon: float, step: float = 0.5) -> str:
    def _round(x: float) -> float:
        return round(x / step) * step
    return f"{_round(lat):.1f},{_round(lon):.1f}"


def get_weather_icon(icon_code: str, pop: Optional[float] = None, is_day: Optional[bool] = None) -> str:
    icon_map = {
        "01d": "\u2600\ufe0f", "01n": "\U0001F319", "02d": "\u26c5", "02n": "\u26c5",
        "03d": "\u2601\ufe0f", "03n": "\u2601\ufe0f", "04d": "\u2601\ufe0f", "04n": "\u2601\ufe0f",
        "09d": "\U0001F327\ufe0f", "09n": "\U0001F327\ufe0f", "10d": "\U0001F326\ufe0f", "10n": "\U0001F327\ufe0f",
        "11d": "\u26c8\ufe0f", "11n": "\u26c8\ufe0f", "13d": "\u2744\ufe0f", "13n": "\u2744\ufe0f",
        "50d": "\U0001F32B\ufe0f", "50n": "\U0001F32B\ufe0f"
    }
    code = str(icon_code or "")
    pop_val: Optional[float] = None
    if pop is not None:
        try:
            pop_val = float(pop)
        except Exception:
            pop_val = None
    if code in icon_map:
        if pop_val is not None and code.startswith(("09", "10", "11", "13")) and pop_val < UMBRELLA_POP_THRESHOLD:
            return "\u2601\ufe0f"
        return icon_map[code]
    if is_day is None:
        is_day = not code.endswith("n")
    if code.startswith(("09", "10", "11", "13")) and pop_val is not None and pop_val < UMBRELLA_POP_THRESHOLD:
        return "\u2601\ufe0f"
    if code.startswith("11"):
        return "\u26c8\ufe0f"
    if code.startswith("13"):
        return "\u2744\ufe0f"
    if code.startswith("50"):
        return "\U0001F32B\ufe0f"
    if code.startswith(("09", "10")):
        return "\U0001F327\ufe0f"
    if code.startswith(("02", "03", "04")):
        return "\u26c5" if code.startswith("02") and is_day else "\u2601\ufe0f"
    if code.startswith("01"):
        return "\u2600\ufe0f" if is_day else "\U0001F319"
    return "\U0001F321\ufe0f"


def get_rain_risk_icon(
    pop: Optional[float],
    icon_code: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    if pop is None:
        return ""
    try:
        pop_val = float(pop)
    except Exception:
        return ""
    if pop_val <= 0:
        return ""
    group = ""
    if description:
        group = condition_group_from_description(str(description))
    if not group and icon_code:
        group = condition_group_from_icon(str(icon_code), pop_val)
    if group in {"neve", "nebbia"}:
        return ""
    if pop_val >= UMBRELLA_POP_THRESHOLD:
        return "\U0001F327\ufe0f"
    if pop_val >= RAIN_POP_THRESHOLD:
        return "\u2614"
    return ""


METEOBLUE_PICTO_HOURLY: Dict[int, str] = {
    1: "Clear, cloudless sky",
    2: "Clear, few cirrus",
    3: "Clear, some cirrus",
    4: "Clear, cirrus",
    5: "Clear with cirrus",
    6: "Mostly clear",
    7: "Partly cloudy",
    8: "Partly cloudy and cirrus",
    9: "Mostly cloudy",
    10: "Cloudy",
    11: "Overcast",
    12: "Overcast with low clouds",
    13: "Fog",
    14: "Fog, sky visible",
    15: "Fog, sky not visible",
    16: "Fog, sky visible",
    17: "Fog, sky not visible",
    18: "Very light precipitation",
    19: "Light precipitation",
    20: "Precipitation",
    21: "Moderate precipitation",
    22: "Heavy precipitation",
    23: "Very heavy precipitation",
    24: "Extreme precipitation",
    25: "Thunderstorm",
    26: "Thunderstorm with hail",
    27: "Slightly dusty",
    28: "Dusty",
    29: "Very dusty",
    30: "Slightly sandy",
    31: "Sandy",
    32: "Very sandy",
    33: "Light rain",
    34: "Rain",
    35: "Heavy rain",
}


def meteoblue_description(code: Optional[int]) -> str:
    if code is None:
        return ""
    return METEOBLUE_PICTO_HOURLY.get(int(code), "")


def meteoblue_icon_from_desc(desc: str, is_day: bool = True) -> str:
    d = (desc or "").lower()
    if any(k in d for k in ["thunder", "storm", "hail"]):
        base = "11"
    elif "snow" in d:
        base = "13"
    elif any(k in d for k in ["rain", "drizzle", "precipitation", "shower"]):
        base = "10" if "light" in d else "09"
    elif any(k in d for k in ["fog", "mist", "haze"]):
        base = "50"
    elif any(k in d for k in ["overcast", "cloudy"]):
        base = "04"
    elif any(k in d for k in ["partly", "mostly", "few clouds", "some clouds", "cirrus"]):
        base = "03"
    elif any(k in d for k in ["clear", "sunny"]):
        base = "01"
    else:
        base = "01"
    return f"{base}{'d' if is_day else 'n'}"


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def is_rain_description(desc: str) -> bool:
    d = desc.lower().strip()
    return any(k in d for k in RAIN_KEYWORDS)


def hour_band(hour: int) -> str:
    if 0 <= hour <= 5:
        return "night"
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    return "evening"


def condition_group_from_description(desc: str) -> str:
    d = desc.lower()
    if any(k in d for k in ["temporale", "thunder", "storm"]):
        return "temporale"
    if any(k in d for k in ["neve", "snow"]):
        return "neve"
    if any(k in d for k in ["nebbia", "foschia", "mist", "fog"]):
        return "nebbia"
    if any(k in d for k in ["pioggia", "rovesci", "acquazzone", "drizzle", "rain", "shower", "precipitation"]):
        return "pioggia"
    if any(k in d for k in ["nubi", "nuvol", "cloud"]):
        return "nuvoloso"
    return "sereno"


def condition_group_from_icon(icon_code: str, pop: float) -> str:
    if pop >= RAIN_POP_THRESHOLD:
        return "pioggia"
    if icon_code.startswith("11"):
        return "temporale"
    if icon_code.startswith("13"):
        return "neve"
    if icon_code.startswith("50"):
        return "nebbia"
    if icon_code.startswith(("09", "10")):
        return "pioggia"
    if icon_code.startswith("01"):
        return "sereno"
    if icon_code.startswith(("02", "03", "04")):
        return "nuvoloso"
    return "sereno"


def rain_intensity_label(pop: float) -> str:
    if pop >= 70:
        return "forte"
    if pop >= 50:
        return "moderata"
    if pop >= 30:
        return "debole"
    return "debole"


def get_tz_offset_sec(
    forecast_ow: Optional[Dict[str, Any]],
    forecast_wa: Optional[Dict[str, Any]],
    forecast_mb: Optional[Dict[str, Any]] = None,
    ow_cur: Optional["ProviderResult"] = None,
    wa_cur: Optional["ProviderResult"] = None,
    mb_cur: Optional["ProviderResult"] = None,
    coords: Optional[Dict[str, Any]] = None,
) -> int:
    if coords and coords.get("tz_offset_sec") is not None:
        return int(coords["tz_offset_sec"])
    if forecast_ow and forecast_ow.get("city", {}).get("timezone") is not None:
        return int(forecast_ow["city"]["timezone"])
    if forecast_wa and forecast_wa.get("location", {}).get("localtime_epoch") is not None:
        local_epoch = int(forecast_wa["location"]["localtime_epoch"])
        return int(local_epoch - int(now_utc().timestamp()))
    if forecast_mb and forecast_mb.get("metadata", {}).get("utc_timeoffset") is not None:
        return int(float(forecast_mb["metadata"]["utc_timeoffset"]) * 3600)
    if ow_cur and ow_cur.tz_offset_sec is not None:
        return int(ow_cur.tz_offset_sec)
    if wa_cur and wa_cur.tz_offset_sec is not None:
        return int(wa_cur.tz_offset_sec)
    if mb_cur and mb_cur.tz_offset_sec is not None:
        return int(mb_cur.tz_offset_sec)
    return 0


def tzinfo_from_offset(tz_offset_sec: int) -> timezone:
    return timezone(timedelta(seconds=int(tz_offset_sec)))


def local_dt_from_ts(ts: int, tz_offset_sec: int) -> datetime:
    tz = tzinfo_from_offset(tz_offset_sec)
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(tz)


def local_now(tz_offset_sec: int) -> datetime:
    tz = tzinfo_from_offset(tz_offset_sec)
    return now_utc().astimezone(tz)


def tz_offset_from_tzid(tz_id: str) -> Optional[int]:
    try:
        tz = ZoneInfo(tz_id)
        return int(datetime.now(tz).utcoffset().total_seconds())
    except Exception:
        return None


def humanize_description(desc: str, temp: float, feels: float, humidity: Optional[float]) -> str:
    diff = feels - temp
    if diff <= -4:
        return f"{desc} (aria pungente)"
    if diff >= 4:
        return f"{desc} (piu caldo percepito)"
    if humidity is not None and humidity >= 75 and temp >= 25:
        return f"{desc} (caldo umido)"
    return desc


def adaptive_weights(
    base: Dict[str, float],
    diff: float,
    err_ow: Optional[float],
    err_wa: Optional[float],
    threshold: float = 2.0
) -> Dict[str, float]:
    if diff <= threshold:
        return base
    if err_ow is not None and err_wa is not None:
        w_ow = 1 / max(err_ow, 0.2)
        w_wa = 1 / max(err_wa, 0.2)
    elif err_ow is not None:
        w_ow, w_wa = 0.7, 0.3
    elif err_wa is not None:
        w_ow, w_wa = 0.3, 0.7
    else:
        return base
    tot = w_ow + w_wa
    return {"OpenWeather": w_ow / tot, "WeatherAPI": w_wa / tot}
