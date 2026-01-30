from __future__ import annotations

import os
import json
import math
import time
import hashlib
import logging
import sqlite3
import asyncio
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, Optional, List, Tuple

import aiohttp
try:
    import tornado.web
    from telegram.ext._utils.webhookhandler import TelegramHandler
except Exception:
    tornado = None
    TelegramHandler = None
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters

# ===================== CONFIG =====================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")
METEOBLUE_API_KEY = os.getenv("METEOBLUE_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g. https://<project>.up.railway.app
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/telegram-webhook")
HEALTH_PATH = os.getenv("HEALTH_PATH", "/health")
PORT = int(os.getenv("PORT", "8080"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "Fight club")
ADMIN_USER_IDS = os.getenv("ADMIN_USER_IDS", "")
HEALTH_PATH_EFFECTIVE = HEALTH_PATH

BOT_VERSION = "2.7"
BOT_RELEASES = [
    {
        "version": "2.7",
        "changes": [
            "Endpoint /health per Railway/monitoraggio",
            "Comando /admin con stato completo (errori, backoff, cache)",
            "Supporto dominio webhook e path configurabili",
        ],
    },
    {
        "version": "2.6",
        "changes": [
            "Fallback automatico con cache anche se scaduta (stale)",
            "Messaggi chiari quando un provider non risponde",
            "Preferenze per unita (C/F, km/h/mph) e fascia notifiche",
            "Logging strutturato errori API e retry",
            "Test base per formattazione e fallback",
            "Backoff intelligente per provider instabili",
            "TTL forecast piu lungo nelle ore notturne",
            "Stato backoff visibile in /info",
        ],
    },
    {
        "version": "2.5",
        "changes": [
            "Multi-citta con switch rapido",
            "Avvisi pioggia personalizzati (60 min e domani)",
        ],
    },
    {
        "version": "2.4",
        "changes": [
            "Accuratezza per fascia usata nelle affidabilita",
            "Calibrazione POP (percentuali pioggia)",
        ],
    },
    {
        "version": "2.3",
        "changes": [
            "Verifica automatica di fine giornata",
            "Messaggio /controlla piu leggibile",
        ],
    },
    {
        "version": "2.2",
        "changes": [
            "Storico Meteoblue usato come fallback in /controlla",
            "Brier score pioggia e nuovi indicatori accuratezza",
            "Pesi contestuali per fascia/condizione/stagione/zona",
        ],
    },
    {
        "version": "2.1",
        "changes": [
            "Messaggi /start e /comandi aggiornati",
            "Nuovo comando /aggiorna",
            "Formato fonti semplificato",
        ],
    },
    {
        "version": "2.0",
        "changes": [
            "Aggiunto provider Meteoblue (terza fonte)",
            "Affidabilita in percentuale nei messaggi (/meteo, /oggi, /domani)",
            "Pioggia mostrata solo dal momento attuale in avanti",
            "Domani: orario per orario con emoji e affidabilita",
            "Fonti semplificate (numero + eventuali mancanti)",
            "Nuovo comando /aggiorna per forzare le API",
            "Start e comandi aggiornati e piu chiari",
        ],
    },
    {
        "version": "1.2",
        "changes": [
            "Nuovo comando /prev (previsione ora per ora fino a mezzanotte)",
            "Oggi: riepilogo giornata (min/max + andamento)",
            "Pioggia mostrata solo se prevista",
            "Aggiunte emoji per dati importanti nel meteo",
        ],
    },
    {
        "version": "1.1",
        "changes": [
            "Testi messaggi piu chiari e non compatti",
            "Pioggia mostrata solo quando prevista",
            "Rimozione pulsanti inline sotto i messaggi",
            "Versioni con storico (ultime 3 release)",
        ],
    },
    {
        "version": "1.0",
        "changes": [
            "Escape Markdown per nomi citta, descrizioni e URL",
            "Fix fuso orario WeatherAPI nella selezione del giorno",
            "Data locale in /oggi e /domani",
            "Cache forecast condivisa + SQLite WAL/busy_timeout",
            "Chiusura sessione HTTP in shutdown",
        ],
    },
    {
        "version": "0.9",
        "changes": [
            "Aggiunti /versione e /info",
            "Migliorie cache e accuratezza",
        ],
    },
]

DB_FILE = "bot_data.sqlite3"

HTTP_TIMEOUT_SEC = 8
CACHE_MAX_ITEMS = 300

PROVIDERS = ["OpenWeather", "WeatherAPI", "Meteoblue"]

# TTL
CACHE_TTL_CURRENT_MIN = 10
CACHE_TTL_FORECAST_MIN = 30
CACHE_TTL_GEOCODE_HOURS = 24
NIGHT_FORECAST_TTL_MULT = 2

# Rain threshold (%)
RAIN_POP_THRESHOLD = 30
UMBRELLA_POP_THRESHOLD = 50
WIND_STRONG_KPH = 35
FEELS_LIKE_DIFF_C = 4
RAIN_KEYWORDS = [
    "pioggia", "pioviggine", "temporale", "rovesci", "acquazzone",
    "rain", "drizzle", "storm", "shower", "precipitation"
]

DECAY_HALF_LIFE_DAYS = 45
OUTLIER_TEMP_C = 4.0
POP_CALIBRATION_DAYS = 60

ALERT_POP_THRESHOLD = 60
ALERT_WIND_KPH = 45
ALERT_TEMP_HOT = 35
ALERT_TEMP_COLD = -5
ALERT_COOLDOWN_MIN = 180
ALERT_CHECK_MIN = 20
DAILY_CHECK_HOUR = 23
DAILY_CHECK_MIN = 0
TOMORROW_RAIN_HOUR = 18
TOMORROW_RAIN_WINDOW_MIN = 60
OFFLINE_TEST_DAYS = 30

DEFAULT_TEMP_UNIT = "C"
DEFAULT_WIND_UNIT = "kmh"
DEFAULT_ALERT_START_HOUR = 7
DEFAULT_ALERT_END_HOUR = 22

MSG_SERVICE_UNAVAILABLE = "Servizio temporaneamente non disponibile, riprova."
MSG_CITY_NOT_FOUND = "Citta non trovata. Controlla il nome o usa /setcitta."
MSG_NEED_CITY = "Specifica una citta o usa /setcitta o /citta."

# Accuracy verification window:
# verifichiamo una previsione quando siamo entro +/- 90 minuti dal target
VERIFY_WINDOW_MIN = 90

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("meteo-bot")


# ===================== UTILS =====================
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
    # Escape Markdown special chars to avoid Telegram parse errors
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
    if ADMIN_SECRET and token and token == ADMIN_SECRET:
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
        update_queue: asyncio.Queue,
        secret_token: str | None = None,
    ):
        if not webhook_path.startswith("/"):
            webhook_path = f"/{webhook_path}"
        health_path = HEALTH_PATH_EFFECTIVE
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
    giorni = ['Lunedi','Martedi','Mercoledi','Giovedi','Venerdi','Sabato','Domenica']
    mesi = ['Gennaio','Febbraio','Marzo','Aprile','Maggio','Giugno','Luglio','Agosto','Settembre','Ottobre','Novembre','Dicembre']
    return f"{giorni[dt.weekday()]} {dt.day} {mesi[dt.month-1]}"

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

def get_weather_icon(icon_code: str) -> str:
    icon_map = {
        '01d': '\u2600\ufe0f', '01n': '\U0001F319', '02d': '\u26c5', '02n': '\u26c5',
        '03d': '\u2601\ufe0f', '03n': '\u2601\ufe0f', '04d': '\u2601\ufe0f', '04n': '\u2601\ufe0f',
        '09d': '\U0001F327\ufe0f', '09n': '\U0001F327\ufe0f', '10d': '\U0001F326\ufe0f', '10n': '\U0001F326\ufe0f',
        '11d': '\u26c8\ufe0f', '11n': '\u26c8\ufe0f', '13d': '\u2744\ufe0f', '13n': '\u2744\ufe0f',
        '50d': '\U0001F32B\ufe0f', '50n': '\U0001F32B\ufe0f'
    }
    return icon_map.get(icon_code, '\U0001F321\ufe0f')

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
    ow_cur: Optional[ProviderResult] = None,
    wa_cur: Optional[ProviderResult] = None,
    mb_cur: Optional[ProviderResult] = None,
    coords: Optional[Dict[str, Any]] = None,
) -> int:
    if coords and coords.get("tz_offset_sec") is not None:
        return int(coords["tz_offset_sec"])
    if forecast_ow:
        try:
            return int(forecast_ow.get("city", {}).get("timezone", 0))
        except Exception:
            pass
    if forecast_wa:
        try:
            local_epoch = int(forecast_wa.get("location", {}).get("localtime_epoch", 0))
            return int(local_epoch - int(now_utc().timestamp()))
        except Exception:
            pass
    if forecast_mb:
        try:
            meta = forecast_mb.get("metadata", {})
            if meta.get("utc_timeoffset") is not None:
                return int(float(meta["utc_timeoffset"]) * 3600)
        except Exception:
            pass
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


# ===================== ROBUST CACHE (SQLite + RAM) =====================
class Storage:
    def __init__(self, path: str):
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=3)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=3000")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
              user_id TEXT PRIMARY KEY,
              city TEXT
            );

            CREATE TABLE IF NOT EXISTS user_prefs (
              user_id TEXT PRIMARY KEY,
              pop_threshold REAL,
              wind_threshold REAL,
              feels_diff REAL,
              alert_rain_60 INTEGER,
              alert_tomorrow_rain INTEGER,
              temp_unit TEXT,
              wind_unit TEXT,
              alert_start_hour INTEGER,
              alert_end_hour INTEGER
            );

            CREATE TABLE IF NOT EXISTS user_cities (
              user_id TEXT NOT NULL,
              city TEXT NOT NULL,
              is_default INTEGER NOT NULL,
              added_at TEXT NOT NULL,
              PRIMARY KEY (user_id, city)
            );

            CREATE TABLE IF NOT EXISTS cache (
              cache_key TEXT PRIMARY KEY,
              payload TEXT NOT NULL,
              created_at TEXT NOT NULL,
              expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS provider_accuracy (
              provider TEXT PRIMARY KEY,
              errors_json TEXT NOT NULL,
              avg_error REAL NOT NULL,
              accuracy REAL NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS errors_log (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              url TEXT,
              status INTEGER,
              detail TEXT
            );

            CREATE TABLE IF NOT EXISTS provider_backoff (
              provider TEXT PRIMARY KEY,
              fail_count INTEGER NOT NULL,
              backoff_until TEXT,
              last_error TEXT,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alerts_log (
              user_id TEXT NOT NULL,
              alert_type TEXT NOT NULL,
              last_sent_utc TEXT NOT NULL,
              PRIMARY KEY (user_id, alert_type)
            );

            CREATE TABLE IF NOT EXISTS predictions (
              id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              city TEXT NOT NULL,
              kind TEXT NOT NULL,                 -- meteo | oggi | domani
              target_dt_utc TEXT NOT NULL,        -- quando dovrebbe valere
              predicted_fused REAL NOT NULL,      -- temp prevista fusa
              predicted_ow REAL,                  -- temp OW (se disponibile)
              predicted_wa REAL,                  -- temp WA (se disponibile)
              predicted_mb REAL,                  -- temp Meteoblue (se disponibile)
              predicted_wind_kph REAL,
              predicted_humidity REAL,
              predicted_pressure REAL,
              predicted_rain INTEGER,             -- 0/1
              predicted_pop REAL,                 -- POP prevista
              error_pop REAL,
              target_hour_local INTEGER,          -- ora locale target
              target_band TEXT,                   -- fascia oraria
              condition_group TEXT,               -- meteo sintetico
              season TEXT,
              zone TEXT,
              verified INTEGER NOT NULL,          -- 0/1
              verified_at_utc TEXT,
              actual_temp REAL,
              actual_rain INTEGER,
              actual_condition_group TEXT,
              error_fused REAL,
              actual_wind_kph REAL,
              actual_humidity REAL,
              actual_pressure REAL,
              error_wind REAL,
              error_humidity REAL,
              error_pressure REAL,
              created_at_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pred_user_verified ON predictions(user_id, verified);
            CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_dt_utc);
            """)
            self._migrate_predictions(conn)
            self._migrate_user_prefs(conn)
            self._migrate_user_cities(conn)
            conn.commit()

    def _migrate_predictions(self, conn: sqlite3.Connection):
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
        if "predicted_rain" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN predicted_rain INTEGER")
        if "predicted_pop" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN predicted_pop REAL")
        if "predicted_mb" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN predicted_mb REAL")
        if "predicted_wind_kph" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN predicted_wind_kph REAL")
        if "predicted_humidity" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN predicted_humidity REAL")
        if "predicted_pressure" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN predicted_pressure REAL")
        if "target_hour_local" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN target_hour_local INTEGER")
        if "target_band" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN target_band TEXT")
        if "condition_group" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN condition_group TEXT")
        if "season" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN season TEXT")
        if "zone" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN zone TEXT")
        if "actual_rain" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN actual_rain INTEGER")
        if "actual_condition_group" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN actual_condition_group TEXT")
        if "error_pop" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN error_pop REAL")
        if "actual_wind_kph" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN actual_wind_kph REAL")
        if "actual_humidity" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN actual_humidity REAL")
        if "actual_pressure" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN actual_pressure REAL")
        if "error_wind" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN error_wind REAL")
        if "error_humidity" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN error_humidity REAL")
        if "error_pressure" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN error_pressure REAL")

    def _migrate_user_prefs(self, conn: sqlite3.Connection):
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(user_prefs)").fetchall()}
        if "alert_rain_60" not in cols:
            conn.execute("ALTER TABLE user_prefs ADD COLUMN alert_rain_60 INTEGER")
        if "alert_tomorrow_rain" not in cols:
            conn.execute("ALTER TABLE user_prefs ADD COLUMN alert_tomorrow_rain INTEGER")
        if "temp_unit" not in cols:
            conn.execute("ALTER TABLE user_prefs ADD COLUMN temp_unit TEXT")
        if "wind_unit" not in cols:
            conn.execute("ALTER TABLE user_prefs ADD COLUMN wind_unit TEXT")
        if "alert_start_hour" not in cols:
            conn.execute("ALTER TABLE user_prefs ADD COLUMN alert_start_hour INTEGER")
        if "alert_end_hour" not in cols:
            conn.execute("ALTER TABLE user_prefs ADD COLUMN alert_end_hour INTEGER")

    # provider backoff
    def get_provider_backoff_until(self, provider: str) -> Optional[datetime]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT backoff_until FROM provider_backoff WHERE provider=?",
                (provider,)
            ).fetchone()
            if not row or not row["backoff_until"]:
                return None
            return from_iso(row["backoff_until"])

    def record_provider_failure(self, provider: str, error: str):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT fail_count FROM provider_backoff WHERE provider=?",
                (provider,)
            ).fetchone()
            fail_count = int(row["fail_count"]) + 1 if row else 1
            backoff_until = None
            if fail_count >= 3:
                exp = min(15 * (2 ** (fail_count - 3)), 120)
                backoff_until = now_utc() + timedelta(minutes=int(exp))
            conn.execute(
                "INSERT INTO provider_backoff(provider, fail_count, backoff_until, last_error, updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(provider) DO UPDATE SET "
                "fail_count=excluded.fail_count, backoff_until=excluded.backoff_until, last_error=excluded.last_error, updated_at=excluded.updated_at",
                (provider, fail_count, iso(backoff_until) if backoff_until else None, error[:120], iso(now_utc()))
            )
            conn.commit()

    def record_provider_success(self, provider: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO provider_backoff(provider, fail_count, backoff_until, last_error, updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(provider) DO UPDATE SET "
                "fail_count=0, backoff_until=NULL, last_error=NULL, updated_at=excluded.updated_at",
                (provider, 0, None, None, iso(now_utc()))
            )
            conn.commit()

    def get_provider_backoff_status(self) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT provider, fail_count, backoff_until, last_error, updated_at FROM provider_backoff ORDER BY provider"
            ).fetchall()

    def _migrate_user_cities(self, conn: sqlite3.Connection):
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(user_cities)").fetchall()}
        if "user_id" not in cols:
            return
        count = conn.execute("SELECT COUNT(*) AS n FROM user_cities").fetchone()["n"]
        if count == 0:
            rows = conn.execute("SELECT user_id, city FROM users WHERE city IS NOT NULL").fetchall()
            for r in rows:
                conn.execute(
                    "INSERT OR IGNORE INTO user_cities(user_id, city, is_default, added_at) VALUES(?,?,?,?)",
                    (r["user_id"], r["city"], 1, iso(now_utc()))
                )

    # users
    def get_user_city(self, user_id: int) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT city FROM user_cities WHERE user_id=? AND is_default=1",
                (str(user_id),)
            ).fetchone()
            if row and row["city"]:
                return row["city"]
            row = conn.execute("SELECT city FROM users WHERE user_id=?", (str(user_id),)).fetchone()
            return row["city"] if row else None

    def set_user_city(self, user_id: int, city: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO user_cities(user_id, city, is_default, added_at) VALUES(?,?,?,?)",
                (str(user_id), city, 0, iso(now_utc()))
            )
            conn.execute(
                "UPDATE user_cities SET is_default=CASE WHEN city=? THEN 1 ELSE 0 END WHERE user_id=?",
                (city, str(user_id))
            )
            conn.execute(
                "INSERT INTO users(user_id, city) VALUES(?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET city=excluded.city",
                (str(user_id), city)
            )
            conn.commit()

    def get_user_cities(self, user_id: int) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT city FROM user_cities WHERE user_id=? ORDER BY is_default DESC, added_at ASC",
                (str(user_id),)
            ).fetchall()
            return [r["city"] for r in rows]

    def set_default_city(self, user_id: int, city: str) -> bool:
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM user_cities WHERE user_id=? AND city=?",
                (str(user_id), city)
            ).fetchone()
            if not exists:
                return False
            conn.execute(
                "UPDATE user_cities SET is_default=CASE WHEN city=? THEN 1 ELSE 0 END WHERE user_id=?",
                (city, str(user_id))
            )
            conn.execute(
                "INSERT INTO users(user_id, city) VALUES(?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET city=excluded.city",
                (str(user_id), city)
            )
            conn.commit()
            return True

    def remove_user_city(self, user_id: int, city: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT is_default FROM user_cities WHERE user_id=? AND city=?",
                (str(user_id), city)
            ).fetchone()
            if not row:
                return False
            was_default = int(row["is_default"]) == 1
            conn.execute(
                "DELETE FROM user_cities WHERE user_id=? AND city=?",
                (str(user_id), city)
            )
            if was_default:
                new_def = conn.execute(
                    "SELECT city FROM user_cities WHERE user_id=? ORDER BY added_at ASC LIMIT 1",
                    (str(user_id),)
                ).fetchone()
                if new_def:
                    conn.execute(
                        "UPDATE user_cities SET is_default=CASE WHEN city=? THEN 1 ELSE 0 END WHERE user_id=?",
                        (new_def["city"], str(user_id))
                    )
                    conn.execute(
                        "INSERT INTO users(user_id, city) VALUES(?, ?) "
                        "ON CONFLICT(user_id) DO UPDATE SET city=excluded.city",
                        (str(user_id), new_def["city"])
                    )
                else:
                    conn.execute("DELETE FROM users WHERE user_id=?", (str(user_id),))
            conn.commit()
            return True

    def get_all_users(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT user_id FROM users").fetchall()
            return [r["user_id"] for r in rows]

    def get_user_prefs(self, user_id: int) -> Dict[str, float]:
        prefs = {
            "pop_threshold": float(UMBRELLA_POP_THRESHOLD),
            "wind_threshold": float(WIND_STRONG_KPH),
            "feels_diff": float(FEELS_LIKE_DIFF_C),
            "alert_rain_60": 0,
            "alert_tomorrow_rain": 1,
            "temp_unit": DEFAULT_TEMP_UNIT,
            "wind_unit": DEFAULT_WIND_UNIT,
            "alert_start_hour": DEFAULT_ALERT_START_HOUR,
            "alert_end_hour": DEFAULT_ALERT_END_HOUR,
        }
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM user_prefs WHERE user_id=?", (str(user_id),)).fetchone()
            if row:
                if row["pop_threshold"] is not None:
                    prefs["pop_threshold"] = float(row["pop_threshold"])
                if row["wind_threshold"] is not None:
                    prefs["wind_threshold"] = float(row["wind_threshold"])
                if row["feels_diff"] is not None:
                    prefs["feels_diff"] = float(row["feels_diff"])
                if row["alert_rain_60"] is not None:
                    prefs["alert_rain_60"] = int(row["alert_rain_60"])
                if row["alert_tomorrow_rain"] is not None:
                    prefs["alert_tomorrow_rain"] = int(row["alert_tomorrow_rain"])
                if row["temp_unit"] is not None:
                    prefs["temp_unit"] = normalize_temp_unit(row["temp_unit"])
                if row["wind_unit"] is not None:
                    prefs["wind_unit"] = normalize_wind_unit(row["wind_unit"])
                if row["alert_start_hour"] is not None:
                    prefs["alert_start_hour"] = int(row["alert_start_hour"])
                if row["alert_end_hour"] is not None:
                    prefs["alert_end_hour"] = int(row["alert_end_hour"])
        return prefs

    def set_user_prefs(
        self,
        user_id: int,
        pop_threshold: Optional[float],
        wind_threshold: Optional[float],
        feels_diff: Optional[float],
        alert_rain_60: Optional[int] = None,
        alert_tomorrow_rain: Optional[int] = None,
        temp_unit: Optional[str] = None,
        wind_unit: Optional[str] = None,
        alert_start_hour: Optional[int] = None,
        alert_end_hour: Optional[int] = None,
    ):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO user_prefs(user_id, pop_threshold, wind_threshold, feels_diff, alert_rain_60, alert_tomorrow_rain, temp_unit, wind_unit, alert_start_hour, alert_end_hour) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "pop_threshold=COALESCE(excluded.pop_threshold, user_prefs.pop_threshold), "
                "wind_threshold=COALESCE(excluded.wind_threshold, user_prefs.wind_threshold), "
                "feels_diff=COALESCE(excluded.feels_diff, user_prefs.feels_diff), "
                "alert_rain_60=COALESCE(excluded.alert_rain_60, user_prefs.alert_rain_60), "
                "alert_tomorrow_rain=COALESCE(excluded.alert_tomorrow_rain, user_prefs.alert_tomorrow_rain), "
                "temp_unit=COALESCE(excluded.temp_unit, user_prefs.temp_unit), "
                "wind_unit=COALESCE(excluded.wind_unit, user_prefs.wind_unit), "
                "alert_start_hour=COALESCE(excluded.alert_start_hour, user_prefs.alert_start_hour), "
                "alert_end_hour=COALESCE(excluded.alert_end_hour, user_prefs.alert_end_hour)",
                (
                    str(user_id),
                    pop_threshold,
                    wind_threshold,
                    feels_diff,
                    alert_rain_60,
                    alert_tomorrow_rain,
                    temp_unit,
                    wind_unit,
                    alert_start_hour,
                    alert_end_hour,
                )
            )
            conn.commit()

    # cache
    def cache_get_with_meta(self, key: str, allow_expired: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[datetime], Optional[datetime], bool]:
        with self._connect() as conn:
            row = conn.execute("SELECT payload, created_at, expires_at FROM cache WHERE cache_key=?", (key,)).fetchone()
            if not row:
                return None, None, None, False
            created = from_iso(row["created_at"])
            expires = from_iso(row["expires_at"])
            expired = now_utc() >= expires
            if expired and not allow_expired:
                conn.execute("DELETE FROM cache WHERE cache_key=?", (key,))
                conn.commit()
                return None, created, expires, True
            return json.loads(row["payload"]), created, expires, expired

    def cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        payload, _, _, expired = self.cache_get_with_meta(key, allow_expired=False)
        if expired:
            return None
        return payload

    def cache_set(self, key: str, payload: Dict[str, Any], ttl: timedelta):
        created = now_utc()
        expires = created + ttl
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cache(cache_key, payload, created_at, expires_at) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload, created_at=excluded.created_at, expires_at=excluded.expires_at",
                (key, json.dumps(payload), iso(created), iso(expires))
            )
            # limit size
            conn.execute("""
              DELETE FROM cache
              WHERE cache_key IN (
                SELECT cache_key FROM cache
                ORDER BY created_at ASC
                LIMIT (SELECT MAX(0, COUNT(*) - ?) FROM cache)
              )
            """, (CACHE_MAX_ITEMS,))
            conn.commit()

    def log_error(self, url: str, status: int, detail: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO errors_log(created_at, url, status, detail) VALUES(?,?,?,?)",
                (iso(now_utc()), url, int(status), detail[:400])
            )
            conn.commit()

    def get_last_alert(self, user_id: int, alert_type: str) -> Optional[datetime]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_sent_utc FROM alerts_log WHERE user_id=? AND alert_type=?",
                (str(user_id), alert_type)
            ).fetchone()
            return from_iso(row["last_sent_utc"]) if row else None

    def set_last_alert(self, user_id: int, alert_type: str, dt: datetime):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO alerts_log(user_id, alert_type, last_sent_utc) VALUES(?,?,?) "
                "ON CONFLICT(user_id, alert_type) DO UPDATE SET last_sent_utc=excluded.last_sent_utc",
                (str(user_id), alert_type, iso(dt))
            )
            conn.commit()

    def get_recent_errors(self, limit: int = 5) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT created_at, url, status, detail FROM errors_log ORDER BY id DESC LIMIT ?",
                (int(limit),)
            ).fetchall()

    def get_offline_stats(self, user_id: Optional[int], days: int = 30) -> Dict[str, Any]:
        cutoff = now_utc() - timedelta(days=int(days))
        query = (
            "SELECT predicted_fused, predicted_ow, predicted_wa, predicted_mb, actual_temp, error_fused, "
            "predicted_rain, actual_rain, condition_group, actual_condition_group "
            "FROM predictions WHERE verified=1 AND actual_temp IS NOT NULL AND created_at_utc >= ?"
        )
        params: List[Any] = [iso(cutoff)]
        if user_id is not None:
            query += " AND user_id=?"
            params.append(str(user_id))
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        if not rows:
            return {"count": 0}
        fused_errs = []
        ow_errs = []
        wa_errs = []
        mb_errs = []
        rain_total = rain_hits = 0
        cond_total = cond_hits = 0
        for r in rows:
            if r["error_fused"] is not None:
                fused_errs.append(float(r["error_fused"]))
            if r["predicted_ow"] is not None and r["actual_temp"] is not None:
                ow_errs.append(abs(float(r["predicted_ow"]) - float(r["actual_temp"])))
            if r["predicted_wa"] is not None and r["actual_temp"] is not None:
                wa_errs.append(abs(float(r["predicted_wa"]) - float(r["actual_temp"])))
            if r["predicted_mb"] is not None and r["actual_temp"] is not None:
                mb_errs.append(abs(float(r["predicted_mb"]) - float(r["actual_temp"])))
            if r["predicted_rain"] is not None and r["actual_rain"] is not None:
                rain_total += 1
                if bool(r["predicted_rain"]) == bool(r["actual_rain"]):
                    rain_hits += 1
            if r["condition_group"] and r["actual_condition_group"]:
                cond_total += 1
                if str(r["condition_group"]) == str(r["actual_condition_group"]):
                    cond_hits += 1
        return {
            "count": len(rows),
            "fused_mae": sum(fused_errs) / len(fused_errs) if fused_errs else None,
            "ow_mae": sum(ow_errs) / len(ow_errs) if ow_errs else None,
            "wa_mae": sum(wa_errs) / len(wa_errs) if wa_errs else None,
            "mb_mae": sum(mb_errs) / len(mb_errs) if mb_errs else None,
            "rain_hits": rain_hits,
            "rain_total": rain_total,
            "cond_hits": cond_hits,
            "cond_total": cond_total,
            "days": int(days),
        }

    # provider accuracy
    def get_provider_accuracy(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM provider_accuracy").fetchall()
            for r in rows:
                out[r["provider"]] = {
                    "errors": json.loads(r["errors_json"]),
                    "avg_error": float(r["avg_error"]),
                    "accuracy": float(r["accuracy"]),
                    "updated_at": r["updated_at"],
                }
        return out

    def get_provider_bias(self, limit: int = 200) -> Dict[str, float]:
        """
        Signed bias = avg(predicted - actual). Positive means provider overestimates.
        """
        bias: Dict[str, float] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT predicted_ow, predicted_wa, predicted_mb, actual_temp FROM predictions "
                "WHERE verified=1 AND actual_temp IS NOT NULL "
                "ORDER BY verified_at_utc DESC LIMIT ?",
                (int(limit),)
            ).fetchall()
        ow_vals = []
        wa_vals = []
        mb_vals = []
        for r in rows:
            if r["predicted_ow"] is not None:
                ow_vals.append(float(r["predicted_ow"]) - float(r["actual_temp"]))
            if r["predicted_wa"] is not None:
                wa_vals.append(float(r["predicted_wa"]) - float(r["actual_temp"]))
            if r["predicted_mb"] is not None:
                mb_vals.append(float(r["predicted_mb"]) - float(r["actual_temp"]))
        if ow_vals:
            bias["OpenWeather"] = sum(ow_vals) / len(ow_vals)
        if wa_vals:
            bias["WeatherAPI"] = sum(wa_vals) / len(wa_vals)
        if mb_vals:
            bias["Meteoblue"] = sum(mb_vals) / len(mb_vals)
        return bias

    def get_provider_bias_context(
        self,
        provider: str,
        band: str,
        condition_group: str,
        kind_group: Optional[str] = None,
        season: Optional[str] = None,
        zone: Optional[str] = None,
        min_samples: int = 5,
        limit: int = 400
    ) -> float:
        if provider == "OpenWeather":
            pred_col = "predicted_ow"
        elif provider == "WeatherAPI":
            pred_col = "predicted_wa"
        else:
            pred_col = "predicted_mb"
        kind_filter = ""
        params = [band, condition_group]
        if kind_group == "now":
            kind_filter = " AND kind='meteo' "
        elif kind_group == "forecast":
            kind_filter = " AND kind IN ('oggi','domani') "
        if season:
            kind_filter += " AND season=? "
            params.append(season)
        if zone:
            kind_filter += " AND zone=? "
            params.append(zone)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {pred_col} AS pred, actual_temp, verified_at_utc FROM predictions "
                "WHERE verified=1 AND actual_temp IS NOT NULL "
                "AND target_band=? AND condition_group=? AND {pred_col} IS NOT NULL "
                f"{kind_filter}"
                "ORDER BY verified_at_utc DESC LIMIT ?".format(pred_col=pred_col),
                (*params, int(limit))
            ).fetchall()
        vals = []
        weights = []
        now = now_utc()
        for r in rows:
            if r["pred"] is None:
                continue
            age_days = (now - from_iso(r["verified_at_utc"])).total_seconds() / 86400 if r["verified_at_utc"] else 0.0
            w = 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)
            vals.append(float(r["pred"]) - float(r["actual_temp"]))
            weights.append(w)
        if len(vals) >= min_samples:
            num = sum(v * w for v, w in zip(vals, weights))
            den = sum(weights) if weights else 0.0
            return num / den if den > 0 else sum(vals) / len(vals)
        # fallback: band only
        kind_filter2 = ""
        params2 = [band]
        if kind_group == "now":
            kind_filter2 = " AND kind='meteo' "
        elif kind_group == "forecast":
            kind_filter2 = " AND kind IN ('oggi','domani') "
        if season:
            kind_filter2 += " AND season=? "
            params2.append(season)
        if zone:
            kind_filter2 += " AND zone=? "
            params2.append(zone)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {pred_col} AS pred, actual_temp, verified_at_utc FROM predictions "
                "WHERE verified=1 AND actual_temp IS NOT NULL "
                "AND target_band=? AND {pred_col} IS NOT NULL "
                f"{kind_filter2}"
                "ORDER BY verified_at_utc DESC LIMIT ?".format(pred_col=pred_col),
                (*params2, int(limit))
            ).fetchall()
        vals = []
        weights = []
        now = now_utc()
        for r in rows:
            if r["pred"] is None:
                continue
            age_days = (now - from_iso(r["verified_at_utc"])).total_seconds() / 86400 if r["verified_at_utc"] else 0.0
            w = 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)
            vals.append(float(r["pred"]) - float(r["actual_temp"]))
            weights.append(w)
        if len(vals) >= min_samples:
            num = sum(v * w for v, w in zip(vals, weights))
            den = sum(weights) if weights else 0.0
            return num / den if den > 0 else sum(vals) / len(vals)
        # fallback: overall
        overall = self.get_provider_bias(limit=limit)
        return float(overall.get(provider, 0.0))

    def get_provider_accuracy_context(
        self,
        provider: str,
        band: str,
        condition_group: str,
        kind_group: Optional[str] = None,
        season: Optional[str] = None,
        zone: Optional[str] = None,
        min_samples: int = 5,
        limit: int = 400
    ) -> Optional[float]:
        if provider == "OpenWeather":
            pred_col = "predicted_ow"
        elif provider == "WeatherAPI":
            pred_col = "predicted_wa"
        else:
            pred_col = "predicted_mb"
        kind_filter = ""
        if kind_group == "now":
            kind_filter = " AND kind='meteo' "
        elif kind_group == "forecast":
            kind_filter = " AND kind IN ('oggi','domani') "
        params = [band, condition_group]
        if season:
            kind_filter += " AND season=? "
            params.append(season)
        if zone:
            kind_filter += " AND zone=? "
            params.append(zone)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {pred_col} AS pred, actual_temp, verified_at_utc FROM predictions "
                "WHERE verified=1 AND actual_temp IS NOT NULL "
                "AND target_band=? AND condition_group=? AND {pred_col} IS NOT NULL "
                f"{kind_filter}"
                "ORDER BY verified_at_utc DESC LIMIT ?".format(pred_col=pred_col),
                (*params, int(limit))
            ).fetchall()
        errs = []
        weights = []
        now = now_utc()
        for r in rows:
            if r["pred"] is None:
                continue
            age_days = (now - from_iso(r["verified_at_utc"])).total_seconds() / 86400 if r["verified_at_utc"] else 0.0
            w = 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)
            errs.append(abs(float(r["pred"]) - float(r["actual_temp"])))
            weights.append(w)
        if len(errs) >= min_samples:
            num = sum(e * w for e, w in zip(errs, weights))
            den = sum(weights) if weights else 0.0
            return num / den if den > 0 else sum(errs) / len(errs)
        return None

    def upsert_provider_accuracy(self, provider: str, errors: List[float], avg_error: float, accuracy: float):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO provider_accuracy(provider, errors_json, avg_error, accuracy, updated_at) VALUES(?, ?, ?, ?, ?) "
                "ON CONFLICT(provider) DO UPDATE SET errors_json=excluded.errors_json, avg_error=excluded.avg_error, accuracy=excluded.accuracy, updated_at=excluded.updated_at",
                (provider, json.dumps(errors), float(avg_error), float(accuracy), iso(now_utc()))
            )
            conn.commit()

    # predictions (accurate verification)
    def save_prediction(
        self,
        user_id: int,
        city: str,
        kind: str,
        target_dt_utc: datetime,
        predicted_fused: float,
        predicted_ow: Optional[float],
        predicted_wa: Optional[float],
        predicted_mb: Optional[float] = None,
        predicted_wind_kph: Optional[float] = None,
        predicted_humidity: Optional[float] = None,
        predicted_pressure: Optional[float] = None,
        predicted_rain: Optional[bool] = None,
        predicted_pop: Optional[float] = None,
        target_hour_local: Optional[int] = None,
        condition_group: Optional[str] = None,
        season: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> str:
        pid = md5(f"{user_id}|{city}|{kind}|{target_dt_utc.timestamp()}|{time.time()}")[:10]
        band = hour_band(target_hour_local) if target_hour_local is not None else None
        with self._connect() as conn:
            conn.execute("""
              INSERT INTO predictions(
                id, user_id, city, kind,
                target_dt_utc, predicted_fused, predicted_ow, predicted_wa, predicted_mb,
                predicted_wind_kph, predicted_humidity, predicted_pressure,
                predicted_rain, predicted_pop, error_pop, target_hour_local, target_band, condition_group, season, zone,
                verified, verified_at_utc, actual_temp, error_fused, created_at_utc
              ) VALUES(?,?,?,?,?,?,?,?,?, ?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?)
            """, (
                pid, str(user_id), city, kind,
                iso(target_dt_utc), float(predicted_fused),
                float(predicted_ow) if predicted_ow is not None else None,
                float(predicted_wa) if predicted_wa is not None else None,
                float(predicted_mb) if predicted_mb is not None else None,
                float(predicted_wind_kph) if predicted_wind_kph is not None else None,
                float(predicted_humidity) if predicted_humidity is not None else None,
                float(predicted_pressure) if predicted_pressure is not None else None,
                int(predicted_rain) if predicted_rain is not None else None,
                float(predicted_pop) if predicted_pop is not None else None,
                None,
                int(target_hour_local) if target_hour_local is not None else None,
                band,
                condition_group,
                season,
                zone,
                0,
                None,
                None,
                None,
                iso(now_utc())
            ))
            conn.commit()
        return pid

    def load_due_unverified_predictions(self, user_id: int) -> List[sqlite3.Row]:
        """
        Prende previsioni non verificate che sono 'verificabili' ora:
        - target_dt_utc <= now + window
        - e now - target_dt_utc <= window
        Quindi siamo vicino al target, non ore dopo.
        """
        now = now_utc()
        min_dt = now - timedelta(minutes=VERIFY_WINDOW_MIN)
        max_dt = now + timedelta(minutes=VERIFY_WINDOW_MIN)
        with self._connect() as conn:
            rows = conn.execute("""
              SELECT * FROM predictions
              WHERE user_id=? AND verified=0
                AND target_dt_utc BETWEEN ? AND ?
              ORDER BY target_dt_utc ASC
            """, (str(user_id), iso(min_dt), iso(max_dt))).fetchall()
        return rows

    def mark_prediction_verified(
        self,
        pid: str,
        actual_temp: float,
        error_fused: float,
        actual_rain: Optional[bool] = None,
        actual_condition_group: Optional[str] = None,
        actual_wind_kph: Optional[float] = None,
        actual_humidity: Optional[float] = None,
        actual_pressure: Optional[float] = None,
        error_wind: Optional[float] = None,
        error_humidity: Optional[float] = None,
        error_pressure: Optional[float] = None,
        error_pop: Optional[float] = None,
    ):
        with self._connect() as conn:
            conn.execute("""
              UPDATE predictions
              SET verified=1, verified_at_utc=?, actual_temp=?, actual_rain=?, actual_condition_group=?, error_fused=?,
                  actual_wind_kph=?, actual_humidity=?, actual_pressure=?,
                  error_wind=?, error_humidity=?, error_pressure=?, error_pop=?
              WHERE id=?
            """, (
                iso(now_utc()),
                float(actual_temp),
                int(actual_rain) if actual_rain is not None else None,
                actual_condition_group,
                float(error_fused),
                float(actual_wind_kph) if actual_wind_kph is not None else None,
                float(actual_humidity) if actual_humidity is not None else None,
                float(actual_pressure) if actual_pressure is not None else None,
                float(error_wind) if error_wind is not None else None,
                float(error_humidity) if error_humidity is not None else None,
                float(error_pressure) if error_pressure is not None else None,
                float(error_pop) if error_pop is not None else None,
                pid,
            ))
            conn.commit()


class RamCache:
    """Micro-cache in RAM (veloce) + fallback SQLite"""
    def __init__(self, storage: Storage, max_items: int = 300):
        self.storage = storage
        self.max_items = max_items
        self.data: Dict[str, Tuple[float, Dict[str, Any]]] = {}  # key -> (expires_ts, payload)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        item = self.data.get(key)
        now_ts = time.time()
        if item:
            exp, payload = item
            if now_ts < exp:
                return payload
            self.data.pop(key, None)
        # fallback SQLite
        payload = self.storage.cache_get(key)
        if payload:
            # non conosciamo esattamente expires_ts, mettiamo un breve cache in RAM (30s)
            self.set(key, payload, ttl_seconds=30)
        return payload

    def set(self, key: str, payload: Dict[str, Any], ttl_seconds: int):
        if len(self.data) >= self.max_items:
            # rimuovi un item qualunque (semplice)
            self.data.pop(next(iter(self.data)))
        self.data[key] = (time.time() + ttl_seconds, payload)


# ===================== HTTP (aiohttp) =====================
class HttpClient:
    def __init__(self, storage: Optional[Storage] = None):
        self.session: Optional[aiohttp.ClientSession] = None
        self.storage = storage

    async def ensure(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SEC)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def get_json(self, url: str, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Tuple[int, Optional[Any]]:
        await self.ensure()
        ctx = context or {}
        provider = ctx.get("provider")
        if self.storage and provider:
            backoff_until = self.storage.get_provider_backoff_until(provider)
            if backoff_until and now_utc() < backoff_until:
                log_event("http_backoff", provider=provider, url=url, until=iso(backoff_until))
                return 0, None
        for attempt in range(2):
            try:
                start = time.monotonic()
                async with self.session.get(url, params=params) as resp:
                    elapsed_ms = int((time.monotonic() - start) * 1000)
                    if resp.status != 200:
                        if resp.status in {429, 500, 502, 503, 504} and attempt == 0:
                            rate_rem = resp.headers.get("X-RateLimit-Remaining") or resp.headers.get("X-Rate-Limit-Remaining")
                            rate_lim = resp.headers.get("X-RateLimit-Limit") or resp.headers.get("X-Rate-Limit-Limit")
                            log_event(
                                "http_retry",
                                provider=ctx.get("provider"),
                                url=url,
                                status=resp.status,
                                ms=elapsed_ms,
                                attempt=attempt + 1,
                                rate_remaining=rate_rem,
                                rate_limit=rate_lim,
                            )
                            await asyncio.sleep(0.6)
                            continue
                        rate_rem = resp.headers.get("X-RateLimit-Remaining") or resp.headers.get("X-Rate-Limit-Remaining")
                        rate_lim = resp.headers.get("X-RateLimit-Limit") or resp.headers.get("X-Rate-Limit-Limit")
                        log_event(
                            "http_error",
                            provider=ctx.get("provider"),
                            url=url,
                            status=resp.status,
                            ms=elapsed_ms,
                            attempt=attempt + 1,
                            rate_remaining=rate_rem,
                            rate_limit=rate_lim,
                        )
                        if self.storage:
                            self.storage.log_error(url, resp.status, f"HTTP {resp.status}")
                            if provider:
                                self.storage.record_provider_failure(provider, f"HTTP {resp.status}")
                        return resp.status, None
                    if self.storage and provider:
                        self.storage.record_provider_success(provider)
                    return resp.status, await resp.json()
            except asyncio.TimeoutError:
                log_event("http_timeout", provider=ctx.get("provider"), url=url, attempt=attempt + 1)
                if attempt == 0:
                    await asyncio.sleep(0.6)
                    continue
                if self.storage:
                    self.storage.log_error(url, 0, "timeout")
                    if provider:
                        self.storage.record_provider_failure(provider, "timeout")
                return 0, None
            except Exception as exc:
                logger.exception("HTTP error")
                log_event(
                    "http_exception",
                    provider=ctx.get("provider"),
                    url=url,
                    error=type(exc).__name__,
                )
                if self.storage:
                    self.storage.log_error(url, 0, f"exception: {type(exc).__name__}")
                    if provider:
                        self.storage.record_provider_failure(provider, type(exc).__name__)
                return 0, None

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


# ===================== PROVIDERS =====================
def map_weatherapi_icon(code: int) -> str:
    icon_map = {
        1000: '01d', 1003: '02d', 1006: '03d', 1009: '04d',
        1030: '50d', 1063: '09d', 1066: '13d', 1069: '13d',
        1087: '11d', 1114: '13d', 1117: '13d',
        1135: '50d', 1147: '50d',
        1180: '09d', 1183: '09d', 1186: '09d', 1189: '09d',
        1192: '09d', 1195: '09d',
        1240: '09d', 1243: '09d', 1246: '09d',
        1273: '11d', 1276: '11d',
        1282: '13d'
    }
    return icon_map.get(code, '01d')


@dataclass
class ProviderResult:
    success: bool
    provider: str
    temp: float = 0.0
    feels_like: float = 0.0
    description: str = ""
    icon: str = "01d"
    wind_kph: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    tz_offset_sec: Optional[int] = None


class OpenWeatherProvider:
    def __init__(self, http: HttpClient, api_key: str):
        self.http = http
        self.api_key = api_key

    async def geocode(self, city: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        url = "https://api.openweathermap.org/geo/1.0/direct"
        params = {"q": city, "limit": 1, "appid": self.api_key}
        status, data = await self.http.get_json(url, params, context={"provider": "OpenWeather", "endpoint": "geocode"})
        if not data:
            return None
        try:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "name": data[0].get("name", city),
                "country": data[0].get("country", "")
            }
        except Exception:
            return None

    async def current(self, lat: float, lon: float) -> ProviderResult:
        if not self.api_key:
            return ProviderResult(False, "OpenWeather")
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric", "lang": "it"}
        status, data = await self.http.get_json(url, params, context={"provider": "OpenWeather", "endpoint": "current"})
        if not data:
            return ProviderResult(False, "OpenWeather")
        try:
            wind_ms = float(data.get("wind", {}).get("speed", 0.0))
            return ProviderResult(
                True, "OpenWeather",
                temp=float(data["main"]["temp"]),
                feels_like=float(data["main"]["feels_like"]),
                description=str(data["weather"][0]["description"]).capitalize(),
                icon=str(data["weather"][0]["icon"]),
                wind_kph=wind_ms * 3.6,
                humidity=float(data["main"].get("humidity")) if data.get("main", {}).get("humidity") is not None else None,
                pressure=float(data["main"].get("pressure")) if data.get("main", {}).get("pressure") is not None else None,
                tz_offset_sec=int(data.get("timezone")) if data.get("timezone") is not None else None,
            )
        except Exception:
            return ProviderResult(False, "OpenWeather")

    async def forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric", "lang": "it", "cnt": 40}
        status, data = await self.http.get_json(url, params, context={"provider": "OpenWeather", "endpoint": "forecast"})
        return data


class WeatherAPIProvider:
    def __init__(self, http: HttpClient, api_key: str):
        self.http = http
        self.api_key = api_key

    async def current(self, lat: float, lon: float) -> ProviderResult:
        if not self.api_key:
            return ProviderResult(False, "WeatherAPI")
        url = "https://api.weatherapi.com/v1/current.json"
        params = {"key": self.api_key, "q": f"{lat},{lon}", "lang": "it"}
        status, data = await self.http.get_json(url, params, context={"provider": "WeatherAPI", "endpoint": "current"})
        if not data:
            return ProviderResult(False, "WeatherAPI")
        try:
            code = int(data["current"]["condition"]["code"])
            local_epoch = int(data.get("location", {}).get("localtime_epoch", 0))
            tz_offset = None
            if local_epoch:
                tz_offset = int(local_epoch - int(now_utc().timestamp()))
            return ProviderResult(
                True, "WeatherAPI",
                temp=float(data["current"]["temp_c"]),
                feels_like=float(data["current"]["feelslike_c"]),
                description=str(data["current"]["condition"]["text"]),
                icon=map_weatherapi_icon(code),
                wind_kph=float(data["current"].get("wind_kph")) if data["current"].get("wind_kph") is not None else None,
                humidity=float(data["current"].get("humidity")) if data["current"].get("humidity") is not None else None,
                pressure=float(data["current"].get("pressure_mb")) if data["current"].get("pressure_mb") is not None else None,
                tz_offset_sec=tz_offset,
            )
        except Exception:
            return ProviderResult(False, "WeatherAPI")

    async def forecast(self, lat: float, lon: float, days: int = 2) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        url = "https://api.weatherapi.com/v1/forecast.json"
        params = {"key": self.api_key, "q": f"{lat},{lon}", "days": days, "aqi": "no", "alerts": "no", "lang": "it"}
        status, data = await self.http.get_json(url, params, context={"provider": "WeatherAPI", "endpoint": "forecast"})
        return data

    async def history(self, lat: float, lon: float, date_str: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        url = "https://api.weatherapi.com/v1/history.json"
        params = {"key": self.api_key, "q": f"{lat},{lon}", "dt": date_str, "lang": "it"}
        status, data = await self.http.get_json(url, params, context={"provider": "WeatherAPI", "endpoint": "history"})
        return data

    async def timezone(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        url = "https://api.weatherapi.com/v1/timezone.json"
        params = {"key": self.api_key, "q": f"{lat},{lon}"}
        status, data = await self.http.get_json(url, params, context={"provider": "WeatherAPI", "endpoint": "timezone"})
        return data

    async def geocode(self, city: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        url = "https://api.weatherapi.com/v1/search.json"
        params = {"key": self.api_key, "q": city}
        status, data = await self.http.get_json(url, params, context={"provider": "WeatherAPI", "endpoint": "geocode"})
        if not data:
            return None
        try:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "name": data[0].get("name", city),
                "country": data[0].get("country", ""),
                "tz_id": data[0].get("tz_id"),
            }
        except Exception:
            return None


# ===================== METEOBLUE =====================
class MeteoblueProvider:
    def __init__(self, http: HttpClient, api_key: str):
        self.http = http
        self.api_key = api_key

    async def current(self, lat: float, lon: float) -> ProviderResult:
        if not self.api_key:
            return ProviderResult(False, "Meteoblue")
        url = "https://my.meteoblue.com/packages/current"
        params = {
            "lat": lat,
            "lon": lon,
            "apikey": self.api_key,
            "format": "json",
            "temperature": "C",
            "windspeed": "kmh",
        }
        status, data = await self.http.get_json(url, params, context={"provider": "Meteoblue", "endpoint": "current"})
        if not data:
            return ProviderResult(False, "Meteoblue")
        try:
            section = data.get("data_current", data)
            code = section.get("pictocode_detailed", section.get("pictocode"))
            desc = meteoblue_description(int(code)) if code is not None else ""
            is_day = bool(section.get("isdaylight", 1))
            icon = meteoblue_icon_from_desc(desc, is_day=is_day)
            tz_offset = None
            meta = data.get("metadata", {})
            if meta.get("utc_timeoffset") is not None:
                tz_offset = int(float(meta["utc_timeoffset"]) * 3600)
            return ProviderResult(
                True, "Meteoblue",
                temp=float(section.get("temperature", 0.0)),
                feels_like=float(section.get("felttemperature", section.get("temperature", 0.0))),
                description=desc,
                icon=icon,
                wind_kph=float(section.get("windspeed")) if section.get("windspeed") is not None else None,
                humidity=float(section.get("relativehumidity")) if section.get("relativehumidity") is not None else None,
                pressure=float(section.get("sealevelpressure")) if section.get("sealevelpressure") is not None else (
                    float(section.get("pressure")) if section.get("pressure") is not None else None
                ),
                tz_offset_sec=tz_offset,
            )
        except Exception:
            return ProviderResult(False, "Meteoblue")

    async def forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        url = "https://my.meteoblue.com/packages/basic-1h"
        params = {
            "lat": lat,
            "lon": lon,
            "apikey": self.api_key,
            "format": "json",
            "temperature": "C",
            "windspeed": "kmh",
            "forecast_days": 2,
        }
        status, data = await self.http.get_json(url, params, context={"provider": "Meteoblue", "endpoint": "forecast"})
        return data

    async def history(self, lat: float, lon: float, history_days: int = 1) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        url = "https://my.meteoblue.com/packages/basic-1h"
        params = {
            "lat": lat,
            "lon": lon,
            "apikey": self.api_key,
            "format": "json",
            "temperature": "C",
            "windspeed": "kmh",
            "history_days": int(history_days),
            "forecast_days": 0,
        }
        status, data = await self.http.get_json(url, params, context={"provider": "Meteoblue", "endpoint": "history"})
        return data

# ===================== WEATHER SERVICE (fusion + rain + accuracy) =====================
class WeatherService:
    def __init__(self, storage: Storage):
        self.storage = storage

    @staticmethod
    def _weighted_value(v1: Optional[float], w1: float, v2: Optional[float], w2: float) -> Optional[float]:
        if v1 is None and v2 is None:
            return None
        if v1 is None:
            return v2
        if v2 is None:
            return v1
        return (v1 * w1) + (v2 * w2)

    def get_dynamic_weights(self) -> Dict[str, float]:
        acc = self.storage.get_provider_accuracy()
        if not acc:
            return {"OpenWeather": 0.34, "WeatherAPI": 0.33, "Meteoblue": 0.33}
        present_vals = [v["accuracy"] for v in acc.values() if v.get("accuracy") is not None]
        avg_acc = sum(present_vals) / len(present_vals) if present_vals else 60.0
        acc_vals: Dict[str, float] = {}
        for p in PROVIDERS:
            val = acc.get(p, {}).get("accuracy")
            if val is None:
                val = max(40.0, avg_acc * 0.9)
            acc_vals[p] = float(val)
        total = sum(acc_vals.values())
        if total <= 0:
            return {p: 1 / len(PROVIDERS) for p in PROVIDERS}
        return {k: v / total for k, v in acc_vals.items()}

    def update_provider_accuracy(self, provider: str, predicted: float, actual: float):
        error = abs(predicted - actual)
        current = self.storage.get_provider_accuracy().get(provider)
        errors = list(current["errors"]) if current else []
        errors.append(float(error))
        errors = errors[-100:]
        avg_error = sum(errors) / len(errors)
        accuracy = clamp(100 - (avg_error * 15), 0, 100)
        self.storage.upsert_provider_accuracy(provider, errors, avg_error, accuracy)

    @staticmethod
    def _apply_outlier_weights(weights: Dict[str, float], temps: Dict[str, Optional[float]]) -> Dict[str, float]:
        vals = [(k, v) for k, v in temps.items() if v is not None and k in weights]
        if len(vals) < 2:
            return weights
        only_vals = sorted(v for _, v in vals)
        median = only_vals[len(only_vals) // 2] if len(only_vals) % 2 == 1 else sum(only_vals[len(only_vals)//2-1:len(only_vals)//2+1]) / 2
        w = dict(weights)
        for k, v in vals:
            if abs(float(v) - float(median)) >= OUTLIER_TEMP_C:
                w[k] = w.get(k, 0.0) * 0.2
        tot = sum(w.values())
        if tot > 0:
            w = {k: v / tot for k, v in w.items()}
        return w

    def fuse(
        self,
        ow: ProviderResult,
        wa: ProviderResult,
        mb: Optional[ProviderResult] = None,
        bias: Optional[Dict[str, float]] = None,
        weights_override: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        mb = mb or ProviderResult(False, "Meteoblue")
        if not ow.success and not wa.success and not mb.success:
            return None

        bias = bias or {}
        ow_bias = float(bias.get("OpenWeather", 0.0))
        wa_bias = float(bias.get("WeatherAPI", 0.0))
        mb_bias = float(bias.get("Meteoblue", 0.0))

        if ow.success and not wa.success and not mb.success:
            ow_temp = ow.temp - ow_bias
            ow_feels = ow.feels_like - ow_bias
            return {
                "temp": ow_temp,
                "feels_like": ow_feels,
                "description": ow.description,
                "icon": ow.icon,
                "wind_kph": ow.wind_kph,
                "humidity": ow.humidity,
                "pressure": ow.pressure,
                "sources": ["OpenWeather"],
                "provider_details": {"OpenWeather": {"temp": ow_temp, "weight": 1.0}}
            }
        if wa.success and not ow.success and not mb.success:
            wa_temp = wa.temp - wa_bias
            wa_feels = wa.feels_like - wa_bias
            return {
                "temp": wa_temp,
                "feels_like": wa_feels,
                "description": wa.description,
                "icon": wa.icon,
                "wind_kph": wa.wind_kph,
                "humidity": wa.humidity,
                "pressure": wa.pressure,
                "sources": ["WeatherAPI"],
                "provider_details": {"WeatherAPI": {"temp": wa_temp, "weight": 1.0}}
            }
        if mb.success and not ow.success and not wa.success:
            mb_temp = mb.temp - mb_bias
            mb_feels = mb.feels_like - mb_bias
            return {
                "temp": mb_temp,
                "feels_like": mb_feels,
                "description": mb.description,
                "icon": mb.icon,
                "wind_kph": mb.wind_kph,
                "humidity": mb.humidity,
                "pressure": mb.pressure,
                "sources": ["Meteoblue"],
                "provider_details": {"Meteoblue": {"temp": mb_temp, "weight": 1.0}}
            }

        weights = weights_override or self.get_dynamic_weights()
        w_map = {
            "OpenWeather": float(weights.get("OpenWeather", 0.0)),
            "WeatherAPI": float(weights.get("WeatherAPI", 0.0)),
            "Meteoblue": float(weights.get("Meteoblue", 0.0)),
        }
        available = {
            "OpenWeather": ow if ow.success else None,
            "WeatherAPI": wa if wa.success else None,
            "Meteoblue": mb if mb.success else None,
        }
        avail_weights = {k: v for k, v in w_map.items() if available.get(k)}
        tot = sum(avail_weights.values())
        if tot <= 0:
            n = len(avail_weights) or 1
            avail_weights = {k: 1 / n for k in avail_weights}
        else:
            avail_weights = {k: v / tot for k, v in avail_weights.items()}
        avail_weights = self._apply_outlier_weights(avail_weights, {
            "OpenWeather": ow.temp if ow.success else None,
            "WeatherAPI": wa.temp if wa.success else None,
            "Meteoblue": mb.temp if mb.success else None,
        })

        ow_temp = (ow.temp - ow_bias) if ow.success else None
        wa_temp = (wa.temp - wa_bias) if wa.success else None
        mb_temp = (mb.temp - mb_bias) if mb.success else None
        ow_feels = (ow.feels_like - ow_bias) if ow.success else None
        wa_feels = (wa.feels_like - wa_bias) if wa.success else None
        mb_feels = (mb.feels_like - mb_bias) if mb.success else None

        def _wavg(values: Dict[str, Optional[float]]) -> Optional[float]:
            num = 0.0
            den = 0.0
            for k, v in values.items():
                if v is None or k not in avail_weights:
                    continue
                w = avail_weights[k]
                num += float(v) * w
                den += w
            if den <= 0:
                return None
            return num / den

        temp = _wavg({"OpenWeather": ow_temp, "WeatherAPI": wa_temp, "Meteoblue": mb_temp})
        feels = _wavg({"OpenWeather": ow_feels, "WeatherAPI": wa_feels, "Meteoblue": mb_feels})
        wind_kph = _wavg({"OpenWeather": ow.wind_kph, "WeatherAPI": wa.wind_kph, "Meteoblue": mb.wind_kph})
        humidity = _wavg({"OpenWeather": ow.humidity, "WeatherAPI": wa.humidity, "Meteoblue": mb.humidity})
        pressure = _wavg({"OpenWeather": ow.pressure, "WeatherAPI": wa.pressure, "Meteoblue": mb.pressure})

        # descrizione/icona dal provider col peso maggiore
        top_provider = max(avail_weights.items(), key=lambda x: x[1])[0] if avail_weights else "OpenWeather"
        if top_provider == "OpenWeather":
            desc, icon = ow.description, ow.icon
        elif top_provider == "WeatherAPI":
            desc, icon = wa.description, wa.icon
        else:
            desc, icon = mb.description, mb.icon

        return {
            "temp": float(temp) if temp is not None else 0.0,
            "feels_like": float(feels) if feels is not None else 0.0,
            "description": desc,
            "icon": icon,
            "wind_kph": wind_kph,
            "humidity": humidity,
            "pressure": pressure,
            "sources": [k for k in avail_weights.keys()],
            "provider_details": {
                "OpenWeather": {"temp": ow_temp, "weight": avail_weights.get("OpenWeather")},
                "WeatherAPI": {"temp": wa_temp, "weight": avail_weights.get("WeatherAPI")},
                "Meteoblue": {"temp": mb_temp, "weight": avail_weights.get("Meteoblue")},
            }
        }

    def analyze_rain(self, ow: ProviderResult, wa: ProviderResult, forecast: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        rain = {"currently_raining": False, "periods": [], "max_prob": 0}
        if ow.success and is_rain_description(ow.description):
            rain["currently_raining"] = True
        if wa.success and is_rain_description(wa.description):
            rain["currently_raining"] = True

        if forecast and "list" in forecast:
            target_date = datetime.now().date()
            day_points = interpolate_hourly_points(extract_day_points(forecast, target_date))
            rain["periods"] = compute_rain_periods(day_points)
            if rain["periods"]:
                rain["max_prob"] = max(p["max_prob"] for p in rain["periods"])

        return rain

    def rain_message(self, rain: Dict[str, Any], now_local: Optional[datetime] = None) -> str:
        periods = rain.get("periods") or []
        if not now_local:
            now_local = datetime.now().astimezone()
        label_now = rain_intensity_label(float(rain.get("max_prob", 0)))
        if rain["currently_raining"]:
            if periods:
                for p in periods:
                    if p["start"] <= now_local <= p["end"]:
                        e = p["end"].strftime("%H:%M")
                        label = rain_intensity_label(float(p["max_prob"]))
                        return f"Piove adesso fino alle {e} (max {p['max_prob']:.0f}%, {label})"
            return f"Piove adesso (brevi rovesci {label_now})"
        if periods:
            future = [p for p in periods if p["start"] > now_local]
            if future:
                chosen = future[0]
                s = chosen["start"].strftime("%H:%M")
                e = chosen["end"].strftime("%H:%M")
                label = rain_intensity_label(float(chosen["max_prob"]))
                return f"Pioggia tra {s}-{e} (max {chosen['max_prob']:.0f}%, {label})"
        return ""


# ===================== CACHE KEYS =====================
def key_geo(city: str) -> str:
    return md5(f"geo:{city.strip().lower()}")

def key_current(lat: float, lon: float) -> str:
    bucket = datetime.now().hour // 2
    return md5(f"cur:{lat:.4f}:{lon:.4f}:{bucket}")

def key_forecast(lat: float, lon: float, days: int) -> str:
    bucket = datetime.now().hour // 2
    return md5(f"fc:{days}:{lat:.4f}:{lon:.4f}:{bucket}")


# ===================== FORECAST HELPERS =====================
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
        # outlier handling per slot
        temps_map = {}
        if ow_p:
            temps_map["OpenWeather"] = ow_p["temp"]
        if wa_p:
            temps_map["WeatherAPI"] = wa_p["temp"]
        if mb_p:
            temps_map["Meteoblue"] = mb_p["temp"]
        if len(temps_map) >= 2:
            vals = sorted(temps_map.values())
            median = vals[len(vals) // 2] if len(vals) % 2 == 1 else sum(vals[len(vals)//2-1:len(vals)//2+1]) / 2
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


# ===================== MESSAGE FORMAT =====================
def format_meteo_message(
    fused: Dict[str, Any],
    city: str,
    country: str,
    min_t: Optional[float],
    max_t: Optional[float],
    rain_msg: str,
    sources_label: str,
    cache_age_min: Optional[int],
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
    icon = get_weather_icon(fused.get("icon", "01d"))
    if local_dt is None:
        local_dt = datetime.now()
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
    lines = [
        f"{icon} METEO {city_md}, {country_md}",
        f"{date_label} {time_label}",
        "",
        f"\U0001F321 Ora: {format_temp(fused.get('temp'), prefs)} (Percepita {format_temp(fused.get('feels_like'), prefs)})",
        f"{icon} Condizioni: {desc_md}",
    ]
    if rain_msg:
        lines.append(f"\U0001F327\ufe0f Pioggia: {rain_msg}")
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
        icon = get_weather_icon(str(p.get("icon", "01d")))
        rain = f" (POP {p['pop']:.0f}%)" if p["pop"] > 20 else ""
        lines.append(f"- {p['ora']}: {format_temp(p.get('temp'), prefs)}{rain} {icon} {conf}%")
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


# ===================== APP STATE =====================
@dataclass
class AppState:
    storage: Storage
    ram_cache: RamCache
    http: HttpClient
    ow: OpenWeatherProvider
    wa: WeatherAPIProvider
    mb: MeteoblueProvider
    weather: WeatherService


async def _on_shutdown(app: Application):
    state: Optional[AppState] = app.bot_data.get("state")
    if state:
        await state.http.close()


# ===================== GEO (cached) =====================
async def get_coords(state: AppState, city: str) -> Optional[Dict[str, Any]]:
    k = key_geo(city)

    # 1) RAM
    cached = state.ram_cache.get(k)
    if cached:
        return cached

    # 2) SQLite
    cached = state.storage.cache_get(k)
    if cached:
        state.ram_cache.set(k, cached, ttl_seconds=60)
        return cached

    # 3) API
    coords = await state.ow.geocode(city)
    if not coords:
        coords = await state.wa.geocode(city)
    if coords:
        tz_offset = None
        tz_id = coords.get("tz_id")
        if tz_id:
            tz_offset = tz_offset_from_tzid(tz_id)
        if tz_offset is None and WEATHERAPI_KEY:
            tz_data = await state.wa.timezone(coords["lat"], coords["lon"])
            if tz_data and tz_data.get("location", {}).get("tz_id"):
                tz_offset = tz_offset_from_tzid(tz_data["location"]["tz_id"])
        if tz_offset is not None:
            coords["tz_offset_sec"] = int(tz_offset)
        state.storage.cache_set(k, coords, ttl=timedelta(hours=CACHE_TTL_GEOCODE_HOURS))
        state.ram_cache.set(k, coords, ttl_seconds=60)
    return coords


# ===================== COMMANDS =====================
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
        "- /setcitta\n"
        "- /citta\n"
        "- /aggiorna\n\n"
        "Per vedere tutti i comandi disponibili: /comandi"
    )

    await update.effective_message.reply_text(msg, parse_mode="Markdown", reply_markup=build_reply_keyboard())


async def comandi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Comandi disponibili\n"
        "- /meteo - meteo attuale\n"
        "- /aggiorna - forza aggiornamento meteo\n"
        "- /oggi - riepilogo giornata\n"
        "- /prev - ora per ora fino a mezzanotte\n"
        "- /domani - previsioni domani\n"
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
                fused, coords["name"], coords["country"], min_t, max_t, rain_msg, sources_label, age_min, local_dt, reliability, fallback_note, details_line, prefs=prefs
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
        fused, coords["name"], coords["country"], min_t, max_t, rain_msg, sources_label, None, local_dt, reliability, fallback_note, details_line, prefs=prefs
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
    else:
        cached_meta, created_at, _, expired = state.storage.cache_get_with_meta(ck, allow_expired=True)
        if cached_meta and not expired:
            forecast_ow = cached_meta.get("forecast_ow")
            forecast_wa = cached_meta.get("forecast_wa")
            forecast_mb = cached_meta.get("forecast_mb")
            cached_used = True
            state.ram_cache.set(ck, cached_meta, ttl_seconds=30)
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

    forecast_ow, forecast_wa, forecast_mb, points_ow, points_wa, points_mb, fused_points, target_date, tz_offset, sources_label, cached_used, cache_stale, _ = await _load_forecast_points(
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
        msg += "\nDato: Cache"
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
        icon = get_weather_icon(str(p.get("icon", "01d")))
        lines.append(f"{h:02d}:00 {format_temp(p.get('temp'), prefs)} {conf}% {icon}")

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
    token = context.args[0] if context.args else None
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
        db_size_kb = int(os.path.getsize(DB_FILE) / 1024) if os.path.exists(DB_FILE) else 0
        lines.append("DB")
        lines.append(f"- Cache: {cache_count}")
        lines.append(f"- Predizioni: {pred_total} (verificate {pred_ok})")
        lines.append(f"- Errori totali: {err_total}")
        lines.append(f"- DB size: {db_size_kb} KB")
    except Exception:
        pass

    rows = state.storage.get_recent_errors(5)
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
            _, _, _, points_ow, points_wa, points_mb, fused_points, target_date, _, _, _ = await _load_forecast_points(
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


# ===================== MAIN =====================

def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN mancante nel .env")
    if not OPENWEATHER_API_KEY and not WEATHERAPI_KEY and not METEOBLUE_API_KEY:
        raise RuntimeError("Configura almeno un provider nel .env")

    storage = Storage(DB_FILE)
    ram_cache = RamCache(storage)
    http = HttpClient(storage)

    ow = OpenWeatherProvider(http, OPENWEATHER_API_KEY or "")
    wa = WeatherAPIProvider(http, WEATHERAPI_KEY or "")
    mb = MeteoblueProvider(http, METEOBLUE_API_KEY or "")
    weather = WeatherService(storage)

    app = Application.builder().token(BOT_TOKEN).post_shutdown(_on_shutdown).build()
    app.bot_data["state"] = AppState(storage, ram_cache, http, ow, wa, mb, weather)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("setcitta", setcitta))
    app.add_handler(CommandHandler("mycitta", mycitta))
    app.add_handler(CommandHandler("citta", citta))
    app.add_handler(CommandHandler("delcitta", delcitta))
    app.add_handler(CommandHandler("meteo", meteo))
    app.add_handler(CommandHandler("aggiorna", aggiorna))
    app.add_handler(CommandHandler("oggi", oggi))
    app.add_handler(CommandHandler("prev", prev))
    app.add_handler(CommandHandler("domani", domani))
    app.add_handler(CommandHandler("controlla", controlla))
    app.add_handler(CommandHandler("comandi", comandi))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("versione", versione))
    app.add_handler(CommandHandler("versioni", versione))
    app.add_handler(CommandHandler("pref", pref))
    app.add_handler(CommandHandler("setpref", setpref))
    app.add_handler(CommandHandler("stat", stat))
    app.add_handler(CommandHandler("admin", admin))
    app.add_handler(CommandHandler("testoffline", testoffline))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_buttons))

    if app.job_queue:
        app.job_queue.run_repeating(check_alerts_job, interval=ALERT_CHECK_MIN * 60, first=60)
        app.job_queue.run_repeating(daily_check_job, interval=20 * 60, first=120)
        app.job_queue.run_repeating(tomorrow_rain_job, interval=20 * 60, first=180)

    logger.info("Bot avviato.")
    if WEBHOOK_URL:
        global HEALTH_PATH_EFFECTIVE
        if WEBHOOK_PATH == HEALTH_PATH:
            logger.warning("HEALTH_PATH uguale a WEBHOOK_PATH, uso /health per healthcheck.")
            health_path = "/health"
        else:
            health_path = HEALTH_PATH
        HEALTH_PATH_EFFECTIVE = health_path
        if TelegramHandler and tornado:
            try:
                import telegram.ext._updater as tg_updater
                tg_updater.WebhookAppClass = CustomWebhookApp  # type: ignore
            except Exception:
                pass
        webhook_url = WEBHOOK_URL.rstrip("/") + WEBHOOK_PATH
        logger.info("Webhook attivo su %s", webhook_url)
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=WEBHOOK_PATH.lstrip("/"),
            webhook_url=webhook_url,
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

