from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


load_dotenv()


def _get_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _get_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except Exception:
        return default


def _get_str(value: Optional[str], default: str) -> str:
    return value if value is not None else default


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


@dataclass
class Config:
    # Tokens / endpoints
    bot_token: Optional[str] = None
    openweather_api_key: Optional[str] = None
    weatherapi_key: Optional[str] = None
    meteoblue_api_key: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_path: str = "/telegram-webhook"
    health_path: str = "/health"
    port: int = 8080
    admin_secret: str = "Fight club"
    admin_user_ids: str = ""

    # Versions / releases
    bot_version: str = "3.0"
    bot_releases: List[Dict[str, Any]] = field(default_factory=list)

    # Storage / HTTP
    db_file: str = "bot_data.sqlite3"
    http_timeout_sec: int = 8
    cache_max_items: int = 300

    # Providers
    providers: List[str] = field(default_factory=lambda: ["OpenWeather", "WeatherAPI", "Meteoblue"])

    # TTL
    cache_ttl_current_min: int = 10
    cache_ttl_forecast_min: int = 30
    cache_ttl_geocode_hours: int = 24
    night_forecast_ttl_mult: int = 2

    # Thresholds / parameters
    rain_pop_threshold: int = 30
    umbrella_pop_threshold: int = 50
    wind_strong_kph: int = 35
    feels_like_diff_c: int = 4
    rain_keywords: List[str] = field(default_factory=lambda: [
        "pioggia", "pioviggine", "temporale", "rovesci", "acquazzone",
        "rain", "drizzle", "storm", "shower", "precipitation"
    ])

    decay_half_life_days: int = 45
    outlier_temp_c: float = 4.0
    pop_calibration_days: int = 60

    alert_pop_threshold: int = 60
    alert_wind_kph: int = 45
    alert_temp_hot: int = 35
    alert_temp_cold: int = -5
    alert_cooldown_min: int = 180
    alert_check_min: int = 20
    daily_check_hour: int = 23
    daily_check_min: int = 0
    tomorrow_rain_hour: int = 18
    tomorrow_rain_window_min: int = 60
    offline_test_days: int = 30
    morning_meteo_hour: int = 8

    default_temp_unit: str = "C"
    default_wind_unit: str = "kmh"
    default_alert_start_hour: int = 7
    default_alert_end_hour: int = 22

    msg_service_unavailable: str = "Servizio temporaneamente non disponibile, riprova."
    msg_city_not_found: str = "Citta non trovata. Controlla il nome o usa /setcitta."
    msg_need_city: str = "Specifica una citta o usa /setcitta o /citta."

    verify_window_min: int = 90

    @classmethod
    def load(cls, json_path: Optional[str] = None) -> "Config":
        base = cls()

        path = json_path or os.getenv("CONFIG_JSON") or os.getenv("CONFIG_PATH")
        if path:
            data = _load_json(path)
            for key, value in data.items():
                if hasattr(base, key):
                    setattr(base, key, value)

        base.bot_token = os.getenv("BOT_TOKEN", base.bot_token)
        base.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", base.openweather_api_key)
        base.weatherapi_key = os.getenv("WEATHERAPI_KEY", base.weatherapi_key)
        base.meteoblue_api_key = os.getenv("METEOBLUE_API_KEY", base.meteoblue_api_key)
        base.webhook_url = os.getenv("WEBHOOK_URL", base.webhook_url)
        base.webhook_path = _get_str(os.getenv("WEBHOOK_PATH"), base.webhook_path)
        base.health_path = _get_str(os.getenv("HEALTH_PATH"), base.health_path)
        base.port = _get_int(os.getenv("PORT"), base.port)
        base.admin_secret = _get_str(os.getenv("ADMIN_SECRET"), base.admin_secret)
        base.admin_user_ids = _get_str(os.getenv("ADMIN_USER_IDS"), base.admin_user_ids)

        base.db_file = _get_str(os.getenv("DB_FILE"), base.db_file)
        base.http_timeout_sec = _get_int(os.getenv("HTTP_TIMEOUT_SEC"), base.http_timeout_sec)
        base.cache_max_items = _get_int(os.getenv("CACHE_MAX_ITEMS"), base.cache_max_items)

        base.cache_ttl_current_min = _get_int(os.getenv("CACHE_TTL_CURRENT_MIN"), base.cache_ttl_current_min)
        base.cache_ttl_forecast_min = _get_int(os.getenv("CACHE_TTL_FORECAST_MIN"), base.cache_ttl_forecast_min)
        base.cache_ttl_geocode_hours = _get_int(os.getenv("CACHE_TTL_GEOCODE_HOURS"), base.cache_ttl_geocode_hours)
        base.night_forecast_ttl_mult = _get_int(os.getenv("NIGHT_FORECAST_TTL_MULT"), base.night_forecast_ttl_mult)

        base.rain_pop_threshold = _get_int(os.getenv("RAIN_POP_THRESHOLD"), base.rain_pop_threshold)
        base.umbrella_pop_threshold = _get_int(os.getenv("UMBRELLA_POP_THRESHOLD"), base.umbrella_pop_threshold)
        base.wind_strong_kph = _get_int(os.getenv("WIND_STRONG_KPH"), base.wind_strong_kph)
        base.feels_like_diff_c = _get_int(os.getenv("FEELS_LIKE_DIFF_C"), base.feels_like_diff_c)
        base.decay_half_life_days = _get_int(os.getenv("DECAY_HALF_LIFE_DAYS"), base.decay_half_life_days)
        base.outlier_temp_c = _get_float(os.getenv("OUTLIER_TEMP_C"), base.outlier_temp_c)
        base.pop_calibration_days = _get_int(os.getenv("POP_CALIBRATION_DAYS"), base.pop_calibration_days)

        base.alert_pop_threshold = _get_int(os.getenv("ALERT_POP_THRESHOLD"), base.alert_pop_threshold)
        base.alert_wind_kph = _get_int(os.getenv("ALERT_WIND_KPH"), base.alert_wind_kph)
        base.alert_temp_hot = _get_int(os.getenv("ALERT_TEMP_HOT"), base.alert_temp_hot)
        base.alert_temp_cold = _get_int(os.getenv("ALERT_TEMP_COLD"), base.alert_temp_cold)
        base.alert_cooldown_min = _get_int(os.getenv("ALERT_COOLDOWN_MIN"), base.alert_cooldown_min)
        base.alert_check_min = _get_int(os.getenv("ALERT_CHECK_MIN"), base.alert_check_min)
        base.daily_check_hour = _get_int(os.getenv("DAILY_CHECK_HOUR"), base.daily_check_hour)
        base.daily_check_min = _get_int(os.getenv("DAILY_CHECK_MIN"), base.daily_check_min)
        base.tomorrow_rain_hour = _get_int(os.getenv("TOMORROW_RAIN_HOUR"), base.tomorrow_rain_hour)
        base.tomorrow_rain_window_min = _get_int(os.getenv("TOMORROW_RAIN_WINDOW_MIN"), base.tomorrow_rain_window_min)
        base.offline_test_days = _get_int(os.getenv("OFFLINE_TEST_DAYS"), base.offline_test_days)
        base.morning_meteo_hour = _get_int(os.getenv("MORNING_METEO_HOUR"), base.morning_meteo_hour)

        base.default_temp_unit = _get_str(os.getenv("DEFAULT_TEMP_UNIT"), base.default_temp_unit)
        base.default_wind_unit = _get_str(os.getenv("DEFAULT_WIND_UNIT"), base.default_wind_unit)
        base.default_alert_start_hour = _get_int(os.getenv("DEFAULT_ALERT_START_HOUR"), base.default_alert_start_hour)
        base.default_alert_end_hour = _get_int(os.getenv("DEFAULT_ALERT_END_HOUR"), base.default_alert_end_hour)

        base.verify_window_min = _get_int(os.getenv("VERIFY_WINDOW_MIN"), base.verify_window_min)
        return base


config = Config.load()

BOT_TOKEN = config.bot_token
OPENWEATHER_API_KEY = config.openweather_api_key
WEATHERAPI_KEY = config.weatherapi_key
METEOBLUE_API_KEY = config.meteoblue_api_key
WEBHOOK_URL = config.webhook_url
WEBHOOK_PATH = config.webhook_path
HEALTH_PATH = config.health_path
PORT = config.port
ADMIN_SECRET = config.admin_secret
ADMIN_USER_IDS = config.admin_user_ids
HEALTH_PATH_EFFECTIVE = HEALTH_PATH

BOT_VERSION = config.bot_version
BOT_RELEASES = config.bot_releases or [
    {
        "version": "3.0",
        "changes": [
            "Riga rischio pioggia (basso/medio/alto) nelle previsioni",
            "Nuovo comando /trend con riassunto rapido giornata",
            "Admin con parametri, pesi e bias provider",
        ],
    },
    {
        "version": "2.9",
        "changes": [
            "Emoji meteo separate: cielo + rischio pioggia",
            "Pioggia non mostrata se solo rischio basso",
            "Dopo /controlla invalida la cache forecast per aggiornare /domani",
        ],
    },
    {
        "version": "2.8",
        "changes": [
            "Comando /news con changelog esteso",
            "Aggiornamenti UX e dashboard admin",
        ],
    },
    {
        "version": "2.7",
        "changes": [
            "Endpoint /health per Railway/monitoraggio",
            "Comando /admin con stato completo (errori, backoff, cache)",
            "Supporto dominio webhook e path configurabili",
            "Comando /aiuto con guida completa",
            "Dashboard admin estesa con utenti e accuracy",
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

DB_FILE = config.db_file
HTTP_TIMEOUT_SEC = config.http_timeout_sec
CACHE_MAX_ITEMS = config.cache_max_items
PROVIDERS = config.providers

CACHE_TTL_CURRENT_MIN = config.cache_ttl_current_min
CACHE_TTL_FORECAST_MIN = config.cache_ttl_forecast_min
CACHE_TTL_GEOCODE_HOURS = config.cache_ttl_geocode_hours
NIGHT_FORECAST_TTL_MULT = config.night_forecast_ttl_mult

RAIN_POP_THRESHOLD = config.rain_pop_threshold
UMBRELLA_POP_THRESHOLD = config.umbrella_pop_threshold
WIND_STRONG_KPH = config.wind_strong_kph
FEELS_LIKE_DIFF_C = config.feels_like_diff_c
RAIN_KEYWORDS = config.rain_keywords

DECAY_HALF_LIFE_DAYS = config.decay_half_life_days
OUTLIER_TEMP_C = config.outlier_temp_c
POP_CALIBRATION_DAYS = config.pop_calibration_days

ALERT_POP_THRESHOLD = config.alert_pop_threshold
ALERT_WIND_KPH = config.alert_wind_kph
ALERT_TEMP_HOT = config.alert_temp_hot
ALERT_TEMP_COLD = config.alert_temp_cold
ALERT_COOLDOWN_MIN = config.alert_cooldown_min
ALERT_CHECK_MIN = config.alert_check_min
DAILY_CHECK_HOUR = config.daily_check_hour
DAILY_CHECK_MIN = config.daily_check_min
TOMORROW_RAIN_HOUR = config.tomorrow_rain_hour
TOMORROW_RAIN_WINDOW_MIN = config.tomorrow_rain_window_min
OFFLINE_TEST_DAYS = config.offline_test_days
MORNING_METEO_HOUR = config.morning_meteo_hour

DEFAULT_TEMP_UNIT = config.default_temp_unit
DEFAULT_WIND_UNIT = config.default_wind_unit
DEFAULT_ALERT_START_HOUR = config.default_alert_start_hour
DEFAULT_ALERT_END_HOUR = config.default_alert_end_hour

MSG_SERVICE_UNAVAILABLE = config.msg_service_unavailable
MSG_CITY_NOT_FOUND = config.msg_city_not_found
MSG_NEED_CITY = config.msg_need_city

VERIFY_WINDOW_MIN = config.verify_window_min
