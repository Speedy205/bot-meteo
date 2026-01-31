from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import aiohttp

from config import HTTP_TIMEOUT_SEC
from storage import Storage
from utils import iso, log_event, logger, meteoblue_description, meteoblue_icon_from_desc, now_utc


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
