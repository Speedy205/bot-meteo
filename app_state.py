from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from telegram.ext import Application

from providers import HttpClient, MeteoblueProvider, OpenWeatherProvider, WeatherAPIProvider
from storage import RamCache, Storage
from weather_service import WeatherService

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
