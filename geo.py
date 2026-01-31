from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

from app_state import AppState
from config import CACHE_TTL_GEOCODE_HOURS, WEATHERAPI_KEY
from forecast import key_geo
from utils import tz_offset_from_tzid

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
