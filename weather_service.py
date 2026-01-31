from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime

from config import OUTLIER_TEMP_C, PROVIDERS
from providers import ProviderResult
from storage import Storage
from utils import clamp, is_rain_description
from forecast import compute_rain_periods, extract_day_points, interpolate_hourly_points


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
        if len(only_vals) % 2 == 1:
            median = only_vals[len(only_vals) // 2]
        else:
            median = sum(only_vals[len(only_vals) // 2 - 1:len(only_vals) // 2 + 1]) / 2
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
        if not rain.get("periods"):
            if rain.get("currently_raining"):
                return "Piove adesso"
            return "Nessuna pioggia prevista"
        periods = rain["periods"]
        top = periods[0]
        start = top["start"]
        end = top["end"]
        max_prob = top["max_prob"]
        duration = int((end - start).total_seconds() / 60)
        if now_local and start <= now_local <= end:
            return f"Piove ora (fino alle {end.strftime('%H:%M')}, max {max_prob}%)"
        return f"Pioggia dalle {start.strftime('%H:%M')} alle {end.strftime('%H:%M')} ({duration}m, max {max_prob}%)"
