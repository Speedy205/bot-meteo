from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from utils import from_iso, iso, now_utc


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

    def cache_delete(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE cache_key=?", (key,))
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

    def delete(self, key: str) -> None:
        self.data.pop(key, None)
