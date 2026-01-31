from __future__ import annotations

from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters

from app_state import AppState, _on_shutdown
from commands import (
    admin, aggiorna, aiuto, citta, comandi, controlla, delcitta, domani, handle_callback,
    handle_text_buttons, info, meteo, mycitta, news, oggi, pref, prev, setcitta, setpref,
    start, stat, testoffline, trend, versione
)
import config as config_module
from config import (
    ALERT_CHECK_MIN, BOT_TOKEN, DB_FILE, HEALTH_PATH, METEOBLUE_API_KEY, OPENWEATHER_API_KEY,
    PORT, WEATHERAPI_KEY, WEBHOOK_PATH, WEBHOOK_URL
)
from jobs import check_alerts_job, daily_check_job, morning_meteo_job, tomorrow_rain_job
from providers import HttpClient, MeteoblueProvider, OpenWeatherProvider, WeatherAPIProvider
from storage import RamCache, Storage
from utils import CustomWebhookApp, TelegramHandler, logger, tornado
from weather_service import WeatherService


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
    app.add_handler(CommandHandler("trend", trend))
    app.add_handler(CommandHandler("domani", domani))
    app.add_handler(CommandHandler("controlla", controlla))
    app.add_handler(CommandHandler("comandi", comandi))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("versione", versione))
    app.add_handler(CommandHandler("versioni", versione))
    app.add_handler(CommandHandler("news", news))
    app.add_handler(CommandHandler("pref", pref))
    app.add_handler(CommandHandler("setpref", setpref))
    app.add_handler(CommandHandler("stat", stat))
    app.add_handler(CommandHandler("admin", admin))
    app.add_handler(CommandHandler("aiuto", aiuto))
    app.add_handler(CommandHandler("testoffline", testoffline))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_buttons))

    if app.job_queue:
        app.job_queue.run_repeating(check_alerts_job, interval=ALERT_CHECK_MIN * 60, first=60)
        app.job_queue.run_repeating(daily_check_job, interval=20 * 60, first=120)
        app.job_queue.run_repeating(tomorrow_rain_job, interval=20 * 60, first=180)
        app.job_queue.run_repeating(morning_meteo_job, interval=60 * 60, first=240)

    logger.info("Bot avviato.")
    if WEBHOOK_URL:
        if WEBHOOK_PATH == HEALTH_PATH:
            logger.warning("HEALTH_PATH uguale a WEBHOOK_PATH, uso /health per healthcheck.")
            health_path = "/health"
        else:
            health_path = HEALTH_PATH
        config_module.HEALTH_PATH_EFFECTIVE = health_path
        if TelegramHandler and tornado:
            try:
                import telegram.ext._updater as tg_updater
                import telegram.ext._utils.webhookhandler as webhookhandler
                tg_updater.WebhookAppClass = CustomWebhookApp  # type: ignore
                webhookhandler.WebhookAppClass = CustomWebhookApp  # type: ignore
            except Exception:
                logger.warning("Impossibile registrare /health nel webhook server.")
        else:
            logger.warning("Webhook attivo senza /health (tornado non disponibile).")
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
