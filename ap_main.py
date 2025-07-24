import os
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from ap_utils import (
    ap_command, p_command, rsi_macd_command,
    add_favorite, delete_favorite, set_alert_threshold
)
from ap_jobs import schedule_daily_jobs

def setup_bot():
    token = os.getenv("TELEGRAM_TOKEN")
    updater = Updater(token, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("ap", ap))
    dp.add_handler(CommandHandler("p", p_price))
    dp.add_handler(CommandHandler("trend", trend))
    dp.add_handler(CommandHandler("add", add_fav))
    dp.add_handler(CommandHandler("delete", del_fav))
    dp.add_handler(CommandHandler("setalert", set_alert))

    schedule_daily_jobs(updater.job_queue)
    return updater

def start_bot(updater):
    updater.start_polling()
    updater.idle()

def ap(update: Update, context: CallbackContext):
    arg1 = context.args[0] if context.args else "24h"
    days = int(arg1.replace("d", "")) if arg1.isdigit() else None
    update.message.reply_text(ap_command(arg1, days))

def p_price(update: Update, context: CallbackContext):
    coins = context.args
    if not coins:
        update.message.reply_text("Kullanım: /p btc eth ...")
        return
    update.message.reply_text(p_command(coins))

def trend(update: Update, context: CallbackContext):
    coins = context.args if context.args else ["btc", "eth"]
    update.message.reply_text(rsi_macd_command(coins))

def add_fav(update: Update, context: CallbackContext):
    if len(context.args) < 2:
        update.message.reply_text("Kullanım: /add F1 /p btc eth ...")
        return
    fav = context.args[0]
    coins = context.args[1:]
    update.message.reply_text(add_favorite(fav, coins))

def del_fav(update: Update, context: CallbackContext):
    if not context.args:
        update.message.reply_text("Kullanım: /delete F1")
        return
    fav = context.args[0]
    update.message.reply_text(delete_favorite(fav))

def set_alert(update: Update, context: CallbackContext):
    if len(context.args) < 2:
        update.message.reply_text("Kullanım: /setalert 20 10")
        return
    update.message.reply_text(set_alert_threshold(context.args[0], context.args[1]))
