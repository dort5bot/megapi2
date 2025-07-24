from datetime import time
from ap_utils import save_daily_history, ALERT_FILE
import json, os

def schedule_daily_jobs(job_queue):
    # Her gün saat 03:05 UTC+3 kayıt
    job_queue.run_daily(daily_record, time(hour=3, minute=5))

def daily_record(context):
    vsbtc, alt, longt = save_daily_history()
    alert = {"level1": 20, "level2": 10}

    if os.path.exists(ALERT_FILE):
        with open(ALERT_FILE, "r") as f:
            alert = json.load(f)

    msgs = []
    if vsbtc < alert['level1']:
        msgs.append(f"⚠ Altların Btc'ye Karşı Gücü: {vsbtc}")
    if alt < alert['level1']:
        msgs.append(f"⚠ Altların Gücü: {alt}")
    if longt < alert['level1']:
        msgs.append(f"⚠ Uzun Vadede Güç: {longt}")

    if vsbtc < alert['level2'] or alt < alert['level2'] or longt < alert['level2']:
        msgs.append("✅ Güçlü alım dönemi")

    if msgs:
        context.bot.send_message(chat_id=context.job.context, text="\n".join(msgs))
