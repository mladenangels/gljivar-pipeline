"""Telegram message to Mladen. Token and chat from TELEGRAM_TOKEN / TELEGRAM_CHAT_ID, or
locally from ~/.claude/telegram-config.json.

Usage: notify.py "text" [photo.png]
"""
import json
import os
import sys
from pathlib import Path

import requests


def creds():
    if os.environ.get("TELEGRAM_TOKEN"):
        return os.environ["TELEGRAM_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    cfg = json.loads((Path.home() / ".claude" / "telegram-config.json").read_text())
    return cfg["bot_token"], cfg["chat_id"]


def send(text, photo=None):
    token, chat = creds()
    api = f"https://api.telegram.org/bot{token}"
    if photo and Path(photo).exists():
        with open(photo, "rb") as f:
            r = requests.post(f"{api}/sendPhoto", data={"chat_id": chat, "caption": text[:1000]},
                              files={"photo": f}, timeout=60)
    else:
        r = requests.post(f"{api}/sendMessage", data={"chat_id": chat, "text": text[:4000]}, timeout=60)
    print("telegram:", r.json().get("ok"))


if __name__ == "__main__":
    send(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
