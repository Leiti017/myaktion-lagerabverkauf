# ki_engine_price.py – MyAktion Lagerabverkauf Preis-Engine
# Zweck: Foto -> Listenpreis (Neupreis-Schätzung in EUR)
# Antwort: float (oder None bei Fehler)

from __future__ import annotations

import base64
import os
import re
from typing import Optional

import requests

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """
Du bist ein extrem präziser Preis-Assistent für MyAktion Lagerabverkauf.
Du siehst ein Produktfoto und schätzt den realistischen, tagesaktuellen NEUPREIS (Listenpreis) in EUR.

Regeln:
- Wenn das Produkt nicht eindeutig erkennbar ist: gib 0.00 zurück (statt zu raten).
- Wenn Preisschilder/UVP/Etiketten klar lesbar sind, nutze diese als starke Evidenz.
- Wenn du schätzen musst: eher leicht höher als zu niedrig.
- Gib NUR eine Zeile aus im Format:
PRICE_EUR=123.45
- Dezimaltrennzeichen ist ein Punkt.
""".strip()

def _encode_image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _parse_price(text: str) -> Optional[float]:
    m = re.search(r"PRICE_EUR\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def estimate_list_price_eur(image_path: str) -> Optional[float]:
    if not OPENAI_API_KEY:
        print("[KI] Fehler: OPENAI_API_KEY fehlt")
        return None

    b64 = _encode_image_to_b64(image_path)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Bitte streng im Format antworten: PRICE_EUR=..."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 60,
    }

    try:
        r = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=25)
        if r.status_code != 200:
            print("[KI-DEBUG] HTTP:", r.status_code)
            print("[KI-DEBUG] BODY:", r.text)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return _parse_price(content)
    except Exception as e:
        print("[KI] Fehler:", e)
        return None
