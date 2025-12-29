import os
import json
import base64
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _prompt_for_price(art_id: str) -> str:
    return (
        "Du bist ein Preis-Analyst für Handel/Marktpreise.\n"
        "Aufgabe: Schätze einen realistischen aktuellen Neupreis (Listen-/Marktpreis) in EUR für das Produkt.\n"
        "Wenn ein Preis/UVP/Etikett im Bild sichtbar ist, nutze diesen.\n"
        "Gib am Ende NUR JSON aus, ohne zusätzlichen Text.\n"
        "Schema:\n"
        "{"
        '"name":"",'
        '"brand":"",'
        '"model":"",'
        '"retail_price": 0'
        "}\n"
    )


def generate_meta(image_path: str, art_id: str = "lagerabverkauf", context=None):
    if not OPENAI_API_KEY:
        return {"error": "missing_openai_api_key"}

    b64 = _b64_image(image_path)

    ctx = ""
    if isinstance(context, dict):
        ctx = json.dumps(context)[:4000]

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _prompt_for_price(art_id)},
            {"role": "user", "content": [
                {"type": "text", "text": f"Kontext (optional, vorherige Fotos): {ctx}" if ctx else "Kein Kontext."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]},
        ],
        "temperature": 0.2,
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )

    if r.status_code != 200:
        return {"error": "openai_http_error", "status": r.status_code, "body": r.text[:800]}

    data = r.json()
    text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
        return {"error": "invalid_json_from_model", "raw": text[:800]}
