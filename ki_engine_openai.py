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
        "Du bist ein Preis-Analyst für Handel/Marktpreise (MyAktion Lagerabverkauf).\n"
        "Aufgabe: Bestimme den realistischen aktuellen NEUPREIS/Marktpreis (Listen-/Marktpreis) in EUR\n"
        "für GENAU DIESE Packungsgröße/Variante, die auf den Fotos zu sehen ist.\n\n"
        "WICHTIG (Multi-Foto-Logik):\n"
        "- Es können mehrere Fotos kommen (Vorderseite + Rückseite + Preisetikett).\n"
        "- Nutze ALLE Fotos zusammen, um Produkt + Größe (g/ml/Stk) sicher zu bestimmen.\n"
        "- Wenn auf einem Foto Gramm/ml/Stückzahl steht und auf einem anderen Foto nur das Produkt: kombiniere.\n"
        "- Wenn ein Preis/UVP/Etikett klar lesbar ist: nutze diesen als 'label_price_eur' und setze retail_price darauf.\n"
        "- Wenn nur Grundpreis (z.B. €/100g, €/kg, €/L) + Füllmenge sichtbar ist: rechne den Gesamtpreis aus.\n"
        "- Wenn Produkt NICHT eindeutig erkennbar ist oder Größe fehlt und du nur raten müsstest: retail_price = 0.\n\n"
        "Gib am Ende NUR JSON aus, ohne zusätzlichen Text.\n"
        "Schema (alle Felder optional, aber retail_price ist Pflicht):\n"
        "{\n"
        '  "name":"",\n'
        '  "brand":"",\n'
        '  "model":"",\n'
        '  "ean":"",\n'
        '  "quantity": 0,\n'
        '  "unit":"g|kg|ml|l|stk",\n'
        '  "label_price_eur": 0,\n'
        '  "unit_price_eur": 0,\n'
        '  "unit_price_basis":"per_100g|per_kg|per_100ml|per_l|per_piece|unknown",\n'
        '  "retail_price": 0,\n'
        '  "confidence": 0.0\n'
        "}\n"
    )



def generate_meta_multi(image_paths: list[str], art_id: str = "lagerabverkauf", context=None):
    """
    Sendet mehrere Fotos in EINEM Call (Front/Back/Etikett) damit Gramm/ml/Preis sauber kombiniert werden.
    Gibt dict gemäß JSON-Schema zurück.
    """
    if not OPENAI_API_KEY:
        return {"error": "missing_openai_api_key"}

    ctx = ""
    if isinstance(context, dict):
        ctx = json.dumps(context)[:4000]

    # Multi-image content
    content = []
    content.append({"type": "text", "text": f"Kontext (optional, vorherige Fotos): {ctx}" if ctx else "Kein Kontext."})
    for p in image_paths[:6]:  # safety limit
        b64 = _b64_image(p)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _prompt_for_price(art_id)},
            {"role": "user", "content": content},
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
