import os
import json
import base64
import requests
import re
from typing import Any, Dict, Optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/chat/completions").strip()


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _prompt_for_price(art_id: str) -> str:
    # Extra-strict JSON-only prompt (helps avoid invalid_json_from_model)
    return (
        "Du bist ein Preis-Analyst für Handel/Marktpreise.\n"
        "Aufgabe: Schätze einen realistischen aktuellen Neupreis (Listen-/Marktpreis) in EUR für das Produkt.\n"
        "WICHTIG:\n"
        "- Wenn ein Preis/UVP/Etikett im Bild klar sichtbar ist: nutze diesen Preis.\n"
        "- Wenn Grundpreis (€/100g, €/kg, €/L) und Füllmenge (g/ml/L/Stk) sichtbar sind: rechne retail_price daraus.\n"
        "- Wenn Produkt nicht eindeutig erkennbar ist: retail_price = 0.\n"
        "- Antworte NUR mit einem JSON-Objekt (kein Text davor/danach).\n"
        "Schema:\n"
        "{"
        "\"name\":\"\","
        "\"brand\":\"\","
        "\"model\":\"\","
        "\"retail_price\":0,"
        "\"notes\":\"\""
        "}\n"
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # try to salvage a JSON object within text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            chunk = text[start : end + 1]
            try:
                return json.loads(chunk)
            except Exception:
                return None
    return None


def _coerce_price(meta: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure retail_price is numeric
    if not isinstance(meta, dict):
        return {"error": "meta_not_dict"}

    rp = meta.get("retail_price", meta.get("price", 0))
    try:
        meta["retail_price"] = float(rp)
    except Exception:
        # try regex fallback
        m = re.search(r"([0-9]+(?:[.,][0-9]{1,2})?)", str(rp))
        if m:
            meta["retail_price"] = float(m.group(1).replace(",", "."))
        else:
            meta["retail_price"] = 0.0
    return meta


def generate_meta(image_path: str, art_id: str = "lagerabverkauf", context=None):
    """
    Single-image extraction (compatible with your existing server code).
    Returns dict like {"name":..., "brand":..., "model":..., "retail_price":..., "notes":...}
    or {"error":..., ...}
    """
    if not OPENAI_API_KEY:
        return {"error": "missing_openai_api_key"}

    b64 = _b64_image(image_path)

    ctx = ""
    if isinstance(context, dict):
        try:
            ctx = json.dumps(context)[:4000]
        except Exception:
            ctx = ""

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _prompt_for_price(art_id)},
            {"role": "user", "content": [
                {"type": "text", "text": f"Kontext (optional, vorherige Fotos): {ctx}" if ctx else "Kein Kontext."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
            ]},
        ],
        "temperature": 0.2,
        "max_tokens": 220,
        # If the model supports it, this forces strict JSON output (many OpenAI chat models do).
        "response_format": {"type": "json_object"},
    }

    try:
        r = requests.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=35,
        )
    except Exception as e:
        return {"error": "openai_request_failed", "detail": str(e)[:400]}

    if r.status_code != 200:
        return {"error": "openai_http_error", "status": r.status_code, "body": (r.text or "")[:800]}

    data = r.json()
    text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    meta = _extract_json(text)
    if meta is None:
        return {"error": "invalid_json_from_model", "raw": text[:800]}

    return _coerce_price(meta)


def generate_meta_multi(image_paths, art_id: str = "lagerabverkauf"):
    """
    Multi-image in ONE request (front/back/label). Returns same schema + optionally "per_image".
    """
    if not OPENAI_API_KEY:
        return {"error": "missing_openai_api_key"}

    contents = [{"type": "text", "text": "Das sind mehrere Fotos desselben Produkts (Vorderseite/Rückseite/Etikett). "
                                        "Kombiniere alle Infos und gib EIN JSON im Schema zurück."}]
    for p in image_paths:
        b64 = _b64_image(p)
        contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _prompt_for_price(art_id)},
            {"role": "user", "content": contents},
        ],
        "temperature": 0.2,
        "max_tokens": 260,
        "response_format": {"type": "json_object"},
    }

    try:
        r = requests.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
    except Exception as e:
        return {"error": "openai_request_failed", "detail": str(e)[:400]}

    if r.status_code != 200:
        return {"error": "openai_http_error", "status": r.status_code, "body": (r.text or "")[:800]}

    data = r.json()
    text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    meta = _extract_json(text)
    if meta is None:
        return {"error": "invalid_json_from_model", "raw": text[:800]}

    return _coerce_price(meta)
