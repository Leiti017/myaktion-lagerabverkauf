import os
import json
import base64
import requests
from typing import List, Union, Dict, Any, Optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _system_prompt_multi() -> str:
    return (
        "Du bist ein extrem präziser Preis-Analyst für Handel/Marktpreise (EUR).\n"
        "Du bekommst 1-4 Fotos vom SELBEN Produkt (Front/Back/Bottom/Barcode/Preisschild).\n"
        "\n"
        "Ziel: Gib einen realistischen aktuellen Neupreis (Listen-/Marktpreis) in EUR.\n"
        "\n"
        "KERNREGEL (wichtig):\n"
        "- retail_price darf NICHT 0 sein, außer das Produkt ist wirklich nicht identifizierbar.\n"
        "- Wenn Marke/Name erkennbar sind, aber Preis/EAN/Menge nicht: gib eine konservative Schätzung (nicht 0)\n"
        "  und setze confidence niedrig.\n"
        "\n"
        "Anti-Fantasie-Regeln (sehr wichtig):\n"
        "- Erfinde KEINE konkreten Produktdetails (Menge/Variante/Preis), wenn nicht im Bild erkennbar.\n"
        "- Nutze harte Evidenz in dieser Priorität:\n"
        "  (1) klar sichtbarer Preis/UVP/Etikett\n"
        "  (2) EAN/Barcode (wenn lesbar)\n"
        "  (3) klar erkennbare Menge/Variante (100g vs 250g, 50ml vs 100ml, Stückzahl, Multipack)\n"
        "  (4) falls Produkt klar, aber Menge unklar: vorsichtige Schätzung, als Annahme markieren\n"
        "\n"
        "Mengen-Anpassung (wenn Menge erkennbar):\n"
        "- Passe den Preis an die erkannte Menge an.\n"
        "- Bei typischen Konsumgütern steigt Preis mit Menge meist weniger als linear (Mengenrabatt).\n"
        "  Beispiel: 2x Menge => meist ~1.5x bis 1.9x Preis (je nach Produkt).\n"
        "\n"
        "Wenn nur unsicher: gib vorsichtigen Preis und setze confidence niedrig.\n"
        "\n"
        "Gib am Ende NUR JSON (ohne Text) mit Schema:\n"
        "{\n"
        '  "name": "",\n'
        '  "brand": "",\n'
        '  "variant": "",\n'
        '  "size_text": "",\n'
        '  "size_value": null,\n'
        '  "size_unit": "",\n'
        '  "ean": "",\n'
        '  "retail_price": 0,\n'
        '  "confidence": 0.0,\n'
        '  "price_basis": "tag|ean|size|guess",\n'
        '  "assumptions": ""\n'
        "}\n"
        "\n"
        "size_value/size_unit: falls möglich, extrahiere Zahl und Einheit (ml/g/kg/l/Stk).\n"
        "\n"
        "confidence Skala:\n"
        "- 0.85-1.0: Preis/UVP klar im Bild\n"
        "- 0.65-0.85: EAN oder Menge+Variante klar\n"
        "- 0.40-0.65: Produkt klar, aber Menge unklar (vorsichtige Annahme)\n"
        "- 0.25-0.40: nur Marke/Name klar, konservative Schätzung\n"
    )


def _system_prompt_ean_refine() -> str:
    return (
        "Du bist ein extrem präziser Preis-Analyst (EUR).\n"
        "Du bekommst eine EAN (Barcode-Nummer) und optional Produkt-Hinweise.\n"
        "Aufgabe: Bestimme den realistischen aktuellen Neupreis (Listen-/Marktpreis) in EUR.\n"
        "\n"
        "KERNREGEL: retail_price darf NICHT 0 sein, außer die EAN ist offensichtlich unbrauchbar.\n"
        "\n"
        "Anti-Fantasie-Regeln:\n"
        "- Wenn EAN nicht eindeutig ist: gib eine konservative Schätzung und setze confidence niedrig.\n"
        "- Erfinde keine Menge/Variante, wenn nicht aus EAN/Hinweisen ableitbar.\n"
        "\n"
        "Gib NUR JSON (ohne Text) aus mit Schema:\n"
        "{\n"
        '  "name":"",\n'
        '  "brand":"",\n'
        '  "variant":"",\n'
        '  "size_text":"",\n'
        '  "size_value": null,\n'
        '  "size_unit":"",\n'
        '  "ean":"",\n'
        '  "retail_price":0,\n'
        '  "confidence":0.0,\n'
        '  "price_basis":"ean|guess",\n'
        '  "assumptions":""\n'
        "}\n"
    )


def _post_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if r.status_code != 200:
        return {"error": "openai_http_error", "status": r.status_code, "body": r.text[:800]}
    return r.json()


def _parse_json_from_model(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
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


def generate_meta(image_paths: Union[str, List[str]], context: Optional[dict] = None) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"error": "missing_openai_api_key"}

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    ctx = ""
    if isinstance(context, dict):
        try:
            ctx = json.dumps(context, ensure_ascii=False)[:3500]
        except Exception:
            ctx = ""

    content: List[Dict[str, Any]] = []
    content.append({"type": "text", "text": "Mehrere Fotos vom gleichen Produkt. Kombiniere alle Infos."})
    if ctx:
        content.append({"type": "text", "text": f"Kontext (optional): {ctx}"})

    for p in image_paths:
        b64 = _b64_image(p)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
            }
        )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt_multi()},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
    }

    data = _post_chat(payload)
    if "error" in data:
        return data

    text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    return _parse_json_from_model(text)


def refine_with_ean(ean: str, hint_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"error": "missing_openai_api_key"}

    ean = (ean or "").strip()
    if not ean:
        return {"error": "missing_ean"}

    hints = {}
    if isinstance(hint_meta, dict):
        for k in ("name", "brand", "variant", "size_text", "size_value", "size_unit"):
            v = hint_meta.get(k)
            if isinstance(v, (str, int, float)) and str(v).strip():
                hints[k] = v

    user_text = {
        "ean": ean,
        "hints": hints,
        "instruction": "Nutze EAN primär. Falls unklar, konservative Schätzung (nicht 0) mit niedriger confidence.",
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt_ean_refine()},
            {"role": "user", "content": f"EAN-Refine Input (JSON): {json.dumps(user_text, ensure_ascii=False)}"},
        ],
        "temperature": 0.2,
    }

    data = _post_chat(payload)
    if "error" in data:
        return data

    text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    return _parse_json_from_model(text)
