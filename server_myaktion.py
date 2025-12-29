# server_myaktion.py – MyAktion Lagerabverkauf (Fotos → Preis)
# Start (Render):
#   uvicorn server_myaktion:app --host 0.0.0.0 --port $PORT

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

try:
    from ki_engine_openai import generate_meta
except Exception:
    generate_meta = None

try:
    from ki_engine_openai import generate_meta_multi
except Exception:
    generate_meta_multi = None

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="MyAktion Preis-Scanner")

# Render Health Check (wichtig)
@app.api_route("/health", methods=["GET", "HEAD"])
def render_health():
    return {"ok": True}

# Static folder (icons, logo, etc.)
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def home():
    p = BASE_DIR / "index.html"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "index.html fehlt im Root-Verzeichnis."}, status_code=500)
    return FileResponse(str(p))


# =========================
# PWA / Manifest
# =========================

# Haupt-Manifest (dein index.html lädt /manifest.json)
@app.get("/manifest.json")
def manifest_json():
    p = static_dir / "manifest.json"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "static/manifest.json fehlt."}, status_code=404)
    return FileResponse(str(p))

# Kompatibilität: falls irgendwas noch /site.webmanifest erwartet
@app.get("/site.webmanifest")
def site_webmanifest():
    p = static_dir / "manifest.json"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "static/manifest.json fehlt."}, status_code=404)
    return FileResponse(str(p))


# =========================
# Favicons / iOS Icons
# =========================

@app.get("/favicon.ico")
def favicon():
    p = static_dir / "favicon.ico"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "static/favicon.ico fehlt."}, status_code=404)
    return FileResponse(str(p))

@app.get("/apple-touch-icon.png")
def apple_touch_icon():
    p = static_dir / "apple-touch-icon.png"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "static/apple-touch-icon.png fehlt."}, status_code=404)
    return FileResponse(str(p))


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "myaktion-price-scan",
        "has_openai_engine": bool(generate_meta is not None),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY", "").strip()),
    }


def _downscale_and_fix_orientation(image_bytes: bytes, max_side: int = 1400) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)  # fix orientation
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        scale = max_side / float(m)
        img = img.resize((int(w * scale), int(h * scale)))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _safe_round_price_eur(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0:
        return 0.0
    return round(v + 1e-9, 2)


def _our_price(list_price: float) -> float:
    # Unser Preis = 80% vom Listenpreis (-20%)
    if not list_price or list_price <= 0:
        return 0.0
    return _safe_round_price_eur(list_price * 0.80)


def _extract_price_from_meta(meta: dict) -> float:
    if not meta:
        return 0.0
    rp = meta.get("retail_price")
    if rp is None:
        rp = meta.get("price")
    if rp is None:
        return 0.0
    try:
        return float(rp)
    except Exception:
        return 0.0

def _as_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _score_meta(meta: Optional[dict]) -> int:
    if not isinstance(meta, dict):
        return 0
    score = 0
    if _as_float(meta.get("retail_price"), 0) > 0:
        score += 4
    if _as_float(meta.get("label_price_eur"), 0) > 0:
        score += 5  # strongest
    if meta.get("name"):
        score += 1
    if meta.get("brand"):
        score += 1
    if meta.get("model"):
        score += 1
    if meta.get("ean"):
        score += 2
    if _as_float(meta.get("quantity"), 0) > 0 and meta.get("unit"):
        score += 2
    # confidence (0..1)
    c = _as_float(meta.get("confidence"), 0)
    if c >= 0.6:
        score += 2
    elif c >= 0.35:
        score += 1
    return score

def _pick_final_price(items: list[dict]) -> tuple[float, str]:
    """
    Items: [{"list_price": float, "source": str, "meta": dict}, ...]
    Regel: wenn irgendwo ein klarer Etiketten-Preis gefunden wird -> der gewinnt.
    Sonst: bestes Meta nach Score, bei Gleichstand höherer Preis.
    """
    best = None
    # 1) Etikettenpreis bevorzugen
    for it in items:
        meta = it.get("meta") if isinstance(it, dict) else None
        lp = _as_float(it.get("list_price"), 0)
        label = _as_float(meta.get("label_price_eur") if isinstance(meta, dict) else 0, 0)
        if label > 0:
            return (round(label + 1e-9, 2), it.get("source", "openai"))
        # manche Modelle liefern label_price direkt als retail_price
        if lp > 0 and isinstance(meta, dict) and (meta.get("unit_price_basis") or meta.get("quantity")):
            # leave for scoring stage
            pass

    # 2) Score-basiert
    for it in items:
        meta = it.get("meta") if isinstance(it, dict) else None
        lp = _as_float(it.get("list_price"), 0)
        sc = _score_meta(meta)
        cand = (sc, lp, it.get("source", "openai"))
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand

    if best and best[1] > 0:
        return (round(best[1] + 1e-9, 2), best[2] or "openai")
    # fallback: max price
    mx = max([_as_float(it.get("list_price"), 0) for it in items], default=0.0)
    return (round(mx + 1e-9, 2), "openai")


@app.post("/api/scan")
async def scan(files: List[UploadFile] = File(...)):
    """
    Multi-Foto Scan:
    - nimmt 1..N Fotos
    - zeigt pro Foto eine Schätzung
    - kombiniert (Front/Back/Etikett) zu einem finalen Listenpreis
      -> wichtig für Drogerie/Tiernahrung (Gramm/ml/Stk)
    """
    t0 = time.time()

    if not files:
        return JSONResponse({"ok": False, "error": "Keine Dateien erhalten."}, status_code=400)

    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    items: list[dict] = []
    tmp_paths: list[Path] = []

    # 1) Bilder normalisieren und temporär speichern
    for f in files:
        try:
            raw = await f.read()
            img = _downscale_and_fix_orientation(raw)
            tmp_path = BASE_DIR / f"_tmp_{uuid.uuid4().hex}.jpg"
            img.save(tmp_path, "JPEG", quality=86)
            tmp_paths.append(tmp_path)
        except Exception:
            continue

    if not tmp_paths:
        return JSONResponse({"ok": False, "error": "Keine gültigen Bilder."}, status_code=400)

    # 2) OpenAI: bevorzugt EIN Call mit allen Bildern (kombiniert Gramm/ml/Etikett)
    context: Optional[dict] = None
    if generate_meta is not None and has_key:
        try:
            if generate_meta_multi is not None and len(tmp_paths) >= 2:
                meta_all = generate_meta_multi([str(p) for p in tmp_paths], art_id="lagerabverkauf", context=context)
                # Wir verwenden dasselbe Meta für den finalen Merge, aber liefern trotzdem pro Foto Einträge
                lp_all = _safe_round_price_eur(_extract_price_from_meta(meta_all))
                for i, p in enumerate(tmp_paths):
                    items.append({
                        "index": i,
                        "list_price": lp_all,
                        "source": "openai-multi",
                        "meta": meta_all if isinstance(meta_all, dict) else {},
                    })
            else:
                # Fallback: pro Foto (mit Context-Weitergabe)
                for i, p in enumerate(tmp_paths):
                    meta = generate_meta(str(p), art_id="lagerabverkauf", context=context)
                    if isinstance(meta, dict):
                        context = meta
                    lp = _safe_round_price_eur(_extract_price_from_meta(meta))
                    items.append({"index": i, "list_price": lp, "source": "openai", "meta": meta if isinstance(meta, dict) else {}})
        except Exception:
            # wenn KI crasht: weiter unten fallback
            items = []
    else:
        # kein Key/kein Engine
        for i, p in enumerate(tmp_paths):
            items.append({"index": i, "list_price": 0.0, "source": "no-openai-key" if not has_key else "no-openai-engine", "meta": {}})

    # 3) Temp löschen
    for p in tmp_paths:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    # 4) finalen Preis bestimmen
    final_list_price, final_source = _pick_final_price(items)
    our_price = _our_price(final_list_price)

    return JSONResponse(
        {
            "ok": True,
            "list_price": final_list_price,
            "our_price": our_price,
            "source": final_source,
            "items": items,
            "runtime_ms": int((time.time() - t0) * 1000),
        }
    )