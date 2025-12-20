# server_myaktion.py – MyAktion Lagerabverkauf (Fotos → Preis)
# Start (Render):
#   uvicorn server_myaktion:app --host 0.0.0.0 --port $PORT

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

# Optional: OpenAI vision engine
try:
    from ki_engine_openai import generate_meta
except Exception:
    generate_meta = None

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="MyAktion Preis-Scanner")

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


@app.get("/site.webmanifest")
def manifest():
    p = BASE_DIR / "site.webmanifest"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "site.webmanifest fehlt."}, status_code=404)
    return FileResponse(str(p))


@app.get("/favicon.ico")
def favicon():
    p = BASE_DIR / "favicon.ico"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "favicon.ico fehlt."}, status_code=404)
    return FileResponse(str(p))


@app.get("/apple-touch-icon.png")
def apple_touch_icon():
    p = BASE_DIR / "apple-touch-icon.png"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "apple-touch-icon.png fehlt."}, status_code=404)
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
    """
    Beispiel-Logik:
    - wenn kein Preis erkannt -> 0
    - sonst 60% vom Listenpreis
    Passe das an dein reales Preis-System an.
    """
    if not list_price or list_price <= 0:
        return 0.0
    return _safe_round_price_eur(list_price * 0.60)


def _extract_price_from_meta(meta: dict) -> float:
    """
    Versucht robust den Preis aus generate_meta() zu holen.
    """
    if not meta:
        return 0.0
    # bevorzugt retail_price
    rp = meta.get("retail_price")
    if rp is None:
        rp = meta.get("price")
    if rp is None:
        return 0.0
    try:
        return float(rp)
    except Exception:
        return 0.0


@app.post("/api/scan")
async def scan(files: List[UploadFile] = File(...)):
    """
    Erwartet Multi-Foto Upload mit Feldnamen: files
    (passt zu deiner index.html)
    """
    t0 = time.time()

    if not files:
        return JSONResponse({"ok": False, "error": "Keine Dateien erhalten."}, status_code=400)

    # Wir nehmen den höchsten erkannten Preis über alle Fotos (oft: 1 Foto zeigt Preisschild)
    best_price = 0.0
    best_source = "manual"

    # Optionale Kontextweitergabe: 2. Foto ergänzt Details.
    # Wir reichen dem Engine-Call den "context" weiter, wenn die Engine das unterstützt.
    context: Optional[dict] = None

    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())

    for f in files:
        try:
            raw = await f.read()
            img = _downscale_and_fix_orientation(raw)

            tmp_path = BASE_DIR / f"_tmp_{uuid.uuid4().hex}.jpg"
            img.save(tmp_path, "JPEG", quality=86)

            list_price = 0.0
            source = "manual"

            if generate_meta is not None and has_key:
                meta = generate_meta(str(tmp_path), art_id="lagerabverkauf", context=context)
                # Kontext updaten, damit Foto2 helfen kann
                if isinstance(meta, dict):
                    context = meta

                list_price = _extract_price_from_meta(meta)
                source = "openai"
            else:
                source = "no-openai-key" if not has_key else "no-openai-engine"

            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

            list_price = _safe_round_price_eur(list_price)

            if list_price > best_price:
                best_price = list_price
                best_source = source

        except Exception:
            # Wenn ein Foto kaputt ist, ignorieren wir es und machen weiter
            continue

    our_price = _our_price(best_price)

    return JSONResponse(
        {
            "ok": True,
            "list_price": best_price,
            "our_price": our_price,
            "source": best_source,
            "runtime_ms": int((time.time() - t0) * 1000),
        }
    )
