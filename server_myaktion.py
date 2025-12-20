# server_myaktion.py – MyAktion Lagerabverkauf (Multi-Foto → Preis)
# Start (Render):
#   uvicorn server_myaktion:app --host 0.0.0.0 --port $PORT

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

# Optional: OpenAI vision engine if OPENAI_API_KEY is set
try:
    from ki_engine_openai import generate_meta  # expects: generate_meta(image_path, art_id, context) -> dict
except Exception:
    generate_meta = None

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="MyAktion Lagerabverkauf")

# Serve static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _safe_round_price_eur(v: float) -> float:
    try:
        return round(float(v), 2)
    except Exception:
        return 0.0


def _our_price(list_price: float) -> float:
    return _safe_round_price_eur(list_price * 0.80)


def _downscale_and_fix_orientation(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    max_side = 1600
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _extract_price_from_meta(meta: dict) -> float:
    # Accept a few likely keys
    for k in ("retail_price", "listenpreis", "list_price", "price"):
        if k in meta and meta[k] not in (None, ""):
            try:
                return float(str(meta[k]).replace(",", "."))
            except Exception:
                continue
    return 0.0


@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(str(BASE_DIR / "index.html"))


@app.get("/health")
@app.get("/api/health")
def health():
    return {"ok": True}


# Root-level convenience routes (many browsers request these at /)
@app.get("/favicon.ico")
def favicon():
    return FileResponse(str(STATIC_DIR / "favicon.ico"))


@app.get("/apple-touch-icon.png")
def apple_touch():
    return FileResponse(str(STATIC_DIR / "apple-touch-icon.png"))


@app.get("/site.webmanifest")
def manifest():
    return FileResponse(str(BASE_DIR / "site.webmanifest"))


@app.post("/api/scan")
async def scan(request: Request):
    """
    Multi-photo upload:
    - Accepts any multipart field names.
    - Collects ALL uploaded files (works with input[multiple]).
    - Runs recognition per photo and chooses the best/most likely price.
    """
    t0 = time.time()

    form = await request.form()
    uploads = []
    for v in form.values():
        if hasattr(v, "filename") and hasattr(v, "file"):
            uploads.append(v)

    if not uploads:
        return JSONResponse({"ok": False, "error": "Kein Bild empfangen. Bitte Foto erneut auswählen."})

    if not (os.getenv("OPENAI_API_KEY", "").strip()):
        # Without key we cannot use vision pricing reliably
        return JSONResponse({"ok": False, "error": "OPENAI_API_KEY fehlt in Render (Environment). Bitte setzen und neu deployen."})

    prices: List[float] = []
    details = []
    source = "openai"

    for up in uploads:
        try:
            data = await up.read()
            img = _downscale_and_fix_orientation(data)
            tmp = BASE_DIR / f"_tmp_{uuid.uuid4().hex}.jpg"
            img.save(tmp, "JPEG", quality=86)

            p = 0.0
            meta = None
            if generate_meta is not None:
                meta = generate_meta(str(tmp), art_id="lagerabverkauf", context=None) or {}
                p = _extract_price_from_meta(meta)

            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

            p = _safe_round_price_eur(p)
            if p > 0:
                prices.append(p)
            details.append({"filename": up.filename, "price": p})
        except Exception:
            details.append({"filename": getattr(up, "filename", "unknown"), "price": 0.0})

    if not prices:
        ms = int((time.time() - t0) * 1000)
        return JSONResponse({
            "ok": True,
            "list_price": 0.0,
            "our_price": 0.0,
            "runtime_ms": ms,
            "source": "no_price_detected",
            "details": details
        })

    # Choose most frequent rounded price; fallback to max if all different
    rounded = [round(p, 2) for p in prices]
    # frequency map
    freq = {}
    for p in rounded:
        freq[p] = freq.get(p, 0) + 1
    best_price = max(freq.items(), key=lambda kv: (kv[1], kv[0]))[0]  # prefer frequent, then higher

    list_price = _safe_round_price_eur(best_price)
    our_price = _our_price(list_price)
    ms = int((time.time() - t0) * 1000)

    return JSONResponse({
        "ok": True,
        "list_price": list_price,
        "our_price": our_price,
        "runtime_ms": ms,
        "source": source,
        "details": details
    })
