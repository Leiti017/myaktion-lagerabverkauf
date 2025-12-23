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

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="MyAktion Preis-Scanner")

# Render Health Check (wichtig)
@app.get("/health")
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


@app.post("/api/scan")
async def scan(files: List[UploadFile] = File(...)):
    t0 = time.time()

    if not files:
        return JSONResponse({"ok": False, "error": "Keine Dateien erhalten."}, status_code=400)

    best_price = 0.0
    best_source = "manual"
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
