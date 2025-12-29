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

# Engines (OpenAI JSON-meta OR simpler PRICE_EUR parser)
try:
    from ki_engine_openai import generate_meta, generate_meta_multi
except Exception:
    generate_meta = None
    generate_meta_multi = None

try:
    from ki_engine_price import estimate_list_price_eur
except Exception:
    estimate_list_price_eur = None


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
@app.get("/manifest.json")
def manifest_json():
    p = static_dir / "manifest.json"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "static/manifest.json fehlt."}, status_code=404)
    return FileResponse(str(p))

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
        "has_openai_multi": bool(generate_meta_multi is not None),
        "has_simple_engine": bool(estimate_list_price_eur is not None),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


# Helpful GET so you don't get confused by 405 in the browser
@app.get("/api/scan")
def scan_get_info():
    return JSONResponse(
        {
            "ok": False,
            "error": "Method Not Allowed: /api/scan erwartet POST mit FormData (files=...). "
                     "Wenn du /api/scan im Browser öffnest, kommt sonst 405.",
        },
        status_code=200,
    )


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
    if not meta or not isinstance(meta, dict):
        return 0.0
    rp = meta.get("retail_price", meta.get("price"))
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

    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())

    # Save temp images once so we can do multi-image in one call
    tmp_paths: List[Path] = []
    per_image = []
    debug_errors = []

    try:
        for f in files:
            raw = await f.read()
            img = _downscale_and_fix_orientation(raw)
            tmp_path = BASE_DIR / f"_tmp_{uuid.uuid4().hex}.jpg"
            img.save(tmp_path, "JPEG", quality=86)
            tmp_paths.append(tmp_path)

        best_price = 0.0
        best_source = "manual"
        meta_used = None

        # 1) Prefer multi-image engine if available and we have 2+ images
        if len(tmp_paths) >= 2 and generate_meta_multi is not None and has_key:
            meta = generate_meta_multi([str(p) for p in tmp_paths], art_id="lagerabverkauf")
            meta_used = meta
            lp = _safe_round_price_eur(_extract_price_from_meta(meta))
            if lp > 0:
                best_price = lp
                best_source = "openai-multi"
            else:
                debug_errors.append(meta if isinstance(meta, dict) else {"error": "unknown_meta"})
        # 2) Otherwise evaluate each image (openai meta OR simple engine) and take the best
        else:
            context: Optional[dict] = None
            for p in tmp_paths:
                list_price = 0.0
                source = "manual"
                meta = None

                if generate_meta is not None and has_key:
                    meta = generate_meta(str(p), art_id="lagerabverkauf", context=context)
                    if isinstance(meta, dict) and "error" not in meta:
                        context = meta
                    list_price = _extract_price_from_meta(meta)
                    source = "openai"
                elif estimate_list_price_eur is not None and has_key:
                    # fallback: older, very robust "PRICE_EUR=.." parser
                    val = estimate_list_price_eur(str(p))
                    list_price = val or 0.0
                    source = "openai-simple"
                else:
                    source = "no-openai-key" if not has_key else "no-engine"

                list_price = _safe_round_price_eur(list_price)
                per_image.append({"file": getattr(p, "name", "image"), "source": source, "list_price": list_price, "meta": meta})

                if isinstance(meta, dict) and meta.get("error"):
                    debug_errors.append(meta)

                if list_price > best_price:
                    best_price = list_price
                    best_source = source
                    meta_used = meta

        our_price = _our_price(best_price)

        return JSONResponse(
            {
                "ok": True,
                "list_price": best_price,
                "our_price": our_price,
                "source": best_source,
                "runtime_ms": int((time.time() - t0) * 1000),
                "per_image": per_image,
                "debug_errors": debug_errors[:3],  # keep it small
                "meta_used": meta_used if isinstance(meta_used, dict) else None,
            }
        )

    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
