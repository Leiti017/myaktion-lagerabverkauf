from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

try:
    from ki_engine_openai import generate_meta, refine_with_ean
except Exception:
    generate_meta = None
    refine_with_ean = None

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="MyAktion Preis-Scanner")


@app.api_route("/health", methods=["GET", "HEAD"])
def render_health():
    return {"ok": True}


static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def home():
    p = BASE_DIR / "index.html"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "index.html fehlt im Root-Verzeichnis."}, status_code=500)
    return FileResponse(str(p))


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


def _downscale_and_fix_orientation(image_bytes: bytes, max_side: int = 1600) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
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
    if not list_price or list_price <= 0:
        return 0.0
    return _safe_round_price_eur(list_price * 0.80)


def _sanity_clamp_price(p: float) -> float:
    # Nur extreme Ausreißer killen (damit "guess" nicht ständig zu 0 wird)
    if not p or p <= 0:
        return 0.0
    if p > 5000:
        return 0.0
    return p


def _extract(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    return {
        "name": meta.get("name", "") or "",
        "brand": meta.get("brand", "") or "",
        "variant": meta.get("variant", "") or "",
        "size_text": meta.get("size_text", "") or "",
        "size_value": meta.get("size_value", None),
        "size_unit": meta.get("size_unit", "") or "",
        "ean": meta.get("ean", "") or "",
        "retail_price": meta.get("retail_price", meta.get("price", 0)) or 0,
        "confidence": meta.get("confidence", 0.0) or 0.0,
        "price_basis": meta.get("price_basis", "") or "",
        "assumptions": meta.get("assumptions", "") or "",
    }


def _merge_keep_best(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    a = dict(primary or {})
    b = dict(secondary or {})
    try:
        ca = float(a.get("confidence") or 0.0)
    except Exception:
        ca = 0.0
    try:
        cb = float(b.get("confidence") or 0.0)
    except Exception:
        cb = 0.0

    winner, loser = a, b
    if cb > ca + 0.03:
        winner, loser = b, a
    elif abs(cb - ca) <= 0.03:
        pref = {"tag": 3, "ean": 2, "size": 1, "guess": 0}
        wa = pref.get((a.get("price_basis") or "").strip(), 0)
        wb = pref.get((b.get("price_basis") or "").strip(), 0)
        if wb > wa:
            winner, loser = b, a

    out = dict(winner)
    for k, v in loser.items():
        if k not in out or out.get(k) in ("", None, 0, 0.0):
            out[k] = v

    if (winner.get("assumptions") or "") and (loser.get("assumptions") or ""):
        out["assumptions"] = f"{winner.get('assumptions')} | {loser.get('assumptions')}"
    return out


@app.post("/api/scan")
async def scan(files: List[UploadFile] = File(...)):
    t0 = time.time()

    if not files:
        return JSONResponse({"ok": False, "error": "Keine Dateien erhalten."}, status_code=400)

    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    if generate_meta is None or not has_key:
        src = "no-openai-engine" if generate_meta is None else "no-openai-key"
        return JSONResponse(
            {"ok": True, "list_price": 0.0, "our_price": 0.0, "source": src, "runtime_ms": int((time.time() - t0) * 1000)}
        )

    tmp_paths: List[str] = []
    ex_final: Dict[str, Any] = {}

    try:
        for f in files:
            raw = await f.read()
            img = _downscale_and_fix_orientation(raw)
            tmp_path = BASE_DIR / f"_tmp_{uuid.uuid4().hex}.jpg"
            img.save(tmp_path, "JPEG", quality=86)
            tmp_paths.append(str(tmp_path))

        meta1 = generate_meta(tmp_paths, context=None)
        ex1 = _extract(meta1)
        ex_final = dict(ex1)

        ean = (ex1.get("ean") or "").strip()
        if refine_with_ean is not None and ean:
            meta2 = refine_with_ean(ean, hint_meta=ex1)
            ex2 = _extract(meta2)
            ex_final = _merge_keep_best(ex1, ex2)

    finally:
        for p in tmp_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    try:
        list_price = float(ex_final.get("retail_price") or 0.0)
    except Exception:
        list_price = 0.0

    list_price = _safe_round_price_eur(_sanity_clamp_price(list_price))
    our_price = _our_price(list_price)

    try:
        conf = float(ex_final.get("confidence") or 0.0)
    except Exception:
        conf = 0.0

    basis = (ex_final.get("price_basis") or "").strip()
    size_text = (ex_final.get("size_text") or "").strip()
    ean = (ex_final.get("ean") or "").strip()
    assumptions = (ex_final.get("assumptions") or "").strip()

    warnings = []
    if basis == "guess" or conf < 0.55:
        warnings.append("Unsicher: Schätzung ohne klare Preisevidenz")
    if not size_text:
        warnings.append("Menge nicht klar erkannt")
    if not ean and (basis in ("guess", "size")) and conf < 0.65:
        warnings.append("EAN/Barcode nicht erkannt")
    if assumptions:
        warnings.append(f"Annahme: {assumptions}")

    warning_text = " · ".join(warnings).strip()
    source = "openai" if conf >= 0.55 else "openai_low_conf"

    return JSONResponse(
        {
            "ok": True,
            "list_price": list_price,
            "our_price": our_price,
            "source": source,
            "confidence": round(conf, 2),
            "price_basis": basis,
            "size_text": size_text,
            "ean": ean,
            "warning": warning_text,
            "runtime_ms": int((time.time() - t0) * 1000),
        }
    )
