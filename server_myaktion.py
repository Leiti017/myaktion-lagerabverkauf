# server_myaktion.py – MyAktion Lagerabverkauf (Foto → Preis)
# Start (Render):
#   uvicorn server_myaktion:app --host 0.0.0.0 --port $PORT
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os, io, math, uuid
from PIL import Image, ImageOps

# Optional: use OpenAI vision engine if OPENAI_API_KEY is set
try:
    from ki_engine_openai import generate_meta
except Exception:
    generate_meta = None

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()

# Serve static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def home():
    return FileResponse(str(BASE_DIR / "index.html"))

@app.get("/site.webmanifest")
def manifest():
    return FileResponse(str(BASE_DIR / "site.webmanifest"), media_type="application/manifest+json")

@app.get("/sw.js")
def sw():
    # Must be at root scope for PWA
    return FileResponse(str(BASE_DIR / "sw.js"), media_type="application/javascript")

@app.get("/health")
def health():
    return {"ok": True}

def _safe_round_price_eur(v: float) -> float:
    # keep 2 decimals, avoid negative
    if not v or v < 0:
        return 0.0
    return round(float(v), 2)

def _our_price(list_price: float) -> float:
    # -20%
    return _safe_round_price_eur(list_price * 0.80)

def _downscale_and_fix_orientation(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    max_side = 1280
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img

@app.post("/api/scan")
async def scan(image: UploadFile = File(...)):
    data = await image.read()
    img = _downscale_and_fix_orientation(data)

    # save temp jpeg (engine expects a path)
    tmp = BASE_DIR / f"_tmp_{uuid.uuid4().hex}.jpg"
    img.save(tmp, "JPEG", quality=86)

    list_price = 0.0
    source = "manual"
    try:
        if generate_meta is not None and os.getenv("OPENAI_API_KEY", "").strip():
            meta = generate_meta(str(tmp), art_id="lagerabverkauf", context=None)
            if meta and meta.get("retail_price"):
                list_price = float(meta["retail_price"])
                source = "openai"
    except Exception:
        # keep 0.0
        source = "failed"
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    list_price = _safe_round_price_eur(list_price)
    our_price = _our_price(list_price)

    return JSONResponse({
        "list_price": list_price,
        "our_price": our_price,
        "source": source
    })
