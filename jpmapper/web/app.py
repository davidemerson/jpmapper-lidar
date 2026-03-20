"""FastAPI application for JPMapper web interface."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import rasterio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from jpmapper.web.routes import router

# Global DSM dataset handle — opened once at startup, closed on shutdown.
dsm_dataset: rasterio.DatasetReader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dsm_dataset
    dsm_path = os.environ.get("JPMAPPER_DSM_PATH", "")
    if not dsm_path:
        raise RuntimeError(
            "DSM path not configured. Set JPMAPPER_DSM_PATH env var or use 'jpmapper web --dsm <path>'"
        )
    dsm_dataset = rasterio.open(dsm_path)
    yield
    if dsm_dataset is not None:
        dsm_dataset.close()
        dsm_dataset = None


app = FastAPI(
    title="JPMapper LOS Analyzer",
    description="Line-of-sight analysis web interface",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api")

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(_static_dir / "index.html"))
