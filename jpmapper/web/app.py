"""FastAPI application for JPMapper web interface."""
from __future__ import annotations

import asyncio
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import rasterio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

import jpmapper.web.routes as _routes
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
    _routes._coverage_cache = None

    # Size the default thread pool to the hardware so concurrent analysis
    # requests can saturate available cores within this worker process.
    cpu_count = multiprocessing.cpu_count()
    n_workers = int(os.environ.get("JPMAPPER_WEB_WORKERS", "1"))
    # Divide cores among uvicorn workers; each gets its own thread pool.
    threads_per_worker = max(4, cpu_count // max(n_workers, 1))
    loop = asyncio.get_event_loop()
    pool = ThreadPoolExecutor(max_workers=threads_per_worker)
    loop.set_default_executor(pool)

    yield

    pool.shutdown(wait=False)
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)
