from contextlib import asynccontextmanager
from pathlib import Path
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ..storage.cache import Cache
from ..storage.database import Database
from .service import Service
from .diagnostics import diagnostic


class ScanBody(BaseModel):
    folder: str = Field(min_length=1)
    recursive: bool = True


class AnalyzeBody(BaseModel):
    backend: str = "features"
    force: bool = False
    sampling: str = "uniform"


class GroupBody(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class AssignmentBody(BaseModel):
    clip_ids: list[int] = Field(min_length=1, max_length=1500)
    group_id: int
    expected_revisions: dict[str, int] | None = None


class RecommendationBody(BaseModel):
    group_id: int
    backend: str = "clip"
    example_ids: list[int] | None = None


class DecisionBody(BaseModel):
    decision: str


def create_app(data_dir=None, demo=False):
    cache = Cache(data_dir)
    db = Database(cache)
    modes = {row["demo"] for row in db.rows("SELECT DISTINCT demo FROM clips")}
    if modes and modes != {int(demo)}:
        raise ValueError("Demo and real libraries must stay separate. Use another --data-dir or launch the demo library with --demo.")
    service = Service(cache, db)
    lock_file = None
    @asynccontextmanager
    async def lifespan(app):
        nonlocal lock_file
        lock_file = cache.output("app.lock").open("a")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeError("This library is already open in another server") from exc
        service.recover_interrupted()
        if demo:
            from ..demo import seed_demo
            seed_demo(cache, db)
        handler = RotatingFileHandler(cache.output("logs/application.log"), maxBytes=2_000_000, backupCount=3)
        logger = logging.getLogger("recipe_sorter")
        logger.addHandler(handler)
        try:
            yield
        finally:
            logger.removeHandler(handler)
            handler.close()
            lock_file.close()
    app = FastAPI(title="Recipe Clip Sorter", lifespan=lifespan)
    app.state.service, app.state.cache, app.state.db = service, cache, db
    frontend = Path(__file__).resolve().parents[1] / "frontend"

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.url.hostname
        if host not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            return JSONResponse({"detail": "Localhost access only"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and (urlsplit(origin).netloc != request.headers.get("host") or urlsplit(origin).scheme not in ("http", "https")):
            return JSONResponse({"detail": "Cross-origin access is disabled"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("x-recipe-sorter") != "local":
            return JSONResponse({"detail": "Local application request header required"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/")
    def home():
        return FileResponse(frontend / "index.html")

    @app.get("/api/library")
    def library():
        return {**db.library(), "demo": demo, "job": service.snapshot()}

    @app.get("/api/job")
    def job():
        return service.snapshot()

    @app.get("/api/diagnostics")
    def environment():
        return {**diagnostic(cache), "demo": demo}

    @app.post("/api/scan")
    def scan(body: ScanBody):
        if demo:
            raise ValueError("Demo library is isolated. Launch without --demo to scan real sources.")
        folder = Path(body.folder).expanduser()
        if not folder.is_dir():
            raise ValueError("Folder does not exist on this workstation")
        cache.check_source(folder)
        return service.start("Scan", service.scan, body.folder, body.recursive)

    @app.post("/api/analyze")
    def analyze(body: AnalyzeBody):
        if demo:
            raise ValueError("Demo proposals are illustrative. Launch without --demo for real analysis.")
        if body.backend not in {"clip", "features", "hybrid"} or body.sampling not in {"uniform", "adaptive"}:
            raise ValueError("Select a valid backend and sampling strategy")
        return service.start("Analysis", service.analyze, body.backend, body.force, body.sampling)

    @app.post("/api/groups")
    def create_group(body: GroupBody):
        return {"id": db.group(body.name)}

    @app.patch("/api/groups/{group_id}")
    def rename_group(group_id: int, body: GroupBody):
        db.rename(group_id, body.name)
        return {"saved": True}

    @app.delete("/api/groups/{group_id}")
    def delete_group(group_id: int):
        db.delete_group(group_id)
        return {"saved": True}

    @app.post("/api/assign")
    def assign(body: AssignmentBody):
        move_id = db.assign(body.clip_ids, body.group_id, body.expected_revisions)
        return {"saved": True, "move_id": move_id}

    @app.post("/api/moves/{move_id}/undo")
    def undo_move(move_id: int):
        db.undo(move_id)
        return {"saved": True}

    from .recommendations import Recommendations
    recommendations = Recommendations(service)

    @app.post("/api/recommendations")
    def recommend(body: RecommendationBody):
        if body.backend not in {"clip", "features", "hybrid"}:
            raise ValueError("Select a supported comparison backend")
        return service.start("Find matches", recommendations.build, body.group_id, body.backend, body.example_ids)

    @app.get("/api/recommendations/{group_id}")
    def recommendation_queue(group_id: int, status: str = "pending"):
        return recommendations.queue(group_id, status)

    @app.post("/api/suggestions/{suggestion_id}/decision")
    def recommendation_decision(suggestion_id: int, body: DecisionBody):
        return recommendations.decide(suggestion_id, body.decision)

    @app.get("/api/clips/{clip_id}")
    def detail(clip_id: int):
        c = next((c for c in db.library()["clips"] if c["id"] == clip_id), None)
        if not c:
            raise HTTPException(404, "Clip not found")
        return {**c, "corrections": db.rows("SELECT * FROM corrections WHERE clip_id=? ORDER BY id DESC", (clip_id,)),
                "proposals": db.rows("SELECT * FROM proposals WHERE clip_id=? ORDER BY id DESC", (clip_id,))}

    @app.post("/api/clips/{clip_id}/preview")
    def preview(clip_id: int):
        db.get_clip(clip_id)
        return service.start("Preview", service.preview, clip_id)

    @app.get("/api/frames/{frame_id}")
    def frame(frame_id: int):
        rows = db.rows("SELECT path FROM frames WHERE id=?", (frame_id,))
        if not rows or not cache.path(rows[0]["path"]).is_file():
            raise HTTPException(404, "Cached frame missing. Re-run analysis.")
        return FileResponse(cache.path(rows[0]["path"]), media_type="image/jpeg")

    @app.get("/api/clips/{clip_id}/video")
    def video(clip_id: int):
        c = db.get_clip(clip_id)
        if not c["demo"]:
            from ..storage.cache import fingerprint
            try:
                if fingerprint(Path(c["path"])) != c["fingerprint"]:
                    raise HTTPException(409, "Source changed. Scan and rebuild preview.")
            except OSError:
                raise HTTPException(409, "Source is offline. Reconnect it before previewing.")
        if not c["preview"] or not cache.path(c["preview"]).is_file():
            raise HTTPException(404, "Generate a preview first")
        return FileResponse(cache.path(c["preview"]), media_type="video/webm")

    app.mount("/static", StaticFiles(directory=frontend), name="static")
    return app
