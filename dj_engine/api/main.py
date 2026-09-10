"""FastAPI-backend voor de web-UI: dun bovenop de bestaande engine, geen
nieuwe logica — alleen HTTP eromheen.

Bewust buiten scope (blijven CLI-only, zie cli.py): import-rekordbox,
normalize-energy, set-api-key — eenmalige/onderhoudsacties, geen onderdeel
van de live pick->suggest->confirm-lus die deze UI ondersteunt.

Start met: dj-engine serve
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from dj_engine import db as db_module
from dj_engine.config import load_config
from dj_engine.enrichment.llm_providers import LLMProviderError
from dj_engine.enrichment.musicbrainz_client import MusicBrainzClient
from dj_engine.ingest.pipeline import analyze_track
from dj_engine.recommend.engine import suggest_next
from dj_engine.recommend.llm_suggest import suggest_next_llm
from dj_engine.recommend.refine import refine_candidates

STATIC_DIR = Path(__file__).parent / "static"


# --- Pydantic-modellen ---------------------------------------------------

class TrackOut(BaseModel):
    id: int
    filepath: str
    title: str | None = None
    artist: str | None = None
    mbid: str | None = None
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    camelot: str | None = None
    loudness: float | None = None
    danceability: float | None = None
    energy: float | None = None
    energy_raw: float | None = None
    mood_happy: float | None = None
    mood_sad: float | None = None
    mood_aggressive: float | None = None
    mood_relaxed: float | None = None
    mood_party: float | None = None
    genre: str | None = None
    analyzed_at: str | None = None


class ScoreBreakdown(BaseModel):
    key: float
    bpm: float
    energy: float
    mood: float


class SuggestionOut(BaseModel):
    track: TrackOut
    total_score: float
    breakdown: ScoreBreakdown


class SuggestResponse(BaseModel):
    current_track: TrackOut
    direction: str
    results: list[SuggestionOut]


class RefineResponse(BaseModel):
    refined_count: int
    suggestions: SuggestResponse


class LLMSuggestionOut(BaseModel):
    track: TrackOut
    reden: str
    gegrond_op_bron: bool


class LLMSuggestResponse(BaseModel):
    provider: str | None
    model: str | None
    used_search: bool
    results: list[LLMSuggestionOut]


class TransitionIn(BaseModel):
    from_track_id: int
    to_track_id: int
    direction: str
    rating: int | None = None


class TransitionOut(BaseModel):
    id: int


# --- App-factory -----------------------------------------------------------

def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(title="dj-engine", description="Lokale DJ track-database & recommendation engine")
    app.state.config = cfg

    def get_config() -> dict[str, Any]:
        return app.state.config

    def get_db(config: dict[str, Any] = Depends(get_config)):
        conn = db_module.connect(config["database"]["path"])
        try:
            yield conn
        finally:
            conn.close()

    def _track_or_404(conn: sqlite3.Connection, track_id: int) -> dict[str, Any]:
        track = db_module.get_track(conn, track_id)
        if track is None:
            raise HTTPException(status_code=404, detail=f"Track {track_id} niet gevonden")
        return track

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/tracks/search", response_model=list[TrackOut])
    def search_tracks(
        q: str | None = None,
        artist: str | None = None,
        genre: str | None = None,
        bpm_min: float | None = None,
        bpm_max: float | None = None,
        limit: int = 50,
        conn: sqlite3.Connection = Depends(get_db),
    ):
        return db_module.search_tracks(
            conn, artist=artist, genre=genre, bpm_min=bpm_min, bpm_max=bpm_max, q=q, limit=limit,
        )

    @app.get("/tracks/{track_id}", response_model=TrackOut)
    def get_track(track_id: int, conn: sqlite3.Connection = Depends(get_db)):
        return _track_or_404(conn, track_id)

    @app.post("/tracks/{track_id}/analyze", response_model=TrackOut)
    def analyze(
        track_id: int,
        musicbrainz: bool = False,
        config: dict[str, Any] = Depends(get_config),
        conn: sqlite3.Connection = Depends(get_db),
    ):
        _track_or_404(conn, track_id)
        mb_client = MusicBrainzClient(config, db_conn=conn) if musicbrainz else None
        try:
            track = analyze_track(track_id, conn, config, mb_client=mb_client)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # Deze track wordt zometeen als referentiepunt gebruikt voor
        # energy_score() (delta t.o.v. de nu-spelende track) — anders dan
        # de CLI's periodieke drempel forceren we hier meteen een
        # normalisatie zodat `energy` direct bruikbaar is (goedkoop: één
        # SQL-pass, geen audio-analyse).
        if track.get("energy_raw") is not None:
            db_module.normalize_energy(conn)
            track = db_module.get_track(conn, track_id)
        return track

    @app.get("/tracks/{track_id}/suggest", response_model=SuggestResponse)
    def suggest(
        track_id: int,
        direction: str,
        top: int = 10,
        config: dict[str, Any] = Depends(get_config),
        conn: sqlite3.Connection = Depends(get_db),
    ):
        current = _track_or_404(conn, track_id)
        try:
            results = suggest_next(conn, track_id, direction, config, top_n=top)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"current_track": current, "direction": direction, "results": results}

    @app.post("/tracks/{track_id}/refine", response_model=RefineResponse)
    def refine(
        track_id: int,
        direction: str,
        top: int = 10,
        config: dict[str, Any] = Depends(get_config),
        conn: sqlite3.Connection = Depends(get_db),
    ):
        current = _track_or_404(conn, track_id)
        try:
            shortlist_n = config.get("suggest", {}).get("shortlist_size", 40)
            shortlist = suggest_next(conn, track_id, direction, config, top_n=shortlist_n)
            shortlist_ids = [r["track"]["id"] for r in shortlist]
            refined_count = refine_candidates(conn, config, shortlist_ids)
            results = suggest_next(conn, track_id, direction, config, top_n=top)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "refined_count": refined_count,
            "suggestions": {"current_track": current, "direction": direction, "results": results},
        }

    @app.post("/tracks/{track_id}/suggest-llm", response_model=LLMSuggestResponse)
    def suggest_llm(
        track_id: int,
        direction: str,
        top: int = 10,
        config: dict[str, Any] = Depends(get_config),
        conn: sqlite3.Connection = Depends(get_db),
    ):
        _track_or_404(conn, track_id)
        try:
            outcome = suggest_next_llm(conn, track_id, direction, config, top_n=top)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LLMProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return outcome

    @app.post("/transitions", response_model=TransitionOut, status_code=201)
    def log_transition(body: TransitionIn, conn: sqlite3.Connection = Depends(get_db)):
        _track_or_404(conn, body.from_track_id)
        _track_or_404(conn, body.to_track_id)
        transition_id = db_module.log_transition(
            conn, body.from_track_id, body.to_track_id, body.direction, rating=body.rating
        )
        return {"id": transition_id}

    return app


# Voor `uvicorn dj_engine.api.main:app` met de default config. De CLI
# (`dj-engine serve`) gebruikt create_app() rechtstreeks met de via
# --config/config.local.yaml geladen config, niet dit module-level object.
app = create_app()
