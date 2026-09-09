"""SQLite-database laag voor dj-engine: schema + CRUD.

Geen ORM — bewust dunne wrapper rond sqlite3 zodat het schema en de
queries transparant en makkelijk te debuggen blijven.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,
    title TEXT,
    artist TEXT,
    mbid TEXT,
    bpm REAL,
    key TEXT,
    scale TEXT,
    camelot TEXT,
    loudness REAL,
    danceability REAL,
    energy REAL,
    -- energy_raw: ongenormaliseerde energie-feature uit essentia_extractor
    -- (RMS + hoogfrequent-energieratio). `energy` is de 0-1 min-max
    -- normalisatie hiervan over de hele bibliotheek (zie ingest/pipeline.py).
    -- Extra kolom t.o.v. BUILD_SPEC.md — die zegt "minimaal deze kolommen".
    energy_raw REAL,
    mood_happy REAL,
    mood_sad REAL,
    mood_aggressive REAL,
    mood_relaxed REAL,
    mood_party REAL,
    genre TEXT,
    analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tracks_camelot ON tracks(camelot);
CREATE INDEX IF NOT EXISTS idx_tracks_bpm ON tracks(bpm);

CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_track_id INTEGER REFERENCES tracks(id),
    to_track_id INTEGER REFERENCES tracks(id),
    direction TEXT,
    used_at TEXT DEFAULT CURRENT_TIMESTAMP,
    rating INTEGER
);

CREATE TABLE IF NOT EXISTS musicbrainz_cache (
    cache_key TEXT PRIMARY KEY,
    mbid TEXT,
    genre_tags TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# Kolommen die via insert_track()/update_track() beschrijfbaar zijn (id en
# analyzed_at worden door de database beheerd).
_TRACK_COLUMNS = [
    "filepath", "title", "artist", "mbid", "bpm", "key", "scale", "camelot",
    "loudness", "danceability", "energy", "energy_raw",
    "mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party",
    "genre",
]


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open een database-connectie en zorg dat het schema bestaat.

    Maakt de bovenliggende map aan als die nog niet bestaat (behalve voor
    in-memory databases).
    """
    db_path_str = str(db_path)
    if db_path_str != ":memory:":
        Path(db_path_str).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path_str)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def track_exists(conn: sqlite3.Connection, filepath: str) -> bool:
    """True als er al een track met dit filepath in de database staat."""
    row = conn.execute(
        "SELECT 1 FROM tracks WHERE filepath = ? LIMIT 1", (filepath,)
    ).fetchone()
    return row is not None


def insert_track(conn: sqlite3.Connection, track: dict[str, Any]) -> int:
    """Voeg een nieuwe track toe. Retourneert het nieuwe track-id.

    `track` mag een subset van de bekende kolommen bevatten; ontbrekende
    velden worden NULL (nooit gefabriceerd).

    Raises:
        sqlite3.IntegrityError: als filepath al bestaat (gebruik track_exists()
            of upsert_track() om dat te voorkomen).
    """
    columns = [c for c in _TRACK_COLUMNS if c in track]
    placeholders = ", ".join("?" for _ in columns)
    values = [track[c] for c in columns]

    cursor = conn.execute(
        f"INSERT INTO tracks ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    return cursor.lastrowid


def upsert_track(conn: sqlite3.Connection, track: dict[str, Any]) -> int:
    """Voeg een track toe, of update de bestaande rij op basis van filepath.

    Idempotent: opnieuw ingesten van dezelfde filepath overschrijft de
    bestaande analyse in plaats van te dupliceren.
    """
    if "filepath" not in track:
        raise ValueError("track dict moet 'filepath' bevatten")

    columns = [c for c in _TRACK_COLUMNS if c in track]
    placeholders = ", ".join("?" for _ in columns)
    values = [track[c] for c in columns]
    update_clause = ", ".join(
        f"{c} = excluded.{c}" for c in columns if c != "filepath"
    )

    cursor = conn.execute(
        f"""
        INSERT INTO tracks ({', '.join(columns)}) VALUES ({placeholders})
        ON CONFLICT(filepath) DO UPDATE SET {update_clause}
        """,
        values,
    )
    conn.commit()

    if cursor.lastrowid and cursor.rowcount == 1:
        existing = conn.execute(
            "SELECT id FROM tracks WHERE filepath = ?", (track["filepath"],)
        ).fetchone()
        return existing["id"]
    return cursor.lastrowid


def get_track(conn: sqlite3.Connection, track_id: int) -> dict[str, Any] | None:
    """Haal één track op via id, of None als die niet bestaat."""
    row = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
    return dict(row) if row else None


def get_track_by_filepath(conn: sqlite3.Connection, filepath: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM tracks WHERE filepath = ?", (filepath,)).fetchone()
    return dict(row) if row else None


def get_all_tracks(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Alle tracks, gesorteerd op id."""
    rows = conn.execute("SELECT * FROM tracks ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def search_tracks(
    conn: sqlite3.Connection,
    artist: str | None = None,
    genre: str | None = None,
    bpm_min: float | None = None,
    bpm_max: float | None = None,
) -> list[dict[str, Any]]:
    """Zoek tracks op combinatie van filters (allemaal optioneel, AND-gecombineerd)."""
    clauses: list[str] = []
    params: list[Any] = []

    if artist:
        clauses.append("artist LIKE ?")
        params.append(f"%{artist}%")
    if genre:
        clauses.append("genre LIKE ?")
        params.append(f"%{genre}%")
    if bpm_min is not None:
        clauses.append("bpm >= ?")
        params.append(bpm_min)
    if bpm_max is not None:
        clauses.append("bpm <= ?")
        params.append(bpm_max)

    query = "SELECT * FROM tracks"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id"

    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def delete_track(conn: sqlite3.Connection, track_id: int) -> bool:
    """Verwijder een track. Retourneert True als er iets verwijderd is."""
    cursor = conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
    conn.commit()
    return cursor.rowcount > 0


def log_transition(
    conn: sqlite3.Connection,
    from_track_id: int,
    to_track_id: int,
    direction: str,
    rating: int | None = None,
) -> int:
    """Log een (mogelijk live gebruikte) overgang tussen twee tracks."""
    cursor = conn.execute(
        """
        INSERT INTO transitions (from_track_id, to_track_id, direction, rating)
        VALUES (?, ?, ?, ?)
        """,
        (from_track_id, to_track_id, direction, rating),
    )
    conn.commit()
    return cursor.lastrowid


def get_transitions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM transitions ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def get_recent_track_ids(conn: sqlite3.Connection, limit: int) -> list[int]:
    """De `limit` meest recent gebruikte to_track_id's uit transitions.

    Gebruikt om herhaling binnen een set te voorkomen bij suggest_next().
    """
    if limit <= 0:
        return []
    rows = conn.execute(
        "SELECT to_track_id FROM transitions ORDER BY used_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r["to_track_id"] for r in rows]


def get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Samenvattende statistieken voor de `stats` CLI-command."""
    total = conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()["n"]

    def _coverage(column: str) -> int:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM tracks WHERE {column} IS NOT NULL"
        ).fetchone()
        return row["n"]

    return {
        "total_tracks": total,
        "with_bpm": _coverage("bpm"),
        "with_key": _coverage("camelot"),
        "with_mood": _coverage("mood_happy"),
        "with_genre": _coverage("genre"),
        "with_mbid": _coverage("mbid"),
        "total_transitions": conn.execute(
            "SELECT COUNT(*) AS n FROM transitions"
        ).fetchone()["n"],
    }


def normalize_energy(conn: sqlite3.Connection) -> int:
    """Herbereken de 0-1 `energy`-kolom via min-max normalisatie van
    `energy_raw` over alle tracks in de bibliotheek.

    Roept geen audio-analyse aan (goedkoop, alleen een SQL-pass) — de
    ingest-pipeline roept dit aan het eind van elke ingest-run aan, zodat
    nieuw toegevoegde tracks de schaal van de hele collectie meenemen.

    Retourneert het aantal bijgewerkte rijen. Tracks zonder energy_raw
    (mislukte analyse) blijven op energy=NULL staan.
    """
    bounds = conn.execute(
        "SELECT MIN(energy_raw) AS lo, MAX(energy_raw) AS hi "
        "FROM tracks WHERE energy_raw IS NOT NULL"
    ).fetchone()
    if bounds is None or bounds["lo"] is None:
        return 0

    lo, hi = bounds["lo"], bounds["hi"]
    if hi == lo:
        # Eén track, of alle tracks hebben identieke ruwe energie:
        # een echte 0-1 spreiding is niet te bepalen, dus zet op het
        # neutrale midden i.p.v. een willekeurige 0 of 1 te verzinnen.
        cursor = conn.execute(
            "UPDATE tracks SET energy = 0.5 WHERE energy_raw IS NOT NULL"
        )
    else:
        cursor = conn.execute(
            "UPDATE tracks SET energy = (energy_raw - ?) / (? - ?) "
            "WHERE energy_raw IS NOT NULL",
            (lo, hi, lo),
        )
    conn.commit()
    return cursor.rowcount


# --- MusicBrainz cache -------------------------------------------------

def get_cached_lookup(conn: sqlite3.Connection, cache_key: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM musicbrainz_cache WHERE cache_key = ?", (cache_key,)
    ).fetchone()
    return dict(row) if row else None


def set_cached_lookup(
    conn: sqlite3.Connection, cache_key: str, mbid: str | None, genre_tags: str | None
) -> None:
    conn.execute(
        """
        INSERT INTO musicbrainz_cache (cache_key, mbid, genre_tags)
        VALUES (?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            mbid = excluded.mbid,
            genre_tags = excluded.genre_tags,
            fetched_at = CURRENT_TIMESTAMP
        """,
        (cache_key, mbid, genre_tags),
    )
    conn.commit()
