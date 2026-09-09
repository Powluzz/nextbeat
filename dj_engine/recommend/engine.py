"""Recommendation-engine: suggest_next() scoort alle tracks in de bibliotheek
t.o.v. de huidige track en geeft de top-N terug, met een uitsplitsing per
dimensie zodat de gebruiker ziet WAAROM een track wordt voorgesteld.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from dj_engine import db as db_module
from dj_engine.recommend.scoring import (
    VALID_DIRECTIONS,
    bpm_score,
    energy_score,
    key_score,
    mood_score,
)


def suggest_next(
    conn: sqlite3.Connection,
    current_track_id: int,
    direction: str,
    config: dict[str, Any],
    top_n: int = 10,
    exclude_recently_played: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Beveel de volgende `top_n` tracks aan vanaf de huidige track.

    Args:
        conn: open sqlite3-connectie.
        current_track_id: id van de track die nu draait.
        direction: "build" | "hold" | "ease" | "surprise".
        config: app-config (zie dj_engine.config.load_config()) — bepaalt
            gewichten, bpm-tolerantie, camelot-scores en energy-targets.
        top_n: aantal suggesties.
        exclude_recently_played: expliciete lijst track-id's om uit te
            sluiten (bv. al gedraaid deze set). None (default) = gebruik
            automatisch de laatste `scoring.default_exclude_recent`
            tracks uit de transitions-tabel. Geef `[]` expliciet mee om
            geen enkele automatische uitsluiting toe te passen.

    Returns:
        Lijst van maximaal `top_n` dicts, gesorteerd op total_score
        (hoog naar laag), elk met:
            {"track": <track-dict>, "total_score": float,
             "breakdown": {"key": float, "bpm": float, "energy": float, "mood": float}}

    Raises:
        ValueError: onbekende richting, of current_track_id bestaat niet.
    """
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Onbekende richting: {direction!r} (verwacht: {VALID_DIRECTIONS})")

    current = db_module.get_track(conn, current_track_id)
    if current is None:
        raise ValueError(f"Track met id {current_track_id} bestaat niet")

    scoring_cfg = config["scoring"]
    weights = scoring_cfg["weights"][direction]
    weight_sum = sum(weights.values()) or 1.0
    bpm_max_dev = scoring_cfg["bpm_max_deviation_pct"]
    camelot_scores = scoring_cfg["camelot_scores"]
    energy_targets = scoring_cfg.get("energy_targets")

    if exclude_recently_played is None:
        exclude_recently_played = db_module.get_recent_track_ids(
            conn, scoring_cfg.get("default_exclude_recent", 0)
        )
    exclude_ids = set(exclude_recently_played) | {current_track_id}

    results = []
    for candidate in db_module.get_all_tracks(conn):
        if candidate["id"] in exclude_ids:
            continue

        breakdown = {
            "key": key_score(current["camelot"], candidate["camelot"], scores=camelot_scores),
            "bpm": bpm_score(current["bpm"], candidate["bpm"], max_deviation_pct=bpm_max_dev),
            "energy": energy_score(
                current["energy"], candidate["energy"], direction, targets=energy_targets
            ),
            "mood": mood_score(current, candidate),
        }
        total_score = sum(weights[dim] * breakdown[dim] for dim in breakdown) / weight_sum

        results.append({
            "track": candidate,
            "total_score": total_score,
            "breakdown": breakdown,
        })

    results.sort(key=lambda r: r["total_score"], reverse=True)
    return results[:top_n]
