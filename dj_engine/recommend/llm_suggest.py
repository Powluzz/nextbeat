"""AI-suggestiebron: hetzelfde vaste prompt/antwoord-'algoritme' voor élke
provider (zie enrichment/llm_providers/), zodat het resultaat zo
voorspelbaar mogelijk blijft ongeacht welke API de gebruiker koppelt.

Ontwerp:
1. De lokale engine (recommend.engine.suggest_next) levert eerst een
   gesloten kandidatenlijst (shortlist) — het model mag NOOIT een track
   noemen die daar niet in staat (voorkomt gefabriceerde suggesties).
2. build_prompt() bouwt exact dezelfde instructie/vraag, ongeacht provider.
3. parse_llm_response() parseert het antwoord robuust (code fences, rommel
   eromheen) en verwijdert elk track_id dat niet in de kandidatenlijst
   staat — een hallucinatie-check die nooit vertrouwt op het model alleen.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from dj_engine import db as db_module
from dj_engine.enrichment.llm_providers import LLMProviderError, get_provider
from dj_engine.recommend.engine import suggest_next
from dj_engine.recommend.scoring import VALID_DIRECTIONS

logger = logging.getLogger(__name__)

_DIRECTION_EXPLANATIONS = {
    "build": "energie omhoog brengen",
    "hold": "energie/sfeer vasthouden",
    "ease": "energie omlaag brengen",
    "surprise": "een verrassende maar verdedigbare afslag nemen",
}

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _format_track(track: dict[str, Any]) -> str:
    bpm = f"{track['bpm']:.0f}" if track.get("bpm") is not None else "?"
    return (
        f"[id={track['id']}] {track.get('artist') or '?'} - {track.get('title') or '?'} "
        f"({bpm} BPM, {track.get('camelot') or '?'}, genre: {track.get('genre') or '?'})"
    )


def build_prompt(current: dict[str, Any], candidates: list[dict[str, Any]], direction: str) -> str:
    """Bouw de vaste prompt — identiek voor elke provider.

    Args:
        current: de huidige track (dict, zie db.get_track()).
        candidates: gesloten kandidatenlijst (dicts, zie db.get_all_tracks()).
        direction: "build" | "hold" | "ease" | "surprise".
    """
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Onbekende richting: {direction!r} (verwacht: {VALID_DIRECTIONS})")

    candidate_lines = "\n".join(f"{i+1}. {_format_track(c)}" for i, c in enumerate(candidates))

    return f"""Je bent een DJ-assistent die vervolgtracks beoordeelt voor een live set of setvoorbereiding.

HUIDIGE TRACK:
{_format_track(current)}

RICHTING: {direction} ({_DIRECTION_EXPLANATIONS[direction]})

KANDIDATEN (kies UITSLUITEND uit onderstaande lijst — noem NOOIT een track die hier niet in staat):
{candidate_lines}

OPDRACHT:
Beoordeel voor elke kandidaat hoe goed die past als vervolgtrack, gegeven de
richting hierboven. Gebruik websearch als je die tool hebt om te checken of
deze track daadwerkelijk in bestaande DJ-sets/tracklists na de huidige track
voorkomt, of om genre-verwantschap/artiestrelaties te checken die niet uit
de titel blijken. Heb je geen websearch, baseer je oordeel dan op je eigen
kennis van deze artiesten/genres/tracks — wees in dat geval extra
terughoudend en markeer je antwoorden daar eerlijk naar.

Antwoord UITSLUITEND met een JSON-array, geen andere tekst ervoor of erna,
in exact dit formaat:
[
  {{"track_id": <int, moet een id uit de kandidatenlijst zijn>,
    "reden": "<max 2 zinnen, Nederlands>",
    "gegrond_op_bron": <true als je dit via websearch geverifieerd hebt, anders false>}}
]

Sorteer van beste naar minst goede match. Neem alleen tracks op die je
daadwerkelijk zou aanraden — korter dan de volledige kandidatenlijst mag."""


def parse_llm_response(text: str, valid_ids: set[int]) -> list[dict[str, Any]]:
    """Parse het modelantwoord robuust naar een lijst suggesties.

    Nooit een gefabriceerd resultaat doorlaten: elk item zonder geldig
    'track_id' (int, aanwezig in `valid_ids`) wordt overgeslagen en gelogd,
    nooit stilzwijgend geaccepteerd. Bij volledig onparseerbare tekst wordt
    een lege lijst teruggegeven (nooit een crash).
    """
    cleaned = _CODE_FENCE_RE.sub("", text.strip()).strip()

    data = _try_parse_json_array(cleaned)
    if data is None:
        logger.warning("Kon geen JSON-array uit het modelantwoord halen: %r", text[:300])
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning("Overgeslagen: geen dict-item in modelantwoord: %r", item)
            continue
        track_id = item.get("track_id")
        if not isinstance(track_id, int) or track_id not in valid_ids:
            logger.warning(
                "Overgeslagen: track_id %r staat niet in de kandidatenlijst (mogelijke hallucinatie)",
                track_id,
            )
            continue
        results.append({
            "track_id": track_id,
            "reden": str(item.get("reden", "")).strip(),
            "gegrond_op_bron": bool(item.get("gegrond_op_bron", False)),
        })
    return results


def _try_parse_json_array(text: str) -> list[Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        pass

    # Fallback: er stond mogelijk tekst vóór/na de array — pak het eerste
    # '['...']'-blok via bracket-matching en probeer dat te parsen.
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start:i + 1])
                    return parsed if isinstance(parsed, list) else None
                except json.JSONDecodeError:
                    return None
    return None


def suggest_next_llm(
    conn,
    current_track_id: int,
    direction: str,
    config: dict[str, Any],
    top_n: int = 10,
) -> dict[str, Any]:
    """AI-suggestie op basis van een lokale shortlist + het vaste prompt-
    contract hierboven. Werkt met elke geconfigureerde provider
    (config: llm_suggest.provider).

    Returns:
        {"results": [{"track": <dict>, "reden": str, "gegrond_op_bron": bool}, ...],
         "used_search": bool, "provider": str, "model": str}

    Raises:
        ValueError: onbekende richting of current_track_id bestaat niet.
        LLMProviderError: provider niet configureerbaar/bereikbaar (geen
            API-key, netwerkfout, ...) — de caller (CLI) toont dit netjes.
    """
    current = db_module.get_track(conn, current_track_id)
    if current is None:
        raise ValueError(f"Track met id {current_track_id} bestaat niet")

    llm_config = config.get("llm_suggest", {})
    shortlist_size = llm_config.get("shortlist_size", 30)

    shortlist = suggest_next(conn, current_track_id, direction, config, top_n=shortlist_size)
    candidates = [r["track"] for r in shortlist]
    if not candidates:
        return {"results": [], "used_search": False, "provider": llm_config.get("provider"), "model": None}

    prompt = build_prompt(current, candidates, direction)
    provider = get_provider(config)
    response = provider.complete(prompt, llm_config)

    valid_ids = {c["id"] for c in candidates}
    parsed = parse_llm_response(response.text, valid_ids)

    by_id = {c["id"]: c for c in candidates}
    results = [
        {"track": by_id[item["track_id"]], "reden": item["reden"], "gegrond_op_bron": item["gegrond_op_bron"]}
        for item in parsed
    ][:top_n]

    return {
        "results": results,
        "used_search": response.used_search,
        "provider": llm_config.get("provider", "anthropic"),
        "model": response.model,
    }
