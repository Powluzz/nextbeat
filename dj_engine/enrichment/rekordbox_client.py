"""Rekordbox XML-catalogusimport: snelle import van track-metadata
(filepath, title, artist, genre, bpm, camelot) uit een Rekordbox
"Export Collection in xml format"-bestand. Geen audio-decode.

Rekordbox's eigen BPM/key-analyse wordt 1-op-1 overgenomen — geen
Essentia-aanroep hier. Tracks die zo binnenkomen hebben dus meteen bpm/
camelot; energy/mood_* blijven NULL tot een latere analyze()-aanroep
(zie ingest/pipeline.py::analyze_track).

Alleen <COLLECTION><TRACK>-elementen worden gelezen — de <PLAYLISTS>-boom
bevat enkel TrackID-referenties (geen metadata) en wordt genegeerd.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

from dj_engine.analysis.camelot import key_to_camelot

logger = logging.getLogger(__name__)

# file://localhost/tidal:tracks:12345 -> Tidal-streaming, geen lokaal bestand.
_STREAMING_LOCATION_RE = re.compile(r"^file://localhost/(tidal|spotify|deezer):(.+)$")
STREAMING_SCHEMES = ("tidal://", "spotify://", "deezer://")

# Rekordbox Tonality: majeur zonder suffix (bv. "D", "F#"), mineur met een
# 'm'-suffix (bv. "Dm", "F#m"). Alles wat hier niet op past (leeg, of —
# geobserveerd in het echt — corrupte tag-data die in dit veld terecht is
# gekomen) levert bewust (None, None, None) op, nooit een gok.
_TONALITY_RE = re.compile(r"^([A-G](?:#|b)?)(m)?$")


def _decode_location(location: str, path_mapping: list[dict[str, str]]) -> str:
    """Zet een Rekordbox Location-URI om naar een filepath.

    Streaming-bronnen krijgen een synthetische 'scheme://'-vorm (bv.
    'tidal://tracks/12345') zodat ze herkenbaar 'geen lokaal bestand' zijn
    — nooit stilzwijgend als een kapot pad behandeld.
    """
    streaming_match = _STREAMING_LOCATION_RE.match(location)
    if streaming_match:
        scheme, rest = streaming_match.groups()
        return f"{scheme}://{rest.replace(':', '/')}"

    parsed = urlparse(location)
    raw_path = unquote(parsed.path)
    # "/C:/Users/..." (URL-pad met leidende slash voor een Windows-drive-letter) -> "C:/Users/..."
    if re.match(r"^/[A-Za-z]:", raw_path):
        raw_path = raw_path[1:]

    for rule in path_mapping:
        prefix = rule.get("from", "")
        if prefix and raw_path.startswith(prefix):
            raw_path = rule["to"] + raw_path[len(prefix):]
            break

    return raw_path


def _parse_tonality(tonality: str | None) -> tuple[str | None, str | None, str | None]:
    """'Dm' -> (key='D', scale='minor', camelot='7A'). Leeg/onherkenbaar
    -> (None, None, None)."""
    if not tonality:
        return None, None, None
    match = _TONALITY_RE.match(tonality.strip())
    if not match:
        logger.warning("Onherkende Tonality-waarde overgeslagen: %r", tonality)
        return None, None, None
    note, minor_suffix = match.groups()
    scale = "minor" if minor_suffix else "major"
    camelot = key_to_camelot(note, scale)
    return note, scale, camelot


def parse_rekordbox_xml(
    xml_path: str, path_mapping: list[dict[str, str]] | None = None
) -> Iterator[dict[str, Any]]:
    """Parse een Rekordbox XML-collectie-export naar track-dicts.

    Args:
        xml_path: pad naar het geëxporteerde .xml-bestand.
        path_mapping: lijst van {"from": ..., "to": ...}-regels om een
            prefix van het opgeslagen (vaak Windows-)pad te herschrijven
            naar een pad dat op dit systeem bestaat. Leeg (default) laat
            paden ongewijzigd.

    Yields:
        dicts met filepath/title/artist/genre/bpm/key/scale/camelot,
        plus 'rekordbox_track_id' en 'is_streaming' (voor de caller —
        worden niet naar db.upsert_track() doorgegeven, dat zijn geen
        kolommen in tracks).

    Raises:
        ValueError: als het bestand geen <COLLECTION> bevat (geen geldige
            Rekordbox-collectie-export).
    """
    mapping = path_mapping or []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    collection = root.find("COLLECTION")
    if collection is None:
        raise ValueError(
            f"Geen <COLLECTION> gevonden in {xml_path} — is dit een Rekordbox "
            "'Export Collection in xml format'-bestand?"
        )

    for track_el in collection.findall("TRACK"):
        location = track_el.get("Location")
        if not location:
            logger.warning(
                "Track zonder Location overgeslagen (TrackID=%s, Name=%s)",
                track_el.get("TrackID"), track_el.get("Name"),
            )
            continue

        filepath = _decode_location(location, mapping)
        is_streaming = filepath.startswith(STREAMING_SCHEMES)

        bpm_raw = track_el.get("AverageBpm")
        try:
            bpm = float(bpm_raw) if bpm_raw else None
        except ValueError:
            bpm = None

        key, scale, camelot = _parse_tonality(track_el.get("Tonality"))

        yield {
            "filepath": filepath,
            "title": track_el.get("Name") or None,
            "artist": track_el.get("Artist") or None,
            "genre": track_el.get("Genre") or None,
            "bpm": bpm,
            "key": key,
            "scale": scale,
            "camelot": camelot,
            "rekordbox_track_id": track_el.get("TrackID"),
            "is_streaming": is_streaming,
        }
