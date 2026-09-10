"""Tests voor dj_engine.enrichment.rekordbox_client.

Draait tegen tests/fixtures/rekordbox_sample.xml — een kleine, zelfgemaakte
fixture die de echte veldvormen nabootst die zijn aangetroffen in een
echte Rekordbox-export (Windows-paden, lege/mineure/corrupte Tonality,
Tidal-streaming-entries, ontbrekende BPM).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dj_engine.enrichment.rekordbox_client import parse_rekordbox_xml

FIXTURE = Path(__file__).parent / "fixtures" / "rekordbox_sample.xml"


def _by_id(tracks, rb_id):
    return next(t for t in tracks if t["rekordbox_track_id"] == rb_id)


def test_parses_all_collection_tracks_not_playlist_refs():
    tracks = list(parse_rekordbox_xml(str(FIXTURE)))
    assert len(tracks) == 6


def test_major_key_parsed_correctly():
    track = _by_id(list(parse_rekordbox_xml(str(FIXTURE))), "171579258")
    assert track["key"] == "D"
    assert track["scale"] == "major"
    assert track["camelot"] == "10B"
    assert track["bpm"] == 120.0
    assert track["title"] == "Blurred lines"
    assert track["artist"] == "Robin Thicke ft. T.I. & Pharrell"
    assert track["genre"] == "Top 40"
    assert track["is_streaming"] is False


def test_minor_key_parsed_correctly():
    track = _by_id(list(parse_rekordbox_xml(str(FIXTURE))), "246103807")
    assert track["key"] == "F#"
    assert track["scale"] == "minor"
    assert track["camelot"] == "11A"
    assert track["bpm"] == 126.5


def test_empty_tonality_yields_none_not_fabricated():
    track = _by_id(list(parse_rekordbox_xml(str(FIXTURE))), "300000001")
    assert track["key"] is None
    assert track["scale"] is None
    assert track["camelot"] is None
    # bpm is wel gewoon aanwezig -- alleen het Tonality-veld ontbrak
    assert track["bpm"] == 128.0


def test_corrupt_tonality_yields_none_and_does_not_crash():
    track = _by_id(list(parse_rekordbox_xml(str(FIXTURE))), "400000002")
    assert track["key"] is None
    assert track["scale"] is None
    assert track["camelot"] is None


def test_streaming_track_gets_synthetic_scheme_path():
    track = _by_id(list(parse_rekordbox_xml(str(FIXTURE))), "453498254")
    assert track["is_streaming"] is True
    assert track["filepath"] == "tidal://tracks/453498254"
    # metadata (bpm/key/genre) is nog steeds bruikbaar, ondanks geen lokaal bestand
    assert track["bpm"] == 124.0
    assert track["key"] == "A"
    assert track["scale"] == "minor"


def test_missing_bpm_yields_none():
    track = _by_id(list(parse_rekordbox_xml(str(FIXTURE))), "500000003")
    assert track["bpm"] is None


def test_windows_path_decoded_and_url_unescaped():
    track = _by_id(list(parse_rekordbox_xml(str(FIXTURE))), "171579258")
    assert track["filepath"] == "C:/Users/paulm/Music/DJ Muziek/Robin Thicke-Blurred lines.mp3"


def test_path_mapping_rewrites_prefix():
    mapping = [{"from": "C:/Users/paulm/Music/DJ Muziek", "to": "/mnt/music/dj"}]
    tracks = list(parse_rekordbox_xml(str(FIXTURE), path_mapping=mapping))
    track = _by_id(tracks, "171579258")
    assert track["filepath"] == "/mnt/music/dj/Robin Thicke-Blurred lines.mp3"


def test_path_mapping_does_not_affect_streaming_tracks():
    mapping = [{"from": "C:/Users/paulm/Music/DJ Muziek", "to": "/mnt/music/dj"}]
    tracks = list(parse_rekordbox_xml(str(FIXTURE), path_mapping=mapping))
    track = _by_id(tracks, "453498254")
    assert track["filepath"] == "tidal://tracks/453498254"


def test_missing_collection_element_raises_value_error(tmp_path):
    bad_xml = tmp_path / "not_rekordbox.xml"
    bad_xml.write_text("<ROOT><FOO/></ROOT>")
    with pytest.raises(ValueError, match="COLLECTION"):
        list(parse_rekordbox_xml(str(bad_xml)))
