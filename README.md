# dj-engine

Lokale, zelfstandige DJ track-database & live recommendation engine.

Analyseert een muziekbibliotheek met Essentia (BPM, key, energie,
danceability, mood, genre), slaat resultaten op in SQLite, verrijkt via
MusicBrainz, en beveelt vervolgtracks aan op basis van een gekozen richting:
**build** / **hold** / **ease** / **surprise**.

Zie `BUILD_SPEC.md` voor de volledige opdracht. Dit project wordt stap voor
stap opgebouwd volgens Fase 1 uit dat document.

## Status

- [x] Projectstructuur
- [x] Database schema + CRUD + tests
- [ ] Camelot-module + tests
- [ ] Essentia-extractor
- [ ] MusicBrainz-client
- [ ] Ingest-pipeline
- [ ] Scoring/recommend-engine
- [ ] CLI

## Installatie (work in progress)

```bash
python3 -m pip install --user --break-system-packages -e ".[analysis,dev]"
```

> Dit systeem heeft geen `python3-venv` beschikbaar; packages zijn daarom
> user-level geïnstalleerd. Gebruik een echte venv (`python3 -m venv .venv`)
> waar mogelijk.

## MusicBrainz gebruiksvoorwaarden

Dit project respecteert de MusicBrainz API rate limit (1 request/seconde) en
stuurt een unieke, identificeerbare `User-Agent` header mee (naam, versie,
contact — configureerbaar in `config.yaml`). Zie de MusicBrainz
[API-documentatie](https://musicbrainz.org/doc/MusicBrainz_API) voor de
volledige voorwaarden. Automatische verrijking via andere bronnen (zoals
tracklist-sites) valt buiten de scope van dit project en is aan de eigen
gebruiksvoorwaarden van die bronnen onderhevig.
