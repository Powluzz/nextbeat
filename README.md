# dj-engine

Lokale, zelfstandige DJ track-database & live recommendation engine.

Analyseert een muziekbibliotheek met Essentia (BPM, key, energie,
danceability, mood/genre optioneel), slaat resultaten op in SQLite,
verrijkt via MusicBrainz, en beveelt vervolgtracks aan op basis van een
gekozen richting: **build** (energie omhoog) / **hold** (vasthouden) /
**ease** (energie omlaag) / **surprise** (verrassende maar verdedigbare
afslag).

Zie `BUILD_SPEC.md` voor de volledige oorspronkelijke opdracht.

## Status: Fase 1 (MVP) — compleet

- [x] Projectstructuur
- [x] Database schema + CRUD + tests
- [x] Camelot-mapping + tests
- [x] Essentia-extractor (BPM/key/loudness/danceability/energy) + tests
- [x] Mood/genre-classificatie (lazy-loaded, optioneel) + tests
- [x] MusicBrainz-client met rate-limiting + caching + tests
- [x] Ingest-pipeline (idempotent, per-bestand foutafhandeling) + tests
- [x] Scoring- en recommendation-engine + tests
- [x] CLI (`ingest`, `suggest`, `search`, `stats`) + tests

169 tests, allemaal groen (`pytest -q`), inclusief echte inference tegen de
gedownloade mood/genre-modellen. Fase 2 (FastAPI + web-UI) en Fase 3
(zelflerende gewichten) zijn nog niet gebouwd — zie BUILD_SPEC.md.

## Installatie

Dit systeem had geen `python3-venv` beschikbaar; gebruik bij voorkeur wél een
echte virtualenv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis,dev]"
```

Zonder venv (zoals in deze projectomgeving gedaan, PEP 668 "externally
managed" systemen):

```bash
python3 -m pip install --user --break-system-packages -e ".[analysis,dev]"
```

Optioneel, voor mood/genre-classificatie (aanbevolen maar niet vereist):

```bash
pip install essentia-tensorflow   # i.p.v. het basis 'essentia'-pakket
bash models/download_models.sh    # vraagt expliciete bevestiging
```

Zonder deze modellen blijven de `mood_*`- en `genre`-kolommen `NULL` — er
wordt nooit data gefabriceerd. Met modellen: getest en werkend, zie
`tests/test_mood_genre_models.py` (de `requires_real_models`-tests draaien
alleen als de modellen daadwerkelijk gedownload zijn).

## Configuratie

Alle instelbare parameters staan in `config.yaml` (database-locatie, audio-
extensies, MusicBrainz rate limit/User-Agent, modelpaden, scoringsgewichten
en -doelwaarden per richting). Gebruik `dj-engine --config pad/naar/eigen.yaml`
om een alternatieve config te gebruiken.

## Gebruik

```bash
# Analyseer een muziekmap (recursief), verrijk via MusicBrainz
dj-engine ingest ~/Music/DJ-set

# Zonder MusicBrainz (offline / sneller)
dj-engine ingest ~/Music/DJ-set --no-musicbrainz

# Vraag vervolgtracks aan voor track met id 42, energie omhoog
dj-engine suggest 42 --direction build --top 10

# Zoeken
dj-engine search --artist "Daft Punk" --genre techno --bpm-range 120-128

# Bibliotheekstatistieken
dj-engine stats
```

`suggest` toont per suggestie een uitsplitsing (`key=`, `bpm=`, `energy=`,
`mood=`, elk 0-100) zodat je ziet waarom een track wordt voorgesteld.

## Testen

```bash
pytest -q
```

Alle Essentia-analysetests draaien op korte, zelfgegenereerde (CC0)
audiofragmenten in `tests/fixtures/` — zie
`tests/fixtures/generate_fixtures.py`. Er wordt nooit copyrighted materiaal
meegeleverd. MusicBrainz-netwerkcalls zijn in alle tests gemockt.

## Ontwerpkeuzes die niet 1-op-1 in BUILD_SPEC.md staan

- **`energy`-kolom**: BUILD_SPEC.md noemt geen specifiek Essentia-algoritme
  hiervoor. Gekozen aanpak (in overleg): een ruwe, ongenormaliseerde
  `energy_raw`-feature (RMS + hoogfrequent-energieratio via
  `EnergyBandRatio`) per track, die na elke ingest-run via min-max
  normalisatie over de hele bibliotheek naar de 0-1 `energy`-kolom
  geschaald wordt (`db.normalize_energy()`). Dit is dus relatief aan je
  eigen collectie, niet een vaste absolute schaal.
- **`energy_raw`-kolom**: extra kolom t.o.v. de minimale schema-lijst in
  BUILD_SPEC.md (die zegt "minimaal deze kolommen") — nodig om de
  bibliotheekbrede energie-normalisatie idempotent te kunnen herhalen
  zonder audio opnieuw te analyseren.
- **TensorFlow-node-namen in `mood_genre_models.py`**: bij het echt
  testen tegen de gedownloade modellen bleken twee node-namen af te wijken
  van Essentia's algemene documentatie/defaults: de mood-classifiers zijn
  2-klasse softmax-koppen (`output="model/Softmax"`, niet de default
  `"model/Sigmoid"`), en `genre_discogs400` is geëxporteerd als SavedModel/
  PartitionedCall (`input="serving_default_model_Placeholder"`,
  `output="PartitionedCall"`) — zie de code-comments in dat bestand.
- **MusicBrainz genre-tags vereisen 2 calls**: `search_recordings()` levert
  in de echte MB-API geen `tag-list` mee (bevestigd tegen musicbrainz.org).
  Een gevonden MBID wordt daarom gevolgd door een tweede, eveneens
  rate-limited `get_recording_by_id(mbid, includes=["tags"])`-call.

## MusicBrainz — gebruiksvoorwaarden

Dit project respecteert de MusicBrainz API-gebruiksvoorwaarden:

- **Rate limit**: maximaal 1 request/seconde, afgedwongen door een eigen
  throttle (`enrichment/musicbrainz_client.RateLimiter`) plus
  musicbrainzngs' interne rate limiter als tweede vangnet.
- **User-Agent**: een unieke, identificeerbare header (naam/versie/contact),
  ingesteld via `config.yaml` (`musicbrainz.user_agent_*`) — vul je eigen
  contactgegevens in voordat je dit tegen de productie-API draait.
- **Caching**: resultaten worden lokaal gecached (`musicbrainz_cache`-tabel)
  om herhaalde lookups te voorkomen.

Zie de volledige voorwaarden op de
[MusicBrainz API-documentatie](https://musicbrainz.org/doc/MusicBrainz_API).

## Licentie/ToS-bewustzijn

Dit project beperkt automatische metadata-verrijking bewust tot open bronnen
(MusicBrainz, en optioneel AcousticBrainz-achtige databases). Verrijking via
andere bronnen (zoals tracklist-sites) valt buiten de scope van dit project
en is onderhevig aan de eigen gebruiksvoorwaarden van die bronnen — voeg dat
zelf toe op eigen risico/verantwoordelijkheid.
