# dj-engine

Lokale, zelfstandige DJ track-database & live recommendation engine.

Analyseert een muziekbibliotheek met Essentia (BPM, key, energie,
danceability, mood/genre optioneel), slaat resultaten op in SQLite,
verrijkt via MusicBrainz, en beveelt vervolgtracks aan op basis van een
gekozen richting: **build** (energie omhoog) / **hold** (vasthouden) /
**ease** (energie omlaag) / **surprise** (verrassende maar verdedigbare
afslag).

Zie `BUILD_SPEC.md` voor de volledige oorspronkelijke opdracht.

## Status

**Fase 1 (MVP) — compleet.**

- [x] Projectstructuur, database schema + CRUD, Camelot-mapping
- [x] Essentia-extractor (BPM/key/loudness/danceability/energy)
- [x] Mood/genre-classificatie (lazy-loaded, optioneel)
- [x] MusicBrainz-client met rate-limiting + caching
- [x] Ingest-pipeline (idempotent, per-bestand foutafhandeling)
- [x] Scoring- en recommendation-engine
- [x] CLI (`ingest`, `suggest`, `search`, `stats`)

**Fase 2 (herzien, Rekordbox-gekoppeld) — in opbouw.** Zie
[Rekordbox-workflow](#rekordbox-workflow) hieronder. Gebouwd:

- [x] Rekordbox XML-catalogusimport (`import-rekordbox`) — snel, geen audio
- [x] On-demand single-track analyse (`analyze`) — bpm/key uit Rekordbox
      blijven leidend, alleen energy/mood/genre worden aangevuld
- [x] Periodieke i.p.v. per-track energy-normalisatie
- [x] `suggest`: eerst snelle score (bpm/key/genre), dan achtergrond-
      verfijning met energie/mood voor de kansrijkste kandidaten
      (`recommend/refine.py`, multiprocessing)
- [x] Handmatige API-key-opslag (`set-api-key`, schrijft naar `.env`)
- [ ] AI-suggestiebron via Claude API + websearch (`--source llm`) — bewust
      nog niet gebouwd, config-infrastructuur staat al klaar (`claude_api`
      in config.yaml)
- [ ] Web-UI/FastAPI

217 tests, allemaal groen (`pytest -q`).

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

**Drie lagen**, hoog naar laag prioriteit: `config.local.yaml` >
`config.yaml` > ingebouwde defaults. `config.local.yaml` (zie
`config.local.yaml.example`) wordt automatisch meegeladen als het bestaat,
staat in `.gitignore`, en is bedoeld voor machine-/persoonsspecifieke
waarden die niet in git horen — op dit moment vooral
`rekordbox.path_mapping` (een absoluut pad naar jouw eigen muziekmap).

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

## Rekordbox-workflow

Voor grote bibliotheken (getest tegen een echte export van 7308 tracks) is
bulk-analyseren met Essentia vooraf niet nodig — Rekordbox heeft zelf al
bpm/key geanalyseerd. Workflow:

```bash
# 1. In Rekordbox: File -> Export Collection in xml format
# 2. Snelle catalogusimport (geen audio-decode, ~1s voor 7000+ tracks):
dj-engine import-rekordbox bibliotheek.xml

# 3. Vul rekordbox.path_mapping in als Rekordbox een ander pad gebruikt
#    (bv. Windows) dan waar dj-engine draait. Dit is persoonlijk/machine-
#    specifiek -- hoort niet in config.yaml (git), maar in config.local.yaml
#    (genegeerd door git, wordt automatisch meegeladen):
cp config.local.yaml.example config.local.yaml
#    # pas config.local.yaml aan:
#    rekordbox:
#      path_mapping:
#        - from: "C:/Users/naam/Music/DJ Muziek"
#          to: "/pad/op/dit/systeem"

# 4. Tijdens draaien/voorbereiden: 1 track door de engine
dj-engine analyze 42

# 5. Suggesties: eerst snel (bpm/key/genre), daarna automatisch ververst
#    met energie/mood voor de beste ~40 kandidaten
dj-engine suggest 42 --direction build
```

`bpm`/`key`/`camelot` die uit Rekordbox komen worden **nooit** overschreven
door Essentia's eigen detectie — alleen `energy`/`mood_*`/`loudness`/
`danceability` komen uit onze eigen analyse. Streaming-tracks (Tidal e.d.,
geen lokaal bestand) worden herkend en overgeslagen, nooit als fout behandeld.

De `energy`-kolom wordt niet na elke losse `analyze` herschaald, maar pas
na een configureerbare drempel (`scoring.energy_renormalize_threshold`,
default 20) — of forceer het direct met `dj-engine normalize-energy`.

### API-key voor de (nog niet gebouwde) AI-suggestiebron

```bash
dj-engine set-api-key sk-ant-...   # schrijft naar .env, staat in .gitignore
```

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
- **`db.upsert_track(..., commit=False)`**: ontdekt bij een echte
  Rekordbox-import van 7308 tracks — één `conn.commit()` per rij duurde
  44,7s; één commit na de hele batch: 0,7s. `commit=True` blijft de
  default (ongewijzigd gedrag voor alle bestaande call sites); bulk-imports
  gebruiken expliciet `commit=False` + één `conn.commit()` na afloop.
- **Multiprocessing met `spawn`, niet `fork`**: `recommend/refine.py`
  gebruikt bewust `multiprocessing.get_context("spawn")` i.p.v. Linux'
  default `fork` — forken ná het laden van TensorFlow in het hoofdproces
  is een bekende bron van hangs/crashes bij native libraries. Essentia/TF
  worden toch al lazy (pas in de worker) geïmporteerd, dus dit kost geen
  extra opstarttijd in de praktijk.

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
