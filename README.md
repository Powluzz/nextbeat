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
- [x] AI-suggestiebron (`suggest ... --source llm`), provider-onafhankelijk:
      Claude (met websearch) of elke eigen OpenAI-compatibele API — zie
      [AI-suggestie](#ai-suggestie-source-llm) hieronder
- [x] Lokale config-override (`config.local.yaml`, niet in git) voor
      machine-specifieke instellingen zoals `rekordbox.path_mapping`
- [x] Web-UI (`dj-engine serve`) — FastAPI + één statische pagina, zie
      [Web-UI](#web-ui) hieronder

286 tests, allemaal groen (`pytest -q`).

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

## AI-suggestie (`--source llm`)

Naast de lokale, deterministische engine kun je optioneel een LLM om een
tweede mening vragen — bv. omdat het via websearch kan checken of een
kandidaat daadwerkelijk in bestaande DJ-tracklists na de huidige track
voorkomt, iets wat de lokale audio-analyse niet kan weten.

**Niet vastgezet op Claude.** Je kiest zelf de provider in `config.yaml`
(of `config.local.yaml`) via `llm_suggest.provider`:

| Provider | Websearch | Vereist |
|---|---|---|
| `anthropic` (default) | ✅ ingebouwd, server-side | `pip install anthropic` (extra: `llm`) |
| `openai_compatible` | ❌ (model-kennis alleen) | alleen `llm_suggest.base_url` — werkt met OpenAI zelf, Groq, Mistral, een lokale Ollama-server, enzovoort |

**Ongeacht de provider krijgt het model exact dezelfde vraag** (vaste
prompt + verplicht JSON-antwoordformaat, zie
`recommend/llm_suggest.py::build_prompt`) — dat maakt het resultaat zo
voorspelbaar mogelijk, en niet afhankelijk van eigenaardigheden van één
specifieke API. Het model mag bovendien nooit een track noemen die niet in
de lokale shortlist staat — elk antwoord wordt tegen die gesloten lijst
gevalideerd (`parse_llm_response`), en niet-bestaande track-id's worden
altijd stilzwijgend verwijderd, nooit vertrouwd.

```bash
# API-key handmatig instellen (schrijft naar .env, genegeerd door git)
dj-engine set-api-key sk-ant-...
# voor een andere provider/variabele:
dj-engine set-api-key sk-... --var-name OPENAI_API_KEY

# gebruik
dj-engine suggest 42 --direction surprise --source llm
```

Voorbeeld `config.local.yaml` voor een eigen OpenAI-compatibele API:

```yaml
llm_suggest:
  provider: "openai_compatible"
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
  api_key_env_var: "OPENAI_API_KEY"
```

De output toont altijd expliciet of websearch daadwerkelijk gebruikt is
(`met websearch` / `zonder websearch (model-kennis alleen)`) en per
suggestie of de onderbouwing op een gevonden bron steunt (`✓ bron`) —
zodat jij weet hoeveel vertrouwen je aan welk antwoord geeft.

## Web-UI

```bash
pip install -e ".[api]"     # fastapi + uvicorn
dj-engine serve              # http://127.0.0.1:8000/
```

Eén pagina, precies de live-lus: zoek/selecteer de huidige track → kies een
richting → snelle suggesties (altijd beschikbaar) → optioneel **Verfijn met
energie/mood** (kan 1-2 minuten duren op een echte shortlist — bewuste,
blocking knop, geen achtergrondtaak/polling) → optioneel **Vraag AI-mening**
→ klik een suggestie om 'm als volgende track te bevestigen (logt de
overgang in `transitions`, wordt de nieuwe huidige track).

Standaard alleen op `127.0.0.1` (geen authenticatie ingebouwd). Backend-only
acties (`import-rekordbox`, `normalize-energy`, `set-api-key`) staan bewust
niet in de UI — dat zijn eenmalige/onderhoudstaken, geen onderdeel van de
live pick→suggest→confirm-lus; gebruik daarvoor de CLI.

API-documentatie (Swagger, automatisch door FastAPI): `http://127.0.0.1:8000/docs`.

## Testen

```bash
pytest -q
```

Alle Essentia-analysetests draaien op korte, zelfgegenereerde (CC0)
audiofragmenten in `tests/fixtures/` — zie
`tests/fixtures/generate_fixtures.py`. Er wordt nooit copyrighted materiaal
meegeleverd. MusicBrainz-netwerkcalls zijn in alle tests gemockt.

## Ontwerpkeuzes die niet 1-op-1 in BUILD_SPEC.md staan

- **`POST /tracks/{id}/analyze` normaliseert energy meteen**, in
  tegenstelling tot de CLI's `analyze`-commando (die de periodieke drempel
  gebruikt, zie hierboven). De web-UI analyseert typisch de nu-spelende
  track vlak vóórdat er suggesties voor gevraagd worden — energy_score()
  vergelijkt altijd t.o.v. díe track, dus moet `energy` daar direct na
  bruikbaar zijn. `normalize_energy()` is een goedkope SQL-pass (geen
  audio-analyse), dus dit kost niets extra's.
- **`uvicorn.run(..., ws="none")`**: dit systeem heeft een verouderd
  systeembreed `websockets`-pakket dat botst met uvicorn's auto-detectie
  (`ImportError: cannot import name 'ServerProtocol'`) — de app gebruikt
  toch geen WebSockets (de blocking-knop-aanpak voor verfijning/AI-
  suggestie heeft dat niet nodig), dus expliciet uitgeschakeld.
- **AI-suggestie is provider-onafhankelijk, niet Claude-only** (op
  expliciet verzoek): `enrichment/llm_providers/` is een kleine
  adapter-laag (`base.py` definieert het contract, elke provider
  implementeert alleen `complete(prompt) -> tekst`). Het prompt/antwoord-
  'algoritme' zelf (`recommend/llm_suggest.py`) is providerloos en dus
  voor élke adapter identiek. Nieuwe providers toevoegen = één klasse
  erbij in `llm_providers/` + een regel in de factory (`__init__.py`).
- **`openai_compatible`-provider gebruikt bewust alleen `urllib`** (geen
  `openai`-SDK-dependency) — dat past bij het doel ("koppel je eigen API")
  beter dan een SDK die zelf weer aannames doet over welke provider het is.
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
