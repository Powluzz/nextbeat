# Bouwopdracht: DJ Track Database & Live Recommendation Engine

## Doel

Bouw een lokale, zelfstandige applicatie die:
1. Een muziekbibliotheek analyseert met Essentia (BPM, key, energie, danceability, mood, genre).
2. De resultaten opslaat in een doorzoekbare database (SQLite).
3. Metadata verrijkt via open bronnen (MusicBrainz, optioneel AcousticBrainz-dump).
4. Tijdens het "draaien" (live) of tijdens setvoorbereiding vervolgtracks aanbeveelt op basis van een gekozen richting: **build** (energie omhoog), **hold** (vasthouden), **ease** (energie omlaag), of **surprise** (verrassende maar verdedigbare afslag).
5. Een simpele lokale web-UI of CLI biedt om tijdens een set snel te werken.

Bouw dit als een production-quality Python-project, geen prototype. Voeg tests, logging, configuratie en documentatie toe.

---

## Technische stack

- **Taal:** Python 3.11+
- **Audio-analyse:** `essentia` (en optioneel `essentia-tensorflow` voor mood/genre-classificatie via pretrained modellen)
- **Metadata tags lezen:** `mutagen`
- **Open metadata enrichment:** `musicbrainzngs` (MusicBrainz API)
- **Database:** SQLite via `sqlite3` (standaard library) of `sqlite-utils` voor gemak
- **CLI:** `argparse` of `click`
- **Web-UI (optioneel, fase 2):** `FastAPI` + eenvoudige HTML/JS frontend (geen zware frontend-framework nodig)
- **Testing:** `pytest`
- **Config:** `.env` of `config.yaml` voor instellingen (rate limits, modelpaden, database-locatie)

---

## Projectstructuur

```
dj-engine/
├── README.md
├── pyproject.toml               # of requirements.txt
├── config.yaml
├── dj_engine/
│   ├── __init__.py
│   ├── db.py                    # database schema + CRUD
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── essentia_extractor.py   # BPM, key, danceability, loudness
│   │   ├── mood_genre_models.py    # TensorFlow mood/genre classificatie (optioneel/lazy-loaded)
│   │   └── camelot.py              # key -> Camelot-code mapping + compatibiliteit
│   ├── enrichment/
│   │   ├── __init__.py
│   │   └── musicbrainz_client.py   # MBID + genre-tag lookup, met rate-limiting
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── pipeline.py             # scan folder -> analyse -> enrichment -> db insert
│   ├── recommend/
│   │   ├── __init__.py
│   │   ├── scoring.py               # key/bpm/energy/mood scores
│   │   └── engine.py                # suggest_next(), setflow-simulatie
│   ├── api/                        # fase 2: FastAPI endpoints
│   │   └── main.py
│   └── cli.py                      # entrypoint: ingest, suggest, serve
├── tests/
│   ├── test_camelot.py
│   ├── test_scoring.py
│   ├── test_db.py
│   └── fixtures/                   # korte test-audiofragmenten (CC0/eigen bezit)
└── models/                          # pretrained Essentia .pb-modellen (los te downloaden)
```

---

## Fase 1: Core engine (MVP)

### 1.1 Database schema (`db.py`)

Implementeer de tabel `tracks` met minimaal deze kolommen:

```sql
CREATE TABLE tracks (
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
    mood_happy REAL,
    mood_sad REAL,
    mood_aggressive REAL,
    mood_relaxed REAL,
    mood_party REAL,
    genre TEXT,
    analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tracks_camelot ON tracks(camelot);
CREATE INDEX idx_tracks_bpm ON tracks(bpm);
```

Voeg ook een tabel `transitions` toe om handmatige of live-gebruikte overgangen te loggen (voor toekomstige zelflerende verbetering):

```sql
CREATE TABLE transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_track_id INTEGER REFERENCES tracks(id),
    to_track_id INTEGER REFERENCES tracks(id),
    direction TEXT,           -- build/hold/ease/surprise
    used_at TEXT DEFAULT CURRENT_TIMESTAMP,
    rating INTEGER            -- optionele handmatige feedback 1-5
);
```

Schrijf CRUD-functies: `insert_track()`, `get_track(id)`, `get_all_tracks()`, `track_exists(filepath)`, `log_transition(...)`.

### 1.2 Audio-analyse (`analysis/essentia_extractor.py`)

Implementeer een functie `analyze_audio(filepath: str) -> dict` die:
- Audio laadt met `essentia.standard.MonoLoader`.
- BPM detecteert met `RhythmExtractor2013(method="multifeature")`.
- Key/scale detecteert met `KeyExtractor()`.
- Loudness berekent met `Loudness()`.
- Danceability berekent met `Danceability()`.
- Alle output retourneert als een dict, inclusief foutafhandeling (bijv. corrupte bestanden loggen en overslaan, niet laten crashen).

Schrijf dit zodanig dat het losstaand testbaar is met een kort audiofragment.

### 1.3 Camelot-mapping (`analysis/camelot.py`)

- Implementeer een volledige mapping van alle 24 toonsoorten (12 majeur + 12 mineur) naar Camelot-codes (1A–12B). Gebruik de standaard Camelot Wheel-indeling.
- Implementeer `camelot_distance(a: str, b: str) -> float` die een compatibiliteitsscore 0–100 teruggeeft volgens deze regels:
  - Zelfde code: 100
  - ±1 stap, zelfde letter (bijv. 8A → 9A of 7A): 90
  - Zelfde nummer, andere letter (relative major/minor, bijv. 8A → 8B): 85
  - ±2 stappen, zelfde letter (energy boost/drop): 75
  - Overig: 30 (of lager, instelbaar)
- Schrijf unit tests die alle edge cases dekken (bijv. wrap-around van 12 naar 1).

### 1.4 Mood/genre classificatie (`analysis/mood_genre_models.py`) — optioneel maar aanbevolen

- Documenteer duidelijk dat dit pretrained TensorFlow-modellen van Essentia vereist (Discogs-EffNet embeddings + downstream classifiers voor mood_happy, mood_sad, mood_aggressive, mood_relaxed, mood_party, genre_discogs400).
- Implementeer lazy loading: als de modelbestanden niet in `models/` aanwezig zijn, sla deze stap over en laat de betreffende database-kolommen `NULL`. Log een duidelijke waarschuwing, verzin geen data.
- Voeg een `download_models.sh`-script toe dat de benodigde `.pb`-bestanden ophaalt van de officiële Essentia-modellenpagina, met instructies dat de gebruiker dit handmatig moet bevestigen/uitvoeren (geen automatische onbeheerde downloads van grote bestanden zonder gebruikersactie).

### 1.5 MusicBrainz-enrichment (`enrichment/musicbrainz_client.py`)

- Implementeer `lookup_track(title, artist) -> (mbid, genre_tags)`.
- **Verplicht:** respecteer MusicBrainz' rate limit van 1 request/seconde. Gebruik een throttle/sleep-mechanisme of een library die dit afdwingt.
- Stel een duidelijke, unieke `User-Agent` header in zoals MusicBrainz vereist (naam + versie + contactgegevens), configureerbaar via `config.yaml`.
- Cache resultaten lokaal (bijv. in de database of een losse cache-tabel) om herhaalde lookups te voorkomen.
- Vang netwerkfouten en timeouts af zonder de hele ingest-pipeline te laten crashen.

### 1.6 Ingest-pipeline (`ingest/pipeline.py`)

- Implementeer `ingest_folder(folder_path, db_connection, config)`:
  - Loop recursief door de map, filter op audio-extensies (.mp3, .wav, .flac, .aiff, .m4a).
  - Sla bestanden over die al in de database staan (idempotent).
  - Lees tags met `mutagen`.
  - Roep audio-analyse aan.
  - Roep MusicBrainz-enrichment aan.
  - Schrijf resultaat naar database.
  - Toon voortgang (bijv. met `tqdm`).
  - Log fouten per bestand naar een logbestand, ga door met de rest.

### 1.7 Scoring & recommendation engine (`recommend/scoring.py`, `recommend/engine.py`)

- Implementeer losse scorefuncties: `key_score()`, `bpm_score()` (met instelbare max BPM-afwijking, standaard 8%), `energy_score(direction)`, `mood_score()`.
- Implementeer gewichtenprofielen per richting (build/hold/ease/surprise) in een configureerbare dict, niet hardcoded diep in de logica.
- Implementeer `suggest_next(current_track_id, direction, top_n=10, exclude_recently_played=[])`:
  - Haal huidige track op.
  - Scoor alle andere tracks in de database.
  - Sluit expliciet tracks uit die recent al gebruikt zijn (voorkom herhaling in een set).
  - Retourneer top N gesorteerd op totaalscore, met een uitsplitsing per dimensie (key/bpm/energy/mood) zodat de gebruiker ziet WAAROM een track wordt voorgesteld.

### 1.8 CLI (`cli.py`)

Implementeer minimaal deze commando's:

```
dj-engine ingest <folder>
dj-engine suggest <track_id> --direction build|hold|ease|surprise --top 10
dj-engine search --artist "..." --genre "..." --bpm-range 120-128
dj-engine stats                     # aantal tracks, dekking mood/genre, etc.
```

---

## Fase 2: Live-modus en UI (na validatie van Fase 1)

### 2.1 Lokale API (`api/main.py`)

- Bouw een lichte FastAPI-app met endpoints:
  - `GET /tracks` — lijst/zoeken
  - `GET /tracks/{id}/suggest?direction=build` — suggesties voor huidig nummer
  - `POST /transitions` — logging van daadwerkelijk gebruikte overgang + optionele rating
- Dit maakt een toekomstige web- of mobiele UI mogelijk zonder de kernlogica te wijzigen.

### 2.2 Eenvoudige lokale web-UI

- Eén pagina: zoek/selecteer huidig nummer → kies richting → toon top-10 suggesties met scoreverdeling.
- Geen zware frontend-stack nodig; simpele HTML + fetch-calls naar de FastAPI-backend is voldoende voor een werkend prototype.

### 2.3 Zelflerende verbetering (optioneel, fase 3)

- Gebruik de `transitions`-tabel om, na voldoende gelogde data, de gewichten per richting te herijken (bijv. via een simpele logistische regressie op "welke dimensies voorspelden een hoge rating").
- Documenteer dit als expliciet toekomstig werk; bouw het niet op fictieve/synthetische trainingsdata.

---

## Kwaliteitseisen

- **Geen gefabriceerde data:** als mood/genre-modellen niet geladen zijn, moet de database `NULL` tonen — nooit een verzonnen waarde.
- **Idempotente ingest:** opnieuw draaien op een al-geanalyseerde map mag niets dupliceren of onnodig herberekenen.
- **Config-gedreven:** rate limits, gewichtenprofielen, modelpaden en database-locatie moeten in `config.yaml` staan, niet hardcoded.
- **Foutafhandeling per bestand:** één corrupt of onleesbaar audiobestand mag de hele ingest-batch niet laten falen.
- **Tests:** unit tests voor Camelot-mapping, scoring-functies en database-CRUD. Mock externe netwerkcalls (MusicBrainz) in tests.
- **Documentatie:** README met installatie-instructies (inclusief hoe Essentia en de optionele TensorFlow-modellen te installeren), gebruiksvoorbeelden per CLI-commando, en een duidelijke sectie over de rate-limit- en gebruiksvoorwaarden van MusicBrainz.
- **Licentie/ToS-bewustzijn:** voeg een korte README-sectie toe die gebruikers waarschuwt dat automatische verrijking via externe bronnen (bijv. tracklist-sites) onderhevig is aan hun eigen gebruiksvoorwaarden, en dat dit project zich beperkt tot MusicBrainz/AcousticBrainz-achtige open bronnen.

## Startopdracht voor Claude Code

Begin met Fase 1, stap voor stap, in deze volgorde: projectstructuur → database schema + tests → Camelot-module + tests → Essentia-extractor (test op 2-3 losse audiofragmenten) → MusicBrainz-client met rate-limiting → ingest-pipeline → scoring/recommend-engine → CLI. Lever na elke stap draaiende, geteste code op voordat je verder gaat naar de volgende stap. Vraag om verduidelijking als een audio-bestandsformaat of modelpad niet duidelijk is, in plaats van aannames te maken over ontbrekende data.
