"""CLI entrypoint voor dj-engine.

Commando's:
    dj-engine ingest <folder>
    dj-engine import-rekordbox <xml-pad>
    dj-engine analyze <track_id>
    dj-engine normalize-energy
    dj-engine suggest <track_id> --direction build|hold|ease|surprise --top 10 [--source engine|llm]
    dj-engine search --artist "..." --genre "..." --bpm-range 120-128
    dj-engine stats
    dj-engine set-api-key <key>
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

import click

from dj_engine import db as db_module
from dj_engine.config import load_config, load_dotenv
from dj_engine.enrichment.llm_providers import LLMProviderError
from dj_engine.enrichment.musicbrainz_client import MusicBrainzClient
from dj_engine.enrichment.rekordbox_client import parse_rekordbox_xml
from dj_engine.ingest.pipeline import analyze_track, ingest_folder
from dj_engine.logging_config import configure_logging
from dj_engine.recommend.engine import suggest_next
from dj_engine.recommend.llm_suggest import suggest_next_llm
from dj_engine.recommend.refine import refine_candidates


def _connect(config: dict[str, Any]) -> sqlite3.Connection:
    return db_module.connect(config["database"]["path"])


def _format_track_line(t: dict[str, Any]) -> str:
    return (
        f"[{t['id']:>4}] {t['artist'] or '?'} - {t['title'] or '?'} "
        f"({_fmt(t.get('bpm'))} BPM, {t.get('camelot') or '?'}, {t.get('genre') or '?'})"
    )


def _fmt(value: float | None, decimals: int = 0) -> str:
    return "?" if value is None else f"{value:.{decimals}f}"


def _print_suggestions(results: list[dict[str, Any]], label: str) -> None:
    click.echo(f"\n-- {label} --")
    if not results:
        click.echo("Geen suggesties gevonden (bibliotheek te klein of alles uitgesloten).")
        return
    for rank, result in enumerate(results, start=1):
        t = result["track"]
        b = result["breakdown"]
        click.echo(f"{rank:2}. {_format_track_line(t)}  totaal={result['total_score']:.1f}")
        click.echo(
            f"      key={b['key']:.0f}  bpm={b['bpm']:.0f}  "
            f"energy={b['energy']:.0f}  mood={b['mood']:.0f}"
        )


@click.group()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Pad naar config.yaml (default: config.yaml in de projectroot).",
)
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """dj-engine: lokale DJ track-database & live recommendation engine."""
    load_dotenv()  # .env (indien aanwezig) in environment laden, vóór config
    config = load_config(config_path)
    configure_logging(config)
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--no-musicbrainz", is_flag=True, help="Sla MusicBrainz-verrijking over (offline/sneller)."
)
@click.option("--no-progress", is_flag=True, help="Geen voortgangsbalk tonen.")
@click.pass_context
def ingest(ctx: click.Context, folder: str, no_musicbrainz: bool, no_progress: bool) -> None:
    """Scan FOLDER recursief en analyseer/verrijk nieuwe tracks."""
    config = ctx.obj["config"]
    conn = _connect(config)
    try:
        mb_client = None if no_musicbrainz else MusicBrainzClient(config, db_conn=conn)
        stats = ingest_folder(folder, conn, config, mb_client=mb_client, progress=not no_progress)
    finally:
        conn.close()

    click.echo(
        f"Klaar: {stats['scanned']} gescand, {stats['ingested']} nieuw toegevoegd, "
        f"{stats['skipped_existing']} overgeslagen (bestond al), {stats['failed']} mislukt."
    )
    if stats["failed"]:
        click.echo(f"Zie logbestand ({config['logging']['file']}) voor details over mislukte bestanden.")


@main.command()
@click.argument("track_id", type=int)
@click.option(
    "--direction",
    type=click.Choice(["build", "hold", "ease", "surprise"]),
    default="hold",
    show_default=True,
)
@click.option("--top", "top_n", type=int, default=10, show_default=True)
@click.option(
    "--no-refine", is_flag=True,
    help="Toon alleen de snelle score (bpm/key/genre), sla de energie/mood-verfijning over.",
)
@click.option(
    "--source",
    type=click.Choice(["engine", "llm"]),
    default="engine",
    show_default=True,
    help="'engine' = lokale score (bpm/key/energy/mood). 'llm' = AI-suggestie "
         "via de geconfigureerde provider (config: llm_suggest), met dezelfde "
         "vaste vraag ongeacht welke API je koppelt.",
)
@click.pass_context
def suggest(
    ctx: click.Context, track_id: int, direction: str, top_n: int, no_refine: bool, source: str
) -> None:
    """Beveel de volgende TOP tracks aan na TRACK_ID in de gekozen richting.

    Standaard (--source engine): toont eerst een snelle score (bpm/key/
    genre — altijd beschikbaar, ook voor tracks die nog niet door de engine
    geanalyseerd zijn), en ververst die daarna met echte energie/mood-data
    voor de meest kansrijke kandidaten (zie recommend/refine.py).

    Met --source llm: vraagt een AI-suggestie op basis van dezelfde lokale
    shortlist, via de in config.yaml gekozen provider (Claude, of je eigen
    OpenAI-compatibele API) — zie recommend/llm_suggest.py.
    """
    config = ctx.obj["config"]
    conn = _connect(config)
    try:
        current = db_module.get_track(conn, track_id)
        if current is None:
            click.echo(f"Track {track_id} niet gevonden.", err=True)
            sys.exit(1)

        click.echo(
            f"Huidige track: [{current['id']}] {current['artist'] or '?'} - {current['title'] or '?'} "
            f"({_fmt(current.get('bpm'))} BPM, {current.get('camelot') or '?'})"
        )
        click.echo(f"Richting: {direction}")

        if source == "llm":
            _run_llm_suggest(conn, track_id, direction, config, top_n)
            return

        fast_results = suggest_next(conn, track_id, direction, config, top_n=top_n)
        _print_suggestions(fast_results, "Snel (bpm/key/genre)")

        if no_refine:
            return

        shortlist_n = config.get("suggest", {}).get("shortlist_size", 40)
        shortlist = suggest_next(conn, track_id, direction, config, top_n=shortlist_n)
        shortlist_ids = [r["track"]["id"] for r in shortlist]

        click.echo("\nVerfijnen met energie/mood-analyse...")
        refined_count = refine_candidates(conn, config, shortlist_ids)
        if refined_count == 0:
            click.echo("(Niets te verfijnen — kandidaten waren al volledig geanalyseerd of niet lokaal beschikbaar.)")
            return

        refreshed_results = suggest_next(conn, track_id, direction, config, top_n=top_n)
        _print_suggestions(refreshed_results, f"Ververst ({refined_count} tracks aangevuld)")
    finally:
        conn.close()


def _run_llm_suggest(conn, track_id: int, direction: str, config: dict[str, Any], top_n: int) -> None:
    try:
        outcome = suggest_next_llm(conn, track_id, direction, config, top_n=top_n)
    except LLMProviderError as exc:
        click.echo(f"AI-suggestie mislukt: {exc}", err=True)
        sys.exit(1)

    search_note = "met websearch" if outcome["used_search"] else "zonder websearch (model-kennis alleen)"
    click.echo(f"\n-- AI-suggestie ({outcome['provider']}/{outcome['model']}, {search_note}) --")

    if not outcome["results"]:
        click.echo("Geen (bruikbare) AI-suggesties ontvangen.")
        return

    for rank, item in enumerate(outcome["results"], start=1):
        t = item["track"]
        bron = "✓ bron" if item["gegrond_op_bron"] else "  géén bron"
        click.echo(f"{rank:2}. {_format_track_line(t)}  [{bron}]")
        click.echo(f"      {item['reden']}")


@main.command()
@click.option("--artist", default=None, help="Substring-match op artiest.")
@click.option("--genre", default=None, help="Substring-match op genre.")
@click.option("--bpm-range", "bpm_range", default=None, help="bv. 120-128")
@click.pass_context
def search(ctx: click.Context, artist: str | None, genre: str | None, bpm_range: str | None) -> None:
    """Zoek tracks op artiest/genre/bpm-bereik."""
    config = ctx.obj["config"]

    bpm_min = bpm_max = None
    if bpm_range:
        try:
            lo, hi = bpm_range.split("-")
            bpm_min, bpm_max = float(lo), float(hi)
        except ValueError:
            click.echo(f"Ongeldig --bpm-range formaat: {bpm_range!r} (verwacht bv. 120-128)", err=True)
            sys.exit(1)

    conn = _connect(config)
    try:
        results = db_module.search_tracks(
            conn, artist=artist, genre=genre, bpm_min=bpm_min, bpm_max=bpm_max
        )
    finally:
        conn.close()

    if not results:
        click.echo("Geen tracks gevonden.")
        return

    for t in results:
        click.echo(_format_track_line(t))
    click.echo(f"\n{len(results)} track(s) gevonden.")


@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Toon bibliotheekstatistieken (aantal tracks, dekking per veld)."""
    config = ctx.obj["config"]
    conn = _connect(config)
    try:
        s = db_module.get_stats(conn)
    finally:
        conn.close()

    total = s["total_tracks"]
    click.echo(f"Totaal aantal tracks: {total}")
    if total > 0:
        for label, key in [
            ("Met BPM", "with_bpm"),
            ("Met key/camelot", "with_key"),
            ("Met mood-data", "with_mood"),
            ("Met genre", "with_genre"),
            ("Met MusicBrainz-ID", "with_mbid"),
        ]:
            pct = 100 * s[key] / total
            click.echo(f"  {label:<20}: {s[key]:>5} ({pct:5.1f}%)")
    click.echo(f"Gelogde overgangen (transitions): {s['total_transitions']}")


@main.command(name="import-rekordbox")
@click.argument("xml_path", type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def import_rekordbox_cmd(ctx: click.Context, xml_path: str) -> None:
    """Importeer een Rekordbox 'Export Collection in xml format'-bestand.

    Snel (geen audio-decode) — vult filepath/title/artist/genre/bpm/
    key/camelot rechtstreeks uit Rekordbox' eigen analyse. energy/mood_*
    blijven NULL tot een latere `analyze`/`suggest`-aanroep. Idempotent:
    opnieuw importeren na wijzigingen in Rekordbox overschrijft, dupliceert niet.
    """
    config = ctx.obj["config"]
    path_mapping = config.get("rekordbox", {}).get("path_mapping", [])
    conn = _connect(config)

    imported = 0
    streaming = 0
    try:
        # commit=False: bij duizenden tracks is los committen per rij de
        # dominante kostenpost (fsync per rij) — één commit aan het eind
        # maakt dit ~10x sneller zonder iets aan het gedrag te veranderen.
        for track in parse_rekordbox_xml(xml_path, path_mapping=path_mapping):
            if track["is_streaming"]:
                streaming += 1
            db_module.upsert_track(
                conn,
                {k: v for k, v in track.items() if k not in ("rekordbox_track_id", "is_streaming")},
                commit=False,
            )
            imported += 1
        conn.commit()
    finally:
        conn.close()

    click.echo(f"Rekordbox-import klaar: {imported} tracks verwerkt.")
    if streaming:
        click.echo(
            f"  waarvan {streaming} streaming-tracks (bv. Tidal) — geen lokaal "
            "bestand, kunnen niet door de engine geanalyseerd worden."
        )
    if not path_mapping:
        click.echo(
            "  Let op: 'rekordbox.path_mapping' is nog leeg in config.yaml — "
            "'analyze'/'suggest' kunnen lokale bestanden pas vinden als je dat invult."
        )


@main.command()
@click.argument("track_id", type=int)
@click.option(
    "--no-musicbrainz", is_flag=True, help="Sla MusicBrainz-verrijking over (offline/sneller)."
)
@click.pass_context
def analyze(ctx: click.Context, track_id: int, no_musicbrainz: bool) -> None:
    """Analyseer (of hernieuw) 1 track: energie + mood/genre.

    Voor een track die al bpm/key/camelot heeft (bv. uit Rekordbox) worden
    die waarden nooit overschreven — alleen energy/mood/loudness/
    danceability komen (opnieuw) uit deze analyse.
    """
    config = ctx.obj["config"]
    conn = _connect(config)
    try:
        mb_client = None if no_musicbrainz else MusicBrainzClient(config, db_conn=conn)
        try:
            track = analyze_track(track_id, conn, config, mb_client=mb_client)
        except ValueError as exc:
            click.echo(f"Fout: {exc}", err=True)
            sys.exit(1)
    finally:
        conn.close()

    click.echo(
        f"Track {track_id} geanalyseerd: energy={_fmt(track.get('energy'), 2)}  "
        f"mood_party={_fmt(track.get('mood_party'), 2)}  genre={track.get('genre') or '?'}"
    )


@main.command(name="normalize-energy")
@click.pass_context
def normalize_energy_cmd(ctx: click.Context) -> None:
    """Forceer herberekening van de energy-kolom (min-max over de hele bibliotheek).

    Gebeurt normaal automatisch en periodiek (zie config: scoring.
    energy_renormalize_threshold) — dit commando is voor als je het direct
    wilt afdwingen, bv. na een grote batch analyze-aanroepen.
    """
    config = ctx.obj["config"]
    conn = _connect(config)
    try:
        n = db_module.normalize_energy(conn)
    finally:
        conn.close()
    click.echo(f"{n} tracks genormaliseerd.")


@main.command(name="set-api-key")
@click.argument("api_key")
@click.option(
    "--var-name", default="ANTHROPIC_API_KEY", show_default=True,
    help="Naam van de environment-variabele om te zetten.",
)
def set_api_key_cmd(api_key: str, var_name: str) -> None:
    """Sla een API-key handmatig op in .env (nooit in git, zie .gitignore).

    Infrastructuur voor de nog niet gebouwde optionele AI-suggestiebron
    (config: claude_api) — dit commando zet alleen de key klaar.
    """
    env_path = Path(".env")
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    lines = [line for line in lines if not line.startswith(f"{var_name}=")]
    lines.append(f"{var_name}={api_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    click.echo(f"{var_name} opgeslagen in .env (niet in git).")


if __name__ == "__main__":
    main()
