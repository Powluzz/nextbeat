"""CLI entrypoint voor dj-engine.

Commando's:
    dj-engine ingest <folder>
    dj-engine suggest <track_id> --direction build|hold|ease|surprise --top 10
    dj-engine search --artist "..." --genre "..." --bpm-range 120-128
    dj-engine stats
"""
from __future__ import annotations

import sqlite3
import sys
from typing import Any

import click

from dj_engine import db as db_module
from dj_engine.config import load_config
from dj_engine.enrichment.musicbrainz_client import MusicBrainzClient
from dj_engine.ingest.pipeline import ingest_folder
from dj_engine.logging_config import configure_logging
from dj_engine.recommend.engine import suggest_next


def _connect(config: dict[str, Any]) -> sqlite3.Connection:
    return db_module.connect(config["database"]["path"])


def _format_track_line(t: dict[str, Any]) -> str:
    return (
        f"[{t['id']:>4}] {t['artist'] or '?'} - {t['title'] or '?'} "
        f"({_fmt(t.get('bpm'))} BPM, {t.get('camelot') or '?'}, {t.get('genre') or '?'})"
    )


def _fmt(value: float | None, decimals: int = 0) -> str:
    return "?" if value is None else f"{value:.{decimals}f}"


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
@click.pass_context
def suggest(ctx: click.Context, track_id: int, direction: str, top_n: int) -> None:
    """Beveel de volgende TOP tracks aan na TRACK_ID in de gekozen richting."""
    config = ctx.obj["config"]
    conn = _connect(config)
    try:
        current = db_module.get_track(conn, track_id)
        if current is None:
            click.echo(f"Track {track_id} niet gevonden.", err=True)
            sys.exit(1)
        results = suggest_next(conn, track_id, direction, config, top_n=top_n)
    finally:
        conn.close()

    click.echo(
        f"Huidige track: [{current['id']}] {current['artist'] or '?'} - {current['title'] or '?'} "
        f"({_fmt(current.get('bpm'))} BPM, {current.get('camelot') or '?'})"
    )
    click.echo(f"Richting: {direction}\n")

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


if __name__ == "__main__":
    main()
