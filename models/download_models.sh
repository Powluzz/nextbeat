#!/usr/bin/env bash
#
# download_models.sh — haalt de optionele pretrained Essentia
# mood/genre-modellen op (Discogs-EffNet embeddings + downstream classifiers).
#
# BELANGRIJK:
# - Dit script downloadt NOOIT automatisch zonder expliciete bevestiging.
# - Samen zijn deze modellen tientallen tot honderden MB groot.
# - Controleer de URL's hieronder tegen https://essentia.upf.edu/models.html
#   vóór gebruik — modelversies/paden kunnen wijzigen.
# - Mood/genre-classificatie vereist bovendien het `essentia-tensorflow`
#   Python-pakket (niet hetzelfde als het basis `essentia`-pakket).
#
# Gebruik:
#   bash models/download_models.sh          # vraagt om bevestiging
#   bash models/download_models.sh --yes     # slaat de bevestigingsvraag over

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://essentia.upf.edu/models"

FILES=(
  "feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
  "classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb"
  "classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb"
  "classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb"
  "classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb"
  "classification-heads/mood_party/mood_party-discogs-effnet-1.pb"
  "classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb"
  "classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json"
)

if [[ "${1:-}" != "--yes" ]]; then
  echo "Dit script downloadt ${#FILES[@]} modelbestanden naar: $SCRIPT_DIR"
  echo "Totale omvang: mogelijk enkele honderden MB."
  echo
  echo "Bestanden:"
  for f in "${FILES[@]}"; do
    echo "  - $BASE_URL/$f"
  done
  echo
  read -r -p "Doorgaan met downloaden? [y/N] " confirm
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Geannuleerd."
    exit 1
  fi
fi

for f in "${FILES[@]}"; do
  fname="$(basename "$f")"
  dest="$SCRIPT_DIR/$fname"
  if [[ -f "$dest" ]]; then
    echo "Bestaat al, overslaan: $fname"
    continue
  fi
  echo "Downloaden: $fname"
  curl -fL --progress-bar -o "$dest" "$BASE_URL/$f"
done

echo
echo "Klaar. Controleer config.yaml -> models: of de bestandsnamen overeenkomen."
echo "Installeer daarnaast 'essentia-tensorflow' (i.p.v. 'essentia') om deze"
echo "modellen daadwerkelijk te kunnen gebruiken."
