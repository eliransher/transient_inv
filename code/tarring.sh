#!/usr/bin/env bash
set -euo pipefail

SRC="/scratch/eliransc/elad_trans/train_data/inv_level"
OUTDIR="/scratch/eliransc/elad_trans/train_data/backups"
TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${OUTDIR}/inv_level_${TS}.tar.gz"

# Optional: chunk size for split archive (e.g., 20G). Leave empty to skip split.
CHUNK_SIZE="${CHUNK_SIZE:-}"

mkdir -p "$OUTDIR"

echo "Creating archive: $ARCHIVE"
tar -C "$(dirname "$SRC")" -czf "$ARCHIVE" "$(basename "$SRC")"

echo "Creating checksum..."
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

if [[ -n "$CHUNK_SIZE" ]]; then
  echo "Splitting archive into chunks of $CHUNK_SIZE ..."
  split -b "$CHUNK_SIZE" -d -a 3 "$ARCHIVE" "${ARCHIVE}.part."
  echo "Chunks created: ${ARCHIVE}.part.000, ..."
  echo "Rebuild command:"
  echo "cat ${ARCHIVE}.part.* > ${ARCHIVE}"
fi

echo "Done."
echo "Archive: $ARCHIVE"
echo "Checksum: ${ARCHIVE}.sha256"
