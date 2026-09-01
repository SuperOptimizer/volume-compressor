#!/usr/bin/env bash
# Fetch pinned 128^3 u8 chunks from the PHercParis4 masked volume (zarr v2, raw
# uncompressed chunks, one HTTPS object per chunk). No Python required.
#
#   tools/fetch_corpus.sh probe LEVEL CZ0 CZ1 CZSTEP CY0 CY1 CX0 CX1 [STEP]
#       print "LEVEL cz cy cx" for every present chunk in the grid (HEAD requests)
#   tools/fetch_corpus.sh fetch SET
#       read corpus/manifest_SET.tsv (id level cz cy cx sha256|-), download into
#       corpus/SET/ID.u8, verify or record sha256, write ID.json sidecars
set -euo pipefail
BASE="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
EXPECT=2097152
cd "$(dirname "$0")/.."

case "${1:-}" in
probe)
  L=$2; CZ0=$3; CZ1=$4; CZS=$5; CY0=$6; CY1=$7; CX0=$8; CX1=$9; ST=${10:-1}
  for ((cz = CZ0; cz <= CZ1; cz += CZS)); do
    for ((cy = CY0; cy <= CY1; cy += ST)); do
      for ((cx = CX0; cx <= CX1; cx += ST)); do
        echo "$L $cz $cy $cx"
      done
    done
  done | xargs -P 16 -n 4 sh -c \
    'code=$(curl -s -o /dev/null -w "%{http_code}" -I "'"$BASE"'/$0/$1/$2/$3"); [ "$code" = 200 ] && echo "$0 $1 $2 $3"' \
    | sort -n -k1 -k2 -k3 -k4
  ;;
fetch)
  SET=$2
  MAN="corpus/manifest_$SET.tsv"
  OUT="corpus/$SET"
  mkdir -p "$OUT"
  fetch_one() {
    id=$1; lvl=$2; cz=$3; cy=$4; cx=$5; sha=$6; cls=${7:-unknown}
    dst="$OUT/$id.u8"
    if [ -s "$dst" ]; then
      have=$(sha256sum "$dst" | cut -d' ' -f1)
    else
      tmp="$dst.tmp"
      curl -sf --retry 3 "$BASE/$lvl/$cz/$cy/$cx" -o "$tmp" || { echo "MISSING $id"; rm -f "$tmp"; return 0; }
      sz=$(stat -c %s "$tmp")
      [ "$sz" = "$EXPECT" ] || { echo "BADSIZE $id $sz"; rm -f "$tmp"; return 0; }
      have=$(sha256sum "$tmp" | cut -d' ' -f1)
      mv "$tmp" "$dst"
    fi
    if [ "$sha" != "-" ] && [ "$sha" != "$have" ]; then
      echo "SHA MISMATCH $id: manifest $sha got $have" >&2
      exit 1
    fi
    printf '{"id":"%s","level":%s,"chunk":[%s,%s,%s],"class":"%s","sha256":"%s","source":"%s/%s/%s/%s/%s"}\n' \
      "$id" "$lvl" "$cz" "$cy" "$cx" "$cls" "$have" "$BASE" "$lvl" "$cz" "$cy" "$cx" > "$OUT/$id.json"
    echo "OK $id $have"
  }
  export -f fetch_one
  export BASE EXPECT OUT
  grep -v '^#' "$MAN" | xargs -P 8 -L 1 bash -c 'fetch_one "$@"' _
  # corpus hash = sha256 over sorted per-file hashes
  (cd "$OUT" && sha256sum *.u8 | sort -k2 | sha256sum | cut -d' ' -f1) > "$OUT/manifest.sha256"
  echo "corpus hash: $(cat "$OUT/manifest.sha256")"
  ;;
*)
  sed -n '2,12p' "$0"
  exit 1
  ;;
esac
