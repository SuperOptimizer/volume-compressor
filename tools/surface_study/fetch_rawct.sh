#!/bin/bash
# raw (uncompressed zarr v2) level-0 chunks of the PHercParis4 2.4um masked volume from the open-data bucket
BASE=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0
name=$1; z0=$2; y0=$3; x0=$4; n=$5
mkdir -p $name
for ((z=z0; z<z0+n; z++)); do for ((y=y0; y<y0+n; y++)); do for ((x=x0; x<x0+n; x++)); do echo "$z $y $x"; done; done; done | \
  xargs -P 8 -n 3 sh -c "[ -s $name/\$0_\$1_\$2.u8 ] || curl -s -f -o $name/\$0_\$1_\$2.u8 $BASE/\$0/\$1/\$2 || echo missing \$0 \$1 \$2"
ls $name | wc -l
