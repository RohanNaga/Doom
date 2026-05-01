#!/bin/bash
# Download trained DiT-XL-2 weights from GitHub Release and re-assemble.
# Each .pt is split into 1.3 GB chunks (release asset size limit).
# Output: results/002-DiT-XL-2/checkpoints/{best.pt, 0090000.pt}
set -e

OUT_DIR=results/002-DiT-XL-2/checkpoints
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

REL=https://github.com/RohanNaga/Doom/releases/download/002-DiT-XL-2-best-90k

echo '[download_weights] fetching best.pt chunks...'
for p in part-00 part-01; do
    [ -f "best.pt.$p" ] || curl -sL -o "best.pt.$p" "$REL/best.pt.$p"
done
echo '[download_weights] reassembling best.pt...'
cat best.pt.part-* > best.pt
echo "  best.pt md5: $(md5sum best.pt | cut -d' ' -f1) (expected: ff54cab2093665f00d997f3e5fd1dd27)"
rm -f best.pt.part-*

echo '[download_weights] fetching 0090000.pt chunks...'
for p in part-00 part-01; do
    [ -f "0090000.pt.$p" ] || curl -sL -o "0090000.pt.$p" "$REL/0090000.pt.$p"
done
echo '[download_weights] reassembling 0090000.pt...'
cat 0090000.pt.part-* > 0090000.pt
echo "  0090000.pt md5: $(md5sum 0090000.pt | cut -d' ' -f1) (expected: 3f1be8922d4ffdfd8cd698fe9dede922)"
rm -f 0090000.pt.part-*

echo '[download_weights] DONE'
