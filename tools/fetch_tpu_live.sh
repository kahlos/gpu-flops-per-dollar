#!/usr/bin/env bash
# Dump live TPU gpu-specs pages via the headed Chromium session (playwright-cli).
# Usage: ./tools/fetch_tpu_live.sh [outdir]
# Requires: playwright-cli -s=tpuheaded open --headed about:blank  (once)
# NOTE: runs in a scratch dir — playwright-cli writes session files to $CWD,
# which must stay out of the repo. Only $OUT is kept.
set -u
cd "${TMPDIR:-/tmp}"
OUT="${1:-/tmp/tpu_live}"
mkdir -p "$OUT"

declare -A URLS=(
  [rtx-3090-ti]="https://www.techpowerup.com/gpu-specs/geforce-rtx-3090-ti.c3829"
  [rtx-3090]="https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622"
  [rtx-3080-ti]="https://www.techpowerup.com/gpu-specs/geforce-rtx-3080-ti.c3735"
  [rtx-3080-12gb]="https://www.techpowerup.com/gpu-specs/geforce-rtx-3080-12-gb.c3834"
  [rtx-3080-10gb]="https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621"
  [rtx-3070-ti]="https://www.techpowerup.com/gpu-specs/geforce-rtx-3070-ti.c3675"
  [rtx-3070]="https://www.techpowerup.com/gpu-specs/geforce-rtx-3070.c3674"
  [rtx-3060-ti]="https://www.techpowerup.com/gpu-specs/geforce-rtx-3060-ti.c3681"
  [rtx-3060-12gb]="https://www.techpowerup.com/gpu-specs/geforce-rtx-3060-12-gb.c3682"
  [rtx-3050-8gb]="https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-8-gb.c3858"
  [rtx-2080-ti]="https://www.techpowerup.com/gpu-specs/geforce-rtx-2080-ti.c3305"
  [rtx-2080-super]="https://www.techpowerup.com/gpu-specs/geforce-rtx-2080-super.c3439"
  [rtx-2080]="https://www.techpowerup.com/gpu-specs/geforce-rtx-2080.c3224"
  [rtx-2070-super]="https://www.techpowerup.com/gpu-specs/geforce-rtx-2070-super.c3440"
  [rtx-2070]="https://www.techpowerup.com/gpu-specs/geforce-rtx-2070.c3252"
  [rtx-2060-super]="https://www.techpowerup.com/gpu-specs/geforce-rtx-2060-super.c3441"
  [rtx-2060-6gb]="https://www.techpowerup.com/gpu-specs/geforce-rtx-2060.c3310"
  # No reference pages exist (AIB-only launches, verified in live DB table):
  # rtx-3060-8gb, rtx-2060-12gb, rtx-3050-6gb (c4188 exists but has no SHS slug)
)

for id in "${!URLS[@]}"; do
  echo "== $id =="
  playwright-cli -s=tpuheaded goto "${URLS[$id]}" 2>&1 | tail -n 2
  sleep 7  # let PoW/challenge settle; be polite to TPU
  title="$(playwright-cli -s=tpuheaded eval "() => document.title" --raw 2>/dev/null | tail -n 1)"
  echo "   title: $title"
  playwright-cli -s=tpuheaded eval "() => document.documentElement.outerHTML" --raw > "$OUT/$id.html" 2>"$OUT/$id.err"
  wc -c "$OUT/$id.html"
done
echo "ALL DONE -> $OUT"
