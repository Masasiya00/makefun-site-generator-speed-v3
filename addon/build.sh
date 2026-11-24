#!/bin/sh
# Build addon/engine.node from addon/engine.c.
#
# Plain build: gcc -O3 -march=x86-64-v2 -flto. When the practice set and the
# PGO runner are available (PGO_RUNNER, PRACTICE_CASES), a profile-guided
# build runs instead: instrumented compile, profiling runs over the practice
# set, then an optimized rebuild with -fprofile-use -fprofile-correction.
# Needs the Node 22 node_api.h headers: set NODE_HEADERS, or let this
# script download them into .build-tmp/ (network needed for that step only).

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

CC=${CC:-gcc}
NODE_VER=22.23.4
HDR_DIR=${NODE_HEADERS:-$ROOT/.build-tmp/node-v$NODE_VER/include/node}

if [ ! -f "$HDR_DIR/node_api.h" ]; then
  for d in "$HOME/workspace/.node22/node-v$NODE_VER/include/node" /tmp/node22-headers ./node-headers; do
    if [ -f "$d/node_api.h" ]; then HDR_DIR=$d; break; fi
  done
fi

if [ ! -f "$HDR_DIR/node_api.h" ]; then
  echo "node_api.h not found; downloading Node $NODE_VER headers..." >&2
  mkdir -p "$ROOT/.build-tmp"
  cd "$ROOT/.build-tmp"
  if [ ! -f "node-v$NODE_VER-headers.tar.xz" ]; then
    curl -sL -o "node-v$NODE_VER-headers.tar.xz" \
      "https://nodejs.org/dist/v$NODE_VER/node-v$NODE_VER-headers.tar.xz"
  fi
  tar xf "node-v$NODE_VER-headers.tar.xz"
  cd "$ROOT"
  HDR_DIR=$ROOT/.build-tmp/node-v$NODE_VER/include/node
fi

CFLAGS="-O3 -march=x86-64-v2 -flto -fPIC -shared -I$HDR_DIR"
OUT=addon/engine.node
PGO_RUNNER=${PGO_RUNNER:-/tmp/pgo-run.mjs}
PRACTICE=${PRACTICE_CASES:-$HOME/workspace/makefun/site-generator-speed-01-kit-dl/site-generator-speed-01-run-20261122-kit/practice/input/cases.json}

if [ -f "$PGO_RUNNER" ] && [ -f "$PRACTICE" ]; then
  PGO_DIR=$ROOT/.build-tmp/pgo
  mkdir -p "$PGO_DIR"
  echo "PGO build: instrumented compile"
  $CC $CFLAGS -fprofile-generate="$PGO_DIR" -o "$OUT" addon/engine.c
  echo "PGO build: profiling runs"
  node "$PGO_RUNNER" "$OUT" "$PRACTICE" 3
  echo "PGO build: optimized rebuild"
  $CC $CFLAGS -fprofile-use="$PGO_DIR" -fprofile-correction -o "$OUT" addon/engine.c
else
  echo "plain build (no PGO runner or practice set found)"
  $CC $CFLAGS -o "$OUT" addon/engine.c
fi

ls -l "$OUT"
