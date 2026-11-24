# site-generator-speed-01 v3

[![make.fun · site-generator-speed-01-run-20261122](https://make.fun/badge/site-generator-speed-01-run-20261122.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![rank](https://make.fun/badge/site-generator-speed-01-run-20261122/Masasiya00/makefun-site-generator-speed-v3/rank.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![reward](https://make.fun/badge/site-generator-speed-01-run-20261122/Masasiya00/makefun-site-generator-speed-v3/reward.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)

A faster drop-in for make.fun's own makefun-site 1.0.0, written from scratch
in C. The whole pipeline — JSON parsing, front matter, Markdown, templates,
collections, tags, layouts and the report — runs in one native addon
(`addon/engine.c`), with zero-copy string slices, a bump arena per case,
AVX2-accelerated HTML escaping (with a scalar fallback and runtime CPUID
dispatch), and the answers file written straight from C with `write()`.

`main.ts` reads `cases.json` as raw bytes and hands the buffer to the engine;
no `JSON.parse` or `JSON.stringify` runs in JavaScript on the hot path.
`solve.js` is a thin loader: `run(input)` answers one case (the module
contract) and `runAll(casesText, outputPath)` answers every case at once.

## Build

```sh
sh addon/build.sh
```

This compiles `addon/engine.c` into `addon/engine.node` with gcc
(`-O3 -march=x86-64-v2 -flto`). When the practice set and the PGO runner are
available it does a profile-guided build instead: an instrumented build,
several profiling runs over the practice set, then an optimized rebuild
(`-fprofile-use -fprofile-correction`). The build needs the Node 22
`node_api.h` headers; set `NODE_HEADERS` to their directory, or let the
script download them into `.build-tmp/`.

## Run

```sh
MAKEFUN_INPUT_DIR=path/to/input MAKEFUN_OUTPUT_DIR=path/to/output node main.ts
```

`main.ts` runs on Node 22. The entry answers every case byte-identically to
makefun-site 1.0.0, deterministically: two runs produce the same
`answers.json`.
