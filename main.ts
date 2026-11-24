// site-generator-speed-01 v3: the driver. It reads every case from
// /input/cases.json as raw bytes (no JSON.parse in JavaScript at all) and
// hands the whole buffer to the native engine, which parses, builds and
// writes /output/answers.json itself. Outside a container
// MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { setFlagsFromString } from 'node:v8';

import { runAll } from './solve.js';

// V8's incremental marking off, as the house baseline runs with it.
setFlagsFromString('--no-incremental-marking');

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const casesText = readFileSync(join(INPUT_DIR, 'cases.json'));
runAll(casesText, join(OUTPUT_DIR, 'answers.json'));
