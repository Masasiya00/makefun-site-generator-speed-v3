// site-generator-speed-01 v3: solve.js is a thin loader for the native
// engine in ./addon/engine.node, which holds the whole makefun-site
// pipeline in C. run(input) answers one case (the module contract);
// runAll(casesText, outputPath) answers every case at once and writes
// answers.json straight from C (the path main.ts takes).

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const addon = require('./addon/engine.node');

/** makefun-site 1.0.0's answer for one case's input, as an object. */
export function run(input) {
  return JSON.parse(addon.runOne(JSON.stringify(input)));
}

/** Answer every case in `casesText` (a Buffer of the raw cases.json bytes,
// or a string) and write the answers array to `outputPath`. Returns the
// number of bytes written. */
export function runAll(casesText, outputPath) {
  return addon.runAll(casesText, outputPath);
}
