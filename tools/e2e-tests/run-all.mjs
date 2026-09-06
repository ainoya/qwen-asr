#!/usr/bin/env node
/**
 * tools/e2e-tests/run-all.mjs
 * Unified command-line runner for Qwen-ASR Mobile Safari E2E Test Suite (Tiers 1 - 4).
 *
 * Usage:
 *   node tools/e2e-tests/run-all.mjs             # Run all 4 tiers (126 tests)
 *   node tools/e2e-tests/run-all.mjs --tier 1    # Run Tier 1 (Feature Coverage)
 *   node tools/e2e-tests/run-all.mjs --tier 2    # Run Tier 2 (Boundary & Corner Cases)
 *   node tools/e2e-tests/run-all.mjs --tier 3    # Run Tier 3 (Cross-Feature Combinations)
 *   node tools/e2e-tests/run-all.mjs --tier 4    # Run Tier 4 (Real-World Scenarios)
 */

import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

const args = process.argv.slice(2);
let targetTier = null;
const tierIndex = args.indexOf("--tier");
if (tierIndex !== -1 && args[tierIndex + 1]) {
  targetTier = parseInt(args[tierIndex + 1], 10);
}

const TIER_FILES = {
  1: "tools/e2e-tests/tier1-feature-coverage.test.mjs",
  2: "tools/e2e-tests/tier2-boundary-corner.test.mjs",
  3: "tools/e2e-tests/tier3-cross-feature.test.mjs",
  4: "tools/e2e-tests/tier4-application-scenarios.test.mjs"
};

const filesToRun = targetTier
  ? [TIER_FILES[targetTier]].filter(Boolean)
  : Object.values(TIER_FILES);

if (!filesToRun.length) {
  console.error(`Unknown tier ${targetTier}. Valid tiers are 1, 2, 3, 4.`);
  process.exit(1);
}

console.log("==================================================================");
console.log("  Qwen-ASR Mobile Safari E2E Test Suite (Dual Track)");
console.log(`  Scope: ${targetTier ? `Tier ${targetTier}` : "All 4 Tiers (Tiers 1-4)"}`);
console.log("==================================================================\n");

for (const file of filesToRun) {
  const fileUrl = pathToFileURL(resolve(process.cwd(), file)).href;
  await import(fileUrl);
}
