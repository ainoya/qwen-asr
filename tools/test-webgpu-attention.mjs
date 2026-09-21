// Run with: node --experimental-default-type=module --test tools/test-webgpu-attention.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { WebGPUDecoder } from "../wasm/demo/webgpu-decoder.js";

function decoder(info, subgroups = true, limits = {}) {
  return new WebGPUDecoder({}, {
    adapter: { info },
    device: {
      features: new Set(subgroups ? ["subgroups"] : []),
      limits: { maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupStorageSize: 16384, ...limits },
    },
  });
}

test("Apple tile selection is independent of optional subgroup support", () => {
  for (const subgroups of [false, true]) {
    const gpu = decoder({ vendor: "apple" }, subgroups);
    assert.equal(gpu.useTiledPrefillScores, true);
    assert.equal(gpu.hasScoreSubgroups, false);
  }
});

test("other vendors, redacted adapters, and insufficient limits keep scalar scores", () => {
  for (const vendor of [undefined, "", "intel", "amd", "nvidia", "qualcomm"]) {
    assert.equal(decoder({ vendor }).useTiledPrefillScores, false);
  }
  for (const limits of [{ maxComputeInvocationsPerWorkgroup: 128 },
    { maxComputeWorkgroupStorageSize: 2048 }]) {
    assert.equal(decoder({ vendor: "apple" }, true, limits).useTiledPrefillScores, false);
  }
});

test("eight-key dispatch is selected only for a guaranteed 32-lane subgroup", () => {
  for (const width of [4, 8, 16, 32, 64, 128]) {
    const info = { vendor: "nvidia", subgroupMinSize: width, subgroupMaxSize: width };
    assert.equal(decoder(info).hasScoreSubgroups, width === 32);
    assert.equal(decoder(info, false).hasScoreSubgroups, false);
  }
  for (const info of [{}, { subgroupMinSize: 4, subgroupMaxSize: 128 },
    { subgroupMinSize: 32 }, { subgroupMaxSize: 32 }]) {
    assert.equal(decoder(info).hasScoreSubgroups, false);
  }
});
