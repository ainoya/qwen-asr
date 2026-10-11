import test from "node:test";
import assert from "node:assert/strict";
import { resolveModelConfig } from "./demo/model-config.mjs";

test("local selection resolves both variants independently", () => {
  for (const id of ["0.6b", "1.7b"]) assert.equal(resolveModelConfig({}, id).modelBase, "../../qwen3-asr-" + id + "-q8");
});
test("per-model URLs and sizes override a legacy fixed deployment", () => {
  const config = {modelBase:"wrong", modelSize:999, models:{"0.6b":{modelBase:"https://example.com/small/",modelSize:840},"1.7b":{modelBase:"https://example.com/large",modelSize:2180}}};
  assert.deepEqual(resolveModelConfig(config,"0.6b"),{id:"0.6b",modelBase:"https://example.com/small",modelSize:840});
  assert.equal(resolveModelConfig(config,"1.7b").modelSize,2180);
});
test("legacy hosted URL never silently substitutes 1.7B for 0.6B", () => {
  const config = {modelBase:"https://example.com/large",modelSize:2180};
  assert.equal(resolveModelConfig(config,"1.7b").modelSize,2180);
  assert.throws(()=>resolveModelConfig(config,"0.6b"),/not configured/);
});
test("missing model and unknown selection fail explicitly", () => {
  assert.throws(()=>resolveModelConfig({models:{"1.7b":{modelBase:"large"}}},"0.6b"),/not configured/);
  assert.throws(()=>resolveModelConfig({},"other"),/Unknown model/);
});
test("missing per-model length does not reuse another model's size", () => {
  assert.equal(resolveModelConfig({modelSize:2180,models:{"0.6b":{modelBase:"small"}}},"0.6b").modelSize,0);
});
