/** Resolve one model atomically: its URL and size must belong to the same variant. */
export function resolveModelConfig(config = {}, selected = "1.7b") {
  if (!["0.6b", "1.7b"].includes(selected)) throw new Error("Unknown model: " + selected);
  const entry = config.models?.[selected];
  if (entry) {
    if (!entry.modelBase) throw new Error("No model URL configured for " + selected);
    return { id: selected, modelBase: entry.modelBase.replace(/\/$/, ""), modelSize: Number(entry.modelSize) || 0 };
  }
  // Legacy hosted deployments supplied a single 1.7B URL. Never apply it to 0.6B.
  if (config.modelBase && selected === (config.modelId || "1.7b")) {
    return { id: selected, modelBase: config.modelBase.replace(/\/$/, ""), modelSize: Number(config.modelSize) || 0 };
  }
  if (config.models || config.modelBase) throw new Error("Model " + selected + " is not configured on this deployment; no other model will be downloaded.");
  return { id: selected, modelBase: "../../qwen3-asr-" + selected + "-q8", modelSize: 0 };
}
