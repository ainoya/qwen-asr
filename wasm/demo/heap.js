/* Fresh wasm heap views for the main thread.
 *
 * With pthreads, growing wasm memory replaces the backing SharedArrayBuffer,
 * and only the thread that grew it updates its views immediately. Module
 * HEAP* properties on the main thread stay pointed at the old buffer until
 * Emscripten's own code next touches the heap there - so a view constructed
 * from a stale buffer throws ("Invalid typed array length") if the pointer
 * now lies past the old size, and worse, writes through a stale view land in
 * the abandoned buffer and silently vanish. Call this before every direct
 * heap access. Costs one comparison when nothing changed. */
export function freshHeap(M) {
  const b = (M.wasmMemory || M.HEAPU8).buffer;  /* old builds: no-op */
  if (M.HEAPU8.buffer !== b) {
    M.HEAPU8 = new Uint8Array(b);
    M.HEAPF32 = new Float32Array(b);
  }
  return M;
}
