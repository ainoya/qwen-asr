/*
 * tick.js - yielding that survives background-tab throttling.
 *
 * Chrome clamps setTimeout in a tab that is not in front to roughly once a
 * second. Every poll loop here waits on work happening off the main thread -
 * a wasm job on a pthread, or a GPU submission - so a loop written with
 * setTimeout runs at the clamp rate rather than the work's rate, and the
 * whole harness appears to hang. Measured in a hidden tab: 200 MessageChannel
 * yields took 1.0 ms, 20 setTimeout(0) yields took over 45 s.
 *
 * MessageChannel messages are not clamped, so they are what the poll loops
 * use. GPU and wasm compute themselves are only mildly affected by being in
 * the background (a storage-buffer read benchmark measured 92 GB/s hidden
 * against 122 GB/s in front), which is why fixing the yield is enough to make
 * the comparison harnesses usable without keeping the window focused.
 */

const chan = new MessageChannel();
const waiting = [];
chan.port1.onmessage = () => {
  const resolve = waiting.shift();
  if (resolve) resolve();
};

/* Yield to the event loop without going through a timer. */
export function tick() {
  return new Promise((resolve) => {
    waiting.push(resolve);
    chan.port2.postMessage(0);
  });
}

/* Poll until the predicate is true. */
export async function until(done) {
  while (!done()) await tick();
}

/* Wall-clock delay. In front this is an ordinary timer; hidden, where the
 * timer would be clamped, it spins on tick() instead - the callers that need
 * it are pacing a simulated live stream, and pacing slower than real time
 * would misrepresent what the engine does. */
export function sleep(ms) {
  if (!document.hidden) return new Promise((resolve) => setTimeout(resolve, ms));
  return (async () => {
    const end = performance.now() + ms;
    while (performance.now() < end) await tick();
  })();
}
