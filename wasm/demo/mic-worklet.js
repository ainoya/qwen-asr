/*
 * mic-worklet.js - collects microphone audio into ~0.25 s mono chunks.
 *
 * The AudioContext is created at 16 kHz, so no resampling is needed here; we
 * just batch the 128-frame render quanta into something worth a postMessage.
 */

const CHUNK = 4000; // 0.25 s at 16 kHz

class MicCollector extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(CHUNK);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;

    let i = 0;
    while (i < ch.length) {
      const take = Math.min(CHUNK - this.n, ch.length - i);
      this.buf.set(ch.subarray(i, i + take), this.n);
      this.n += take;
      i += take;
      if (this.n === CHUNK) {
        const out = this.buf.slice(0);
        this.port.postMessage(out, [out.buffer]);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-collector", MicCollector);
