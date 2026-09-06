/*
 * mic-worklet.js - Real-time stateful microphone collector & resampler.
 *
 * Mobile Safari on iOS 18 (A18 Pro / iPhone 16 Pro Max) locks hardware capture
 * to native rates (typically 48,000 Hz or 44,100 Hz). This AudioWorkletProcessor
 * downsamples incoming audio to exactly 16,000 Hz in real time using a 4th-order
 * Butterworth anti-aliasing low-pass filter (cutoff 7.2 kHz) and fractional
 * stateful interpolation across 128-frame render quanta.
 *
 * Resampled 16 kHz Float32 samples are accumulated into chunks of 4,000 samples
 * (0.25 s) and transferred to the main thread via postMessage.
 */

const TARGET_SAMPLE_RATE = 16000;
const CHUNK = 4000; // 0.25 s at 16 kHz
const FILTER_CUTOFF = 7200; // Anti-aliasing cutoff (Hz) for speech Mel-filterbank

class MicCollector extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = TARGET_SAMPLE_RATE;
    this.sourceRate =
      (options && options.processorOptions && options.processorOptions.sourceRate) ||
      (typeof sampleRate !== "undefined" ? sampleRate : TARGET_SAMPLE_RATE);
    this.ratio = this.sourceRate / this.targetRate;

    this.buf = new Float32Array(CHUNK);
    this.n = 0;
    this.phase = 0.0;
    this.prevSample = 0.0;
    this.filteredBuf = new Float32Array(1024);

    // Design 4th-order Butterworth low-pass filter (cascaded Direct Form II Transposed biquads)
    if (this.sourceRate > this.targetRate) {
      const fc = Math.min(FILTER_CUTOFF, (this.targetRate / 2) * 0.95);
      const w0 = 2 * Math.PI * (fc / this.sourceRate);
      const cosw0 = Math.cos(w0);
      const sinw0 = Math.sin(w0);
      // Butterworth 4th-order Q values:
      // Q1 = 1 / (2 * cos(3pi/8)) = 1.3065629648763765
      // Q2 = 1 / (2 * cos(pi/8))  = 0.541196100146197
      const Qs = [1.3065629648763765, 0.541196100146197];
      this.biquads = Qs.map((Q) => {
        const alpha = sinw0 / (2 * Q);
        const b0 = (1 - cosw0) / 2;
        const b1 = 1 - cosw0;
        const b2 = (1 - cosw0) / 2;
        const a0 = 1 + alpha;
        const a1 = -2 * cosw0;
        const a2 = 1 - alpha;
        return {
          b0: b0 / a0,
          b1: b1 / a0,
          b2: b2 / a0,
          a1: a1 / a0,
          a2: a2 / a0,
          s1: 0,
          s2: 0,
        };
      });
    } else {
      this.biquads = null;
    }
  }

  filterSample(x) {
    let y = x;
    const bqs = this.biquads;
    for (let i = 0; i < bqs.length; i++) {
      const bq = bqs[i];
      const out = bq.b0 * y + bq.s1;
      bq.s1 = bq.b1 * y - bq.a1 * out + bq.s2;
      bq.s2 = bq.b2 * y - bq.a2 * out;
      y = out;
    }
    return y;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || ch.length === 0) return true;

    // Passthrough optimization when audio hardware is already at 16 kHz
    if (this.sourceRate === this.targetRate) {
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

    // 1. Anti-aliasing filtering
    const L = ch.length;
    if (this.filteredBuf.length < L) {
      this.filteredBuf = new Float32Array(L);
    }
    const filtered = this.filteredBuf;
    if (this.biquads) {
      for (let i = 0; i < L; i++) {
        filtered[i] = this.filterSample(ch[i]);
      }
    } else {
      filtered.set(ch);
    }

    // 2. Continuous fractional interpolation across quantum boundaries
    while (this.phase < L - 1) {
      let s0, s1, alpha;
      if (this.phase < 0) {
        // Between last sample of previous quantum and index 0 of current quantum
        s0 = this.prevSample;
        s1 = filtered[0];
        alpha = this.phase + 1.0;
      } else {
        const idx = Math.floor(this.phase);
        s0 = filtered[idx];
        s1 = filtered[idx + 1];
        alpha = this.phase - idx;
      }

      const out = s0 + alpha * (s1 - s0);
      this.buf[this.n++] = out;

      if (this.n === CHUNK) {
        const chunk = this.buf.slice(0);
        this.port.postMessage(chunk, [chunk.buffer]);
        this.n = 0;
      }

      this.phase += this.ratio;
    }

    // Preserve boundary sample and advance phase origin to next quantum
    this.prevSample = filtered[L - 1];
    this.phase -= L;
    return true;
  }
}

registerProcessor("mic-collector", MicCollector);
