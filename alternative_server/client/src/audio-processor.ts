// @ts-nocheck
function asMs(samples) {
  return (samples * 1000 / sampleRate).toFixed(1);
}

function asSamples(mili) {
  return Math.round(mili * sampleRate / 1000);
}

class MoshiProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    console.log("Moshi processor lives", currentFrame, sampleRate);
    console.log(currentTime);

    // Buffer length definitions - massively increased for entire TTS responses
    let frameSize = asSamples(80);
    // initialBufferSamples: we wait to have at least that many samples before starting to play
    this.initialBufferSamples = 100 * frameSize;  // 100 frames (8 seconds at 48kHz)
    // once we have enough samples, we further wait that long before starting to play.
    // This allows to have buffer lengths that are not a multiple of frameSize.
    this.partialBufferSamples = asSamples(500);  // 500ms
    // If the buffer length goes over that many, we will drop the oldest packets until
    // we reach back initialBufferSamples + partialBufferSamples.
    this.maxBufferSamples = asSamples(5000);  // 5 seconds - should never drop
    // increments
    this.partialBufferIncrement = asSamples(100);
    this.maxPartialWithIncrements = asSamples(1000);
    this.maxBufferSamplesIncrement = asSamples(100);
    this.maxMaxBufferWithIncrements = asSamples(5000);  // 5 seconds

    // State and metrics
    this.initState();

    this.port.onmessage = (event) => {
      if (event.data.type == "reset") {
        console.log("Reset audio processor state - clearing buffer!");
        this.initState();
        return;
      }
      let frame = event.data.frame;
      console.log(this.timestamp(), "Got frame", this.pidx++, asMs(this.currentSamples()), asMs(frame.length))
      this.frames.push(frame);
      
      // Don't start until we have at least 2 seconds buffered
      if (this.currentSamples() >= asSamples(2000) && !this.started) {
        console.log(this.timestamp(), "Starting playback with", asMs(this.currentSamples()), "buffered");
        this.start();
      }
      if (this.currentSamples() >= this.totalMaxBufferSamples()) {
        console.log(this.timestamp(), "Buffer full but NOT dropping packets - keeping all audio")
        // Don't drop packets - let buffer grow to prevent truncation
        // This ensures complete audio playback even for long responses
        return;
      }
      let delay = this.currentSamples() / sampleRate;
      this.port.postMessage({
        totalAudioPlayed: this.totalAudioPlayed,
        actualAudioPlayed: this.actualAudioPlayed,
        delay: event.data.micDuration - this.timeInStream,
        minDelay: this.minDelay,
        maxDelay: this.maxDelay,
      });
    };
  }

  initState() {
    this.frames = new Array();
    this.offsetInFirstBuffer = 0;
    this.firstOut = false;
    this.remainingPartialBufferSamples = 0;
    this.timeInStream = 0.;
    this.resetStart();

    // Metrics
    this.totalAudioPlayed = 0.;
    this.actualAudioPlayed = 0.;
    this.maxDelay = 0.;
    this.minDelay = 2000.;
    // Debug
    this.pidx = 0;

    // For now let's reset the buffer params.
    this.partialBufferSamples = asSamples(10);
    this.maxBufferSamples = asSamples(10);
  }

  totalMaxBufferSamples() {
    return this.maxBufferSamples + this.partialBufferSamples + this.initialBufferSamples;
  }

  timestamp() {
    return Date.now() % 1000;
  }

  currentSamples() {
    let samples = 0;
    for (let k = 0; k < this.frames.length; k++) {
      samples += this.frames[k].length
    }
    samples -= this.offsetInFirstBuffer;
    return samples;
  }

  resetStart() {
    this.started = false;
  }

  start() {
    this.started = true;
    this.remainingPartialBufferSamples = this.partialBufferSamples;
    this.firstOut = true;
  }

  canPlay() {
    return this.started && this.frames.length > 0 && this.remainingPartialBufferSamples <= 0;
  }

  process(inputs, outputs, parameters) {
    let delay = this.currentSamples() / sampleRate;
    if (this.canPlay()) {
      this.maxDelay = Math.max(this.maxDelay, delay);
      this.minDelay = Math.min(this.minDelay, delay);
    }
    const output = outputs[0][0];
    if (!this.canPlay()) {
      if (this.actualAudioPlayed > 0) {
        this.totalAudioPlayed += output.length / sampleRate;
      }
      this.remainingPartialBufferSamples -= output.length;
      return true;
    }
    if (this.firstOut) {
      console.log(this.timestamp(), "Audio resumed", asMs(this.currentSamples()), this.remainingPartialBufferSamples);
    }
    let first = this.frames[0];
    let out_idx = 0;
    while (out_idx < output.length && this.frames.length) {
      let first = this.frames[0];
      let to_copy = Math.min(first.length - this.offsetInFirstBuffer, output.length - out_idx);
      output.set(first.subarray(this.offsetInFirstBuffer, this.offsetInFirstBuffer + to_copy), out_idx);
      this.offsetInFirstBuffer += to_copy;
      out_idx += to_copy;
      if (this.offsetInFirstBuffer == first.length) {
        this.offsetInFirstBuffer = 0;
        this.frames.shift();
      }
    }
    if (this.firstOut) {
      this.firstOut = false;
      for (let i = 0; i < out_idx; i++) {
        output[i] *= i / out_idx;
      }
    }
    if (out_idx < output.length) {
      console.log(this.timestamp(), "Missed some audio", output.length - out_idx);
      this.partialBufferSamples += this.partialBufferIncrement;
      this.partialBufferSamples = Math.min(this.partialBufferSamples, this.maxPartialWithIncrements);
      console.log("Increased partial buffer to", asMs(this.partialBufferSamples));
      // We ran out of a buffer, let's revert to the started state to replenish it.
      this.resetStart();
      for (let i = 0; i < out_idx; i++) {
        output[i] *= (out_idx - i) / out_idx;
      }
    }
    this.totalAudioPlayed += output.length / sampleRate;
    this.actualAudioPlayed += out_idx / sampleRate;
    this.timeInStream += out_idx / sampleRate;
    return true;
  }
}
registerProcessor("moshi-processor", MoshiProcessor);
