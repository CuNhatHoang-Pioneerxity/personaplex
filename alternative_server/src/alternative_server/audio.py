"""Audio processing utilities for Opus encoding/decoding."""

import asyncio
from typing import Optional
import numpy as np
from io import BytesIO


class OpusCodec:
    """Opus audio codec for real-time streaming."""
    
    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        frame_duration: float = 20,  # ms
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self._encoder = None
        self._decoder = None
    
    def init_encoder(self):
        """Initialize Opus encoder."""
        try:
            import opuslib
            self._encoder = opuslib.Encoder(
                self.sample_rate,
                self.channels,
                opuslib.APPLICATION_AUDIO,
            )
        except ImportError:
            raise RuntimeError("opuslib not installed. Install with: pip install opuslib")
    
    def init_decoder(self):
        """Initialize Opus decoder."""
        try:
            import opuslib
            self._decoder = opuslib.Decoder(self.sample_rate, self.channels)
        except ImportError:
            raise RuntimeError("opuslib not installed. Install with: pip install opuslib")
    
    def encode(self, pcm: np.ndarray) -> bytes:
        """
        Encode PCM audio to Opus.
        
        Args:
            pcm: Float32 PCM audio samples
            
        Returns:
            Opus-encoded bytes
        """
        if self._encoder is None:
            self.init_encoder()
        
        # Convert float32 to int16
        pcm_int16 = (pcm * 32767).astype(np.int16)
        
        # Encode
        opus_data = self._encoder.encode(pcm_int16.tobytes(), self.frame_size)
        return opus_data
    
    def decode(self, opus_data: bytes) -> np.ndarray:
        """
        Decode Opus to PCM audio.
        
        Args:
            opus_data: Opus-encoded bytes
            
        Returns:
            Float32 PCM audio samples
        """
        if self._decoder is None:
            self.init_decoder()
        
        # Decode
        pcm_int16 = self._decoder.decode(opus_data, self.frame_size)
        
        # Convert int16 to float32
        pcm = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32767
        return pcm


class AudioBuffer:
    """Circular buffer for audio accumulation."""
    
    def __init__(self, max_seconds: float = 30.0, sample_rate: int = 24000):
        self.max_samples = int(max_seconds * sample_rate)
        self.sample_rate = sample_rate
        self._buffer = np.array([], dtype=np.float32)
    
    def append(self, audio: np.ndarray):
        """Append audio to buffer, trimming if necessary."""
        self._buffer = np.concatenate([self._buffer, audio])
        
        # Trim to max size
        if len(self._buffer) > self.max_samples:
            self._buffer = self._buffer[-self.max_samples:]
    
    def get_all(self) -> np.ndarray:
        """Get all buffered audio."""
        return self._buffer.copy()
    
    def get_last(self, seconds: float) -> np.ndarray:
        """Get last N seconds of audio."""
        samples = int(seconds * self.sample_rate)
        return self._buffer[-samples:].copy() if len(self._buffer) > samples else self._buffer.copy()
    
    def clear(self):
        """Clear the buffer."""
        self._buffer = np.array([], dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self._buffer)


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    
    try:
        import scipy.signal as signal
        
        # Calculate number of samples in output
        num_samples = int(len(audio) * target_sr / orig_sr)
        
        # Resample
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)
        
    except ImportError:
        # Fallback: simple linear interpolation
        ratio = target_sr / orig_sr
        indices = np.arange(0, len(audio), 1/ratio)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def convert_to_16khz(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Convert audio to 16kHz for Whisper."""
    return resample_audio(audio, orig_sr, 16000)


def convert_to_24khz(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Convert audio to 24kHz for PersonaPlex compatibility."""
    return resample_audio(audio, orig_sr, 24000)
