"""Text-to-Speech using Kokoro with Vietnamese support."""

import asyncio
from typing import Optional, AsyncIterator
import numpy as np
import torch

# Try new kokoro API (0.9.4+)
try:
    from kokoro import KPipeline, KModel
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False


class KokoroTTS:
    """Kokoro-based text-to-speech (new API 0.9.4+)."""
    
    # Available voices - kokoro 0.9.4 uses voice packs from HF
    VOICES = {
        # American Female voices
        "af_bella": "af_bella",
        "af_sarah": "af_sarah",
        "af_sky": "af_sky",
        "af_nicole": "af_nicole",
        # American Male voices
        "am_adam": "am_adam",
        "am_michael": "am_michael",
        # British voices
        "bf_emma": "bf_emma",
        "bf_isabella": "bf_isabella",
        "bm_george": "bm_george",
        "bm_lewis": "bm_lewis",
    }
    
    def __init__(
        self,
        voice: str = "af_bella",
        device: str = "cuda",
        sample_rate: int = 24000,  # Match PersonaPlex
    ):
        self.voice = voice
        self.device = device
        self.sample_rate = sample_rate
        self._pipeline: Optional[KPipeline] = None
        self._model: Optional[KModel] = None
        
        # Force model loading during initialization
        print(f"Initializing Kokoro TTS with voice '{voice}' on device '{device}'")
        _ = self.model  # This will trigger model loading
        print("Kokoro TTS initialized successfully")
    
    @property
    def pipeline(self) -> KPipeline:
        if not KOKORO_AVAILABLE:
            raise RuntimeError(
                "Kokoro is not installed. Install with: pip install kokoro"
            )
        if self._pipeline is None:
            # Use 'a' for American English, 'b' for British
            lang_code = 'b' if self.voice.startswith('b') else 'a'
            # KPipeline with model=False to avoid auto-loading, we'll pass model manually
            self._pipeline = KPipeline(lang_code=lang_code, model=False)
        return self._pipeline
    
    @property
    def model(self) -> KModel:
        if not KOKORO_AVAILABLE:
            raise RuntimeError(
                "Kokoro is not installed. Install with: pip install kokoro"
            )
        if self._model is None:
            # KModel is lightweight, loads actual weights on first inference
            # Note: KModel doesn't accept device parameter, it uses torch device management
            self._model = KModel()
            # Move to device after creation
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()
        return self._model
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            voice: Voice ID (uses default if None)
            
        Returns:
            Audio samples as numpy array (float32, mono, 24kHz)
        """
        voice = voice or self.voice
        
        # KPipeline returns generator of (graphemes, phonemes, audio)
        # Audio is None if model=False, but we'll pass model on call
        generator = self.pipeline(text, voice=voice, model=self.model, speed=1.0)
        
        # Collect all audio chunks
        audio_chunks = []
        for graphemes, phonemes, audio in generator:
            if audio is not None:
                # Audio is torch tensor, convert to numpy
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.cpu().numpy()
                else:
                    audio_np = audio
                audio_chunks.append(audio_np)
        
        if not audio_chunks:
            # Return silence if no audio generated
            return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
        
        # Concatenate all chunks
        combined = np.concatenate(audio_chunks)
        duration = len(combined) / self.sample_rate
        print(f"TTS generated {duration:.2f}s of audio from {len(text)} chars")
        return combined
    
    async def synthesize_async(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> np.ndarray:
        """Async wrapper for synthesis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.synthesize,
            text,
            voice,
        )
    
    async def synthesize_stream(
        self,
        text: str,
        chunk_size: int = 100,  # Characters per chunk
    ) -> AsyncIterator[np.ndarray]:
        """
        Stream synthesis for lower latency.
        
        Splits text into sentences/chunks and synthesizes each.
        """
        # Split into sentences for streaming
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            if sentence.strip():
                audio = await self.synthesize_async(sentence)
                yield audio
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences for streaming synthesis."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class FallbackTTS:
    """Fallback TTS using edge-tts (browser-based) if Kokoro unavailable."""
    
    def __init__(
        self,
        voice: str = "en-US-AvaNeural",
        sample_rate: int = 24000,
    ):
        self.voice = voice
        self.sample_rate = sample_rate
    
    async def synthesize(
        self,
        text: str,
    ) -> np.ndarray:
        """Synthesize using edge-tts (requires edge-tts package)."""
        try:
            import edge_tts
            import io
            from scipy.io import wavfile
            
            communicate = edge_tts.Communicate(text, self.voice)
            audio_bytes = io.BytesIO()
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes.write(chunk["data"])
            
            audio_bytes.seek(0)
            # Convert to numpy
            # Note: edge-tts outputs mp3, need conversion
            # This is a simplified version
            return audio_bytes.getvalue()
            
        except ImportError:
            raise RuntimeError(
                "edge-tts not installed. Install with: pip install edge-tts"
            )


# Vietnamese TTS configuration
VIETNAMESE_TTS_CONFIG = {
    # Kokoro doesn't have native Vietnamese voices yet
    # Use fallback with Vietnamese voice
    "fallback_voice": "vi-VN-HoaiMyNeural",
    "language": "vi",
}


def get_tts_for_language(language: str = "en") -> KokoroTTS:
    """Get appropriate TTS for language."""
    if language == "vi":
        # Vietnamese - use edge-tts fallback
        return FallbackTTS(voice="vi-VN-HoaiMyNeural")
    else:
        # English and others - use Kokoro
        return KokoroTTS()
