"""Text-to-Speech using Kokoro with Vietnamese support."""

import asyncio
from typing import Optional, AsyncIterator
import numpy as np

# Kokoro imports - will be available after installation
try:
    from kokoro import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    Kokoro = None


class KokoroTTS:
    """Kokoro-based text-to-speech."""
    
    # Available voices
    VOICES = {
        # English voices
        "af_bella": "American Female - Bella",
        "af_sarah": "American Female - Sarah", 
        "am_adam": "American Male - Adam",
        "bf_emma": "British Female - Emma",
        "bm_george": "British Male - George",
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
        self._model: Optional[Kokoro] = None
    
    @property
    def model(self) -> Kokoro:
        if not KOKORO_AVAILABLE:
            raise RuntimeError(
                "Kokoro is not installed. Install with: pip install kokoro"
            )
        if self._model is None:
            self._model = Kokoro(device=self.device)
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
        audio = self.model.create(
            text,
            voice_id=voice,
            sample_rate=self.sample_rate,
        )
        return audio
    
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
