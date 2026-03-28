"""Speech-to-Text using faster-whisper with Vietnamese support."""

import asyncio
from typing import AsyncIterator, Optional
import numpy as np
from faster_whisper import WhisperModel


class WhisperSTT:
    """Whisper-based speech-to-text with multilingual support."""
    
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,  # Auto-detect if None
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: Optional[WhisperModel] = None
    
    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[str, str]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        
        text = " ".join(segment.text.strip() for segment in segments)
        return text, info.language
    
    async def transcribe_async(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[str, str]:
        """Async wrapper for transcription."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.transcribe,
            audio,
            sample_rate,
        )
    
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
    ) -> AsyncIterator[tuple[str, str]]:
        """
        Stream transcription for real-time processing.
        
        Note: This accumulates chunks and transcribes when silence is detected.
        """
        accumulated = np.array([], dtype=np.float32)
        
        async for chunk in audio_chunks:
            accumulated = np.concatenate([accumulated, chunk])
            
            # Check for silence/pause (simple energy-based)
            if len(accumulated) > sample_rate * 2:  # At least 2 seconds
                energy = np.sqrt(np.mean(accumulated ** 2))
                if energy < 0.01:  # Silence threshold
                    if len(accumulated) > sample_rate * 0.5:  # At least 0.5s of speech
                        text, lang = self.transcribe(accumulated, sample_rate)
                        yield text, lang
                    accumulated = np.array([], dtype=np.float32)
        
        # Process remaining audio
        if len(accumulated) > sample_rate * 0.5:
            text, lang = self.transcribe(accumulated, sample_rate)
            yield text, lang


# Vietnamese-optimized configuration
VIETNAMESE_WHISPER_CONFIG = {
    "model_size": "small",  # or "medium" for better accuracy
    "language": "vi",
    "device": "cuda",
    "compute_type": "float16",
}
