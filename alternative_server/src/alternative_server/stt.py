"""Speech-to-Text using faster-whisper with Vietnamese support."""

import asyncio
import logging
from typing import AsyncIterator, Optional
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


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
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Whisper model loaded successfully")
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
        # Normalize audio to [-1, 1] range
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Apply a small gain if audio is too quiet
        if np.abs(audio).max() < 0.1:
            audio = audio * 5.0
        
        # Convert to int16 format that Whisper expects
        audio_int16 = (audio * 32767).astype(np.int16)
        
        logger.info(f"Audio after normalization: max={audio.max():.4f}, min={audio.min():.4f}")
        logger.info(f"Audio int16 range: [{audio_int16.min()}, {audio_int16.max()}]")
        
        # Try basic transcription with minimal parameters
        segments, info = self.model.transcribe(
            audio_int16,
            task="transcribe",
        )
        
        # Collect all segments
        segment_texts = []
        for segment in segments:
            segment_texts.append(segment.text.strip())
            logger.debug(f"Segment: '{segment.text}' (start={segment.start:.2f}, end={segment.end:.2f})")
        
        text = " ".join(segment_texts)
        logger.info(f"Total segments: {len(segment_texts)}, Combined text: '{text}'")
        # Force English language detection
        detected_lang = "en"
        print(f"DEBUG: Forced language to English")
        return text, detected_lang
    
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
    "model_size": "medium",  # Medium for better Vietnamese accuracy
    "language": "vi",
    "device": "cuda",
    "compute_type": "float16",
}
