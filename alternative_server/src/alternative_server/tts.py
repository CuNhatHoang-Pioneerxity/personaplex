"""Text-to-Speech using Piper and Kokoro with Vietnamese support."""

import asyncio
from typing import Optional, AsyncIterator
import numpy as np
import wave
import io
import os

# Piper TTS
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

# Kokoro TTS
try:
    from kokoro import KPipeline, KModel
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

try:
    import torch
except ImportError:
    torch = None


class PiperTTS:
    """Piper-based text-to-speech with Vietnamese support."""
    
    # Available voices - Piper has Vietnamese models
    # Format: {lang}_{region}-{voice_name}-{quality}
    # Quality: low, medium, high
    VOICES = {
        # Vietnamese - use vais1000 for best quality
        "vi_vais": "vi_VN-vais1000-medium",
        "vi_VN-vais1000": "vi_VN-vais1000-medium",
        "vi_25hours": "vi_VN-25hours_single-low",
        "vi_VN-25hours_single": "vi_VN-25hours_single-low",
        "vi_vivos": "vi_VN-vivos-x_low",
        "vi_VN-vivos": "vi_VN-vivos-x_low",
        # English
        "en_US-lessac": "en_US-lessac-medium",  # English female
        "en_US-amy": "en_US-amy-medium",  # English female
        "en_US-danny": "en_US-danny-medium",  # English male
        "en_GB-alba": "en_GB-alba-medium",  # British female
        "en_GB-cori": "en_GB-cori-medium",  # British female
    }
    
    # Default voice per language
    DEFAULT_VOICES = {
        "vi": "vi_VN-vais1000-medium",  # Best Vietnamese quality
        "en": "en_US-lessac-medium",
    }
    
    def __init__(
        self,
        voice: str = "vi_VN-vais1000",
        device: str = "cuda",
        sample_rate: int = 24000,  # Output sample rate (resampled from Piper's 22050Hz)
        model_dir: Optional[str] = None,
    ):
        self.voice = voice
        self.device = device
        self.sample_rate = sample_rate
        self.piper_sample_rate = 22050  # Piper's native output rate
        self.model_dir = model_dir or os.path.expanduser("~/.local/share/piper")
        self._voice: Optional[PiperVoice] = None
        
        # Force model loading during initialization
        print(f"Initializing Piper TTS with voice '{voice}' on device '{device}'")
        _ = self.voice_model  # This will trigger model loading
        print("Piper TTS initialized successfully")
    
    @property
    def voice_model(self) -> PiperVoice:
        if not PIPER_AVAILABLE:
            raise RuntimeError(
                "Piper is not installed. Install with: pip install piper-tts"
            )
        if self._voice is None:
            # Load the voice model
            model_name = self.VOICES.get(self.voice, self.voice)
            print(f"Loading Piper model: {model_name}")
            
            # Ensure model directory exists
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Check if model exists locally, if not download
            model_onnx = os.path.join(self.model_dir, f"{model_name}.onnx")
            model_json = os.path.join(self.model_dir, f"{model_name}.onnx.json")  # Piper uses .onnx.json
            
            if not os.path.exists(model_onnx) or not os.path.exists(model_json):
                print(f"Downloading Piper model {model_name}...")
                import urllib.request
                
                # Parse model name: en_US-lessac-medium -> lang=en, region=en_US, name=lessac, quality=medium
                parts = model_name.split('-')
                if len(parts) >= 3:
                    region = parts[0]  # en_US
                    voice_name = parts[1]  # lessac
                    quality = parts[2]  # medium
                    lang = region.split('_')[0]  # en from en_US
                else:
                    lang = "en"
                    region = "en_US"
                    voice_name = "lessac"
                    quality = "medium"
                
                # Piper model URL format - use ?download=true to get raw file
                base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{region}/{voice_name}/{quality}/{model_name}"
                
                print(f"Downloading from: {base_url}.onnx?download=true")
                
                # Download ONNX model
                urllib.request.urlretrieve(f"{base_url}.onnx?download=true", model_onnx)
                print(f"Downloaded {model_onnx}")
                
                # Download config JSON - Piper uses .onnx.json suffix
                urllib.request.urlretrieve(f"{base_url}.onnx.json?download=true", model_json)
                print(f"Downloaded {model_json}")
            
            # Load the voice
            self._voice = PiperVoice.load(model_onnx, use_cuda=False)
        return self._voice
    
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
            Audio samples as numpy array (float32, mono, 22050Hz)
        """
        # Use different voice if specified
        if voice and voice != self.voice:
            old_voice = self.voice
            self.voice = voice
            self._voice = None  # Reset to load new voice
            try:
                model = self.voice_model
            finally:
                self.voice = old_voice
                self._voice = None  # Reset back to original voice
        else:
            model = self.voice_model
        
        # Piper outputs WAV bytes, we need to extract PCM
        # synthesize returns a generator of AudioChunk objects
        print(f"DEBUG: voice_model type: {type(self.voice_model)}")
        
        wav_chunks = []
        
        # synthesize returns a generator
        print("DEBUG: Using synthesize")
        for chunk in self.voice_model.synthesize(text):
            print(f"DEBUG: chunk type: {type(chunk)}")
            # chunk is an AudioChunk object with audio_int16_bytes attribute
            if hasattr(chunk, 'audio_int16_bytes'):
                wav_chunks.append(chunk.audio_int16_bytes)
                print(f"DEBUG: Got chunk.audio_int16_bytes {len(chunk.audio_int16_bytes)} bytes")
            elif hasattr(chunk, 'audio'):
                wav_chunks.append(chunk.audio)
                print(f"DEBUG: Got chunk.audio {len(chunk.audio)} bytes")
            elif isinstance(chunk, bytes):
                wav_chunks.append(chunk)
                print(f"DEBUG: Got chunk bytes {len(chunk)} bytes")
        
        print(f"DEBUG: Total chunks: {len(wav_chunks)}")
        
        if not wav_chunks:
            # Return silence if no audio generated
            print("DEBUG: No audio chunks, returning silence")
            return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
        
        # Combine all chunks
        wav_bytes = b''.join(wav_chunks)
        print(f"DEBUG: Combined PCM bytes: {len(wav_bytes)}")
        
        # audio_int16_bytes is raw PCM, no WAV header
        audio_data = wav_bytes
        
        if not audio_data:
            # Return silence if no audio generated
            return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
        
        # Convert int16 bytes to float32 numpy array
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Resample from Piper's 22050Hz to target sample rate (24000Hz)
        if self.sample_rate != self.piper_sample_rate:
            ratio = self.sample_rate / self.piper_sample_rate
            new_len = int(len(audio_float) * ratio)
            indices = np.linspace(0, len(audio_float) - 1, new_len)
            audio_float = np.interp(indices, np.arange(len(audio_float)), audio_float)
        
        duration = len(audio_float) / self.sample_rate
        print(f"TTS generated {duration:.2f}s of audio from {len(text)} chars")
        return audio_float
    
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
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class FallbackTTS:
    """Fallback TTS using edge-tts (browser-based) if Piper unavailable."""
    
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
            if self.device == "cuda" and torch and torch.cuda.is_available():
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
                if isinstance(audio, torch.Tensor) and torch:
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


# Vietnamese TTS configuration
VIETNAMESE_TTS_CONFIG = {
    "default_voice": "vi_VN-vais1000-medium",
    "language": "vi",
}


def get_tts_for_language(language: str = "en", device: str = "cuda", engine: str = "piper") -> PiperTTS | KokoroTTS:
    """Get appropriate TTS for language and engine."""
    if engine == "kokoro":
        # Kokoro doesn't have Vietnamese voices, use fallback for Vietnamese
        if language == "vi":
            return FallbackTTS(voice="vi-VN-HoaiMyNeural")
        else:
            # English and others - use Kokoro
            voice = KokoroTTS.VOICES.get("af_bella", "af_bella")
            return KokoroTTS(voice=voice, device=device)
    else:
        # Piper (default)
        voice = PiperTTS.DEFAULT_VOICES.get(language, "en_US-lessac")
        return PiperTTS(voice=voice, device=device)
