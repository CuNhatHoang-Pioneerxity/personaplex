"""Main WebSocket server matching PersonaPlex protocol."""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional
import numpy as np

# Add nvidia cublas DLL path before importing CUDA-dependent libraries
cublas_bin = Path(__file__).parent.parent.parent / ".venv" / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
if cublas_bin.exists():
    os.add_dll_directory(str(cublas_bin))
    os.environ["PATH"] = str(cublas_bin) + os.pathsep + os.environ.get("PATH", "")

from aiohttp import web

from .protocol import (
    decode_message,
    encode_message,
    HandshakeMessage,
    AudioMessage,
    TextMessage,
    ControlMessage,
    MetadataMessage,
    ErrorMessage,
    PingMessage,
    ControlAction,
)
from .stt import WhisperSTT
from .llm import OllamaLLM, PERSONA_PROMPTS
from .tts import PiperTTS, FallbackTTS, get_tts_for_language
from .audio import AudioBuffer, OpusCodec, convert_to_16khz

logger = logging.getLogger(__name__)


class VoiceSession:
    """Manages a single voice conversation session."""
    
    def __init__(
        self,
        stt: WhisperSTT,
        llm: OllamaLLM,
        tts,  # PiperTTS or FallbackTTS
        opus_codec: OpusCodec,
    ):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.opus = opus_codec
        
        # Audio buffer for accumulating speech
        self.audio_buffer = AudioBuffer(max_seconds=30.0, sample_rate=24000)
        
        # State
        self.is_speaking = False
        self.is_paused = False
        self.processing = False  # Flag to prevent re-entry during processing
        self.detected_language = "en"
        self.text_prompt = ""
        self.voice_prompt = "en_US-lessac"
        
        # Accumulated text for display
        self.transcribed_text = ""
        self.response_text = ""
        self.last_sent_transcription = ""  # Track already sent text
        self.last_sent_response = ""  # Track already sent response
    
    def set_persona(self, text_prompt: str, voice_prompt: str):
        """Set the persona for this session."""
        self.text_prompt = text_prompt
        self.voice_prompt = voice_prompt
        
        # Map voice prompts to TTS voices (Piper voice names)
        voice_mapping = {
            # New persona IDs
            "NATF0": "en_US-lessac-medium",  # Female
            "NATF1": "en_US-amy-medium",      # Female
            "NATM0": "en_US-danny-medium",     # Male
            "NATM1": "en_US-danny-medium",     # Male
            # Old Kokoro voice names -> Piper equivalents
            "af_bella": "en_US-lessac-medium",
            "af_sarah": "en_US-amy-medium",
            "af_sky": "en_US-lessac-medium",
            "af_nicole": "en_US-lessac-medium",
            "am_adam": "en_US-danny-medium",
            "am_michael": "en_US-danny-medium",
            "bf_emma": "en_GB-alba-medium",
            "bf_isabella": "en_GB-cori-medium",
            "bm_george": "en_GB-cori-medium",
            "bm_lewis": "en_GB-cori-medium",
            # Vietnamese voices
            "vi_VN-vais1000-medium": "vi_VN-vais1000-medium",
            "vi_VN-25hours_single-low": "vi_VN-25hours_single-low",
            "vi_VN-vivos-x_low": "vi_VN-vivos-x_low",
        }
        
        # If voice_prompt is already a Piper voice name, use it directly
        if voice_prompt in voice_mapping.values():
            self.voice_prompt = voice_prompt
        else:
            self.voice_prompt = voice_mapping.get(voice_prompt, "en_US-lessac-medium")  # Default fallback
        
        # Set LLM system prompt
        if text_prompt:
            self.llm.set_system_prompt(text_prompt)
    
    async def process_audio(self, audio_data: bytes) -> Optional[bytes]:
        """
        Process incoming audio and return response audio.
        
        This is the main pipeline:
        1. Decode Opus -> PCM
        2. Accumulate in buffer
        3. Detect silence/end-of-speech
        4. Transcribe with Whisper
        5. Generate response with Ollama
        6. Synthesize with Kokoro
        7. Encode to Opus and return
        """
        try:
            # Decode Opus to PCM
            pcm = self.opus.decode(audio_data)
            self.audio_buffer.append(pcm)
            
            # Check for end of speech (simple energy-based)
            # In real implementation, use VAD
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None
    
    async def process_text(self, text: str) -> Optional[bytes]:
        """Process text input directly (bypasses STT)."""
        self.transcribed_text = text
        return await self._generate_response(text)
    
    async def end_turn(self) -> Optional[bytes]:
        """
        Signal end of user's turn.
        
        Transcribe accumulated audio and generate response.
        """
        if len(self.audio_buffer) < 8000:  # Less than 0.33s at 24kHz
            logger.info("Audio buffer too short, skipping")
            self.audio_buffer.clear()
            return None
        
        # Get buffered audio
        audio = self.audio_buffer.get_all()
        self.audio_buffer.clear()  # Clear after taking audio
        
        # Convert to 16kHz for Whisper
        audio_16khz = convert_to_16khz(audio, 24000)
        
        # Transcribe
        logger.info("Transcribing audio...")
        text, language = await self.stt.transcribe_async(audio_16khz, 16000)
        self.detected_language = language
        
        if not text.strip():
            logger.info("No speech detected, buffer cleared")
            return None
        
        # Only update if this is new text (not same as last)
        if text == self.transcribed_text:
            logger.info("Same text detected, skipping")
            return None
        
        self.transcribed_text = text
        logger.info(f"Transcribed ({language}): {text}")
        
        # Generate response
        return await self._generate_response(text)
    
    async def _generate_response(self, user_input: str) -> Optional[bytes]:
        """Generate TTS response for user input."""
        try:
            # Use the session's TTS (already has correct voice)
            tts = self.tts
            
            # Stream LLM response
            full_response = ""
            audio_chunks = []
            sentence_buffer = ""
            
            async for chunk in self.llm.generate_with_vietnamese(
                user_input,
                self.detected_language,
                stream=True,
            ):
                full_response += chunk
                sentence_buffer += chunk
                
                # Check if we have a complete sentence
                if sentence_buffer.endswith(('.', '!', '?', '。', '！', '？')):
                    # Only synthesize if sentence has meaningful content (not just punctuation)
                    if len(sentence_buffer.strip()) > 1:
                        logger.info(f"TTS synthesizing: '{sentence_buffer}'")
                        audio = await tts.synthesize_async(sentence_buffer)
                        audio_chunks.append(audio)
                    sentence_buffer = ""  # Reset for next sentence
            
            # Handle any remaining text without ending punctuation
            if sentence_buffer.strip() and len(sentence_buffer.strip()) > 1:
                logger.info(f"TTS synthesizing final: '{sentence_buffer}'")
                audio = await tts.synthesize_async(sentence_buffer)
                audio_chunks.append(audio)
            
            self.response_text = full_response
            logger.info(f"Response: {full_response}")
            
            # Combine audio chunks
            if audio_chunks:
                combined = np.concatenate(audio_chunks)
                duration = len(combined) / 24000  # 24kHz sample rate
                logger.info(f"Combined audio: {duration:.2f}s from {len(audio_chunks)} chunks")
                # Encode to Opus
                opus_data = self.opus.encode(combined)
                return opus_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    def reset(self):
        """Reset session state."""
        self.audio_buffer.clear()
        self.is_speaking = False
        self.is_paused = False
        self.transcribed_text = ""
        self.response_text = ""
        self.last_sent_transcription = ""
        self.last_sent_response = ""
        self.llm.clear_history()


class AlternativeServer:
    """Alternative voice server compatible with PersonaPlex client."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8998,
        static_path: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        whisper_model: str = "small",
        whisper_device: str = "cuda",
    ):
        self.host = host
        self.port = port
        self.static_path = static_path
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        
        # Shared components (lazy init)
        self._stt: Optional[WhisperSTT] = None
        self._opus: Optional[OpusCodec] = None
    
    @property
    def stt(self) -> WhisperSTT:
        if self._stt is None:
            self._stt = WhisperSTT(
                model_size=self.whisper_model,
                device=self.whisper_device,
            )
        return self._stt
    
    @property
    def opus(self) -> OpusCodec:
        if self._opus is None:
            self._opus = OpusCodec(sample_rate=24000)
            self._opus.init_decoder()
            self._opus.init_encoder()
        return self._opus
    
    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connection for voice chat."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        logger.info(f"New WebSocket connection from {request.remote}")
        
        # Parse query params FIRST
        text_prompt = request.query.get("text_prompt", "")
        voice_prompt = request.query.get("voice_prompt", "en_US-lessac")
        forced_lang = request.query.get("lang", None)
        engine = request.query.get("engine", "piper")  # piper or kokoro
        
        if forced_lang == "vi":
            # Use selected Vietnamese voice, default to vais1000 if not Vietnamese
            if voice_prompt.startswith("vi_VN-"):
                voice = voice_prompt
            else:
                voice = "vi_VN-vais1000-medium"  # Default Vietnamese voice
        elif forced_lang == "en":
            # Use selected English voice, default to lessac if not English
            if voice_prompt.startswith("en_") or voice_prompt.startswith("af_") or voice_prompt.startswith("am_") or voice_prompt.startswith("bf_") or voice_prompt.startswith("bm_"):
                voice = voice_prompt
            else:
                voice = "en_US-lessac-medium"  # Default English voice
        else:
            voice = voice_prompt  # Use selected voice
        
        # Create TTS based on engine
        if engine == "kokoro":
            from .tts import KokoroTTS
            tts = KokoroTTS(voice=voice, device="cuda")
        else:
            from .tts import PiperTTS
            tts = PiperTTS(voice=voice, device="cuda")
        
        logger.info(f"Initializing {engine.upper()} TTS with voice '{voice}'")
        
        llm = OllamaLLM(
            base_url=self.ollama_url,
            model=self.ollama_model,
        )
        
        session = VoiceSession(
            stt=self.stt,
            llm=llm,
            tts=tts,
            opus_codec=self.opus,
        )
        
        session.set_persona(text_prompt, voice_prompt)
        logger.info(f"Language forced to: {forced_lang}")
        logger.info(f"request.query: {request.query}")

        # Store forced language in session
        if forced_lang:
            session.detected_language = forced_lang

        # Send handshake
        handshake = encode_message(HandshakeMessage())
        await ws.send_bytes(handshake)
        
        # Start ping task to keep connection alive
        async def ping_task():
            while not ws.closed:
                await asyncio.sleep(5)  # Ping every 5 seconds
                if not ws.closed:
                    try:
                        await ws.send_bytes(encode_message(PingMessage()))
                        logger.debug("Sent ping")
                    except:
                        break
        
        ping_task_handle = asyncio.create_task(ping_task())
        
        # Message loop
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    await self._handle_binary_message(ws, session, msg.data)
                elif msg.type == web.WSMsgType.TEXT:
                    logger.warning(f"Unexpected text message: {msg.data}")
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
                    break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            ping_task_handle.cancel()
            await llm.close()
            logger.info("WebSocket connection closed")
        
        return ws
    
    async def _handle_binary_message(
        self,
        ws: web.WebSocketResponse,
        session: VoiceSession,
        data: bytes,
    ):
        """Handle binary protocol message."""
        try:
            message = decode_message(data)
            logger.debug(f"Received message: {type(message).__name__}, size: {len(data)}")
            
            if isinstance(message, HandshakeMessage):
                # Already sent handshake, ignore
                pass
            
            elif isinstance(message, AudioMessage):
                # Process audio - accumulate and check if we have enough
                logger.info(f"Audio message received, {len(message.data)} bytes")
                decoded = session.opus.decode(message.data)
                logger.info(f"Decoded audio: {len(decoded)} samples, range: [{decoded.min():.4f}, {decoded.max():.4f}], mean: {decoded.mean():.4f}")
                session.audio_buffer.append(decoded)
                
                # Check if we have enough audio to process (1.5 seconds)
                buffer_samples = len(session.audio_buffer)
                # Only process if not already processing (cooldown)
                if buffer_samples >= 36000 and not session.processing:  # 1.5s at 24kHz
                    session.processing = True  # Set flag to prevent re-entry
                    logger.info(f"Processing audio buffer: {buffer_samples} samples")
                    response_audio = await session.end_turn()
                    session.processing = False  # Clear flag
                    if response_audio:
                        logger.info(f"Sending response audio: {len(response_audio)} bytes")
                        await ws.send_bytes(encode_message(AudioMessage(data=response_audio)))
                    else:
                        logger.info("No response audio generated")
                    
                    # Send text updates - only if new
                    if session.transcribed_text and session.transcribed_text != session.last_sent_transcription:
                        await ws.send_bytes(encode_message(TextMessage(data=f"You: {session.transcribed_text}")))
                        session.last_sent_transcription = session.transcribed_text
                    if session.response_text and session.response_text != session.last_sent_response:
                        await ws.send_bytes(encode_message(TextMessage(data=f"Assistant: {session.response_text}")))
                        session.last_sent_response = session.response_text
            
            elif isinstance(message, TextMessage):
                # Direct text input
                response_audio = await session.process_text(message.data)
                if response_audio:
                    await ws.send_bytes(encode_message(AudioMessage(data=response_audio)))
                
                # Send transcribed text for display
                await ws.send_bytes(encode_message(TextMessage(data=session.transcribed_text)))
                await ws.send_bytes(encode_message(TextMessage(data=session.response_text)))
            
            elif isinstance(message, ControlMessage):
                if message.action == ControlAction.END_TURN:
                    # User finished speaking
                    response_audio = await session.end_turn()
                    if response_audio:
                        await ws.send_bytes(encode_message(AudioMessage(data=response_audio)))
                    
                    # Send text for display
                    await ws.send_bytes(encode_message(TextMessage(data=f"User: {session.transcribed_text}")))
                    await ws.send_bytes(encode_message(TextMessage(data=f"Assistant: {session.response_text}")))
                
                elif message.action == ControlAction.RESTART:
                    session.reset()
                
                elif message.action == ControlAction.PAUSE:
                    session.is_paused = True
            
            elif isinstance(message, PingMessage):
                # Pong
                await ws.send_bytes(encode_message(PingMessage()))
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await ws.send_bytes(encode_message(ErrorMessage(data=str(e))))
    
    async def handle_root(self, request: web.Request) -> web.FileResponse:
        """Serve index.html."""
        if self.static_path:
            return web.FileResponse(os.path.join(self.static_path, "index.html"))
        return web.Response(text="PersonaPlex Alternative Server", content_type="text/html")
    
    def create_app(self) -> web.Application:
        """Create aiohttp application."""
        app = web.Application()
        
        # WebSocket endpoint
        app.router.add_get("/api/chat", self.handle_chat)
        
        # Static files
        if self.static_path:
            app.router.add_get("/", self.handle_root)
            app.router.add_static("/", path=self.static_path, name="static")
        
        return app
    
    def run(self):
        """Run the server."""
        app = self.create_app()
        web.run_app(app, host=self.host, port=self.port)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="PersonaPlex Alternative Voice Server")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str, help="Path to static client files")
    parser.add_argument("--ollama-url", default="http://localhost:11434", type=str)
    parser.add_argument("--model", default="llama3.2", type=str, help="Ollama model name")
    parser.add_argument("--whisper-model", default="small", type=str, 
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    server = AlternativeServer(
        host=args.host,
        port=args.port,
        static_path=args.static,
        ollama_url=args.ollama_url,
        ollama_model=args.model,
        whisper_model=args.whisper_model,
        whisper_device=args.device,
    )
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    server.run()


if __name__ == "__main__":
    main()
