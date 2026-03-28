"""Audio processing utilities for Opus encoding/decoding."""

import asyncio
from typing import Optional, List
import numpy as np
from io import BytesIO
import ctypes
import struct
import random


class OggPageEncoder:
    """Creates Ogg pages from Opus packets for streaming to client."""
    
    # Use serial 1 to match client decoder warmup stream
    FIXED_SERIAL = 1
    
    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.serial = self.FIXED_SERIAL  # Match client decoder
        self.page_seq = 0
        self.granule_pos = 0
        
    def _crc32_lookup(self) -> List[int]:
        """Generate CRC32 lookup table."""
        lookup = []
        for i in range(256):
            r = i << 24
            for _ in range(8):
                if r & 0x80000000:
                    r = ((r << 1) ^ 0x04c11db7) & 0xFFFFFFFF
                else:
                    r = (r << 1) & 0xFFFFFFFF
            lookup.append(r)
        return lookup
    
    def _crc32(self, data: bytes, crc: int = 0) -> int:
        """Calculate CRC32 for Ogg page."""
        lookup = self._crc32_lookup()
        for byte in data:
            crc = ((crc << 8) ^ lookup[((crc >> 24) ^ byte) & 0xFF]) & 0xFFFFFFFF
        return crc
    
    def create_page(self, packets: List[bytes], samples: int = 0, eos: bool = False) -> bytes:
        """
        Create an Ogg page containing Opus packets.
        
        Args:
            packets: List of Opus packets to include
            samples: Number of samples in this page (for granule position)
            eos: End of stream flag
            
        Returns:
            Ogg page as bytes
        """
        # Build segment table
        segment_table = []
        for packet in packets:
            size = len(packet)
            while size >= 255:
                segment_table.append(255)
                size -= 255
            segment_table.append(size)
        
        # Update granule position with actual samples
        self.granule_pos += samples
        
        # Build header
        header_type = 0x04 if eos else 0x00  # BOS=0x02, EOS=0x04
        if self.page_seq == 0:
            header_type |= 0x02  # First page
        
        header = bytearray()
        header.extend(b'OggS')  # Capture pattern
        header.append(0)  # Version
        header.append(header_type)  # Header type
        header.extend(struct.pack('<Q', self.granule_pos))  # Granule position
        header.extend(struct.pack('<I', self.serial))  # Serial number
        header.extend(struct.pack('<I', self.page_seq))  # Page sequence
        header.extend(struct.pack('<I', 0))  # CRC (placeholder)
        header.append(len(segment_table))  # Number of segments
        header.extend(segment_table)  # Segment table
        
        # Add packets
        for packet in packets:
            header.extend(packet)
        
        # Calculate and insert CRC
        crc = self._crc32(bytes(header))
        header[22:26] = struct.pack('<I', crc)
        
        self.page_seq += 1
        return bytes(header)
    
    def create_header_page(self, sample_rate: int = 24000, channels: int = 1) -> bytes:
        """Create Ogg Opus header page (OpusHead)."""
        # OpusHead packet format - MUST match what client decoder expects
        # Client expects: "OpusHead" + version(1) + channels(1) + preskip(312) + samplerate(48000) + gain(0) + mapping(0)
        
        opus_head = bytearray()
        opus_head.extend(b'OpusHead')
        opus_head.append(1)  # Version
        opus_head.append(channels)  # Channels
        opus_head.extend(struct.pack('<H', 312))  # Pre-skip: 312 samples (0x38, 0x01)
        opus_head.extend(struct.pack('<I', 48000))  # Sample rate: 48kHz (0x80, 0xBB, 0x00, 0x00)
        opus_head.extend(struct.pack('<h', 0))  # Gain
        opus_head.append(0)  # Channel mapping
        
        print(f"DEBUG: OpusHead packet: {bytes(opus_head).hex()}")
        print(f"DEBUG: Expected: 4f707573486561640101380180bb00000000000")
        
        return self.create_page([bytes(opus_head)], samples=0, eos=False)
    
    def create_tag_page(self) -> bytes:
        """Create Ogg Opus comment page (OpusTags)."""
        # OpusTags packet format
        # - "OpusTags" (8 bytes)
        # - Vendor string length (4 bytes LE)
        # - Vendor string
        # - User comment list length (4 bytes LE) = 0
        
        vendor = b'PersonaPlex-Alternative'
        opus_tags = bytearray()
        opus_tags.extend(b'OpusTags')
        opus_tags.extend(struct.pack('<I', len(vendor)))
        opus_tags.extend(vendor)
        opus_tags.extend(struct.pack('<I', 0))  # No user comments
        
        return self.create_page([bytes(opus_tags)], eos=False)


class OggOpusDecoder:
    """Decodes Ogg-wrapped Opus streams from opus-recorder client."""
    
    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._opus_decoder = None
        self._libopus = None
        self._serialno = None
        self._page_granule = 0
        self._init_lib()
    
    def _init_lib(self):
        """Initialize opus library."""
        try:
            from pyogg.opus import (
                libopus,
                opus_int16,
                OPUS_APPLICATION_AUDIO,
            )
            self._libopus = type('LibOpus', (), {
                'libopus': libopus,
                'opus_int16': opus_int16,
                'OPUS_APPLICATION_AUDIO': OPUS_APPLICATION_AUDIO,
            })()
            self._init_decoder()
        except ImportError as e:
            print(f"Warning: Could not load opus: {e}")
    
    def _init_decoder(self):
        """Initialize raw Opus decoder."""
        if self._libopus is None:
            return
        try:
            error = ctypes.c_int()
            self._opus_decoder = self._libopus.libopus.opus_decoder_create(
                48000,  # Opus always uses 48kHz internally
                self.channels,
                ctypes.byref(error)
            )
            if error.value == 0:
                print("Raw Opus decoder initialized")
            else:
                print(f"Opus decoder init failed: {error.value}")
                self._opus_decoder = None
        except Exception as e:
            print(f"Opus decoder init error: {e}")
            self._opus_decoder = None
    
    def _parse_ogg_page(self, data: bytes):
        """Parse an Ogg page and extract Opus packets."""
        if len(data) < 27 or data[:4] != b'OggS':
            return None, None
        
        # Ogg page header structure:
        # 0-3: "OggS"
        # 4: version (should be 0)
        # 5: type flag
        # 6-13: granule position
        # 14-17: bitstream serial number
        # 18-21: page sequence number
        # 22-25: CRC checksum
        # 26: number of page segments
        # 27+: segment table
        
        version = data[4]
        flags = data[5]
        granule_pos = int.from_bytes(data[6:14], 'little', signed=True)
        serialno = int.from_bytes(data[14:18], 'little')
        pageno = int.from_bytes(data[18:22], 'little')
        num_segments = data[26]
        
        # Parse segment table
        segment_table = data[27:27 + num_segments]
        header_size = 27 + num_segments
        
        # Extract packets
        packets = []
        offset = header_size
        packet_len = 0
        
        for seg_len in segment_table:
            packet_len += seg_len
            if seg_len < 255:  # End of packet
                if offset + packet_len <= len(data):
                    packets.append(data[offset:offset + packet_len])
                offset += packet_len
                packet_len = 0
        
        # Handle final packet if table ends with 255
        if packet_len > 0 and offset + packet_len <= len(data):
            packets.append(data[offset:offset + packet_len])
        
        return packets, granule_pos
    
    def _decode_raw_opus(self, opus_packet: bytes) -> Optional[np.ndarray]:
        """Decode raw Opus packet to PCM."""
        if self._opus_decoder is None or not opus_packet:
            return None
        
        try:
            # Frame size for 20ms at 48kHz
            frame_size = 960
            
            # Output buffer
            output = (self._libopus.opus_int16 * (frame_size * self.channels))()
            
            # Create input buffer
            input_data = (ctypes.c_ubyte * len(opus_packet))(*opus_packet)
            
            # Decode
            num_samples = self._libopus.libopus.opus_decode(
                self._opus_decoder,
                input_data,
                len(opus_packet),
                output,
                frame_size,
                0  # no FEC
            )
            
            if num_samples < 0:
                return None
            
            # Convert to numpy
            pcm_int16 = np.ctypeslib.as_array(output, shape=(num_samples * self.channels,))
            pcm = pcm_int16.astype(np.float32) / 32767
            return pcm
            
        except Exception as e:
            print(f"Raw opus decode error: {e}")
            return None
    
    def decode_ogg_page(self, ogg_data: bytes) -> Optional[np.ndarray]:
        """
        Decode Ogg page containing Opus packets.
        
        Args:
            ogg_data: Ogg-wrapped Opus bytes (starts with 'OggS')
            
        Returns:
            Float32 PCM audio or None if decode fails
        """
        if self._libopus is None:
            # Fallback - treat as raw PCM
            try:
                pcm = np.frombuffer(ogg_data, dtype=np.int16).astype(np.float32) / 32767
                return pcm
            except:
                return None
        
        # Check if this is an Ogg page
        if len(ogg_data) < 4 or ogg_data[:4] != b'OggS':
            # Not an Ogg page, try raw PCM fallback
            try:
                pcm = np.frombuffer(ogg_data, dtype=np.int16).astype(np.float32) / 32767
                return pcm
            except:
                return np.zeros(480, dtype=np.float32)
        
        # Parse Ogg page and extract Opus packets
        packets, granule = self._parse_ogg_page(ogg_data)
        
        if not packets:
            return np.zeros(480, dtype=np.float32)
        
        # Skip header pages (identification and comment headers)
        decoded_chunks = []
        
        for packet in packets:
            # Skip header packets
            if packet[:8] == b'OpusHead' or packet[:7] == b'OpusTags':
                continue
            
            # Try to decode as raw Opus
            pcm = self._decode_raw_opus(packet)
            if pcm is not None and len(pcm) > 0:
                decoded_chunks.append(pcm)
        
        if not decoded_chunks:
            return np.zeros(480, dtype=np.float32)
        
        # Concatenate all decoded chunks
        combined = np.concatenate(decoded_chunks)
        
        # Resample from 48kHz to target rate if needed
        if self.sample_rate != 48000:
            combined = self._resample(combined, 48000, self.sample_rate)
        
        return combined
    
    def _resample(self, pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple linear resampling."""
        if from_rate == to_rate:
            return pcm
        ratio = to_rate / from_rate
        new_len = int(len(pcm) * ratio)
        indices = np.linspace(0, len(pcm) - 1, new_len)
        return np.interp(indices, np.arange(len(pcm)), pcm)


class OpusCodec:
    """Opus audio codec for real-time streaming using pyogg ctypes."""
    
    OPUS_APPLICATION_AUDIO = 2049
    
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
        self._libopus = None
        self._ogg_decoder = None
    
    def _load_libopus(self):
        """Load opus library via pyogg."""
        if self._libopus is not None:
            return True
        
        try:
            from pyogg.opus import (
                libopus,
                opus_int16,
                opus_int32,
                OPUS_APPLICATION_AUDIO,
            )
            self._libopus = type('LibOpus', (), {
                'libopus': libopus,
                'opus_int16': opus_int16,
                'opus_int32': opus_int32,
                'OPUS_APPLICATION_AUDIO': OPUS_APPLICATION_AUDIO,
            })()
            return True
        except ImportError as e:
            print(f"Warning: Could not load opus via pyogg: {e}")
            return False
    
    def init_encoder(self):
        """Initialize Opus encoder at 48kHz (Opus internal rate)."""
        if not self._load_libopus():
            return
        
        try:
            error = ctypes.c_int()
            # Always initialize at 48kHz for Opus standard
            self._encoder = self._libopus.libopus.opus_encoder_create(
                48000,  # Opus works internally at 48kHz
                self.channels,
                self._libopus.OPUS_APPLICATION_AUDIO,
                ctypes.byref(error)
            )
            
            if error.value != 0:
                print(f"Warning: Opus encoder creation failed with error {error.value}")
                self._encoder = None
            else:
                print("Opus encoder initialized at 48kHz successfully")
        except Exception as e:
            print(f"Warning: Opus encoder init failed: {e}")
            self._encoder = None
    
    def init_decoder(self):
        """Initialize Ogg Opus decoder."""
        self._ogg_decoder = OggOpusDecoder(self.sample_rate, self.channels)
        print("Ogg Opus decoder initialized")
    
    def encode(self, pcm: np.ndarray) -> bytes:
        """Encode PCM audio to Ogg-wrapped Opus for client."""
        # Resample to 48kHz for Opus encoding
        if self.sample_rate != 48000:
            ratio = 48000 / self.sample_rate
            new_len = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(indices, np.arange(len(pcm)), pcm)
        
        pcm_int16 = (pcm * 32767).astype(np.int16)
        
        if self._encoder is None:
            self.init_encoder()
        
        if self._encoder is None:
            # Fallback: return raw PCM
            return pcm_int16.tobytes()
        
        try:
            # Encode in frames at 48kHz
            # 20ms at 48kHz = 960 samples per frame
            frame_size_48k = int(48000 * 20 / 1000)  # 960 samples
            all_opus_packets = []
            
            # Process audio in frame_size chunks
            for i in range(0, len(pcm_int16), frame_size_48k):
                frame = pcm_int16[i:i + frame_size_48k]
                
                # Pad last frame if needed
                if len(frame) < frame_size_48k:
                    frame = np.pad(frame, (0, frame_size_48k - len(frame)), mode='constant')
                
                max_bytes = 4000
                output = (ctypes.c_ubyte * max_bytes)()
                
                num_bytes = self._libopus.libopus.opus_encode(
                    self._encoder,
                    frame.ctypes.data_as(ctypes.POINTER(self._libopus.opus_int16)),
                    frame_size_48k,
                    output,
                    max_bytes
                )
                
                if num_bytes > 0:
                    all_opus_packets.append(bytes(output[:num_bytes]))
            
            # Wrap in Ogg container with serial=1 to match decoder warmup
            if not all_opus_packets:
                return b''
            
            ogg_encoder = OggPageEncoder(48000, self.channels)
            ogg_encoder.serial = 1  # Must match decoder warmup stream serial
            ogg_encoder.page_seq = 0  # Reset sequence
            ogg_encoder.granule_pos = 0  # Reset granule
            
            # Build Ogg stream: header page, tag page, audio pages
            ogg_data = bytearray()
            header_page = ogg_encoder.create_header_page(48000, self.channels)
            print(f"DEBUG: Header page serial={ogg_encoder.serial}, seq={ogg_encoder.page_seq-1}")
            ogg_data.extend(header_page)
            
            tag_page = ogg_encoder.create_tag_page()
            print(f"DEBUG: Tag page serial={ogg_encoder.serial}, seq={ogg_encoder.page_seq-1}")
            ogg_data.extend(tag_page)
            
            # Calculate samples per frame (20ms at 48kHz = 960 samples)
            samples_per_frame_48k = frame_size_48k
            
            # Put all packets in one audio page (or split if too many)
            max_packets_per_page = 50
            for i in range(0, len(all_opus_packets), max_packets_per_page):
                packet_chunk = all_opus_packets[i:i + max_packets_per_page]
                is_last = (i + max_packets_per_page >= len(all_opus_packets))
                chunk_samples = len(packet_chunk) * samples_per_frame_48k
                audio_page = ogg_encoder.create_page(packet_chunk, samples=chunk_samples, eos=is_last)
                print(f"DEBUG: Audio page serial={ogg_encoder.serial}, seq={ogg_encoder.page_seq-1}, samples={chunk_samples}, eos={is_last}")
                ogg_data.extend(audio_page)
            
            result = bytes(ogg_data)
            print(f"DEBUG: Custom Ogg Opus: {len(result)} bytes, serial=1")
            return result
            
        except Exception as e:
            print(f"Warning: Opus encode failed: {e}")
            import traceback
            traceback.print_exc()
            return pcm_int16.tobytes()
    
    def decode(self, opus_data: bytes) -> np.ndarray:
        """Decode Ogg-wrapped Opus (from opus-recorder) to PCM."""
        if self._ogg_decoder is None:
            self.init_decoder()
        
        # Check if this is Ogg-wrapped (starts with 'OggS')
        if len(opus_data) >= 4 and opus_data[:4] == b'OggS':
            pcm = self._ogg_decoder.decode_ogg_page(opus_data)
            if pcm is not None:
                return pcm
            # Fallback to empty audio
            return np.zeros(self.frame_size, dtype=np.float32)
        
        # Raw PCM fallback
        try:
            pcm = np.frombuffer(opus_data, dtype=np.int16).astype(np.float32) / 32767
            return pcm
        except:
            return np.zeros(self.frame_size, dtype=np.float32)


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
