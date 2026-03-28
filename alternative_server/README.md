# PersonaPlex Alternative Voice Stack

Alternative backend for PersonaPlex using:

- **Whisper** (faster-whisper) - Speech-to-Text with Vietnamese support
- **Ollama** - LLM with RAG context support
- **Kokoro** - Text-to-Speech

## Features

- Vietnamese language support via Whisper multilingual models
- RAG context injection via Ollama
- Same WebSocket protocol as original PersonaPlex
- Compatible with PersonaPlex client

## Installation

```bash
cd alternative_server
uv sync
```

## Requirements

- Ollama running locally (default: http://localhost:11434)
- Pull a model: `ollama pull llama3.2` or `ollama pull qwen2.5`

## Usage

```bash
# Run on port 8998 (same as PersonaPlex)
uv run alternative-server --host 0.0.0.0 --port 8998

# With custom Ollama endpoint
uv run alternative-server --host 0.0.0.0 --port 8998 --ollama-url http://localhost:11434

# With custom model
uv run alternative-server --host 0.0.0.0 --port 8998 --model qwen2.5
```

## Client Configuration

The PersonaPlex client can connect to this server the same way:

```
http://localhost:8998
```

## API Endpoints

- `GET /` - Serve static client files
- `GET /api/chat` - WebSocket endpoint for voice conversation

## WebSocket Protocol

Binary protocol matching PersonaPlex:

| Byte | Type | Description |
|------|------|-------------|
| 0x00 | handshake | Version + model info |
| 0x01 | audio | Opus-encoded audio |
| 0x02 | text | Text message |
| 0x03 | control | Control commands |
| 0x04 | metadata | JSON metadata |
| 0x05 | error | Error message |
| 0x06 | ping | Keepalive |

## Environment Variables

- `OLLAMA_URL` - Ollama API endpoint (default: http://localhost:11434)
- `WHISPER_MODEL` - Whisper model size (default: small, options: tiny, base, small, medium, large)
- `KOKORO_VOICE` - Kokoro voice ID (default: af_bella)
