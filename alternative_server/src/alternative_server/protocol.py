"""WebSocket protocol matching PersonaPlex binary format."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Union
import json


class MessageType(IntEnum):
    HANDSHAKE = 0x00
    AUDIO = 0x01
    TEXT = 0x02
    CONTROL = 0x03
    METADATA = 0x04
    ERROR = 0x05
    PING = 0x06


class ControlAction(IntEnum):
    START = 0x00
    END_TURN = 0x01
    PAUSE = 0x02
    RESTART = 0x03


@dataclass
class HandshakeMessage:
    version: int = 0
    model: int = 0


@dataclass
class AudioMessage:
    data: bytes


@dataclass
class TextMessage:
    data: str


@dataclass
class ControlMessage:
    action: ControlAction


@dataclass
class MetadataMessage:
    data: dict


@dataclass
class ErrorMessage:
    data: str


@dataclass
class PingMessage:
    pass


WSMessage = Union[
    HandshakeMessage,
    AudioMessage,
    TextMessage,
    ControlMessage,
    MetadataMessage,
    ErrorMessage,
    PingMessage,
]


def encode_message(message: WSMessage) -> bytes:
    """Encode a message to binary format."""
    if isinstance(message, HandshakeMessage):
        return bytes([MessageType.HANDSHAKE, message.version, message.model])
    elif isinstance(message, AudioMessage):
        return bytes([MessageType.AUDIO]) + message.data
    elif isinstance(message, TextMessage):
        return bytes([MessageType.TEXT]) + message.data.encode("utf-8")
    elif isinstance(message, ControlMessage):
        return bytes([MessageType.CONTROL, message.action])
    elif isinstance(message, MetadataMessage):
        return bytes([MessageType.METADATA]) + json.dumps(message.data).encode("utf-8")
    elif isinstance(message, ErrorMessage):
        return bytes([MessageType.ERROR]) + message.data.encode("utf-8")
    elif isinstance(message, PingMessage):
        return bytes([MessageType.PING])
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def decode_message(data: bytes) -> WSMessage:
    """Decode binary data to a message."""
    if len(data) == 0:
        raise ValueError("Empty message")
    
    msg_type = data[0]
    payload = data[1:]
    
    if msg_type == MessageType.HANDSHAKE:
        version = payload[0] if len(payload) > 0 else 0
        model = payload[1] if len(payload) > 1 else 0
        return HandshakeMessage(version=version, model=model)
    elif msg_type == MessageType.AUDIO:
        return AudioMessage(data=payload)
    elif msg_type == MessageType.TEXT:
        return TextMessage(data=payload.decode("utf-8"))
    elif msg_type == MessageType.CONTROL:
        action = ControlAction(payload[0]) if len(payload) > 0 else ControlAction.START
        return ControlMessage(action=action)
    elif msg_type == MessageType.METADATA:
        return MetadataMessage(data=json.loads(payload.decode("utf-8")))
    elif msg_type == MessageType.ERROR:
        return ErrorMessage(data=payload.decode("utf-8"))
    elif msg_type == MessageType.PING:
        return PingMessage()
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
