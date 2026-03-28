"""LLM integration using Ollama with RAG context support."""

import asyncio
from typing import AsyncIterator, Optional, List, Dict, Any
import httpx


class OllamaLLM:
    """Ollama-based LLM with context/RAG support."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational.",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self._client: Optional[httpx.AsyncClient] = None
        self._context: List[Dict[str, str]] = []
        self._rag_context: str = ""
    
    @property
    async def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt (persona)."""
        self.system_prompt = prompt
    
    def set_rag_context(self, context: str):
        """Set RAG context to inject into prompts."""
        self._rag_context = context
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self._context.append({"role": role, "content": content})
        # Keep last 20 messages
        if len(self._context) > 20:
            self._context = self._context[-20:]
    
    def clear_history(self):
        """Clear conversation history."""
        self._context = []
    
    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Build messages list with system prompt and RAG context."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Inject RAG context if available
        if self._rag_context:
            messages.append({
                "role": "system",
                "content": f"Relevant context:\n{self._rag_context}"
            })
        
        # Add conversation history
        messages.extend(self._context)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    async def generate(
        self,
        user_input: str,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Generate response from Ollama.
        
        Args:
            user_input: User's text input
            stream: Whether to stream the response
            
        Yields:
            Text chunks (if streaming) or complete response
        """
        messages = self._build_messages(user_input)
        
        client = await self.client
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        
        full_response = ""
        
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                import json
                data = json.loads(line)
                
                if "message" in data and "content" in data["message"]:
                    chunk = data["message"]["content"]
                    full_response += chunk
                    if stream:
                        yield chunk
                
                if data.get("done", False):
                    break
        
        # Add to history
        self.add_to_history("user", user_input)
        self.add_to_history("assistant", full_response)
        
        if not stream:
            yield full_response
    
    async def generate_with_vietnamese(
        self,
        user_input: str,
        detected_language: str = "en",
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Generate response with language awareness.
        
        If Vietnamese is detected, instruct model to respond in Vietnamese.
        """
        if detected_language == "vi":
            # Modify system prompt for Vietnamese
            original_prompt = self.system_prompt
            self.system_prompt = f"{self.system_prompt}\nRespond in Vietnamese (Tiếng Việt)."
            
            async for chunk in self.generate(user_input, stream):
                yield chunk
            
            self.system_prompt = original_prompt
        else:
            async for chunk in self.generate(user_input, stream):
                yield chunk
    
    async def check_health(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            client = await self.client
            response = await client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m["name"] == self.model for m in models)
            return False
        except Exception:
            return False
    
    async def pull_model(self) -> bool:
        """Pull the model if not available."""
        try:
            client = await self.client
            response = await client.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
            )
            return response.status_code == 200
        except Exception:
            return False


# Persona prompts matching PersonaPlex style
PERSONA_PROMPTS = {
    "assistant": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    "casual": "You enjoy having a good conversation.",
    "service": "You are a helpful customer service representative. Be professional and courteous.",
    "vietnamese": "Bạn là một trợ lý hữu ích. Trả lời bằng tiếng Việt một cách tự nhiên và thân thiện.",
}
