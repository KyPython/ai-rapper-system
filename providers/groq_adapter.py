"""
Groq Adapter
Optional cloud provider for fast inference
"""

import time
import logging
from typing import Optional
from .abstraction_layer import GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


class GroqAdapter:
    """Adapter for Groq API (ultra-fast inference)"""
    
    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get("groq_api_key")
        self.model_name = "llama-3.1-70b-versatile"  # Fast and capable
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not configured")
        
        try:
            from groq import AsyncGroq
            self.client = AsyncGroq(api_key=self.api_key)
            logger.info("✅ Groq client initialized")
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")
    
    async def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using Groq"""
        
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences if config.stop_sequences else None,
            )
            
            text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            latency_ms = (time.time() - start_time) * 1000
            
            return GenerationResult(
                text=text,
                provider="groq",
                model=self.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                metadata={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason,
                },
            )
            
        except Exception as e:
            logger.error(f"❌ Groq generation failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Groq API is accessible"""
        try:
            return bool(self.api_key and self.client)
        except Exception:
            return False
