"""
OpenAI Adapter
Optional cloud provider for alternative perspective
"""

import time
import logging
from typing import Optional
from .abstraction_layer import GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """Adapter for OpenAI API (ChatGPT)"""
    
    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get("openai_api_key")
        self.model_name = "gpt-4o-mini"  # Cost-effective model
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info("✅ OpenAI client initialized")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    async def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using OpenAI"""
        
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
                provider="openai",
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
            logger.error(f"❌ OpenAI generation failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            return bool(self.api_key and self.client)
        except Exception:
            return False
