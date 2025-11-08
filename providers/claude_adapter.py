"""
Claude Adapter
Optional cloud provider for creative reasoning and complex wordplay
"""

import time
import logging
from typing import Optional
from .abstraction_layer import GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


class ClaudeAdapter:
    """Adapter for Anthropic Claude API"""
    
    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get("anthropic_api_key")
        self.model_name = "claude-3-5-sonnet-20241022"  # Latest model
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=self.api_key)
            logger.info("✅ Claude client initialized")
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    async def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using Claude"""
        
        start_time = time.time()
        
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            text = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            latency_ms = (time.time() - start_time) * 1000
            
            return GenerationResult(
                text=text,
                provider="claude",
                model=self.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "stop_reason": response.stop_reason,
                },
            )
            
        except Exception as e:
            logger.error(f"❌ Claude generation failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Claude API is accessible"""
        try:
            # Simple check - could be enhanced with actual API ping
            return bool(self.api_key and self.client)
        except Exception:
            return False
