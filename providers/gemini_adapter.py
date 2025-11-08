"""
Google Gemini Adapter
Optional cloud provider for another alternative
"""

import time
import logging
from typing import Optional
from .abstraction_layer import GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


class GeminiAdapter:
    """Adapter for Google Gemini API"""
    
    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get("google_api_key")
        self.model_name = "gemini-1.5-flash"  # Fast and free tier available
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not configured")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("✅ Gemini client initialized")
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    async def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using Gemini"""
        
        start_time = time.time()
        
        try:
            generation_config = {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_output_tokens": config.max_tokens,
                "stop_sequences": config.stop_sequences if config.stop_sequences else None,
            }
            
            response = await self.model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            
            text = response.text
            # Gemini doesn't always provide token counts in free tier
            tokens_used = len(text.split())  # Approximate
            latency_ms = (time.time() - start_time) * 1000
            
            return GenerationResult(
                text=text,
                provider="gemini",
                model=self.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                metadata={
                    "candidates": len(response.candidates),
                    "finish_reason": str(response.candidates[0].finish_reason) if response.candidates else None,
                },
            )
            
        except Exception as e:
            logger.error(f"❌ Gemini generation failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Gemini API is accessible"""
        try:
            return bool(self.api_key and self.model)
        except Exception:
            return False
