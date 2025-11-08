"""
Provider Abstraction Layer
Universal API wrapper for ANY LLM provider - local or cloud
"""

from typing import Optional, Dict, Any, List
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported provider types"""
    LOCAL = "local"
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA = "ollama"
    MANUAL = "manual"


class GenerationConfig:
    """Universal configuration for text generation"""
    def __init__(
        self,
        max_tokens: int = 512,
        temperature: float = 0.9,
        top_p: float = 0.95,
        frequency_penalty: float = 0.3,
        presence_penalty: float = 0.3,
        stop_sequences: Optional[List[str]] = None,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop_sequences = stop_sequences or []


class GenerationResult:
    """Standardized result from any provider"""
    def __init__(
        self,
        text: str,
        provider: str,
        model: str,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.provider = provider
        self.model = model
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class LyricEngine:
    """
    Universal lyric generation engine
    Works with ANY provider or none at all (manual mode)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.primary_provider = config.get("primary_provider", "local")
        self.fallback_enabled = config.get("fallback_to_local", True)
        self.offline_mode = config.get("offline_mode", False)
        
        # Initialize providers
        self._init_providers()
        
        logger.info(f"LyricEngine initialized with primary provider: {self.primary_provider}")
        logger.info(f"Available providers: {list(self.providers.keys())}")

    def _init_providers(self):
        """Initialize all enabled providers"""
        from .local_adapter import LocalAdapter
        
        # Local is ALWAYS available (primary system)
        try:
            self.providers["local"] = LocalAdapter(self.config)
            logger.info("✅ Local provider initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize local provider: {e}")
            raise RuntimeError("Local provider is required but failed to initialize")
        
        # Cloud providers (optional)
        if not self.offline_mode:
            self._init_cloud_providers()

    def _init_cloud_providers(self):
        """Initialize cloud providers if enabled and configured"""
        # Claude
        if self.config.get("enable_claude", False) and self.config.get("anthropic_api_key"):
            try:
                from .claude_adapter import ClaudeAdapter
                self.providers["claude"] = ClaudeAdapter(self.config)
                logger.info("✅ Claude provider initialized")
            except Exception as e:
                logger.warning(f"⚠️  Claude provider failed: {e}")
        
        # OpenAI
        if self.config.get("enable_openai", False) and self.config.get("openai_api_key"):
            try:
                from .openai_adapter import OpenAIAdapter
                self.providers["openai"] = OpenAIAdapter(self.config)
                logger.info("✅ OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"⚠️  OpenAI provider failed: {e}")
        
        # Gemini
        if self.config.get("enable_gemini", False) and self.config.get("google_api_key"):
            try:
                from .gemini_adapter import GeminiAdapter
                self.providers["gemini"] = GeminiAdapter(self.config)
                logger.info("✅ Gemini provider initialized")
            except Exception as e:
                logger.warning(f"⚠️  Gemini provider failed: {e}")
        
        # Groq
        if self.config.get("enable_groq", False) and self.config.get("groq_api_key"):
            try:
                from .groq_adapter import GroqAdapter
                self.providers["groq"] = GroqAdapter(self.config)
                logger.info("✅ Groq provider initialized")
            except Exception as e:
                logger.warning(f"⚠️  Groq provider failed: {e}")

    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate lyrics using specified provider
        
        Args:
            prompt: The generation prompt
            provider: Provider to use (defaults to primary_provider)
            config: Generation configuration
            
        Returns:
            GenerationResult with generated text
        """
        provider = provider or self.primary_provider
        config = config or GenerationConfig()
        
        # Try primary provider
        if provider in self.providers:
            try:
                result = await self.providers[provider].generate(prompt, config)
                logger.info(f"✅ Generation successful via {provider}")
                return result
            except Exception as e:
                logger.error(f"❌ {provider} generation failed: {e}")
                if not self.fallback_enabled:
                    raise
        else:
            logger.warning(f"⚠️  Provider '{provider}' not available")
        
        # Fallback to local if enabled
        if self.fallback_enabled and provider != "local":
            logger.info("🔄 Falling back to local provider")
            try:
                return await self.providers["local"].generate(prompt, config)
            except Exception as e:
                logger.error(f"❌ Local fallback failed: {e}")
                raise
        
        raise RuntimeError(f"No available provider could generate lyrics")

    async def generate_ensemble(
        self,
        prompt: str,
        providers: Optional[List[str]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> List[GenerationResult]:
        """
        Generate lyrics from multiple providers simultaneously
        Useful for comparing outputs and selecting best lines
        
        Args:
            prompt: The generation prompt
            providers: List of providers to use (defaults to all available)
            config: Generation configuration
            
        Returns:
            List of GenerationResults from each provider
        """
        providers = providers or list(self.providers.keys())
        config = config or GenerationConfig()
        
        results = []
        for provider in providers:
            if provider in self.providers:
                try:
                    result = await self.generate(prompt, provider, config)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"⚠️  Ensemble generation failed for {provider}: {e}")
        
        if not results:
            raise RuntimeError("No providers succeeded in ensemble generation")
        
        logger.info(f"✅ Ensemble generation complete: {len(results)}/{len(providers)} providers succeeded")
        return results

    def get_manual_input(self, prompt: str) -> GenerationResult:
        """
        Manual mode - YOU write the lyrics
        AI is completely bypassed
        """
        print("\n" + "="*60)
        print("MANUAL MODE - Write your own lyrics")
        print("="*60)
        print(f"\nPrompt: {prompt}\n")
        print("Enter your lyrics (press Ctrl+D or Ctrl+Z when done):")
        print("-"*60)
        
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        text = "\n".join(lines)
        
        return GenerationResult(
            text=text,
            provider="manual",
            model="human",
            tokens_used=len(text.split()),
            latency_ms=0.0,
            metadata={"source": "manual_input"},
        )

    def health_check(self) -> Dict[str, Any]:
        """Check health status of all providers"""
        status = {
            "primary_provider": self.primary_provider,
            "offline_mode": self.offline_mode,
            "fallback_enabled": self.fallback_enabled,
            "providers": {}
        }
        
        for name, provider in self.providers.items():
            try:
                is_healthy = provider.health_check()
                status["providers"][name] = {
                    "available": True,
                    "healthy": is_healthy,
                    "model": provider.model_name
                }
            except Exception as e:
                status["providers"][name] = {
                    "available": False,
                    "healthy": False,
                    "error": str(e)
                }
        
        return status

    def switch_provider(self, provider: str) -> bool:
        """
        Switch primary provider in under 5 seconds
        
        Args:
            provider: New primary provider
            
        Returns:
            True if switch successful
        """
        if provider not in self.providers:
            logger.error(f"❌ Provider '{provider}' not available")
            return False
        
        old_provider = self.primary_provider
        self.primary_provider = provider
        logger.info(f"🔄 Switched from {old_provider} → {provider}")
        return True

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
