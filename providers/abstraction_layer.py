"""
Provider Abstraction Layer
Universal API wrapper for ANY LLM provider - local or cloud
NOW WITH DARK CONNECTICUT MUSIC FRAMEWORK ENFORCEMENT
"""

from typing import Optional, Dict, Any, List
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DARK CONNECTICUT MUSIC FRAMEWORK CONSTITUTION
# ============================================================================

DARK_CT_FRAMEWORK = """
# Dark Connecticut Hip-Hop Music Framework

## CORE THESIS (Non-Negotiable)
Defiant hope in the face of struggle. Every line must serve this.

## SONIC SIGNATURE
- BPM: 135-150 (default 140)
- Key: Minor (struggle focus, not celebration)
- Drums: Grimy 808s, muffled kicks (250-400 Hz low-pass), double-tap snare, swung hi-hats
- Melody: Dark melodic loops (minor key bells/piano), NO bright synths
- Tone: Anthem for survival (NOT dance, NOT happy, NOT celebration)

## SONGWRITING STRUCTURE
Hook: Thematic vow. Conjunct motion (stepwise) + ONE heroic disjunct leap (4th/5th/octave). Resolve to root.
Verse 1: Introduce CHARACTER with TRUTH. Specific details (Bridgeport, family, real experience). Slow flow, conversational.
Verse 2: Show CHALLENGE/BREAKPOINT. Use flow changes (fast→slow, bouncy→rigid) to represent instability.
Bridge: Introspection, vulnerability, build tension.
Outro: RESOLVE with ambiguity. End on IV or VI chord (NOT I). Create "hunger" for next chapter.

## AUTHENTICITY MANDATE (Non-Negotiable)
✓ Every line derives from direct personal experience
✓ Specific place names (Bridgeport, Merritt Parkway, Westport, Connecticut)
✓ Concrete details (family struggles, financial pressure, identity)
✓ Real voice (no fake dialogue, no imagined scenarios, no dramatization)
✗ NO fake tough guy energy
✗ NO invented stories
✗ NO generic hip-hop tropes

## REJECTION MANDATE (Non-Negotiable)
✗ NO negativity, drama, or toxicity
✗ NO promiscuity, drugs, addictions, thuggish behavior
✗ NO dissing others or flexing
✗ NO scum-bagging narratives
INSTEAD: Focus on MOTIVATIONAL content that shows LOVE for culture and others

## FLOW & CADENCE RULES
- Verses: Start slow/conversational, build intensity where emotion shifts
- Hooks: Most memorable, immediately singable, 7-14 syllables at 140 BPM
- Transitions: Use silence, stuttering, or pause to mark emotional shifts
- Endings: Never resolve to tonic (I). Always ambiguous (IV or VI).

## EVALUATION CHECKLIST (Before every generation)
✓ Does this serve "defiant hope" thesis?
✓ Minor key harmony implied (or compatible)?
✓ Specific personal details (Bridgeport-grounded)?
✓ NO toxic/negative framing?
✓ Authentic voice (no fake dialogue)?
✓ Flow changes where narrative breaks?
✓ Ends on IV or VI (cadential ambiguity)?
✓ Could realistically come from this person's life?

## SUNO-READY FORMAT REQUIREMENTS
[Hook]
[Verse 1]
[Verse 2]
[Bridge]
[Outro]

With flow annotations: *(slow flow)*, *(bounce)*, *(rapid-fire)*, *(stretching words)*
Line length: 7-14 syllables per line at 140 BPM
"""

# ============================================================================
# CLASS DEFINITIONS
# ============================================================================


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
        enforce_framework: bool = True,  # NEW: Framework enforcement toggle
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop_sequences = stop_sequences or []
        self.enforce_framework = enforce_framework  # NEW


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
        framework_compliant: bool = False,  # NEW
        framework_violations: Optional[List[str]] = None,  # NEW
    ):
        self.text = text
        self.provider = provider
        self.model = model
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
        self.framework_compliant = framework_compliant  # NEW
        self.framework_violations = framework_violations or []  # NEW

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "framework_compliant": self.framework_compliant,  # NEW
            "framework_violations": self.framework_violations,  # NEW
        }


# ============================================================================
# FRAMEWORK VALIDATOR
# ============================================================================

class FrameworkValidator:
    """
    Validates lyrics against Dark Connecticut Music Framework
    Checks: Authenticity, Defiant Hope, No Toxicity, Structure, etc.
    """
    
    @staticmethod
    def validate(text: str) -> tuple[bool, List[str]]:
        """
        Validate lyrics against framework
        
        Returns:
            (compliant: bool, violations: List[str])
        """
        violations = []
        
        # Check 1: Defiant Hope Thesis
        toxic_words = ["bitch", "hoe", "kill", "die", "drug", "flex", "diss", "hate", "stupid"]
        if any(word.lower() in text.lower() for word in toxic_words):
            violations.append("Contains toxic/negative framing (rejection mandate violated)")
        
        # Check 2: Authenticity - Look for specific details
        ct_locations = ["bridgeport", "merritt", "westport", "connecticut", "ct"]
        personal_indicators = ["i", "me", "my", "our", "we"]
        
        has_specific_details = any(loc in text.lower() for loc in ct_locations)
        has_personal_voice = any(indicator in text.lower() for indicator in personal_indicators)
        
        if not has_personal_voice:
            violations.append("Lacks personal voice/first-person perspective")
        
        # Check 3: Structure (basic check - looking for section markers)
        has_structure = "[Hook]" in text or "[Verse" in text or "[Bridge]" in text
        
        # Check 4: Fake dialogue/dramatization
        fake_dialogue_markers = ['said "', 'told me "', 'they said "', 'he said "', 'she said "']
        if any(marker in text for marker in fake_dialogue_markers):
            violations.append("Contains fake dialogue (authenticity mandate violated)")
        
        # Check 5: Syllable count (basic - should be mostly 7-14 syllables per line)
        lines = [line.strip() for line in text.split('\n') if line.strip() and not line.startswith('[')]
        if lines:
            avg_syllables = sum(len(line.split()) for line in lines) / len(lines)
            if avg_syllables < 5 or avg_syllables > 18:
                violations.append(f"Average line length suspicious ({avg_syllables:.1f} words/line, expected 7-14 syllables)")
        
        # Check 6: Flow changes indicated
        has_flow_annotations = "*(slow flow)*" in text or "*(bounce)*" in text or "*(rapid" in text
        
        # Determine overall compliance
        compliant = len(violations) == 0
        
        if violations:
            logger.warning(f"⚠️  Framework violations detected: {violations}")
        else:
            logger.info("✅ Lyrics framework compliant")
        
        return compliant, violations


# ============================================================================
# LYRIC ENGINE WITH FRAMEWORK ENFORCEMENT
# ============================================================================

class LyricEngine:
    """
    Universal lyric generation engine with Dark CT Framework enforcement
    Works with ANY provider or none at all (manual mode)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.primary_provider = config.get("primary_provider", "local")
        self.fallback_enabled = config.get("fallback_to_local", True)
        self.offline_mode = config.get("offline_mode", False)
        self.enforce_framework = config.get("enforce_framework", True)  # NEW
        
        # Initialize providers
        self._init_providers()
        
        logger.info(f"LyricEngine initialized with primary provider: {self.primary_provider}")
        logger.info(f"Framework enforcement: {'ENABLED' if self.enforce_framework else 'DISABLED'}")
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

    def _get_framework_prompt(self) -> str:
        """Get framework prompt for this generation"""
        return DARK_CT_FRAMEWORK

    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate lyrics using specified provider
        WITH DARK CT FRAMEWORK ENFORCEMENT
        
        Args:
            prompt: The generation prompt
            provider: Provider to use (defaults to primary_provider)
            config: Generation configuration
            
        Returns:
            GenerationResult with generated text + framework compliance
        """
        provider = provider or self.primary_provider
        config = config or GenerationConfig()
        
        # Add framework to prompt if enforcement enabled
        if config.enforce_framework and self.enforce_framework:
            full_prompt = f"""{self._get_framework_prompt()}

---

GENERATION TASK:
{prompt}

Generate lyrics that STRICTLY follow the Dark Connecticut Music Framework above.
Every line must serve the "defiant hope" thesis.
NO toxic content, NO fake dialogue, NO generic tropes.
Authentic voice ONLY.
"""
        else:
            full_prompt = prompt
        
        # Try primary provider
        if provider in self.providers:
            try:
                result = await self.providers[provider].generate(full_prompt, config)
                
                # Validate against framework
                if config.enforce_framework and self.enforce_framework:
                    compliant, violations = FrameworkValidator.validate(result.text)
                    result.framework_compliant = compliant
                    result.framework_violations = violations
                    
                    if not compliant:
                        logger.warning(f"⚠️  Generated text has framework violations: {violations}")
                
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
                return await self.generate(prompt, "local", config)
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
        All outputs validated against Dark CT Framework
        
        Args:
            prompt: The generation prompt
            providers: List of providers to use (defaults to all available)
            config: Generation configuration
            
        Returns:
            List of GenerationResults (framework-validated)
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
        
        # Sort by framework compliance
        results.sort(key=lambda r: (r.framework_compliant, len(r.framework_violations)), reverse=True)
        
        logger.info(f"✅ Ensemble generation complete: {len(results)}/{len(providers)} providers succeeded")
        logger.info(f"Framework compliant: {sum(1 for r in results if r.framework_compliant)}/{len(results)}")
        return results

    def get_manual_input(self, prompt: str) -> GenerationResult:
        """
        Manual mode - YOU write the lyrics
        AI is completely bypassed
        Framework validation still applies
        """
        print("\n" + "="*60)
        print("MANUAL MODE - Write your own lyrics")
        print("="*60)
        print(f"\nPrompt: {prompt}\n")
        print("Framework Requirements:")
        print("- Defiant hope in struggle")
        print("- Authentic personal experience (Bridgeport, real details)")
        print("- NO toxic framing, fake dialogue, or generic tropes")
        print("- Structure: [Hook] → [Verse 1] → [Verse 2] → [Bridge] → [Outro]")
        print("\nEnter your lyrics (press Ctrl+D or Ctrl+Z when done):")
        print("-"*60)
        
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        text = "\n".join(lines)
        
        # Validate even manual input
        compliant, violations = FrameworkValidator.validate(text)
        
        if violations:
            print("\n⚠️  Framework violations detected:")
            for v in violations:
                print(f"  - {v}")
        else:
            print("\n✅ Your lyrics are framework compliant!")
        
        return GenerationResult(
            text=text,
            provider="manual",
            model="human",
            tokens_used=len(text.split()),
            latency_ms=0.0,
            metadata={"source": "manual_input"},
            framework_compliant=compliant,
            framework_violations=violations,
        )

    def health_check(self) -> Dict[str, Any]:
        """Check health status of all providers"""
        status = {
            "primary_provider": self.primary_provider,
            "offline_mode": self.offline_mode,
            "fallback_enabled": self.fallback_enabled,
            "framework_enforcement": self.enforce_framework,  # NEW
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

    def get_framework(self) -> str:
        """Get the Dark CT framework constitution"""
        return DARK_CT_FRAMEWORK

    def set_framework_enforcement(self, enabled: bool) -> None:
        """Toggle framework enforcement globally"""
        self.enforce_framework = enabled
        logger.info(f"Framework enforcement: {'ENABLED' if enabled else 'DISABLED'}")
