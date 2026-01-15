"""
FastAPI Server with Dark Connecticut Music Framework Enforcement
Endpoints for lyric generation with framework validation
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
import asyncio

from providers.abstraction_layer import (
    LyricEngine, 
    GenerationConfig, 
    FrameworkValidator,
    DARK_CT_FRAMEWORK
)

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Dark Connecticut AI Rapper System",
    description="Framework-enforced lyric generation with multi-provider support",
    version="1.0.0"
)

# Initialize LyricEngine
config = {
    "primary_provider": os.getenv("PRIMARY_PROVIDER", "local"),
    "local_model_path": os.getenv("LOCAL_MODEL_PATH", "./models/my_model.gguf"),
    "fallback_to_local": os.getenv("FALLBACK_TO_LOCAL", "true").lower() == "true",
    "offline_mode": os.getenv("OFFLINE_MODE", "false").lower() == "true",
    "enforce_framework": os.getenv("ENFORCE_FRAMEWORK", "true").lower() == "true",
    
    # Cloud providers
    "enable_claude": os.getenv("ENABLE_CLAUDE", "false").lower() == "true",
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    
    "enable_openai": os.getenv("ENABLE_OPENAI", "false").lower() == "true",
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    
    "enable_gemini": os.getenv("ENABLE_GEMINI", "false").lower() == "true",
    "google_api_key": os.getenv("GOOGLE_API_KEY"),
    
    "enable_groq": os.getenv("ENABLE_GROQ", "false").lower() == "true",
    "groq_api_key": os.getenv("GROQ_API_KEY"),
}

engine = LyricEngine(config)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class GenerateRequest(BaseModel):
    """Request for lyric generation"""
    prompt: str
    provider: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.9
    enforce_framework: Optional[bool] = True


class GenerateResponse(BaseModel):
    """Response with generated lyrics"""
    lyrics: str
    provider: str
    model: str
    framework_compliant: bool
    framework_violations: List[str]
    tokens_used: int
    latency_ms: float


class EnsembleRequest(BaseModel):
    """Request for multi-provider ensemble generation"""
    prompt: str
    providers: Optional[List[str]] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.9
    enforce_framework: Optional[bool] = True
    return_best_only: Optional[bool] = False


class ValidateRequest(BaseModel):
    """Request to validate existing lyrics"""
    lyrics: str


class ValidateResponse(BaseModel):
    """Validation results"""
    compliant: bool
    violations: List[str]
    score: float


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "name": "Dark Connecticut AI Rapper System",
        "version": "1.0.0",
        "framework_enforced": True,
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "generate_ensemble": "/generate/ensemble",
            "validate": "/validate",
            "framework": "/framework",
            "providers": "/providers",
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "engine": engine.health_check()
    }


@app.get("/framework")
async def get_framework():
    """Get the Dark Connecticut Music Framework"""
    return {
        "framework": engine.get_framework(),
        "enforcement_enabled": engine.enforce_framework
    }


@app.post("/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate lyrics with Dark CT Framework enforcement
    
    Example:
    {
        "prompt": "Write a hook about breaking the family curse through sacrifice",
        "provider": "local",
        "enforce_framework": true
    }
    """
    try:
        config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            enforce_framework=request.enforce_framework
        )
        
        result = await engine.generate(request.prompt, request.provider, config)
        
        return GenerateResponse(
            lyrics=result.text,
            provider=result.provider,
            model=result.model,
            framework_compliant=result.framework_compliant,
            framework_violations=result.framework_violations,
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/ensemble")
async def generate_ensemble(request: EnsembleRequest) -> dict:
    """
    Generate from multiple providers simultaneously
    All outputs framework-validated and sorted by compliance
    
    Example:
    {
        "prompt": "Write Verse 1 about the Merritt Parkway commute",
        "providers": ["local", "claude", "openai"],
        "enforce_framework": true,
        "return_best_only": false
    }
    """
    try:
        config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            enforce_framework=request.enforce_framework
        )
        
        results = await engine.generate_ensemble(request.prompt, request.providers, config)
        
        # Format responses
        formatted_results = []
        for result in results:
            formatted_results.append({
                "lyrics": result.text,
                "provider": result.provider,
                "model": result.model,
                "framework_compliant": result.framework_compliant,
                "framework_violations": result.framework_violations,
                "tokens_used": result.tokens_used,
                "latency_ms": result.latency_ms,
            })
        
        if request.return_best_only and formatted_results:
            return {
                "best_result": formatted_results[0],
                "total_attempts": len(formatted_results),
                "framework_compliant_count": sum(1 for r in formatted_results if r["framework_compliant"])
            }
        
        return {
            "results": formatted_results,
            "total_attempts": len(formatted_results),
            "framework_compliant_count": sum(1 for r in formatted_results if r["framework_compliant"]),
            "best_result": formatted_results[0] if formatted_results else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def validate(request: ValidateRequest) -> ValidateResponse:
    """
    Validate existing lyrics against Dark CT Framework
    
    Example:
    {
        "lyrics": "[Hook]\nCruising up the Merritt, same road, same scene..."
    }
    """
    try:
        compliant, violations = FrameworkValidator.validate(request.lyrics)
        
        # Calculate compliance score (0-100)
        score = 100 if not violations else max(0, 100 - (len(violations) * 15))
        
        return ValidateResponse(
            compliant=compliant,
            violations=violations,
            score=score
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/providers")
async def list_providers():
    """List available providers"""
    return {
        "available": engine.get_available_providers(),
        "primary": engine.primary_provider,
        "details": engine.health_check()["providers"]
    }


@app.post("/providers/switch")
async def switch_provider(provider: str):
    """Switch primary provider"""
    success = engine.switch_provider(provider)
    
    if not success:
        raise HTTPException(
            status_code=400, 
            detail=f"Provider '{provider}' not available"
        )
    
    return {
        "status": "switched",
        "new_primary": engine.primary_provider,
        "available": engine.get_available_providers()
    }


@app.post("/framework/toggle")
async def toggle_framework(enabled: bool):
    """Enable/disable framework enforcement globally"""
    engine.set_framework_enforcement(enabled)
    
    return {
        "framework_enforcement": engine.enforce_framework,
        "status": "updated"
    }


@app.get("/framework/checklist")
async def get_framework_checklist():
    """Get framework compliance checklist"""
    return {
        "checklist": [
            "✓ Does this serve 'defiant hope' thesis?",
            "✓ Minor key harmony implied (or compatible)?",
            "✓ Specific personal details (Bridgeport-grounded)?",
            "✓ NO toxic/negative framing?",
            "✓ Authentic voice (no fake dialogue)?",
            "✓ Flow changes where narrative breaks?",
            "✓ Ends on IV or VI (cadential ambiguity)?",
            "✓ Could realistically come from this person's life?"
        ],
        "sonic_signature": {
            "bpm": "135-150 (default 140)",
            "key": "Minor (struggle focus)",
            "drums": "Grimy 808s, muffled kicks (250-400 Hz), double-tap snare, swung hi-hats",
            "melody": "Dark melodic loops (minor key), NO bright synths",
            "tone": "Anthem for survival (NOT dance/celebration)"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_type": type(exc).__name__}
    )


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup info"""
    print("\n" + "="*60)
    print("Dark Connecticut AI Rapper System")
    print("="*60)
    print(f"Primary Provider: {engine.primary_provider}")
    print(f"Framework Enforcement: {engine.enforce_framework}")
    print(f"Available Providers: {engine.get_available_providers()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
