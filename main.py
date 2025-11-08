"""
FastAPI Backend - Main Application
Provider-agnostic API with health checks and routing
"""

import os
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from providers import LyricEngine, GenerationConfig
from scripts.evaluate import LyricEvaluator, compare_lyrics
from scripts.sentiment_analysis import SentimentAnalyzer, EthosDataManager
from scripts.database import Database

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global instances
engine = None
evaluator = None
sentiment_analyzer = None
ethos_manager = None
database = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management"""
    global engine, evaluator, sentiment_analyzer, ethos_manager, database
    
    # Startup
    logger.info("🚀 Starting AI Rapper System...")
    
    # Load configuration
    config = {
        "primary_provider": os.getenv("PRIMARY_PROVIDER", "local"),
        "offline_mode": os.getenv("OFFLINE_MODE", "false").lower() == "true",
        "fallback_to_local": os.getenv("FALLBACK_TO_LOCAL", "true").lower() == "true",
        "local_model_path": os.getenv("LOCAL_MODEL_PATH", "./models/local_lyric_generator.gguf"),
        "use_gpu": os.getenv("USE_GPU", "false").lower() == "true",
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
    
    # Initialize components
    try:
        engine = LyricEngine(config)
        evaluator = LyricEvaluator(database_path=os.getenv("DATABASE_PATH", "./data/metrics.db"))
        sentiment_analyzer = SentimentAnalyzer()
        ethos_manager = EthosDataManager()
        database = Database(db_path=os.getenv("DATABASE_PATH", "./data/metrics.db"))
        logger.info("✅ All components initialized")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down AI Rapper System...")


# Create FastAPI app
app = FastAPI(
    title="AI Rapper System",
    description="AI-agnostic rapper system with local-first architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The generation prompt")
    provider: Optional[str] = Field(None, description="Specific provider to use")
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.9, ge=0.0, le=2.0)
    evaluate: bool = Field(True, description="Whether to evaluate the output")


class EnsembleRequest(BaseModel):
    prompt: str = Field(..., description="The generation prompt")
    providers: Optional[List[str]] = Field(None, description="Providers to use")
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.9, ge=0.0, le=2.0)


class SentimentRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    motivational: bool = Field(False, description="Include motivational analysis")


class EthosContentRequest(BaseModel):
    content: str = Field(..., description="Motivational content")
    category: str = Field(..., description="Category (motivational_quotes, battle_phrases, confidence_builders)")
    tags: Optional[List[str]] = Field(None)


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# API Endpoints
@app.get("/")
async def root():
    """Serve the web interface"""
    return FileResponse("static/index.html")


@app.get("/api")
async def api_root():
    """API status endpoint"""
    return {
        "message": "AI Rapper System - Provider-Agnostic Lyric Generation",
        "version": "1.0.0",
        "status": "online",
    }


@app.get("/health")
async def health_check():
    """Health check for all providers"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return engine.health_check()


@app.get("/providers")
async def list_providers():
    """List available providers"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {
        "primary": engine.primary_provider,
        "available": engine.get_available_providers(),
        "offline_mode": engine.offline_mode,
        "fallback_enabled": engine.fallback_enabled,
    }


@app.post("/providers/switch")
async def switch_provider(provider: str):
    """Switch primary provider"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    success = engine.switch_provider(provider)
    if not success:
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' not available")
    
    return {
        "message": f"Switched to {provider}",
        "primary_provider": engine.primary_provider,
    }


@app.post("/generate")
async def generate_lyrics(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate lyrics using specified provider"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Create generation config
        config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        # Generate
        result = await engine.generate(
            prompt=request.prompt,
            provider=request.provider,
            config=config,
        )
        
        # Save to database
        generation_id = database.save_generation(
            prompt=request.prompt,
            lyrics=result.text,
            provider=result.provider,
            model=result.model,
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms,
            metadata=result.metadata,
        )
        
        # Update provider usage
        database.update_provider_usage(
            provider=result.provider,
            model=result.model,
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms,
            success=True,
        )
        
        # Evaluate if requested
        evaluation = None
        if request.evaluate and evaluator:
            metrics = evaluator.evaluate(result.text)
            evaluation = metrics.to_dict()
            
            # Save evaluation
            background_tasks.add_task(database.save_evaluation, generation_id, evaluation)
        
        return {
            "generation_id": generation_id,
            "lyrics": result.text,
            "provider": result.provider,
            "model": result.model,
            "tokens_used": result.tokens_used,
            "latency_ms": result.latency_ms,
            "evaluation": evaluation,
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/ensemble")
async def generate_ensemble(request: EnsembleRequest):
    """Generate lyrics from multiple providers and compare"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        results = await engine.generate_ensemble(
            prompt=request.prompt,
            providers=request.providers,
            config=config,
        )
        
        # Compare results
        lyrics_list = [r.text for r in results]
        comparison = compare_lyrics(lyrics_list, evaluator)
        
        # Format response
        return {
            "generations": [
                {
                    "lyrics": r.text,
                    "provider": r.provider,
                    "model": r.model,
                    "tokens_used": r.tokens_used,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
            "comparison": comparison,
        }
        
    except Exception as e:
        logger.error(f"Ensemble generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate_lyrics(lyrics: str):
    """Evaluate existing lyrics"""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    try:
        metrics = evaluator.evaluate(lyrics)
        return metrics.to_dict()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of text"""
    if not sentiment_analyzer:
        raise HTTPException(status_code=503, detail="Sentiment analyzer not initialized")
    
    try:
        if request.motivational:
            return sentiment_analyzer.analyze_motivational_content(request.text)
        else:
            return sentiment_analyzer.analyze(request.text)
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ethos/add")
async def add_ethos_content(request: EthosContentRequest):
    """Add motivational content to ethos database"""
    if not ethos_manager:
        raise HTTPException(status_code=503, detail="Ethos manager not initialized")
    
    try:
        ethos_manager.add_motivational_content(
            content=request.content,
            category=request.category,
            tags=request.tags,
        )
        return {"message": "Content added successfully"}
    except Exception as e:
        logger.error(f"Failed to add ethos content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ethos/prompt")
async def get_ethos_prompt(mood: Optional[str] = None):
    """Get motivational prompt for enhancement"""
    if not ethos_manager:
        raise HTTPException(status_code=503, detail="Ethos manager not initialized")
    
    return {"prompt": ethos_manager.get_motivational_prompt(mood)}


@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    if not database:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    stats = database.get_statistics()
    
    if ethos_manager:
        stats["ethos"] = ethos_manager.get_statistics()
    
    return stats


@app.get("/generations/recent")
async def get_recent_generations(limit: int = 10):
    """Get recent generations"""
    if not database:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    return database.get_recent_generations(limit)


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
