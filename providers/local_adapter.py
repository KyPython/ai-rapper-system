"""
Local Model Adapter
PRIMARY SYSTEM - Always available, works offline
Supports GGUF models (llama.cpp) and Hugging Face Transformers
"""

import time
import logging
from typing import Optional
from .abstraction_layer import GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


class LocalAdapter:
    """
    Adapter for local model inference
    Supports both GGUF (quantized) and HF Transformers models
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get("local_model_path", "./models/local_lyric_generator.gguf")
        self.use_gpu = config.get("use_gpu", False)
        self.lazy_load = config.get("lazy_load", True)  # Load on first request, not at startup
        self.model = None
        self.model_type = None
        self.model_name = "local"
        self._model_loaded = False

        if not self.lazy_load:
            self._load_model()
    
    def _load_model(self):
        """Load local model - try GGUF first, fallback to HF Transformers"""
        
        # Try loading GGUF model (llama-cpp-python)
        if self.model_path.endswith(".gguf"):
            try:
                from llama_cpp import Llama
                
                logger.info(f"Loading GGUF model from {self.model_path}")
                
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,  # Context window
                    n_gpu_layers=-1 if self.use_gpu else 0,  # Use GPU if available
                    verbose=False,
                )
                
                self.model_type = "gguf"
                self.model_name = f"local-gguf ({self.model_path.split('/')[-1]})"
                logger.info(f"✅ GGUF model loaded successfully")
                return
                
            except Exception as e:
                logger.warning(f"⚠️  Could not load GGUF model: {e}")
        
        # Fallback to Hugging Face Transformers
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading HF Transformers model from {self.model_path}")
            
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,  # Optimize memory usage during model loading
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.model_type = "transformers"
            self.model_name = f"local-hf ({self.model_path.split('/')[-1]})"
            logger.info(f"✅ HF Transformers model loaded on {device}")
            return
            
        except Exception as e:
            logger.error(f"❌ Could not load HF Transformers model: {e}")

            # Check if we should allow dummy mode (only for explicit testing)
            allow_dummy = self.config.get("allow_dummy_mode", False)

            if allow_dummy:
                logger.warning("⚠️  No model loaded - using dummy mode for development")
                self.model_type = "dummy"
                self.model_name = "local-dummy (no model)"
            else:
                raise RuntimeError(
                    f"Failed to load local model from {self.model_path}. "
                    f"Tried both GGUF and HuggingFace Transformers formats. "
                    f"Please ensure the model exists and is in the correct format. "
                    f"Set 'allow_dummy_mode': True in config to use dummy mode for testing."
                )
    
    async def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using local model"""

        # Lazy load model on first request if not already loaded
        if self.lazy_load and not self._model_loaded:
            logger.info("🔄 Lazy loading model on first request...")
            self._load_model()
            self._model_loaded = True

        start_time = time.time()

        if self.model_type == "gguf":
            text = await self._generate_gguf(prompt, config)
        elif self.model_type == "transformers":
            text = await self._generate_transformers(prompt, config)
        else:
            text = await self._generate_dummy(prompt, config)
        
        latency_ms = (time.time() - start_time) * 1000
        tokens_used = len(text.split())
        
        return GenerationResult(
            text=text,
            provider="local",
            model=self.model_name,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            metadata={"model_type": self.model_type, "offline": True},
        )
    
    async def _generate_gguf(self, prompt: str, config: GenerationConfig) -> str:
        """Generate using llama.cpp GGUF model"""
        import asyncio

        def _sync_generate():
            response = self.model(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repeat_penalty=1.0 + config.frequency_penalty,
                stop=config.stop_sequences,
                echo=False,
            )
            return response["choices"][0]["text"].strip()

        # Run CPU-bound work in thread pool to avoid blocking event loop
        return await asyncio.to_thread(_sync_generate)
    
    async def _generate_transformers(self, prompt: str, config: GenerationConfig) -> str:
        """Generate using Hugging Face Transformers"""
        import torch
        import asyncio

        def _sync_generate():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=1.0 + config.frequency_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated part (not the prompt)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        # Run CPU-bound work in thread pool to avoid blocking event loop
        return await asyncio.to_thread(_sync_generate)
    
    async def _generate_dummy(self, prompt: str, config: GenerationConfig) -> str:
        """Dummy generation for development/testing"""
        
        return f"""[DUMMY MODE - No model loaded]

Prompt: {prompt[:100]}...

This is placeholder text. To enable real generation:
1. Train a model using scripts/train_local.py
2. Download a pre-trained GGUF model
3. Configure the model path in .env

For now, write your lyrics manually using manual mode!
"""
    
    def health_check(self) -> bool:
        """Check if local model is ready"""
        return self.model is not None or self.model_type == "dummy"
