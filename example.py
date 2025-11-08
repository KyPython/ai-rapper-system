"""Example CLI client for testing the system"""

import asyncio
import json
from providers import LyricEngine, GenerationConfig


async def main():
    """Example usage of the AI Rapper System"""
    
    # Configuration
    config = {
        "primary_provider": "local",
        "offline_mode": False,
        "fallback_to_local": True,
        "local_model_path": "./models/local_lyric_generator.gguf",
        "use_gpu": False,
    }
    
    print("🎤 AI Rapper System - Example Usage\n")
    
    # Initialize engine
    print("Initializing engine...")
    engine = LyricEngine(config)
    
    # Health check
    print("\n📊 System Health:")
    health = engine.health_check()
    print(json.dumps(health, indent=2))
    
    # Example 1: Generate with local model
    print("\n" + "="*60)
    print("Example 1: Local Generation")
    print("="*60)
    
    prompt1 = "Write aggressive battle rap bars about confidence and determination"
    print(f"\nPrompt: {prompt1}")
    print("\nGenerating...")
    
    try:
        result1 = await engine.generate(
            prompt=prompt1,
            provider="local",
            config=GenerationConfig(
                max_tokens=200,
                temperature=0.9,
            )
        )
        
        print(f"\n🎯 Result (from {result1.provider}):")
        print("-" * 60)
        print(result1.text)
        print("-" * 60)
        print(f"Tokens: {result1.tokens_used} | Latency: {result1.latency_ms:.2f}ms")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Evaluate lyrics
    print("\n" + "="*60)
    print("Example 2: Evaluation")
    print("="*60)
    
    from scripts.evaluate import LyricEvaluator
    
    evaluator = LyricEvaluator()
    
    sample_lyrics = """
I'm the definition of what confidence is
Every line I drop, yo, you know it's prestigious
My flow is infectious, my wordplay is genius
Standing at the top, yeah, they all wanna be this
    """.strip()
    
    print("\nEvaluating lyrics...")
    metrics = evaluator.evaluate(sample_lyrics)
    
    print(f"\n📈 Evaluation Results:")
    print(json.dumps(metrics.to_dict(), indent=2))
    
    # Example 3: Sentiment analysis
    print("\n" + "="*60)
    print("Example 3: Sentiment Analysis")
    print("="*60)
    
    from scripts.sentiment_analysis import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    
    print("\nAnalyzing sentiment...")
    sentiment = analyzer.analyze_motivational_content(sample_lyrics)
    
    print(f"\n💭 Sentiment Analysis:")
    print(json.dumps(sentiment, indent=2))
    
    # Example 4: Manual mode
    print("\n" + "="*60)
    print("Example 4: Manual Mode (Skipped in example)")
    print("="*60)
    print("\nIn manual mode, you write completely without AI:")
    print("result = engine.get_manual_input(prompt)")
    print("This bypasses all AI and captures your raw creativity.")
    
    # Example 5: Switch providers
    print("\n" + "="*60)
    print("Example 5: Provider Switching")
    print("="*60)
    
    print("\nAvailable providers:")
    print(engine.get_available_providers())
    
    print(f"\nCurrent primary: {engine.primary_provider}")
    print("\nYou can switch instantly:")
    print("engine.switch_provider('claude')")
    print("engine.switch_provider('local')")
    
    print("\n✅ Examples complete!")
    print("\n📝 Next steps:")
    print("   1. Train your own model with your lyrics")
    print("   2. Add cloud provider API keys for ensemble mode")
    print("   3. Use manual mode daily to maintain your skills")
    print("   4. Track your progress in manual/writing_practice.md")


if __name__ == "__main__":
    asyncio.run(main())
