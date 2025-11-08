"""
Quick test script to verify installation
Run this after setup to ensure everything works
"""

import sys


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_imports():
    """Check if required packages are installed"""
    print("\nChecking package imports...")
    
    packages = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "pydantic": "Pydantic",
        "dotenv": "python-dotenv",
        "nltk": "NLTK",
        "vaderSentiment": "VADER",
        "textblob": "TextBlob",
        "pronouncing": "Pronouncing",
        "sqlalchemy": "SQLAlchemy",
    }
    
    failed = []
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_directories():
    """Check if required directories exist"""
    print("\nChecking directories...")
    
    import os
    from pathlib import Path
    
    dirs = ["./data", "./models", "./manual", "./providers", "./scripts"]
    
    failed = []
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path}")
            failed.append(dir_path)
    
    if failed:
        print(f"\n⚠️  Missing directories: {', '.join(failed)}")
        print("Run: python utils.py create-dirs")
        return False
    
    return True


def check_database():
    """Check if database can be initialized"""
    print("\nChecking database...")
    
    try:
        from scripts.database import Database
        db = Database("./data/test.db")
        
        import os
        os.remove("./data/test.db")
        
        print("✅ Database initialization works")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


def check_providers():
    """Check if provider system can be imported"""
    print("\nChecking provider system...")
    
    try:
        from providers import LyricEngine, GenerationConfig
        print("✅ Provider system imports correctly")
        return True
    except Exception as e:
        print(f"❌ Provider import error: {e}")
        return False


def check_evaluation():
    """Check if evaluation system works"""
    print("\nChecking evaluation system...")
    
    try:
        from scripts.evaluate import LyricEvaluator
        evaluator = LyricEvaluator()
        
        test_lyrics = "Test line one, having fun\nTest line two, nothing new"
        metrics = evaluator.evaluate(test_lyrics)
        
        print(f"✅ Evaluation system works (score: {metrics.overall_score():.2f})")
        return True
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False


def check_sentiment():
    """Check if sentiment analysis works"""
    print("\nChecking sentiment analysis...")
    
    try:
        from scripts.sentiment_analysis import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        
        result = analyzer.analyze("I'm confident and strong")
        
        print(f"✅ Sentiment analysis works")
        return True
    except Exception as e:
        print(f"❌ Sentiment analysis error: {e}")
        return False


def test_basic_generation():
    """Test basic generation (dummy mode)"""
    print("\nTesting basic generation...")
    
    try:
        import asyncio
        from providers import LyricEngine, GenerationConfig
        
        config = {
            "primary_provider": "local",
            "local_model_path": "./models/dummy.gguf",  # Won't exist, will use dummy
            "offline_mode": True,
        }
        
        engine = LyricEngine(config)
        
        async def test():
            result = await engine.generate(
                "Test prompt",
                config=GenerationConfig(max_tokens=50)
            )
            return result
        
        result = asyncio.run(test())
        print(f"✅ Generation system works (dummy mode)")
        return True
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("AI Rapper System - Installation Test")
    print("="*60)
    print()
    
    tests = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("Directory Structure", check_directories),
        ("Database System", check_database),
        ("Provider System", check_providers),
        ("Evaluation System", check_evaluation),
        ("Sentiment Analysis", check_sentiment),
        ("Generation System", test_basic_generation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Configure .env file")
        print("  2. Start server: python main.py")
        print("  3. Visit: http://localhost:8000/docs")
    else:
        print("\n⚠️  Some tests failed. Please fix issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Run setup: python utils.py setup")
        print("  - Check Python version: python --version")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
