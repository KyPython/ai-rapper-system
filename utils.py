"""Utility script for common tasks"""

import os
import sys
from pathlib import Path


def create_data_dirs():
    """Create necessary data directories"""
    dirs = [
        "./data",
        "./models",
        "./manual",
        "./logs",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}")


def check_dependencies():
    """Check if required packages are installed"""
    required = [
        "fastapi",
        "uvicorn",
        "transformers",
        "nltk",
        "vaderSentiment",
        "textblob",
        "pronouncing",
        "sqlalchemy",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def download_nltk_data():
    """Download required NLTK data"""
    import nltk
    
    packages = [
        'punkt',
        'cmudict',
        'averaged_perceptron_tagger',
        'words',
    ]
    
    for package in packages:
        try:
            nltk.download(package, quiet=True)
            print(f"✅ Downloaded {package}")
        except Exception as e:
            print(f"⚠️  {package}: {e}")


def initialize_database():
    """Initialize SQLite database"""
    from scripts.database import Database
    
    db = Database("./data/metrics.db")
    print("✅ Database initialized at ./data/metrics.db")


def create_example_training_data():
    """Create example training data file"""
    import json
    
    data = {
        "metadata": {
            "description": "Example training data for AI Rapper System",
            "created": "2025-11-07"
        },
        "training_data": [
            {
                "prompt": "Write aggressive battle rap bars",
                "lyrics": "I'm the definition of determination\nEvery line I spit is a new revelation\nYou can't match my flow or my dedication\nI'm building my empire, no hesitation"
            },
            {
                "prompt": "Write motivational bars",
                "lyrics": "Started from the bottom, now I'm reaching for the peak\nEvery setback made me stronger, never weak\nI turn my pain into power, that's my technique\nThe future that I'm building is unique"
            },
        ]
    }
    
    with open("./data/training_lyrics.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print("✅ Example training data created at ./data/training_lyrics.json")


def run_health_check():
    """Run system health check"""
    print("🔍 Running system health check...\n")
    
    # Check directories
    print("📁 Checking directories...")
    for dir_path in ["./data", "./models", "./manual"]:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
    
    # Check files
    print("\n📄 Checking configuration files...")
    for file_path in [".env", "./data/metrics.db"]:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} - Not found (may need setup)")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    check_dependencies()
    
    print("\n✅ Health check complete!")


def setup_system():
    """Complete system setup"""
    print("🚀 Setting up AI Rapper System...\n")
    
    print("1️⃣ Creating directories...")
    create_data_dirs()
    
    print("\n2️⃣ Checking dependencies...")
    if not check_dependencies():
        print("\n⚠️  Please install dependencies first:")
        print("   pip install -r requirements.txt")
        return
    
    print("\n3️⃣ Downloading NLTK data...")
    download_nltk_data()
    
    print("\n4️⃣ Initializing database...")
    initialize_database()
    
    print("\n5️⃣ Creating example training data...")
    create_example_training_data()
    
    print("\n6️⃣ Checking environment file...")
    if not Path(".env").exists():
        print("  ⚠️  .env not found")
        print("  Copy .env.example to .env and configure:")
        print("     cp .env.example .env")
    else:
        print("  ✅ .env exists")
    
    print("\n✅ Setup complete!")
    print("\n📝 Next steps:")
    print("   1. Configure .env file (if not done)")
    print("   2. Add your lyrics to data/training_lyrics.json")
    print("   3. Train your model: python scripts/train_local.py --action train-gpt2")
    print("   4. Start server: python main.py")


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("AI Rapper System - Utilities")
        print("\nUsage:")
        print("  python utils.py <command>")
        print("\nCommands:")
        print("  setup          - Complete system setup")
        print("  health         - Run health check")
        print("  create-dirs    - Create data directories")
        print("  check-deps     - Check dependencies")
        print("  init-db        - Initialize database")
        print("  example-data   - Create example training data")
        return
    
    command = sys.argv[1]
    
    commands = {
        "setup": setup_system,
        "health": run_health_check,
        "create-dirs": create_data_dirs,
        "check-deps": check_dependencies,
        "init-db": initialize_database,
        "example-data": create_example_training_data,
    }
    
    if command in commands:
        commands[command]()
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python utils.py' for help")


if __name__ == "__main__":
    main()
