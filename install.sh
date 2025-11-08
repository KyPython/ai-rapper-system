#!/bin/bash

# AI Rapper System - Quick Installation Script
# Run this to set up everything in one go

set -e  # Exit on error

echo "🎤 AI Rapper System - Installation"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 8 ]); then
    echo "❌ Python 3.8+ required. You have Python $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✅ Pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('cmudict', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"
echo "✅ NLTK data downloaded"
echo ""

# Run setup utility
echo "Running system setup..."
python utils.py setup
echo "✅ System setup complete"
echo ""

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✅ .env created (please configure your API keys)"
else
    echo "⚠️  .env already exists (skipping)"
fi
echo ""

# Final message
echo "=================================="
echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Configure .env file with your settings"
echo "   2. Add your lyrics to data/training_lyrics.json"
echo "   3. Start the server: python main.py"
echo ""
echo "📚 Documentation:"
echo "   - README.md: Complete documentation"
echo "   - QUICKSTART.md: Quick start guide"
echo "   - PROJECT_SUMMARY.md: System overview"
echo ""
echo "🚀 Quick commands:"
echo "   python main.py           # Start server"
echo "   python example.py        # Run examples"
echo "   python utils.py health   # System check"
echo ""
echo "🎤 Happy creating!"
