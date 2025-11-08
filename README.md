# AI Rapper System

**Provider-Agnostic Lyric Generation with Local-First Architecture**

> Your voice. Your skills. AI as a tool, never a crutch.

---

## 🎯 Core Philosophy

**TRUE INDEPENDENCE:** Quality lyrics even if every AI company shuts down tomorrow.

- **Local model is PRIMARY** - always available, works offline
- **Cloud AIs are OPTIONAL** - use any or none
- **Manual skills are NON-NEGOTIABLE** - 30min daily practice
- **Zero vendor lock-in** - switch providers in under 5 seconds
- **$0 recurring costs** - everything can run completely free forever

---

## ✨ Features

### Provider Abstraction

- ✅ Universal API wrapper for ANY LLM provider
- ✅ Seamlessly switch between local, Claude, ChatGPT, Gemini, Groq
- ✅ Automatic fallback to local if cloud providers fail
- ✅ Full offline capability

### Local Model System (PRIMARY)

- ✅ Train on Google Colab free tier (GPT-2, Phi-3)
- ✅ Run on CPU with quantized GGUF models (1-2GB)
- ✅ Complete lyric generation without internet
- ✅ Your unique style, trained on your data

### Evaluation System (LBCM)

- ✅ Rhyme density and complexity analysis
- ✅ Syllable consistency scoring
- ✅ Sentiment analysis (VADER + TextBlob)
- ✅ Uniqueness checking vs previous work
- ✅ Battle-specific metrics (punchlines, metaphors)

### Data & Ethos Module

- ✅ Motivational content database
- ✅ Sentiment analysis for confidence building
- ✅ CSV templates for manual tagging
- ✅ GitHub versioning + SQLite storage

### Multi-Provider Ensemble

- ✅ Generate from multiple providers simultaneously
- ✅ Compare outputs side-by-side
- ✅ Vote on best lines
- ✅ Learn which provider excels at what

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-rapper-system.git
cd ai-rapper-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger')"
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# At minimum, set PRIMARY_PROVIDER=local
# Cloud provider keys are OPTIONAL
```

### 3. Initialize Database

```bash
python -c "from scripts.database import Database; Database('./data/metrics.db')"
```

### 4. Run the System

```bash
# Start FastAPI server
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test It

```bash
# Health check
curl http://localhost:8000/health

# Generate lyrics (local model)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write aggressive battle rap bars about confidence",
    "provider": "local",
    "max_tokens": 200,
    "evaluate": true
  }'
```

---

## 📚 Usage Guide

### Generate Lyrics

**Using Python:**

```python
from providers import LyricEngine, GenerationConfig

# Initialize engine
config = {
    "primary_provider": "local",
    "local_model_path": "./models/your_model.gguf",
    "fallback_to_local": True,
}
engine = LyricEngine(config)

# Generate
result = await engine.generate(
    prompt="Write motivational bars about persistence",
    provider="local",  # or "claude", "openai", "gemini"
    config=GenerationConfig(temperature=0.9, max_tokens=512)
)

print(result.text)
```

**Using API:**

```bash
# Single generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt here",
    "temperature": 0.9,
    "evaluate": true
  }'

# Ensemble mode (multiple providers)
curl -X POST http://localhost:8000/generate/ensemble \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt here",
    "providers": ["local", "claude", "openai"]
  }'
```

### Switch Providers

**Instant switching (< 5 seconds):**

```python
# Python
engine.switch_provider("claude")  # Switch to Claude
engine.switch_provider("local")   # Back to local

# API
curl -X POST http://localhost:8000/providers/switch?provider=claude
```

### Evaluate Lyrics

```python
from scripts.evaluate import LyricEvaluator

evaluator = LyricEvaluator()
metrics = evaluator.evaluate(your_lyrics)

print(f"Overall Score: {metrics.overall_score()}")
print(f"Rhyme Density: {metrics.rhyme_density}")
print(f"Uniqueness: {metrics.uniqueness}")
```

### Manual Mode (No AI)

```python
# Write completely manually
result = engine.get_manual_input("Write about your journey")
# System prompts you to write, no AI involved
```

---

## 🎓 Training Your Local Model

### Prepare Training Data

```bash
# Create template
python scripts/train_local.py --action prepare

# Edit data/training_lyrics.json with your lyrics
# Aim for 100-1000 examples
```

### Train on Google Colab

**1. Upload to Colab:**

```python
# In Colab notebook
!git clone https://github.com/yourusername/ai-rapper-system.git
%cd ai-rapper-system
!pip install -r requirements.txt

# Upload your training data to Colab
# Or mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

**2. Train GPT-2:**

```python
!python scripts/train_local.py \
  --action train-gpt2 \
  --data ./data/training_lyrics.json \
  --output ./models/my_rapper_model \
  --epochs 3 \
  --batch-size 4
```

**3. Train Phi-3 (faster):**

```python
!python scripts/train_local.py \
  --action train-phi3 \
  --data ./data/training_lyrics.json \
  --output ./models/my_rapper_model \
  --epochs 3 \
  --batch-size 2
```

**4. Download Model:**

```python
# Zip and download
!zip -r my_model.zip ./models/my_rapper_model
from google.colab import files
files.download('my_model.zip')
```

**5. Convert to GGUF (Optional for CPU):**

Follow instructions in `scripts/train_local.py` to quantize for smaller file size and CPU inference.

---

## 🔌 Provider Setup

### Local (Always Available)

```env
PRIMARY_PROVIDER=local
LOCAL_MODEL_PATH=./models/your_model.gguf
USE_GPU=false
```

No API keys needed. Works offline.

### Claude (Optional)

```env
ENABLE_CLAUDE=true
ANTHROPIC_API_KEY=your_key_here
```

Get free tier at: https://console.anthropic.com/

### OpenAI (Optional)

```env
ENABLE_OPENAI=true
OPENAI_API_KEY=your_key_here
```

Get API key at: https://platform.openai.com/

### Gemini (Optional)

```env
ENABLE_GEMINI=true
GOOGLE_API_KEY=your_key_here
```

Get free tier at: https://makersuite.google.com/

### Groq (Optional)

```env
ENABLE_GROQ=true
GROQ_API_KEY=your_key_here
```

Get free tier at: https://console.groq.com/

---

## 📱 Mobile Setup

### Android (Termux)

```bash
# Install Termux from F-Droid
# In Termux:
pkg install python git
git clone https://github.com/yourusername/ai-rapper-system.git
cd ai-rapper-system
pip install -r requirements.txt
python main.py
```

### iOS (a-Shell)

```bash
# Install a-Shell from App Store
# Download repository as ZIP
# Extract and run:
pip install -r requirements.txt
python main.py
```

---

## 🎯 Daily Practice Routine

**Your human skills are NON-NEGOTIABLE:**

1. **30 minutes manual writing** - No AI at all
2. **15 minutes freestyle** - Maintain improvisational skills
3. **Study one great verse** - Learn from masters
4. **Document progress** - Track your growth

See `manual/writing_practice.md` for complete templates.

---

## 📊 System Architecture

```
Your Manual Skills (PRIMARY)
         ↓
Local Model (ALWAYS AVAILABLE)
         ↓
Provider Abstraction Layer
    ↙    ↓    ↘
Claude  ChatGPT  Gemini  [Any LLM]
         ↓
Evaluation System (LBCM)
         ↓
Database (SQLite + GitHub)
```

---

## 🛠️ Project Structure

```
ai-rapper-system/
├── providers/
│   ├── abstraction_layer.py     # Universal API wrapper
│   ├── local_adapter.py          # Local model (PRIMARY)
│   ├── claude_adapter.py         # Claude integration
│   ├── openai_adapter.py         # OpenAI integration
│   ├── gemini_adapter.py         # Gemini integration
│   └── groq_adapter.py           # Groq integration
├── scripts/
│   ├── evaluate.py               # LBCM evaluation system
│   ├── sentiment_analysis.py    # VADER + TextBlob
│   ├── database.py               # SQLite management
│   └── train_local.py            # Training scripts
├── data/
│   ├── training_lyrics.json      # Your training data
│   ├── metrics.db                # SQLite database
│   └── ethos_data.json           # Motivational content
├── models/
│   └── [your trained models]
├── manual/
│   ├── writing_practice.md       # Daily practice templates
│   └── freestyle_logs.md         # Freestyle tracking
├── main.py                       # FastAPI application
├── requirements.txt              # Dependencies
├── .env.example                  # Configuration template
└── README.md                     # This file
```

---

## 🧪 Testing Independence

**You should be able to:**

- [ ] Generate quality lyrics with local model only (no internet)
- [ ] Generate quality lyrics manually (no AI at all)
- [ ] Switch from Claude to ChatGPT in under 5 seconds
- [ ] Run entire system on your phone offline
- [ ] Produce a complete song in 24 hours if ALL cloud AIs are down
- [ ] Train and improve your model without any cloud service

---

## 🚀 Deployment

### Render (Free Tier)

```bash
# 1. Push to GitHub
# 2. Connect to Render
# 3. Set environment variables
# 4. Deploy
```

See `docs/deployment.md` for detailed instructions.

### Vercel (Frontend)

```bash
vercel deploy
```

### Self-Hosted

```bash
# Using systemd
sudo systemctl enable rapper-system
sudo systemctl start rapper-system
```

---

## 📈 Roadmap

### Phase 0 ✅

- [x] Data & Ethos Module (Sentiment analysis)
- [x] CSV templates for manual tagging
- [x] SQLite + GitHub setup

### Phase 1 ✅

- [x] Provider Abstraction Layer
- [x] Universal API wrapper
- [x] FastAPI backend

### Phase 2 ✅

- [x] Local model training scripts
- [x] GPT-2 and Phi-3 support
- [x] GGUF conversion guide

### Phase 3 ✅

- [x] LBCM Evaluation System
- [x] Rhyme density calculator
- [x] Syllable counter
- [x] Uniqueness checker

### Phase 4 ✅

- [x] Multi-provider output
- [x] Ensemble mode
- [x] Provider comparison

### Phase 5 ✅

- [x] Provider management
- [x] Health checks
- [x] Usage tracking

### Phase 6 (In Progress)

- [ ] Mobile deployment guides
- [ ] Beat blueprint module
- [ ] DAW integration (LMMS, Reaper)

### Phase 7 ✅

- [x] Manual practice templates
- [x] Freestyle logging
- [x] Skill tracking

---

## 🤝 Contributing

This is a personal project, but contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - Use freely, build freely, stay independent.

---

## 💡 Philosophy

> **"AI should amplify what's already in you, not replace it."**

This system is built on the principle of **true independence**:

- Your skills come first
- Local models are primary
- Cloud AIs are optional enhancements
- You own your voice
- No vendor can lock you in
- Zero cost is always possible

If every AI company disappeared tomorrow, you'd still be able to:

- Generate quality lyrics with your local model
- Write quality lyrics manually
- Evaluate your work objectively
- Continue improving your craft

That's true independence.

---

## 🆘 Troubleshooting

### Local model won't load

```bash
# Install llama-cpp-python with CPU support
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Or with GPU support (CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Out of memory during training

```bash
# Reduce batch size
--batch-size 1

# Use gradient accumulation
# Edit scripts/train_local.py: gradient_accumulation_steps=8
```

### NLTK data missing

```python
import nltk
nltk.download('all')
```

### Can't connect to cloud provider

```bash
# Check API keys
# Verify provider is enabled in .env
# Test with: curl -X GET http://localhost:8000/health
```

---

## 📞 Support

- **Issues:** https://github.com/yourusername/ai-rapper-system/issues
- **Discussions:** https://github.com/yourusername/ai-rapper-system/discussions

---

## 🙏 Acknowledgments

Built on the philosophy of independence and powered by:

- FastAPI
- Hugging Face Transformers
- llama.cpp
- NLTK, VADER, TextBlob
- All the open-source tools that make this possible

---

**Remember: Your voice is unique. AI is a tool. The goal is to amplify what's already in you, never replace it.**

🎤 **Now go create something amazing.**
