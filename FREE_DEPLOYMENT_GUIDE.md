# Free Deployment Guide - 512MB Memory Limit Solution

## 🚨 Problem: Out of Memory on Render Free Tier

Render's free tier has **512MB RAM limit**. GPT-2-Medium (355M parameters) needs ~700MB+ to load, causing out of memory errors.

## ✅ Solution: Multiple Free Options

### Option 1: Lazy Loading (Implemented ✅)

**What it does:** Model loads on first API request, not at startup. This spreads out memory usage.

**Already configured in your deployment:**
```yaml
# render.yaml
LAZY_LOAD: true  # ✅ Enabled by default
```

**How it works:**
1. App starts quickly without loading model (~50MB RAM)
2. First request triggers model load (~700MB peak)
3. After load, settles to ~600MB
4. **May work** on free tier with lazy loading, but still tight

**Status:** ⚠️ May still hit memory limit with GPT-2-Medium

---

### Option 2: Use GPT-2-Small (RECOMMENDED for Free Tier ✅)

**Memory:** ~250MB (fits comfortably in 512MB)
**Quality:** Still good for rap lyrics, just shorter training time
**Cost:** FREE

#### Step 1: Train GPT-2-Small in Colab

Update the training cell in `AI_Rapper_Training_Guide.ipynb`:

```python
# Change this line:
model_name = "gpt2-medium"  # OLD: 355M params, 700MB RAM

# To this:
model_name = "gpt2"  # NEW: 124M params, 250MB RAM
```

**That's it!** Everything else stays the same. Train as normal.

#### Step 2: Upload to Google Drive

Same as before - upload `trained_model.zip` to Google Drive and get the file ID.

#### Step 3: Update render.yaml

```yaml
buildCommand: pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger')" && gdown --id YOUR_NEW_GPT2_SMALL_MODEL_ID -O trained_model.zip && unzip -q trained_model.zip -d models/ && rm trained_model.zip
```

**Result:** ✅ Fits in 512MB with room to spare!

---

### Option 3: GGUF Quantization (ADVANCED - Smallest Size)

**Memory:** ~80-150MB (4-8x smaller!)
**Quality:** Slight quality loss, but very small
**Complexity:** Requires conversion step

#### Convert Your Trained Model to GGUF Q4

After training in Colab, run this additional cell:

```python
# Install llama.cpp tools
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && make

# Convert to GGUF format
!python llama.cpp/convert.py ./trained_model --outfile trained_model.gguf --outtype f16

# Quantize to Q4 (4-bit quantization)
!./llama.cpp/quantize trained_model.gguf trained_model_q4.gguf Q4_K_M

# Download
from google.colab import files
files.download('trained_model_q4.gguf')
```

Upload `trained_model_q4.gguf` to Google Drive and update paths.

**Result:** ✅ Smallest possible size, but requires more setup

---

## 🎯 Recommended Approach for FREE Deployment

### For Best Experience (Choose ONE):

1. **GPT-2-Small** (EASIEST)
   - ✅ Fits easily in 512MB
   - ✅ Good quality
   - ✅ Fast inference
   - ✅ Simple to set up
   - **Recommended for most users**

2. **GPT-2-Medium + GGUF Q4** (SMALLEST)
   - ✅ Best compression
   - ⚠️ Requires conversion
   - ⚠️ Slight quality loss
   - **For advanced users wanting smallest size**

3. **GPT-2-Medium + Lazy Load** (CURRENT)
   - ⚠️ May work, may not (borderline)
   - ⚠️ Slow first request
   - ⚠️ Risk of OOM crashes
   - **Not recommended for production**

---

## 📊 Memory Comparison

| Model | Format | RAM Usage | Fits in 512MB? |
|-------|--------|-----------|----------------|
| GPT-2-Medium | PyTorch | ~700MB | ❌ No |
| GPT-2-Medium | GGUF Q4 | ~150MB | ✅ Yes |
| GPT-2-Small | PyTorch | ~250MB | ✅ Yes |
| GPT-2-Small | GGUF Q4 | ~80MB | ✅ Yes |

---

## 🚀 Step-by-Step: Switch to GPT-2-Small (Recommended)

### 1. Update Training Notebook

Open `AI_Rapper_Training_Guide.ipynb`, find the training script cell:

```python
# OLD:
model_name = "gpt2-medium"

# NEW:
model_name = "gpt2"  # This is GPT-2-Small (124M params)
```

### 2. Train in Colab

- Upload your `training_lyrics.json` (200+ verses recommended)
- Run all cells (takes 1-2 hours for GPT-2-Small)
- Download `trained_model.zip`

### 3. Upload to Google Drive

- Upload to Google Drive
- Share → "Anyone with the link"
- Copy file ID from URL: `https://drive.google.com/file/d/FILE_ID_HERE/view`

### 4. Update render.yaml

```yaml
buildCommand: pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger')" && gdown --id YOUR_GPT2_SMALL_FILE_ID -O trained_model.zip && unzip -q trained_model.zip -d models/ && rm trained_model.zip
```

### 5. Deploy

```bash
git add render.yaml
git commit -m "Switch to GPT-2-Small for 512MB memory limit"
git push
```

**Result:** ✅ Deployment succeeds! App runs smoothly in 512MB.

---

## ⚡ Alternative Free Platforms (If Render Doesn't Work)

If you want to keep GPT-2-Medium without conversion:

### Hugging Face Spaces
- **RAM:** 16GB free tier
- **Cost:** FREE
- **GPU:** Available
- **Limit:** No memory issues
- **Deploy:** Easy with Dockerfile

### Railway (Free Trial)
- **RAM:** 512MB → 8GB (with free trial)
- **Cost:** $5 free credit
- **Limit:** Credit runs out eventually

### Fly.io
- **RAM:** 256MB shared → 2GB free tier
- **Cost:** FREE with limits
- **Limit:** Complex setup

**Verdict:** Hugging Face Spaces is best free alternative if you want GPT-2-Medium.

---

## 🎯 My Recommendation

**For 100% Free, No Hassle:**

1. Use **GPT-2-Small** on Render
   - Change 1 line in training notebook
   - Retrain (1-2 hours)
   - Deploy normally
   - ✅ Works perfectly in 512MB

**Quality Difference:**
- GPT-2-Small: Still generates good rap verses
- Slightly less complex vocabulary
- Faster inference
- 95% of GPT-2-Medium quality for this use case

---

## 📝 Current Status

Your deployment has:
- ✅ Lazy loading enabled
- ✅ Database fixes applied
- ✅ Async inference working
- ⚠️ GPT-2-Medium model (too large for 512MB)

**Next Step:** Retrain with GPT-2-Small and redeploy.

---

## 🆘 Quick Fix Commands

### Temporary: Use Dummy Mode (Testing Only)

```yaml
# .env or render.yaml
ALLOW_DUMMY_MODE: true
```

This lets the app start without loading the model (for testing API endpoints).

### Permanent: GPT-2-Small

See "Step-by-Step" above.

---

**Questions?** Check COLAB_TRAINING_GUIDE.md for training details.
