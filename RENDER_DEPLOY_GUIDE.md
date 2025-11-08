# 🚀 Render.com Deployment Guide

## Why Render?

- ✅ **Actually works** (no capacity issues)
- ✅ Free tier available
- ✅ Deploy from GitHub in minutes
- ✅ Automatic HTTPS
- ✅ 24/7 uptime
- ⚠️ Free tier sleeps after 15 min inactivity (wakes in ~30 seconds)

---

## Quick Deploy (15 minutes)

### Step 1: Push to GitHub (5 min)

If you don't have a GitHub repo yet:

```bash
cd /private/tmp/ai-rapper-system

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - AI Rapper System"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/ai-rapper-system.git
git branch -M main
git push -u origin main
```

---

### Step 2: Create Render Web Service (5 min)

1. **Go to**: https://dashboard.render.com
2. **Sign in** with GitHub
3. **Click**: "New +" → "Web Service"
4. **Connect your repo**: `ai-rapper-system`
5. **Configure**:

   - **Name**: `ai-rapper-system`
   - **Region**: Oregon (or closest to you)
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**:
     ```
     pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger')"
     ```
   - **Start Command**:
     ```
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan**: Free

6. **Click**: "Create Web Service"

---

### Step 3: Upload Model Files (5 min)

**Option A: Use Render Disk (Recommended for Free Tier)**

Render's free tier doesn't include persistent disk, so model will be in the deployment. We need to either:

1. **Include model in Git** (not ideal - 1.2GB is large)
2. **Download model on startup** (add to build command)
3. **Use external storage** (S3/R2 free tier)

**Option B: Download on Build (Best for Free Tier)**

Update your build command in Render to download the model:

```bash
# In Render dashboard, update Build Command to:
pip install -r requirements.txt && \
python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger')" && \
mkdir -p models/trained_model
```

Then upload your model to a free file host (Cloudflare R2, Backblaze B2, or even Google Drive with public link), and add a download step.

**Option C: Git LFS (Best if you have it)**

```bash
# Install Git LFS
brew install git-lfs
git lfs install

# Track large model file
git lfs track "models/trained_model/model.safetensors"
git add .gitattributes
git add models/trained_model/
git commit -m "Add trained model with LFS"
git push
```

---

### Step 4: Configure Environment Variables

In Render dashboard → Environment:

```
PRIMARY_PROVIDER=local
LOCAL_MODEL_PATH=./models/trained_model
USE_GPU=false
API_HOST=0.0.0.0
OFFLINE_MODE=false
FALLBACK_TO_LOCAL=true
```

---

### Step 5: Deploy & Test

1. Render automatically deploys after pushing to GitHub
2. Wait 5-10 minutes for build
3. Your app will be live at: `https://ai-rapper-system.onrender.com`
4. Visit from phone and bookmark!

---

## ⚠️ Free Tier Limitations

### Sleep After Inactivity:

- Free tier apps sleep after 15 minutes of no requests
- Wake up in ~30 seconds on first request
- **Solution**: Use a free uptime monitor like UptimeRobot to ping every 14 minutes

### Cold Starts:

- First generation after sleep: ~30-40 seconds
- Subsequent generations: 5-10 seconds (normal)

### Monthly Limits:

- 750 hours/month (enough for 24/7 with one app)
- 100GB bandwidth/month

---

## 🚀 Going Live

Once deployed:

1. **Visit**: `https://your-app-name.onrender.com`
2. **Test generation** in the web UI
3. **Bookmark on phone**
4. **Optional**: Set up custom domain

---

## 📈 Upgrading Later

If you want better performance:

- **Starter Plan ($7/month)**: No sleep, faster CPU
- **Standard Plan ($25/month)**: More RAM, faster builds

---

## 🔄 Updating Your Model

When you retrain:

1. Train model on Google Colab (same as before)
2. Download `trained_model.zip`
3. Extract and replace files in `models/trained_model/`
4. Commit and push to GitHub:
   ```bash
   git add models/trained_model/
   git commit -m "Updated model with new training data"
   git push
   ```
5. Render auto-deploys in 5-10 minutes

---

## 🆘 Troubleshooting

### Build fails with "Out of memory"

- Render free tier has limited RAM during builds
- Remove unnecessary dependencies
- Use `torch` CPU-only version

### Model loading fails

- Check `LOCAL_MODEL_PATH` in environment variables
- Ensure model files are in the repo or downloaded on build

### App sleeps too much

- Set up UptimeRobot (free) to ping every 14 minutes
- Or upgrade to Starter plan ($7/mo) for always-on

---

## 📱 Phone Access

Your Render app is accessible at:

```
https://your-app-name.onrender.com
```

Works from:

- ✅ Any phone (iOS/Android)
- ✅ Any WiFi/cellular connection
- ✅ Anywhere in the world
- ✅ No Mac needed!

**Bookmark it and you're done!** 🎉
