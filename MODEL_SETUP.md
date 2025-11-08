# 🤖 Model Setup for Render Deployment

## Why Model is Not in GitHub

Your trained model (`model.safetensors`) is **1.3GB** - GitHub rejects files over 100MB. This is why it's excluded from the repo.

---

## Option 1: Git LFS (Recommended for Frequent Updates)

If you plan to retrain often and want easy model versioning:

```bash
# Install Git LFS
brew install git-lfs
git lfs install

# Track the model file
git lfs track "models/trained_model/model.safetensors"
git lfs track "models/trained_model/*.safetensors"

# Add and commit
git add .gitattributes
git add models/trained_model/
git commit -m "Add trained model with Git LFS"
git push

# Render will automatically pull the LFS files on deploy
```

**Cost**: Free for up to 1GB bandwidth/month on GitHub, then $5/month for 50GB

---

## Option 2: Cloudflare R2 Storage (Free, Best for Render)

1. **Create R2 bucket** at https://dash.cloudflare.com/

   - Free tier: 10GB storage, 10M Class A operations/month
   - Create bucket: `ai-rapper-models`

2. **Upload your model**:

   - Zip the model: `cd models && zip -r trained_model.zip trained_model/`
   - Upload `trained_model.zip` to R2 bucket
   - Make it public or get a presigned URL

3. **Update Render build command** to download on startup:
   ```bash
   pip install -r requirements.txt && \
   python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger')" && \
   curl -L -o trained_model.zip YOUR_R2_PUBLIC_URL && \
   unzip -q trained_model.zip -d models/ && \
   rm trained_model.zip
   ```

---

## Option 3: Google Drive (Quick & Easy)

1. **Upload model to Google Drive**:

   - Zip: `cd models && zip -r trained_model.zip trained_model/`
   - Upload to Google Drive
   - Right-click → Share → Anyone with link can view
   - Copy link (will look like: `https://drive.google.com/file/d/FILE_ID/view`)

2. **Get direct download link**:

   ```
   https://drive.google.com/uc?export=download&id=FILE_ID
   ```

   Replace `FILE_ID` with the actual ID from your share link

3. **Update Render build command**:
   ```bash
   pip install -r requirements.txt && \
   python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger')" && \
   curl -L -o trained_model.zip "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID" && \
   unzip -q trained_model.zip -d models/ && \
   rm trained_model.zip
   ```

---

## Option 4: Include in Render Persistent Disk

Render paid plans ($7/month Starter) include persistent disk:

1. Add persistent disk to your Render service
2. Upload model via SSH or SFTP
3. Model persists across deploys

---

## Recommended Approach

**For your use case (retraining occasionally):**

**Use Option 2 (Cloudflare R2)** or **Option 3 (Google Drive)**:

- Free
- Simple
- Works with Render free tier
- Model auto-downloads on every deploy
- Easy to update (just replace the file)

---

## Current Model Location

Your trained model is currently at:

```
/private/tmp/ai-rapper-system/models/trained_model/
```

Files (1.2GB total):

- `model.safetensors` (1.3GB)
- `config.json`
- `generation_config.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `special_tokens_map.json`

**Next step**: Choose an option above and set it up before deploying to Render!
