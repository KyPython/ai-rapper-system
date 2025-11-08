# Google Colab Training Guide - AI Rapper System

## 📋 Overview

This guide will help you retrain your AI rapper model in Google Colab with the correct configuration and upload it to replace the existing model.

## 🚀 Training in Google Colab

### Step 1: Open the Training Notebook

1. Open `AI_Rapper_Training_Guide.ipynb` in Google Colab
2. **IMPORTANT**: Go to `Runtime` → `Change runtime type` → Select `T4 GPU`
3. Run all cells in order

### Step 2: Prepare Your Training Data

Create a `training_lyrics.json` file with your lyrics:

```json
{
  "training_data": [
    {
      "prompt": "Write aggressive battle rap bars",
      "lyrics": "Your best verse here\nMultiple lines\nWith your unique style"
    },
    {
      "prompt": "Write motivational bars",
      "lyrics": "Another complete verse\nWith your flow\nAnd wordplay"
    }
  ]
}
```

**Tips for Good Training Data:**
- Aim for 100-500 examples for best results
- Each verse should be complete (8-16 bars)
- Include variety in topics, styles, and moods
- Focus on YOUR unique voice and style

### Step 3: Train the Model

The training will take 2-4 hours on Google Colab's free GPU. The notebook will:

1. ✅ Install dependencies
2. ✅ Load your training data
3. ✅ Fine-tune GPT-2-Medium (355M parameters)
4. ✅ Test the model
5. ✅ Zip and download the trained model

**Training Parameters (optimized):**
- Base model: `gpt2-medium` (good quality/size balance)
- Epochs: 3
- Batch size: 2 (fits in free tier)
- Learning rate: 5e-5
- Max sequence length: 512 tokens
- **Generation max_tokens: 512** (allows full verses)

## 📤 Uploading Your New Model

### Option 1: Google Drive (Recommended for Render)

1. **Upload to Google Drive:**
   - After training completes, the notebook downloads `trained_model.zip`
   - Upload this file to your Google Drive
   - Right-click → Share → Change to "Anyone with the link"
   - Copy the file ID from the share link: `https://drive.google.com/file/d/FILE_ID_HERE/view`

2. **Update Render Configuration:**

   Edit `render.yaml` and replace the old file ID:

   ```yaml
   buildCommand: pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger')" && gdown --id YOUR_NEW_FILE_ID_HERE -O trained_model.zip && unzip -q trained_model.zip -d models/ && rm trained_model.zip
   ```

3. **Deploy:**
   ```bash
   git add render.yaml
   git commit -m "Update model with new training"
   git push
   ```

   Render will automatically rebuild with your new model!

### Option 2: Local Testing

1. **Extract the model:**
   ```bash
   unzip trained_model.zip
   ```

2. **Place in your project:**
   ```bash
   cp -r trained_model ./models/trained_model
   ```

3. **Test locally:**
   ```bash
   uvicorn main:app --reload
   ```

## 🔄 Model Versioning Best Practice

Keep track of your models:

```
models/
├── trained_model/          # Current production model
├── trained_model_v1/       # Backup of previous version
├── trained_model_v2/       # Latest experimental
└── README.md              # Training notes
```

**Document each version:**
```markdown
# Model Version History

## v2 - 2025-01-15
- Training data: 500 verses
- Training time: 3.5 hours
- Improvements: Better flow, less repetition
- Google Drive ID: 1_NEW_FILE_ID_HERE

## v1 - 2025-01-10
- Training data: 200 verses
- Training time: 2 hours
- Google Drive ID: 1_OLD_FILE_ID_HERE
```

## 🧪 Testing Your New Model

Before deploying, test the model quality:

1. **Generate test lyrics:**
   ```bash
   curl http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write aggressive battle rap bars", "max_tokens": 256}'
   ```

2. **Check evaluation metrics:**
   - Rhyme density > 0.7
   - Syllable consistency > 0.75
   - Uniqueness > 0.6
   - Overall score > 0.75

3. **Compare with previous version:**
   ```bash
   curl http://localhost:8000/compare \
     -H "Content-Type: application/json" \
     -d '{
       "lyrics1": "old model output",
       "lyrics2": "new model output"
     }'
   ```

## ⚠️ Troubleshooting

### "Model not loading" error
- Ensure `trained_model/` contains: `config.json`, `pytorch_model.bin`, `tokenizer.json`
- Check file permissions
- Verify model was saved correctly during training

### "Out of memory" during training
- Reduce batch size to 1
- Use gradient accumulation (already configured)
- Reduce max_sequence_length to 256

### Poor quality output
- Needs more training data (aim for 200+ examples)
- Check training data quality
- Try training for more epochs (4-5)
- Ensure training data has complete verses, not fragments

## 📊 Training Tips

**For Best Results:**
- 📝 Quality over quantity - 200 great verses > 1000 mediocre ones
- 🎭 Include variety - different moods, topics, styles
- 🔄 Don't overfit - Stop if training loss gets too low (<0.1)
- ✅ Test frequently - Generate samples every 500 steps
- 💾 Save checkpoints - Keep multiple versions

**Training Data Format:**
```python
# GOOD
{
  "prompt": "Write battle rap bars",
  "lyrics": "Complete verse with 8-16 bars\nWith rhyme schemes and flow\nThat demonstrates your style\nAnd unique voice you know"
}

# BAD
{
  "prompt": "rap",
  "lyrics": "just one line"  # Too short!
}
```

## 🎯 Expected Results

After fine-tuning on 200-500 quality verses:
- **Rhyme Density**: 0.75-0.85 (excellent)
- **Flow Consistency**: 0.80-0.90 (smooth)
- **Uniqueness**: 0.70-0.85 (distinctive voice)
- **Overall Quality**: 0.75-0.85 (professional level)

## 📝 Next Steps

1. ✅ Train your model in Colab
2. ✅ Test locally
3. ✅ Upload to Google Drive
4. ✅ Update render.yaml
5. ✅ Deploy to production
6. 📈 Monitor metrics
7. 🔄 Iterate and improve

---

**Questions?** Check the main README.md or open an issue on GitHub.
