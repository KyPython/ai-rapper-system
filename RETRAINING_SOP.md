# 🔄 Retraining SOP (Standard Operating Procedure)

## When to Retrain

- ✅ After adding 20-30+ new verses
- ✅ Output quality drops or becomes repetitive
- ✅ Want to learn new styles/themes
- ✅ Monthly or quarterly to incorporate new material

**Frequency:** As often as you want! No limits, no costs.

---

## ⚡ Quick Retraining (30 min active time, 2-4 hours total)

### Step 1: Update Training Data (10-30 min)

```bash
# Open your training data
cd /private/tmp/ai-rapper-system
open data/training_lyrics.json
```

Add your new verses in this format:

```json
{
  "prompt": "Write [style/theme description]",
  "lyrics": "Your bars here\nLine by line\n8-16 bars per verse"
}
```

**Pro tip:** Add the `total_verses` count in metadata for tracking.

---

### Step 2: Upload to Google Colab (2 min)

1. Go to: https://colab.research.google.com
2. Click: **File > Upload notebook**
3. Upload: `COLAB_TRAINING_GUIDE.ipynb` from your project
4. **CRITICAL:** Runtime > Change runtime type > **GPU** > Save

---

### Step 3: Run Training Cells (2-4 hours, mostly automated)

**Run each cell in order:**

#### Cell 1: Upload Training Data

- Click the upload button
- Select `data/training_lyrics.json` from your project
- Verify verse count

#### Cell 2: Install Dependencies (~2 min)

- Installs transformers, torch, datasets
- Wait for green checkmark

#### Cell 3: Create Training Script

- Writes `train_model.py` to Colab environment
- Should see "Writing train_model.py"

#### Cell 4: Train Model (~30s to 4 hours depending on data size)

- **This is the long part - GPU does the work**
- With 18 verses: ~65 seconds
- With 50 verses: ~3-5 minutes
- With 100 verses: ~10-15 minutes
- Loss should decrease (4.5 → 3.9 is good)

#### Cell 5: Test Generation (30s)

- Generates sample lyrics
- Check if output looks good
- If terrible, add more diverse training data

#### Cell 6: Download Model (~1-2 min)

- Creates `trained_model.zip` (1.2GB)
- Downloads to your computer
- Automatic download should start

---

### Step 4: Deploy Updated Model to Render (5-10 min)

**Simple Git Push - Render Auto-Deploys!**

```bash
# Navigate to Downloads
cd ~/Downloads

# Unzip the new model
unzip trained_model.zip

# Navigate to your project
cd /private/tmp/ai-rapper-system

# Backup old model (optional but recommended)
mv models/trained_model models/trained_model_backup_$(date +%Y%m%d)

# Move new model to project
mv ~/Downloads/trained_model models/

# Add to git and push
git add models/trained_model/
git commit -m "Updated model with new training data - $(date +%Y-%m-%d)"
git push

# Render automatically detects the push and rebuilds (5-10 min)
```

**That's it!** Render will:

1. Detect your git push
2. Rebuild the app with new model
3. Deploy automatically
4. Your app at `https://ai-rapper-system.onrender.com` will have the new model

---

### Step 5: Test New Model (2 min)

Open your Render app URL: `https://ai-rapper-system.onrender.com`

Try prompts from your new training data to verify it learned the new style.

**Note:** First request after deploy may take 30-40 seconds (cold start), then normal speed.

Try prompts from your new training data to verify it learned the new style.

---

## 📊 Training Tips

### Data Quality:

- **Variety > Quantity:** 50 diverse verses > 100 repetitive ones
- **Clear prompts:** Describe each verse style accurately
- **Best work only:** Don't include filler or weak bars
- **Mix styles:** Battle, motivational, technical, storytelling

### Training Parameters (in Colab notebook):

- **Epochs (default 3):** How many times model sees the data
  - Too few (1-2): Underfit, doesn't learn patterns
  - Too many (5+): Overfit, memorizes instead of learning
  - Sweet spot: 3-4 epochs
- **Batch size (default 2):** Memory vs speed tradeoff
  - Keep at 2 for Colab free tier (GPU memory limits)
- **Learning rate (default 5e-5):** How fast model learns
  - Current setting works well, don't change unless expert

### Monitoring Training:

Watch the loss values:

- Starting loss: ~4.5 (random GPT-2)
- Target loss: ~3.0-4.0 (learned your style)
- If loss increases: Stop, something wrong
- If loss plateaus early: Add more diverse data

---

## 🔧 Troubleshooting

### "Model generates gibberish"

- **Cause:** Not enough training data or too diverse prompts
- **Fix:** Add 20-30 more verses with similar styles

### "Training takes too long"

- **Cause:** CPU runtime instead of GPU
- **Fix:** Runtime > Change runtime type > GPU

### "Out of memory error"

- **Cause:** Batch size too high or max_length too long
- **Fix:** In training script, reduce `per_device_train_batch_size=1`

### "Loss stays high (>4.0)"

- **Cause:** Model not learning effectively
- **Fix:**
  - Increase epochs to 4-5
  - Check training data format (must match exactly)
  - Ensure prompts are descriptive

### "Downloaded model doesn't work"

- **Cause:** Incomplete download or wrong folder structure
- **Fix:**
  - Verify `model.safetensors` is ~1.2GB
  - Check all 7 files are present
  - Re-download if corrupted

---

## 📈 Version Control (Recommended)

Keep track of your model versions:

```bash
# Create models archive
mkdir -p /private/tmp/ai-rapper-system/models/archive

# Before retraining, archive current model
cd /private/tmp/ai-rapper-system/models
tar -czf archive/trained_model_v1_$(date +%Y%m%d).tar.gz trained_model/

# Document what changed
echo "v1 - 18 verses, technical focus" >> archive/VERSION_LOG.txt
echo "v2 - 50 verses, added motivational themes" >> archive/VERSION_LOG.txt
```

This way you can roll back if new training doesn't improve quality.

---

## 🎯 Retraining Schedule (Suggested)

### Casual Use:

- **Monthly:** Add 10-20 new verses, retrain
- **Time:** 2-3 hours per month

### Active Development:

- **Weekly:** Add 20-30 verses, retrain
- **Time:** 1-2 hours per week

### Professional:

- **Daily practice:** Write 1-2 verses daily
- **Weekly compilation:** Retrain on best 10-20
- **Time:** 30 min daily practice + 1 hour weekly training

---

## ✅ Success Checklist

After each retraining:

- [ ] Training loss decreased by at least 0.3-0.5
- [ ] Test generation produces coherent lyrics
- [ ] New styles/themes are reflected in output
- [ ] Model doesn't generate gibberish on simple prompts
- [ ] Old styles still work (no catastrophic forgetting)
- [ ] Backed up previous model version
- [ ] Updated version log

---

## 💡 Pro Tips

1. **Keep a "Best Bars" file:** When you write something fire, immediately add to training data
2. **Retrain iteratively:** Don't wait for 100 verses, retrain every 20-30
3. **Test before/after:** Generate same prompt before and after retraining to compare
4. **Use Colab Pro** (optional, $10/month): Faster GPUs, longer runtime, priority access
5. **Mobile workflow:** Write on phone → AirDrop to Mac → Add to training_lyrics.json

---

## 🆘 Need Help?

1. Check Colab output for error messages
2. Verify training data JSON is valid: `python -m json.tool data/training_lyrics.json`
3. Test model locally before uploading: `python example.py`
4. Check GitHub issues or community forums

**Remember:** Retraining is simple, fast, and free. You can do it as often as you want!
