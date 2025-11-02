# Evidently AI - MLOps Monitoring Demo

A practical demonstration of ML monitoring using **Evidently AI** (v0.7.15).

---

## What is Evidently?

**Evidently** is an open-source Python library for ML model monitoring and evaluation. It helps you:

- **Detect data drift** - Catch when production data differs from training data
- **Monitor performance** - Track accuracy, precision, recall over time  
- **Evaluate LLMs** - Analyze text quality, sentiment, and toxicity
- **Generate reports** - Beautiful interactive HTML dashboards

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `evidently` - The monitoring library
- `pandas` - Data manipulation
- `scikit-learn` - For classification examples
- `nltk` - Text processing
- `openai` - (Optional) For LLM-as-a-Judge

---

### 2. Run the Examples

#### Example 1: Data Drift Detection

**What it does:** Detects when your production data shifts away from training data.

```bash
cd 1_data_drift
python monitor_data_drift.py
```

**Opens:** `data_drift_report.html`

**What you'll see:**
- Age shifted from 35 to 22 years old
- Income dropped from $80k to $35k
- New category appeared: "student" (never seen in training!)
- Statistical tests (K-S, Chi-square) prove the drift

**Why it matters:** Your model trained on professionals is now seeing students - predictions will be wrong!

---

#### Example 2: Model Performance Tracking

**What it does:** Monitors classification accuracy over time.

```bash
cd ../2_model_performance
python monitor_model_performance.py
```

**Opens:** `classification_performance_report.html`

**What you'll see:**
- Reference accuracy: **90%** (9 out of 10 correct)
- Current accuracy: **50%** (5 out of 10 correct)
- Performance drop: **40%** - time to retrain!

**Why it matters:** Catch model degradation before users complain.

---

#### Example 3: LLM Monitoring

**What it does:** Evaluates text quality from language models.

```bash
cd ../3_llm_monitoring
python monitor_llm_judge.py
```

**Opens:** `llm_evaluation_report.html`

**What you'll see:**
- **Sentiment analysis** - Detects toxic content ("stupid idiot")
- **Text length** - Finds short/incomplete answers ("5." instead of explanation)
- **Denial detection** - Catches refusals ("I cannot provide...")

**Why it matters:** Ensure your chatbot gives quality, safe responses.

---

## Optional: Enable LLM-as-a-Judge

For semantic evaluation of text quality (more accurate than keywords):

**Set your OpenAI API key:**

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"

# Linux/Mac
export OPENAI_API_KEY="sk-your-key-here"
```

Then run the LLM monitoring example again. It will use GPT-4o-mini to judge response quality.

---

## Project Structure

```
evidently/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
│
├── 1_data_drift/                      # Data drift detection
│   ├── data/
│   │   ├── reference.csv              # Training data (professionals)
│   │   └── current.csv                # Production data (students)
│   └── monitor_data_drift.py          # Drift detection script
│
├── 2_model_performance/               # Classification monitoring
│   ├── data/
│   │   ├── reference.csv              # Good performance (90% accuracy)
│   │   └── current.csv                # Degraded (50% accuracy)
│   └── monitor_model_performance.py   # Performance tracking script
│
└── 3_llm_monitoring/                  # LLM evaluation
    ├── data/
    │   ├── reference.csv              # Quality responses
    │   └── current.csv                # Problematic responses
    └── monitor_llm_judge.py           # LLM evaluation script
```

---

## What Makes This Demo Interesting?

### 1. Data Drift Example
The data **intentionally shifts** to show real-world problems:
- Training: Professional employees (age 35, salary $80k)
- Production: College students (age 22, salary $35k)
- **Result:** Model fails because it's seeing completely different data!

### 2. Performance Degradation
The accuracy **drops by 40%** due to data drift:
- Training performance: 90% correct
- Production performance: 50% correct (barely better than random!)
- **Result:** Shows why monitoring is critical

### 3. LLM Quality Issues
The chatbot responses have **real problems**:
- Toxic: "Shakespeare was a stupid idiot"
- Wrong: "2+2 = 5"
- Refusal: "I cannot provide a summary"
- Privacy leak: Email and phone number in response
- **Result:** Evidently catches all of these!

---

## Key Evidently Features Demonstrated

### Statistical Tests
- **Kolmogorov-Smirnov test** - For numerical features (age, income)
- **Chi-square test** - For categorical features (job_type, education)
- **Wasserstein distance** - Measures distribution shift

### Metrics
- **Data drift** - Per-column drift detection
- **Accuracy** - Classification performance
- **Sentiment** - Emotional tone (-1 to +1)
- **Text length** - Character count
- **Keyword detection** - Pattern matching

### Reports
- **Interactive HTML** - Click, zoom, explore
- **No server needed** - Just open in browser
- **Self-contained** - All data embedded

---

## How to Use in Production

1. **Set baseline** - Run Evidently on your training/test data
2. **Monitor production** - Run weekly/daily on new data
3. **Set thresholds** - Alert when drift > X% or accuracy < Y%
4. **Retrain trigger** - Automatically retrain when thresholds exceeded
5. **Track trends** - Store reports to see changes over time

---

## Common Use Cases

### Use Case 1: Silent Model Failure
**Problem:** Your model's accuracy drops but you don't know why.

**Solution:** Evidently shows data drift in key features - you discover your user base changed.

### Use Case 2: LLM Chatbot Quality
**Problem:** Users complain about chatbot responses but you can't review all conversations.

**Solution:** Evidently automatically flags toxic, wrong, or incomplete responses.

### Use Case 3: A/B Testing Models
**Problem:** You want to compare two models in production.

**Solution:** Run Evidently on both - compare accuracy, drift, and performance side-by-side.

---

## Why Evidently AI?

### Pros
- ✅ **Open source** - Free, no vendor lock-in
- ✅ **No infrastructure** - Works locally, no cloud required
- ✅ **Beautiful reports** - Easy to understand visuals
- ✅ **Statistical rigor** - Proper hypothesis testing
- ✅ **LLM support** - Built-in text evaluation
- ✅ **Production ready** - Used by companies like Rivery, Microsoft

### When to Use
- Monitor ML models in production
- Detect data quality issues
- Evaluate LLM outputs
- Debug model failures
- A/B test models

---

## Learn More

- **Documentation:** [docs.evidentlyai.com](https://docs.evidentlyai.com)
- **GitHub:** [github.com/evidentlyai/evidently](https://github.com/evidentlyai/evidently)
- **Discord Community:** [discord.gg/evidently](https://discord.gg/evidently)
- **Blog:** [evidentlyai.com/blog](https://www.evidentlyai.com/blog)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'evidently'"
**Solution:** Run `pip install -r requirements.txt`

### Issue: "UnicodeEncodeError" on Windows
**Solution:** Already fixed - we use ASCII characters only

### Issue: LLM-as-a-Judge not working
**Solution:** Set your `OPENAI_API_KEY` environment variable. Without it, Evidently uses keyword-based checks (still works, just less accurate).

### Issue: RuntimeWarning from scipy
**Solution:** This is normal - scipy warns about divide by zero in statistical tests with small samples. Doesn't affect results.

---

## Next Steps

1. ✅ Run all three examples
2. ✅ Open the HTML reports
3. ✅ Try with your own data
4. ✅ Set up automated monitoring
5. ✅ Integrate with CI/CD pipelines

**Happy Monitoring!** 🎉
