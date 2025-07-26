# PlaceboRx Validation Pipeline

**Solo entrepreneur execution - 2-4 hours runtime**

A streamlined pipeline to validate PlaceboRx's clinical evidence and market demand within hours, not weeks.

## 🎯 What This Pipeline Does

1. **Clinical Validation**: Searches ClinicalTrials.gov for open-label placebo (OLP) studies with clinically significant results
2. **Market Validation**: Analyzes Reddit communities for desperation signals, openness to alternatives, and engagement
3. **Framing Optimization**: Tests different PlaceboRx messaging approaches for resonance
4. **Go-to-Market Insights**: Provides actionable recommendations for MVP development

## 🚀 Quick Start (30 minutes setup)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Reddit API Credentials
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" as the app type
4. Note your `client_id` and `client_secret`

### 3. Set Up Environment
```bash
# Copy the example file
cp env_example.txt .env

# Edit .env with your credentials
nano .env
```

### 4. Run the Pipeline
```bash
python main_pipeline.py
```

## 📊 Expected Outputs

After 2-4 hours, you'll get:

- **`clinical_trials_results.csv`** - Raw trial data with significance analysis
- **`clinical_trials_report.md`** - Clinical evidence summary
- **`market_analysis_results.csv`** - Reddit post analysis with sentiment scores
- **`market_validation_report.md`** - Market demand insights
- **`placeborx_validation_report.md`** - Consolidated validation report

## 🔬 Clinical Validation Focus

The pipeline specifically looks for:
- **Statistically significant** OLP trials (p < 0.05)
- **Clinically relevant** effect sizes (≥20% improvement)
- **Digital interventions** (apps, online platforms)
- **High-impact conditions** (pain, anxiety, depression, etc.)

## 📈 Market Validation Signals

Analyzes Reddit posts for:
- **Desperation signals**: "nothing works", "tried everything", "desperate"
- **Openness to alternatives**: "natural", "holistic", "non-pharma"
- **Engagement patterns**: upvotes, comments, discussion quality
- **Framing resonance**: which PlaceboRx messaging resonates most

## 🎯 Validation Criteria

### Clinical Validation
- ✅ **STRONG**: Multiple statistically significant OLP trials
- ✅ **MODERATE**: Some clinically relevant trials with digital components
- ⚠️ **WEAK**: Limited or no significant OLP evidence

### Market Validation
- ✅ **STRONG**: >40% high desperation + openness + engagement
- ✅ **MODERATE**: >25% across key signals
- ⚠️ **WEAK**: <25% market demand signals

## 🛠️ Customization

### Modify Target Conditions
Edit `config.py`:
```python
SUBREDDITS = [
    'chronicpain', 'CFS', 'fibromyalgia',  # Add your target conditions
    'anxiety', 'depression', 'mentalhealth'
]
```

### Adjust Search Terms
```python
OLP_SEARCH_TERMS = [
    'open-label placebo', 'digital placebo',  # Add specific terms
    'app placebo', 'online placebo'
]
```

### Change Framing Tests
In `market_analyzer.py`, modify the `framings` dictionary:
```python
framings = {
    'your_framing': "your messaging approach",
    # Add more framings to test
}
```

## ⚡ Performance Optimization

For faster execution:
- Reduce `limit_per_subreddit` in `market_analyzer.py` (default: 50)
- Focus on fewer subreddits initially
- Use cached results for repeated runs

## 🔍 Troubleshooting

### Reddit API Issues
- Verify your credentials in `.env`
- Check Reddit's API status: https://www.redditstatus.com/
- Ensure your app has "script" type permissions

### Clinical Trials API Issues
- The ClinicalTrials.gov API is free and doesn't require authentication
- If rate limited, the pipeline includes built-in delays

### Memory Issues
- For large datasets, consider processing in batches
- Monitor system resources during execution

## 📋 Next Steps After Validation

1. **If validation is STRONG**: Proceed with MVP development
2. **If validation is MODERATE**: Refine approach and re-run with adjusted parameters
3. **If validation is WEAK**: Reconsider product-market fit or pivot strategy

## 🤝 Support

This pipeline is designed for solo entrepreneurs. For issues:
1. Check the troubleshooting section above
2. Verify your API credentials
3. Ensure stable internet connection during execution

---

**Built for speed and validation. Get your results in hours, not weeks.** 