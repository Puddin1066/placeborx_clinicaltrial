# Automated PlaceboRx Hypothesis Testing Pipeline

This automation system executes the complete hypothesis testing workflow and deploys updates to Vercel automatically.

## 🚀 Quick Start

### Option 1: One-Command Execution (Recommended)
```bash
./run_hypothesis_pipeline.sh
```

### Option 2: Step-by-Step Execution
```bash
# 1. Setup environment only
./run_hypothesis_pipeline.sh --setup-only

# 2. Check prerequisites only
./run_hypothesis_pipeline.sh --check-only

# 3. Run complete pipeline
python3 automated_hypothesis_pipeline.py
```

## 📋 What the Pipeline Does

The automated pipeline executes these steps in sequence:

### 1. **Data Analysis** 🔬
- Runs `enhanced_main_pipeline.py` for comprehensive clinical trial analysis
- Executes individual analyzers: `clinical_trials_analyzer.py`, `market_analyzer.py`, `pubmed_analyzer.py`
- Generates visualizations with `visualization_engine.py`
- Processes data through OpenAI for AI-powered insights

### 2. **API Update** 📊
- Reads analysis results and generates new hypothesis data
- Updates `pages/api/hypothesis-data.js` with fresh data
- Creates backup of previous API file
- Ensures data consistency and proper formatting

### 3. **Content Update** 📝
- Updates `vercel_landing_content.js` with new metrics
- Refreshes key performance indicators
- Updates timestamps and version information
- Creates backup of previous content file

### 4. **Deployment** 🚀
- Commits all changes to git with timestamp
- Pushes to main branch to trigger Vercel deployment
- Handles git operations safely with error checking

### 5. **Verification** ✅
- Waits for Vercel deployment to complete
- Tests site accessibility
- Verifies API endpoint functionality
- Confirms data freshness

## 🛠️ Prerequisites

### Required Software
- **Python 3.8+** with pip
- **Git** for version control
- **Node.js** (for Vercel deployment)
- **Vercel CLI** (optional, for advanced features)

### Required Environment Variables
Create a `.env` file in the project root:

```bash
# OpenAI API (required for AI analysis)
OPENAI_API_KEY=your_openai_api_key_here

# Reddit API (required for market analysis)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=PlaceboRx_Validation_Bot/1.0

# Optional: Vercel configuration
VERCEL_PROJECT_ID=your_vercel_project_id
VERCEL_TOKEN=your_vercel_token
```

### Required Python Packages
The script automatically installs dependencies from:
- `requirements_enhanced.txt` (preferred)
- `requirements.txt` (fallback)
- Basic packages if no requirements file exists

## 📁 File Structure

```
placeborx_clinicaltrial/
├── automated_hypothesis_pipeline.py    # Main automation script
├── run_hypothesis_pipeline.sh          # Shell wrapper script
├── enhanced_main_pipeline.py           # Enhanced analysis pipeline
├── clinical_trials_analyzer.py         # Clinical trials analysis
├── market_analyzer.py                  # Market validation analysis
├── pubmed_analyzer.py                  # Literature analysis
├── openai_processor.py                 # AI-powered insights
├── visualization_engine.py             # Data visualization
├── pages/
│   ├── api/
│   │   └── hypothesis-data.js          # API endpoint (auto-updated)
│   ├── index.js                        # Landing page
│   └── dashboard.js                    # Dashboard page
├── vercel_landing_content.js           # Landing content (auto-updated)
├── vercel.json                         # Vercel configuration
├── package.json                        # Node.js dependencies
└── requirements_enhanced.txt           # Python dependencies
```

## 🔧 Configuration

### Pipeline Configuration
Edit `automated_hypothesis_pipeline.py` to customize:

```python
self.config = {
    'git_repo': '.',                    # Git repository path
    'vercel_project_name': 'placeborx-clinicaltrial',  # Vercel project name
    'api_endpoint': '/api/hypothesis-data',            # API endpoint path
    'verification_timeout': 300,        # Deployment verification timeout
    'max_retries': 3                    # Maximum retry attempts
}
```

### Analysis Configuration
Modify analysis parameters in:
- `enhanced_config.py` - Main configuration
- `config.py` - API keys and search terms
- Individual analyzer files for specific settings

## 📊 Output and Logging

### Log Files
- `automation.log` - Main automation log
- `pipeline.log` - Analysis pipeline log
- Backup files with timestamps for API and content files

### Generated Data
- `clinical_analysis_results.json` - Clinical trials analysis
- `market_analysis_results.json` - Market validation data
- `pubmed_analysis_results.json` - Literature analysis
- Visualizations in `output/` directory

### Success Indicators
- ✅ All steps completed successfully
- 📊 Updated metrics in dashboard
- 🌐 Live site with fresh data
- 🔄 Automatic deployment triggered

## 🚨 Troubleshooting

### Common Issues

#### 1. **Environment Variables Missing**
```bash
# Error: Missing environment variables
# Solution: Create .env file with required API keys
```

#### 2. **Python Dependencies**
```bash
# Error: Module not found
# Solution: Run setup first
./run_hypothesis_pipeline.sh --setup-only
```

#### 3. **Git Issues**
```bash
# Error: Not in git repository
# Solution: Initialize git or check directory
git init
git remote add origin your_repo_url
```

#### 4. **Vercel Deployment Fails**
```bash
# Error: Deployment timeout
# Solution: Check Vercel project configuration
vercel ls
vercel link
```

#### 5. **API Endpoint Issues**
```bash
# Error: API not accessible
# Solution: Check API file syntax and Vercel deployment
curl https://your-site.vercel.app/api/hypothesis-data
```

### Debug Mode
Enable verbose logging:
```bash
./run_hypothesis_pipeline.sh --verbose
```

### Manual Recovery
If automation fails, you can run steps manually:
```bash
# 1. Run analysis only
python3 enhanced_main_pipeline.py

# 2. Update API manually
# Edit pages/api/hypothesis-data.js

# 3. Update content manually
# Edit vercel_landing_content.js

# 4. Deploy manually
git add .
git commit -m "Manual update"
git push origin main
```

## 🔄 Scheduling

### Cron Job (Linux/Mac)
Add to crontab for daily execution:
```bash
# Edit crontab
crontab -e

# Add line for daily execution at 2 AM
0 2 * * * cd /path/to/placeborx_clinicaltrial && ./run_hypothesis_pipeline.sh >> /var/log/placeborx_pipeline.log 2>&1
```

### GitHub Actions (Recommended)
Create `.github/workflows/hypothesis-pipeline.yml`:
```yaml
name: Automated Hypothesis Testing
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:     # Manual trigger

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: |
          pip install -r requirements_enhanced.txt
          python3 automated_hypothesis_pipeline.py
```

## 📈 Monitoring

### Success Metrics
- Pipeline execution time < 10 minutes
- All steps complete successfully
- Site accessibility verified
- API endpoint responding correctly
- Data freshness confirmed

### Alerting
Set up monitoring for:
- Pipeline failures
- Deployment timeouts
- API endpoint errors
- Data quality issues

## 🔒 Security Considerations

### API Key Management
- Store API keys in environment variables
- Never commit `.env` files to git
- Use Vercel environment variables for production
- Rotate API keys regularly

### Data Privacy
- Anonymize patient data in analysis
- Follow HIPAA guidelines for clinical data
- Implement proper data retention policies
- Secure API endpoints with authentication if needed

## 📞 Support

### Getting Help
1. Check the logs: `tail -f automation.log`
2. Verify prerequisites: `./run_hypothesis_pipeline.sh --check-only`
3. Test individual components manually
4. Review error messages in the logs

### Contributing
To improve the automation:
1. Fork the repository
2. Make changes to automation scripts
3. Test thoroughly
4. Submit pull request with detailed description

---

**Last Updated:** $(date)
**Version:** 1.0.0
**Maintainer:** PlaceboRx Development Team 