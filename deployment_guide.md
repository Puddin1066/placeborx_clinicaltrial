# 🚀 PlaceboRx Deployment Guide

## 📋 **Quick Deployment Decision Matrix**

| Use Case | Recommended Platform | Complexity | Cost | Setup Time |
|----------|---------------------|------------|------|------------|
| **Research Demo** | Streamlit Cloud | Low | Free | 1-2 hours |
| **Technical Portfolio** | GitHub Pages + Binder | Low | Free | 2-4 hours |
| **Academic Presentation** | Hugging Face Spaces | Medium | Free | 3-6 hours |
| **Professional Demo** | AWS/GCP/Azure | High | $20-50/month | 1-2 days |
| **Production Research Tool** | Custom Cloud Infrastructure | Very High | $100+/month | 1-2 weeks |

---

## 🎯 **Deployment Scenario 1: Research Demonstration (RECOMMENDED)**

### **Platform: Streamlit Cloud**
- ✅ **Best for**: Academic demos, hypothesis validation, research presentations
- ✅ **Pros**: Easy setup, interactive, professional appearance
- ⚠️ **Limitations**: Limited compute resources, public visibility

### **Step-by-Step Deployment**

#### **Step 1: Prepare Repository**
```bash
# 1. Ensure your repository has these files:
streamlit_app.py              # Main web app
requirements_streamlit.txt    # Streamlit dependencies
enhanced_config.py           # Configuration
.streamlit/config.toml       # Streamlit config (optional)
README.md                    # Documentation
```

#### **Step 2: Create Streamlit Configuration**
```bash
mkdir .streamlit
cat > .streamlit/config.toml << EOF
[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[browser]
gatherUsageStats = false
EOF
```

#### **Step 3: Deploy to Streamlit Cloud**
1. **Fork/Push to GitHub**: Ensure your code is in a public GitHub repository
2. **Visit**: https://share.streamlit.io/
3. **Connect GitHub**: Authenticate with your GitHub account
4. **Deploy App**: 
   - Repository: `your-username/placebo-rx-pipeline`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
5. **Configure Secrets** (if needed):
   - Add API keys in Streamlit secrets management
   - Format: `REDDIT_CLIENT_ID = "your_key_here"`

#### **Step 4: Verify Deployment**
```bash
# Your app will be available at:
https://share.streamlit.io/your-username/placebo-rx-pipeline/main/streamlit_app.py
```

---

## 💻 **Deployment Scenario 2: Technical Portfolio**

### **Platform: GitHub Pages + Jupyter Notebooks**
- ✅ **Best for**: Technical demonstrations, code showcasing, developer portfolio
- ✅ **Pros**: Version controlled, professional, code-focused

### **Setup Instructions**

#### **Step 1: Create Portfolio Structure**
```bash
# Create portfolio branch
git checkout -b gh-pages

# Create portfolio structure
mkdir -p docs/notebooks docs/assets docs/reports
```

#### **Step 2: Generate Interactive Notebooks**
```python
# Create demo_notebook.ipynb
"""
PlaceboRx Pipeline Demonstration
Interactive analysis and visualization
"""

# Cell 1: Setup and imports
import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')

# Cell 2: Load and display sample data
sample_data = {
    'condition': ['Chronic Pain', 'Anxiety', 'Depression'],
    'baseline_placebo': [0.28, 0.24, 0.31],
    'digital_enhancement': [1.3, 1.2, 1.4],
    'estimated_effect': [0.36, 0.29, 0.43]
}
df = pd.DataFrame(sample_data)
display(df)

# Cell 3: Interactive visualization
fig = px.bar(df, x='condition', y=['baseline_placebo', 'estimated_effect'],
             title='Baseline vs Digital-Enhanced Placebo Effects',
             barmode='group')
fig.show()

# Cell 4: Analysis results
print("🎯 Key Findings:")
print("- Digital enhancement shows 20-40% improvement")
print("- Strongest effects in IBS and Depression") 
print("- Market validation confirms demand")
```

#### **Step 3: Configure GitHub Pages**
```yaml
# Create _config.yml
title: "PlaceboRx Validation Pipeline"
description: "Advanced analytics for digital placebo research"
theme: jekyll-theme-minimal
plugins:
  - jekyll-sitemap
  - jekyll-feed

# Navigation
navigation:
  - title: "Demo"
    url: "/demo"
  - title: "Documentation" 
    url: "/docs"
  - title: "Code"
    url: "https://github.com/your-username/placebo-rx-pipeline"
```

#### **Step 4: Deploy**
```bash
# Push to gh-pages branch
git add .
git commit -m "Deploy portfolio site"
git push origin gh-pages

# Enable GitHub Pages
# Go to Settings > Pages > Source: gh-pages branch
```

---

## 🤗 **Deployment Scenario 3: Hugging Face Spaces**

### **Platform: Hugging Face Spaces (Streamlit)**
- ✅ **Best for**: ML-focused demonstrations, academic community sharing
- ✅ **Pros**: ML-focused audience, good performance, community features

### **Setup Instructions**

#### **Step 1: Create Hugging Face Space**
1. **Visit**: https://huggingface.co/spaces
2. **Create New Space**:
   - Name: `placebo-rx-validation`
   - License: `Apache 2.0`
   - SDK: `Streamlit`
   - Hardware: `CPU basic` (free tier)

#### **Step 2: Configure Space**
```python
# Create app.py (entry point)
import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import your main app
from streamlit_app import main

if __name__ == "__main__":
    main()
```

```yaml
# Create requirements.txt
streamlit==1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
```

#### **Step 3: Upload Files**
```bash
# Clone the space repository
git clone https://huggingface.co/spaces/your-username/placebo-rx-validation
cd placebo-rx-validation

# Copy your files
cp ../streamlit_app.py .
cp ../enhanced_config.py .
cp ../requirements_streamlit.txt requirements.txt

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

---

## ☁️ **Deployment Scenario 4: Professional Cloud Platform**

### **Platform: AWS/GCP/Azure**
- ✅ **Best for**: Production demos, scalable research tools, professional presentations
- ⚠️ **Requires**: Cloud platform knowledge, ongoing costs

### **AWS Deployment (Example)**

#### **Step 1: Containerize Application**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **Step 2: Deploy to AWS ECS/Fargate**
```bash
# Build and push Docker image
docker build -t placebo-rx-pipeline .
docker tag placebo-rx-pipeline:latest your-account.dkr.ecr.region.amazonaws.com/placebo-rx-pipeline:latest
docker push your-account.dkr.ecr.region.amazonaws.com/placebo-rx-pipeline:latest

# Deploy using AWS CLI or CloudFormation
aws ecs create-service \
    --cluster your-cluster \
    --service-name placebo-rx-service \
    --task-definition placebo-rx-task \
    --desired-count 1 \
    --launch-type FARGATE
```

---

## 🔧 **Configuration for Different Deployment Types**

### **Research Demo Configuration**
```python
# streamlit_config.py
DEMO_MODE = True
ENABLE_FULL_PIPELINE = False
SHOW_DISCLAIMERS = True
API_RATE_LIMITS = True
MAX_ANALYSIS_TIME = 300  # 5 minutes
SAMPLE_DATA_ONLY = True
```

### **Full Pipeline Configuration**
```python
# production_config.py
DEMO_MODE = False
ENABLE_FULL_PIPELINE = True
SHOW_DISCLAIMERS = True
API_RATE_LIMITS = False
MAX_ANALYSIS_TIME = 3600  # 60 minutes
REAL_DATA_ANALYSIS = True
```

---

## 🛡️ **Security and Compliance Considerations**

### **For Research Demonstrations**
```python
# security_config.py
SECURITY_MEASURES = {
    'api_key_protection': 'Use environment variables',
    'data_privacy': 'No PII collection',
    'rate_limiting': 'Prevent API abuse',
    'disclaimer_visibility': 'Prominent research-only warnings',
    'audit_logging': 'Track usage patterns'
}
```

### **Environment Variables Setup**
```bash
# .env file (DO NOT commit to version control)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
OPENAI_API_KEY=your_openai_key_optional
DEPLOYMENT_MODE=research_demo
```

---

## 📊 **Monitoring and Analytics**

### **Basic Monitoring Setup**
```python
# monitoring.py
import logging
import streamlit as st
from datetime import datetime

def setup_monitoring():
    """Setup basic monitoring for deployment"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Track usage
    if 'usage_stats' not in st.session_state:
        st.session_state.usage_stats = {
            'page_views': 0,
            'analyses_run': 0,
            'last_activity': datetime.now()
        }

def log_user_activity(action):
    """Log user activity"""
    logging.info(f"User action: {action} at {datetime.now()}")
    st.session_state.usage_stats[action] = st.session_state.usage_stats.get(action, 0) + 1
```

---

## 🚀 **Quick Start Commands**

### **Option 1: Streamlit Cloud (Recommended)**
```bash
# 1. Push to GitHub
git add .
git commit -m "Deploy PlaceboRx pipeline"
git push origin main

# 2. Deploy at: https://share.streamlit.io/
# 3. Your app will be live in 2-3 minutes
```

### **Option 2: Local Testing**
```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run locally
streamlit run streamlit_app.py

# App will open at: http://localhost:8501
```

### **Option 3: Docker Deployment**
```bash
# Build container
docker build -t placebo-rx-app .

# Run container
docker run -p 8501:8501 placebo-rx-app

# Access at: http://localhost:8501
```

---

## ⚠️ **Important Deployment Considerations**

### **Legal and Ethical**
1. **Research Disclaimer**: Always prominent and clear
2. **Data Privacy**: No collection of personal health information
3. **Academic Use**: Clearly state research/educational purpose only
4. **Liability Limitation**: Not for clinical decision-making

### **Technical**
1. **API Rate Limits**: Implement proper rate limiting
2. **Resource Management**: Monitor compute usage on free tiers
3. **Backup Strategy**: Regular backup of configurations and results
4. **Version Control**: Tag stable releases for deployment

### **User Experience**
1. **Loading Times**: Optimize for reasonable response times
2. **Error Handling**: Graceful degradation when services are unavailable
3. **Mobile Responsiveness**: Ensure usability on different devices
4. **Accessibility**: Follow web accessibility guidelines

---

## 📞 **Support and Troubleshooting**

### **Common Issues**

**Issue**: Streamlit app won't start
```bash
# Solution: Check requirements
pip install -r requirements_streamlit.txt
streamlit --version
```

**Issue**: Missing API credentials
```bash
# Solution: Set environment variables
export REDDIT_CLIENT_ID="your_id"
export REDDIT_CLIENT_SECRET="your_secret"
```

**Issue**: Memory limits on free tier
```bash
# Solution: Enable lightweight mode
LIGHTWEIGHT_MODE=True streamlit run streamlit_app.py
```

### **Getting Help**
- 📧 **Technical Issues**: Create GitHub issues
- 💬 **Questions**: Use GitHub Discussions
- 📖 **Documentation**: Check enhanced_README.md
- 🔍 **Debugging**: Enable debug logging in configuration

---

**🎯 Bottom Line**: Start with **Streamlit Cloud** for a research demonstration. It's free, fast to deploy, and perfect for academic/research use cases. The app includes comprehensive disclaimers and focuses on hypothesis validation rather than clinical application.