# Data Authenticity Guide: Real vs Mocked Data

This guide helps you identify which data in the PlaceboRx repository is **real empirical data** vs **mocked/simulated data**.

## 🔍 **How to Identify Data Authenticity**

### **1. Check the Source Code**

#### **Mock Data Indicators:**
```python
# Look for these keywords in the code:
- "mock" or "_mock_"
- "fake" or "dummy"
- "simulated" or "placeholder"
- "demonstration" or "example"
- "fallback" or "backup"
```

#### **Real Data Indicators:**
```python
# Look for these keywords in the code:
- "api" or "API"
- "request" or "response"
- "live" or "real-time"
- "actual" or "genuine"
- "empirical" or "authentic"
```

### **2. Check Environment Variables**

#### **Required for Real Data:**
```bash
# Check if these are set:
echo $OPENAI_API_KEY      # Should be a real key (starts with sk-)
echo $REDDIT_CLIENT_ID    # Should be a real ID (not "your_reddit_client_id_here")
echo $REDDIT_CLIENT_SECRET # Should be a real secret
```

#### **Test Environment Variables:**
```bash
# Run this command to check:
python3 test_automation.py
```

### **3. Check Generated Files**

#### **File Size Indicators:**
- **Large files** (>1KB) = Likely real data
- **Small files** (<100 bytes) = Likely mock/empty data
- **Template files** = Mock data

#### **Content Indicators:**
- **Real data**: Varied, realistic values, timestamps
- **Mock data**: Round numbers, placeholder text, "example" values

## 📊 **Data Source Analysis**

### **✅ REAL EMPIRICAL DATA SOURCES**

#### **1. ClinicalTrials.gov API** 
**File**: `clinical_trials_analyzer.py`
**Status**: ✅ **REAL**
**Evidence**:
```python
# Real API calls
self.api_url = "https://clinicaltrials.gov/api/v2/studies"
response = requests.get(self.api_url, params=params)
```
**How to verify**: Check `clinical_trials_results.csv` - if it contains real trial data, it's empirical.

#### **2. OpenAI API**
**File**: `openai_processor.py`
**Status**: ✅ **REAL** (if API key is set)
**Evidence**:
```python
# Real OpenAI calls
client = OpenAI(api_key=self.openai_api_key)
response = client.chat.completions.create(...)
```
**How to verify**: Check if your OpenAI API key is real (starts with `sk-`)

#### **3. Reddit API**
**File**: `market_analyzer.py`
**Status**: ⚠️ **CONDITIONAL** (real if API keys set)
**Evidence**:
```python
# Real Reddit API calls
self.reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)
```
**How to verify**: Check if Reddit API keys are real (not placeholders)

### **⚠️ MOCKED/FALLBACK DATA SOURCES**

#### **1. PubMed API**
**File**: `pubmed_analyzer.py`
**Status**: ⚠️ **MIXED** (real API calls, mock fallback)
**Evidence**:
```python
# Mock fallback when API fails
def _get_mock_pubmed_data(self, query: str, max_results: int) -> List[Dict]:
    """Provide mock PubMed data for demonstration"""
    mock_articles = [
        {
            'pmid': '12345678',  # Fake PMID
            'title': 'Open-label placebo treatment in chronic pain...',
            # ... more mock data
        }
    ]
```
**How to verify**: Check if PMIDs are real (8-digit numbers) or fake (like 12345678)

#### **2. OpenAI Fallback**
**File**: `openai_processor.py`
**Status**: ⚠️ **MOCK** (when API key missing)
**Evidence**:
```python
# Mock insights when OpenAI unavailable
print("⚠️ OpenAI not available. Using mock AI insights for demonstration.")
insights = self._generate_mock_insights(clinical_data, market_data, pubmed_data)
```

#### **3. API Endpoint Template Data**
**File**: `pages/api/hypothesis-data.js`
**Status**: ⚠️ **TEMPLATE** (hardcoded values)
**Evidence**:
```javascript
// Template values (not real data)
clinicalEvidence: {
    totalTrials: 5234,  // This is a template number
    digitalInterventions: 156,
    // ... more template data
}
```

## 🔧 **How to Check Data Authenticity**

### **Step 1: Run the Test Suite**
```bash
python3 test_automation.py
```
This will show you which APIs are working and which are using fallbacks.

### **Step 2: Check Generated Files**
```bash
# Check file sizes and content
ls -la *.csv *.json *.md
head -5 pubmed_analysis_results.csv
cat clinical_trials_report.md
```

### **Step 3: Check Logs**
```bash
# Check automation logs
tail -20 automation.log
```

### **Step 4: Verify API Calls**
```bash
# Check if real API calls were made
grep -i "api" automation.log
grep -i "request" automation.log
```

## 📋 **Data Authenticity Checklist**

### **Before Running Pipeline:**
- [ ] OpenAI API key is real (starts with `sk-`)
- [ ] Reddit Client ID is real (not placeholder)
- [ ] Reddit Client Secret is real (not placeholder)
- [ ] Python dependencies installed (pandas, requests, etc.)

### **After Running Pipeline:**
- [ ] Check `automation.log` for API call success/failure
- [ ] Verify file sizes (real data = larger files)
- [ ] Check for realistic values vs template numbers
- [ ] Look for timestamps and real identifiers

### **Real Data Indicators:**
- [ ] Varied, non-round numbers
- [ ] Real timestamps (current dates)
- [ ] Real identifiers (PMID, NCT IDs)
- [ ] API call success messages in logs
- [ ] Large file sizes with detailed content

### **Mock Data Indicators:**
- [ ] Round numbers (100, 1000, etc.)
- [ ] Placeholder text ("example", "demo")
- [ ] Fake identifiers (12345678, "test")
- [ ] API failure messages in logs
- [ ] Small file sizes or empty content

## 🎯 **Current Status Analysis**

Based on your current setup:

| Data Source | Status | Authenticity | Evidence |
|-------------|--------|--------------|----------|
| **ClinicalTrials.gov** | ✅ Real | High | Real API calls made |
| **OpenAI** | ✅ Real | High | Real API key configured |
| **Reddit** | ❌ Mock | Low | Missing API keys |
| **PubMed** | ⚠️ Mixed | Medium | API fails, uses mock fallback |
| **API Endpoint** | ⚠️ Template | Low | Hardcoded template values |

## 🚀 **To Get Fully Real Data:**

1. **Get Reddit API Keys**:
   ```bash
   # Update .env file with real Reddit Client ID
   REDDIT_CLIENT_ID=your_real_client_id_here
   ```

2. **Run Full Pipeline**:
   ```bash
   source venv/bin/activate
   python3 automated_hypothesis_pipeline.py
   ```

3. **Verify Results**:
   ```bash
   # Check for real data indicators
   ls -la *.csv *.json
   grep -i "real\|actual" *.log
   ```

## 📝 **Example: Real vs Mock Data**

### **Real Data Example:**
```json
{
  "pmid": "34567890",
  "title": "Open-label placebo effects in chronic pain",
  "publication_date": "2023-06-15",
  "effect_size": 0.347,
  "p_value": 0.023
}
```

### **Mock Data Example:**
```json
{
  "pmid": "12345678",
  "title": "Example study for demonstration",
  "publication_date": "2023",
  "effect_size": 0.35,
  "p_value": 0.001
}
```

**Key differences**: Real PMIDs are varied, mock ones are sequential. Real dates are specific, mock dates are generic.

---

**Remember**: The system is designed to work with real data but gracefully falls back to realistic mock data when APIs are unavailable. This ensures the pipeline always runs, even in demonstration mode. 