# PlaceboRx Data Authenticity Report
Generated: 2025-07-27 14:53:41

## 🔑 Environment Variables
- **openai**: ✅ Real API Key
- **reddit_id**: ❌ Missing/Placeholder Client ID
- **reddit_secret**: ❌ Missing/Placeholder Client Secret

## 📊 File Size Analysis
- **clinical_trials_results.csv**: 200 bytes - ⚠️ Possibly Mock (small)
- **pubmed_analysis_results.csv**: 11930 bytes - ✅ Very Likely Real (large)
- **market_validation_report.md**: 25 bytes - ❌ Likely Mock (too small)
- **placeborx_validation_report.md**: 493 bytes - ⚠️ Possibly Mock (small)
- **openai_analysis_results.json**: 14466 bytes - ✅ Very Likely Real (large)

## 🔬 Data Source Analysis
### PubMed Literature Data
- **Status**: ⚠️ Mock Data Detected
  - Sequential fake PMIDs detected
  - No real PMIDs found
  - Generic publication dates

### Clinical Trials Data
- **Status**: ⚠️ Empty Data
  - No trials found - this could be real (no OLP trials exist) or mock

### Market Validation Data
- **Status**: ❌ No Market Data
  - Missing Reddit API keys

### API Endpoint Data
- **Status**: ⚠️ Template Data Detected
  - Template number: 5234
  - Template number: 156
  - Template number: 3
  - Template number: 3247
  - Template number: 18

## 📈 Authenticity Summary
- **Real Data Sources**: 0
- **Mock Data Sources**: 2
- **Unknown/Empty**: 2

⚠️ **WARNING**: More mock data than real data detected!
To get real data, ensure all API keys are configured properly.