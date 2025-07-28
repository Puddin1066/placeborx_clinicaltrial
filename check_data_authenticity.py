#!/usr/bin/env python3
"""
Data Authenticity Checker for PlaceboRx
Automatically identifies real vs mocked data in the repository
"""

import os
import json
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Tuple

class DataAuthenticityChecker:
    """Check the authenticity of data in the PlaceboRx repository"""
    
    def __init__(self):
        self.results = {}
        
    def check_file_sizes(self) -> Dict[str, str]:
        """Check file sizes to determine likely authenticity"""
        print("📊 Checking file sizes...")
        
        file_sizes = {}
        data_files = [
            'clinical_trials_results.csv',
            'pubmed_analysis_results.csv', 
            'market_validation_report.md',
            'placeborx_validation_report.md',
            'openai_analysis_results.json'
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                file_sizes[file_path] = {
                    'size': size,
                    'authenticity': self._assess_size_authenticity(size)
                }
        
        return file_sizes
    
    def _assess_size_authenticity(self, size: int) -> str:
        """Assess authenticity based on file size"""
        if size < 100:
            return "❌ Likely Mock (too small)"
        elif size < 1000:
            return "⚠️ Possibly Mock (small)"
        elif size < 10000:
            return "✅ Likely Real (medium)"
        else:
            return "✅ Very Likely Real (large)"
    
    def check_pubmed_data(self) -> Dict[str, str]:
        """Check PubMed data for mock indicators"""
        print("🔬 Checking PubMed data...")
        
        if not os.path.exists('pubmed_analysis_results.csv'):
            return {'status': '❌ File not found'}
        
        try:
            df = pd.read_csv('pubmed_analysis_results.csv')
            
            # Check PMIDs for mock patterns
            mock_indicators = []
            
            if 'pmid' in df.columns:
                pmids = df['pmid'].astype(str).tolist()
                
                # Check for sequential fake PMIDs
                fake_pmids = ['12345678', '12345679', '12345680', '12345681', '12345682']
                if any(pmid in fake_pmids for pmid in pmids):
                    mock_indicators.append("Sequential fake PMIDs detected")
                
                # Check for realistic PMID patterns (8 digits, varied)
                real_pmids = [p for p in pmids if re.match(r'^\d{8}$', p) and p not in fake_pmids]
                if len(real_pmids) == 0:
                    mock_indicators.append("No real PMIDs found")
            
            # Check publication dates
            if 'publication_date' in df.columns:
                dates = df['publication_date'].astype(str).tolist()
                generic_dates = [d for d in dates if d in ['2023', '2022', '2021']]
                if len(generic_dates) > len(dates) * 0.5:
                    mock_indicators.append("Generic publication dates")
            
            # Check effect sizes for round numbers
            if 'effect_size' in df.columns:
                effect_sizes = df['effect_size'].dropna().tolist()
                round_effects = [e for e in effect_sizes if e in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]]
                if len(round_effects) > len(effect_sizes) * 0.5:
                    mock_indicators.append("Round effect sizes")
            
            if mock_indicators:
                return {
                    'status': '⚠️ Mock Data Detected',
                    'indicators': mock_indicators,
                    'authenticity': 'Mock'
                }
            else:
                return {
                    'status': '✅ Real Data Detected',
                    'authenticity': 'Real'
                }
                
        except Exception as e:
            return {'status': f'❌ Error: {str(e)}'}
    
    def check_clinical_trials_data(self) -> Dict[str, str]:
        """Check clinical trials data"""
        print("🏥 Checking clinical trials data...")
        
        if not os.path.exists('clinical_trials_results.csv'):
            return {'status': '❌ File not found'}
        
        try:
            df = pd.read_csv('clinical_trials_results.csv')
            
            if len(df) == 0:
                return {
                    'status': '⚠️ Empty Data',
                    'note': 'No trials found - this could be real (no OLP trials exist) or mock',
                    'authenticity': 'Unknown'
                }
            
            # Check for real NCT IDs
            if 'nct_id' in df.columns:
                nct_ids = df['nct_id'].dropna().tolist()
                real_nct_ids = [n for n in nct_ids if re.match(r'^NCT\d{8}$', str(n))]
                
                if len(real_nct_ids) > 0:
                    return {
                        'status': '✅ Real Clinical Data',
                        'real_nct_count': len(real_nct_ids),
                        'authenticity': 'Real'
                    }
                else:
                    return {
                        'status': '⚠️ No Real NCT IDs',
                        'authenticity': 'Mock'
                    }
            
            return {'status': '⚠️ Unknown format', 'authenticity': 'Unknown'}
            
        except Exception as e:
            return {'status': f'❌ Error: {str(e)}'}
    
    def check_market_data(self) -> Dict[str, str]:
        """Check market validation data"""
        print("📈 Checking market data...")
        
        if not os.path.exists('market_validation_report.md'):
            return {'status': '❌ File not found'}
        
        try:
            with open('market_validation_report.md', 'r') as f:
                content = f.read()
            
            if "No market data available" in content:
                return {
                    'status': '❌ No Market Data',
                    'reason': 'Missing Reddit API keys',
                    'authenticity': 'Mock'
                }
            
            # Check for real Reddit data indicators
            if "subreddit" in content.lower() or "reddit" in content.lower():
                return {
                    'status': '✅ Real Market Data',
                    'authenticity': 'Real'
                }
            
            return {'status': '⚠️ Unknown format', 'authenticity': 'Unknown'}
            
        except Exception as e:
            return {'status': f'❌ Error: {str(e)}'}
    
    def check_environment_variables(self) -> Dict[str, str]:
        """Check environment variables for API keys"""
        print("🔑 Checking environment variables...")
        
        env_status = {}
        
        # Check OpenAI
        openai_key = os.getenv('OPENAI_API_KEY', '')
        if openai_key and openai_key.startswith('sk-'):
            env_status['openai'] = '✅ Real API Key'
        elif openai_key:
            env_status['openai'] = '⚠️ Invalid API Key Format'
        else:
            env_status['openai'] = '❌ No API Key'
        
        # Check Reddit
        reddit_id = os.getenv('REDDIT_CLIENT_ID', '')
        reddit_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        
        if reddit_id and reddit_id != 'your_reddit_client_id_here':
            env_status['reddit_id'] = '✅ Real Client ID'
        else:
            env_status['reddit_id'] = '❌ Missing/Placeholder Client ID'
        
        if reddit_secret and reddit_secret != 'your_reddit_client_secret_here':
            env_status['reddit_secret'] = '✅ Real Client Secret'
        else:
            env_status['reddit_secret'] = '❌ Missing/Placeholder Client Secret'
        
        return env_status
    
    def check_api_endpoint_data(self) -> Dict[str, str]:
        """Check API endpoint for template data"""
        print("🌐 Checking API endpoint data...")
        
        if not os.path.exists('pages/api/hypothesis-data.js'):
            return {'status': '❌ API file not found'}
        
        try:
            with open('pages/api/hypothesis-data.js', 'r') as f:
                content = f.read()
            
            # Check for template values
            template_indicators = []
            
            # Round numbers that are likely template
            template_numbers = ['5234', '156', '3', '3247', '18']
            for num in template_numbers:
                if num in content:
                    template_indicators.append(f"Template number: {num}")
            
            # Check for hardcoded dates
            if 'new Date().toISOString()' in content:
                template_indicators.append("Dynamic timestamp (good)")
            elif '2023' in content or '2024' in content:
                template_indicators.append("Hardcoded year")
            
            if template_indicators:
                return {
                    'status': '⚠️ Template Data Detected',
                    'indicators': template_indicators,
                    'authenticity': 'Template'
                }
            else:
                return {
                    'status': '✅ Real Data',
                    'authenticity': 'Real'
                }
                
        except Exception as e:
            return {'status': f'❌ Error: {str(e)}'}
    
    def generate_authenticity_report(self) -> str:
        """Generate comprehensive authenticity report"""
        print("📋 Generating authenticity report...")
        
        report = []
        report.append("# PlaceboRx Data Authenticity Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Check environment variables
        report.append("## 🔑 Environment Variables")
        env_status = self.check_environment_variables()
        for key, status in env_status.items():
            report.append(f"- **{key}**: {status}")
        report.append("")
        
        # Check file sizes
        report.append("## 📊 File Size Analysis")
        file_sizes = self.check_file_sizes()
        for file_path, info in file_sizes.items():
            report.append(f"- **{file_path}**: {info['size']} bytes - {info['authenticity']}")
        report.append("")
        
        # Check individual data sources
        report.append("## 🔬 Data Source Analysis")
        
        # PubMed
        pubmed_status = self.check_pubmed_data()
        report.append(f"### PubMed Literature Data")
        report.append(f"- **Status**: {pubmed_status['status']}")
        if 'indicators' in pubmed_status:
            for indicator in pubmed_status['indicators']:
                report.append(f"  - {indicator}")
        report.append("")
        
        # Clinical Trials
        clinical_status = self.check_clinical_trials_data()
        report.append(f"### Clinical Trials Data")
        report.append(f"- **Status**: {clinical_status['status']}")
        if 'note' in clinical_status:
            report.append(f"  - {clinical_status['note']}")
        report.append("")
        
        # Market Data
        market_status = self.check_market_data()
        report.append(f"### Market Validation Data")
        report.append(f"- **Status**: {market_status['status']}")
        if 'reason' in market_status:
            report.append(f"  - {market_status['reason']}")
        report.append("")
        
        # API Endpoint
        api_status = self.check_api_endpoint_data()
        report.append(f"### API Endpoint Data")
        report.append(f"- **Status**: {api_status['status']}")
        if 'indicators' in api_status:
            for indicator in api_status['indicators']:
                report.append(f"  - {indicator}")
        report.append("")
        
        # Summary
        report.append("## 📈 Authenticity Summary")
        
        real_count = 0
        mock_count = 0
        unknown_count = 0
        
        for status in [pubmed_status, clinical_status, market_status, api_status]:
            if 'authenticity' in status:
                if status['authenticity'] == 'Real':
                    real_count += 1
                elif status['authenticity'] == 'Mock':
                    mock_count += 1
                else:
                    unknown_count += 1
        
        report.append(f"- **Real Data Sources**: {real_count}")
        report.append(f"- **Mock Data Sources**: {mock_count}")
        report.append(f"- **Unknown/Empty**: {unknown_count}")
        report.append("")
        
        if mock_count > real_count:
            report.append("⚠️ **WARNING**: More mock data than real data detected!")
            report.append("To get real data, ensure all API keys are configured properly.")
        elif real_count > 0:
            report.append("✅ **GOOD**: Some real data sources are working.")
        else:
            report.append("❌ **ISSUE**: No real data sources detected.")
        
        return "\n".join(report)
    
    def run_full_check(self):
        """Run complete authenticity check"""
        print("🔍 PlaceboRx Data Authenticity Checker")
        print("=" * 50)
        
        report = self.generate_authenticity_report()
        
        # Save report
        with open('data_authenticity_report.md', 'w') as f:
            f.write(report)
        
        print("\n" + "=" * 50)
        print("✅ Authenticity check completed!")
        print("📄 Report saved to: data_authenticity_report.md")
        print("\n" + report)

def main():
    """Main execution"""
    checker = DataAuthenticityChecker()
    checker.run_full_check()

if __name__ == "__main__":
    main() 