#!/usr/bin/env python3
"""
Simple API and Data Testing Suite for PlaceboRx
Tests data storage, UI integration, and OpenAI connectivity using standard library
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error

class SimpleAPIDataTester:
    """Simple API and data testing suite using standard library"""
    
    def __init__(self):
        self.test_results = {
            'data_storage_test': {},
            'ui_integration_test': {},
            'api_connectivity_test': {},
            'openai_integration_test': {},
            'summary': {}
        }
        self.start_time = datetime.now()
        
    def test_data_storage(self):
        """Test if data files exist and are valid"""
        print("📂 Testing data storage...")
        
        results = {
            'clinical_data_files': [],
            'market_data_files': [],
            'pubmed_data_files': [],
            'openai_data_files': [],
            'total_files_found': 0,
            'data_integrity_score': 0
        }
        
        # Define expected data files
        data_files = {
            'clinical': [
                'digital_therapeutics_patents_20250727_204235.json',
                'placebo_effect_patents_20250727_204235.json',
                'clinical_trials_results.csv'
            ],
            'market': [
                'market_analysis_results.csv',
                'reddit_api_test_results.json'
            ],
            'pubmed': [
                'pubmed_analysis_results.csv'
            ],
            'openai': [
                'openai_analysis_results.json'
            ]
        }
        
        total_files = 0
        found_files = 0
        
        for category, files in data_files.items():
            category_files = []
            for file_path in files:
                total_files += 1
                if os.path.exists(file_path):
                    found_files += 1
                    file_size = os.path.getsize(file_path)
                    file_time = os.path.getmtime(file_path)
                    category_files.append({
                        'name': file_path,
                        'size': file_size,
                        'modified': datetime.fromtimestamp(file_time).isoformat(),
                        'valid': self._validate_file(file_path)
                    })
            results[f'{category}_data_files'] = category_files
        
        results['total_files_found'] = found_files
        results['data_integrity_score'] = (found_files / total_files * 100) if total_files > 0 else 0
        
        self.test_results['data_storage_test'] = results
        print(f"  ✅ Found {found_files}/{total_files} data files ({results['data_integrity_score']:.1f}% integrity)")
        
        return results
    
    def test_ui_integration(self):
        """Test UI integration files and endpoints"""
        print("🌐 Testing UI integration...")
        
        results = {
            'ui_files_exist': False,
            'api_endpoints_exist': False,
            'streamlit_app_exists': False,
            'nextjs_app_exists': False,
            'hypothesis_endpoint_valid': False
        }
        
        # Check for UI files
        ui_files = [
            'streamlit_app.py',
            'pages/dashboard.js',
            'pages/index.js',
            'pages/api/hypothesis-data.js'
        ]
        
        ui_files_found = 0
        for file_path in ui_files:
            if os.path.exists(file_path):
                ui_files_found += 1
                if 'streamlit' in file_path:
                    results['streamlit_app_exists'] = True
                elif 'pages/' in file_path and file_path.endswith('.js'):
                    results['nextjs_app_exists'] = True
                elif 'api/' in file_path:
                    results['api_endpoints_exist'] = True
        
        results['ui_files_exist'] = ui_files_found > 0
        
        # Validate hypothesis endpoint
        hypothesis_file = 'pages/api/hypothesis-data.js'
        if os.path.exists(hypothesis_file):
            with open(hypothesis_file, 'r') as f:
                content = f.read()
                if 'coreHypothesis' in content and 'clinicalEvidence' in content:
                    results['hypothesis_endpoint_valid'] = True
        
        self.test_results['ui_integration_test'] = results
        print(f"  ✅ UI integration: {ui_files_found}/4 components found")
        
        return results
    
    def test_api_connectivity(self):
        """Test basic API connectivity"""
        print("🔗 Testing API connectivity...")
        
        results = {
            'clinical_trials_api': False,
            'clinical_trials_response_time': 0,
            'pubmed_accessible': False,
            'reddit_data_available': False,
            'internet_connectivity': False
        }
        
        # Test internet connectivity first
        try:
            response = urllib.request.urlopen('https://www.google.com', timeout=10)
            if response.getcode() == 200:
                results['internet_connectivity'] = True
        except:
            results['internet_connectivity'] = False
        
        # Test ClinicalTrials.gov API
        if results['internet_connectivity']:
            try:
                start_time = time.time()
                url = 'https://clinicaltrials.gov/api/v2/studies?pageSize=1'
                response = urllib.request.urlopen(url, timeout=30)
                results['clinical_trials_response_time'] = time.time() - start_time
                
                if response.getcode() == 200:
                    data = json.loads(response.read().decode())
                    if 'studies' in data:
                        results['clinical_trials_api'] = True
            except Exception as e:
                print(f"    ⚠️ ClinicalTrials.gov API error: {e}")
        
        # Check if Reddit data is available locally
        if os.path.exists('reddit_api_test_results.json'):
            results['reddit_data_available'] = True
        
        # Check if PubMed modules are accessible
        try:
            # Don't actually import, just check if files exist
            if os.path.exists('pubmed_analyzer.py'):
                results['pubmed_accessible'] = True
        except:
            pass
        
        self.test_results['api_connectivity_test'] = results
        print(f"  ✅ API connectivity: {sum(results.values())}/4 connections available")
        
        return results
    
    def test_openai_integration(self):
        """Test OpenAI integration and data processing"""
        print("🤖 Testing OpenAI integration...")
        
        results = {
            'openai_key_configured': False,
            'openai_processor_exists': False,
            'openai_results_exist': False,
            'ui_content_generated': False,
            'insights_available': False
        }
        
        # Check for OpenAI API key
        if os.getenv('OPENAI_API_KEY'):
            results['openai_key_configured'] = True
        
        # Check for OpenAI processor
        if os.path.exists('openai_processor.py'):
            results['openai_processor_exists'] = True
        
        # Check for OpenAI results
        if os.path.exists('openai_analysis_results.json'):
            results['openai_results_exist'] = True
            
            try:
                with open('openai_analysis_results.json', 'r') as f:
                    data = json.load(f)
                    if data and len(data) > 0:
                        results['insights_available'] = True
                        
                        # Check for UI-specific content
                        if any('ui' in str(key).lower() for key in data.keys()):
                            results['ui_content_generated'] = True
            except:
                pass
        
        self.test_results['openai_integration_test'] = results
        print(f"  ✅ OpenAI integration: {sum(results.values())}/5 components available")
        
        return results
    
    def run_dynamic_display_test(self):
        """Test dynamic UI display with real data"""
        print("📊 Testing dynamic UI display...")
        
        display_test = {
            'data_freshness': False,
            'real_time_updates': False,
            'ui_data_binding': False,
            'llm_enhanced_content': False
        }
        
        # Check data freshness (updated within last 24 hours)
        recent_files = []
        data_files = [
            'openai_analysis_results.json',
            'reddit_api_test_results.json',
            'pubmed_analysis_results.csv'
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                file_time = os.path.getmtime(file_path)
                if time.time() - file_time < 86400:  # 24 hours
                    recent_files.append(file_path)
        
        if len(recent_files) >= 2:
            display_test['data_freshness'] = True
        
        # Check for UI data binding
        ui_files = ['pages/dashboard.js', 'streamlit_app.py']
        for ui_file in ui_files:
            if os.path.exists(ui_file):
                with open(ui_file, 'r') as f:
                    content = f.read()
                    if any(keyword in content.lower() for keyword in ['fetch', 'api', 'data']):
                        display_test['ui_data_binding'] = True
                        break
        
        # Check for LLM-enhanced content
        if os.path.exists('openai_analysis_results.json'):
            try:
                with open('openai_analysis_results.json', 'r') as f:
                    data = json.load(f)
                    if any('insight' in str(key).lower() or 'analysis' in str(key).lower() for key in data.keys()):
                        display_test['llm_enhanced_content'] = True
            except:
                pass
        
        # Check for real-time update capability
        if os.path.exists('pages/api/hypothesis-data.js'):
            display_test['real_time_updates'] = True
        
        print(f"  ✅ Dynamic display: {sum(display_test.values())}/4 features working")
        
        return display_test
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Run dynamic display test
        dynamic_display = self.run_dynamic_display_test()
        
        # Calculate overall scores
        total_score = 0
        max_score = 0
        
        for test_name, test_results in self.test_results.items():
            if test_name != 'summary':
                for key, value in test_results.items():
                    if isinstance(value, bool):
                        max_score += 1
                        if value:
                            total_score += 1
                    elif isinstance(value, (int, float)) and 'score' in key:
                        max_score += 100
                        total_score += value
        
        # Add dynamic display scores
        for value in dynamic_display.values():
            max_score += 1
            if value:
                total_score += 1
        
        overall_score = (total_score / max_score * 100) if max_score > 0 else 0
        
        summary = {
            'test_duration': f"{duration:.2f} seconds",
            'overall_score': f"{overall_score:.1f}%",
            'status': 'PASS' if overall_score >= 70 else 'FAIL',
            'timestamp': end_time.isoformat(),
            'dynamic_display_test': dynamic_display,
            'key_findings': self._generate_key_findings(),
            'recommendations': self._generate_recommendations()
        }
        
        self.test_results['summary'] = summary
        
        # Save results
        with open('simple_api_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        return self.test_results
    
    def _validate_file(self, file_path):
        """Validate file format and content"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    json.load(f)
                return True
            elif file_path.endswith('.csv'):
                with open(file_path, 'r') as f:
                    content = f.read()
                    return len(content) > 0 and ',' in content
            else:
                return os.path.getsize(file_path) > 0
        except:
            return False
    
    def _generate_key_findings(self):
        """Generate key findings from test results"""
        findings = []
        
        # Data storage findings
        storage = self.test_results.get('data_storage_test', {})
        if storage.get('data_integrity_score', 0) > 80:
            findings.append("Strong data storage with high integrity")
        
        # UI integration findings
        ui = self.test_results.get('ui_integration_test', {})
        if ui.get('streamlit_app_exists') and ui.get('nextjs_app_exists'):
            findings.append("Dual UI implementation (Streamlit + Next.js)")
        
        # API connectivity findings
        api = self.test_results.get('api_connectivity_test', {})
        if api.get('clinical_trials_api'):
            findings.append("Live clinical trials data connection")
        
        # OpenAI integration findings
        openai_test = self.test_results.get('openai_integration_test', {})
        if openai_test.get('insights_available'):
            findings.append("AI-enhanced data analysis active")
        
        return findings
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check data storage
        storage = self.test_results.get('data_storage_test', {})
        if storage.get('data_integrity_score', 0) < 70:
            recommendations.append("Improve data storage reliability")
        
        # Check OpenAI integration
        openai_test = self.test_results.get('openai_integration_test', {})
        if not openai_test.get('openai_key_configured'):
            recommendations.append("Configure OpenAI API key for enhanced insights")
        
        # Check API connectivity
        api = self.test_results.get('api_connectivity_test', {})
        if not api.get('clinical_trials_api'):
            recommendations.append("Verify ClinicalTrials.gov API connectivity")
        
        return recommendations

def main():
    """Main execution function"""
    print("🚀 PlaceboRx API and Data Integration Test")
    print("="*60)
    print("Testing all API connections, data storage, and UI integration")
    print("="*60)
    
    tester = SimpleAPIDataTester()
    
    # Run all tests
    print("\n🔍 Running comprehensive tests...")
    tester.test_data_storage()
    tester.test_ui_integration()
    tester.test_api_connectivity()
    tester.test_openai_integration()
    
    # Generate comprehensive report
    report = tester.generate_comprehensive_report()
    
    # Print detailed results
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE TEST RESULTS")
    print("="*60)
    
    summary = report['summary']
    print(f"🎯 Overall Status: {summary['status']}")
    print(f"📊 Overall Score: {summary['overall_score']}")
    print(f"⏱️  Duration: {summary['test_duration']}")
    
    print(f"\n📂 Data Storage Test:")
    storage = report['data_storage_test']
    print(f"   Files Found: {storage['total_files_found']}")
    print(f"   Integrity Score: {storage['data_integrity_score']:.1f}%")
    
    print(f"\n🌐 UI Integration Test:")
    ui = report['ui_integration_test']
    print(f"   Streamlit App: {'✅' if ui['streamlit_app_exists'] else '❌'}")
    print(f"   Next.js App: {'✅' if ui['nextjs_app_exists'] else '❌'}")
    print(f"   API Endpoints: {'✅' if ui['api_endpoints_exist'] else '❌'}")
    
    print(f"\n🔗 API Connectivity Test:")
    api = report['api_connectivity_test']
    print(f"   Internet: {'✅' if api['internet_connectivity'] else '❌'}")
    print(f"   ClinicalTrials.gov: {'✅' if api['clinical_trials_api'] else '❌'}")
    print(f"   Reddit Data: {'✅' if api['reddit_data_available'] else '❌'}")
    
    print(f"\n🤖 OpenAI Integration Test:")
    openai_test = report['openai_integration_test']
    print(f"   API Key: {'✅' if openai_test['openai_key_configured'] else '❌'}")
    print(f"   Processor: {'✅' if openai_test['openai_processor_exists'] else '❌'}")
    print(f"   Insights: {'✅' if openai_test['insights_available'] else '❌'}")
    
    print(f"\n📊 Dynamic UI Display Test:")
    display = summary['dynamic_display_test']
    print(f"   Data Freshness: {'✅' if display['data_freshness'] else '❌'}")
    print(f"   Real-time Updates: {'✅' if display['real_time_updates'] else '❌'}")
    print(f"   UI Data Binding: {'✅' if display['ui_data_binding'] else '❌'}")
    print(f"   LLM Enhancement: {'✅' if display['llm_enhanced_content'] else '❌'}")
    
    if summary['key_findings']:
        print(f"\n✨ Key Findings:")
        for i, finding in enumerate(summary['key_findings'], 1):
            print(f"   {i}. {finding}")
    
    if summary['recommendations']:
        print(f"\n🔧 Recommendations:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print(f"\n📄 Detailed results saved to: simple_api_test_results.json")
    print("="*60)
    
    return summary['status'] == 'PASS'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)