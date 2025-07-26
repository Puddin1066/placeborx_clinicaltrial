#!/usr/bin/env python3
"""
Test script to run only the clinical trials analysis portion
This bypasses the Reddit API requirements to test clinical functionality
"""

import sys
import time
from clinical_trials_analyzer import ClinicalTrialsAnalyzer

def test_clinical_analysis():
    """Test the clinical trials analysis independently"""
    print("🔬 Testing Clinical Trials Analysis")
    print("="*50)
    
    start_time = time.time()
    
    try:
        # Initialize analyzer
        analyzer = ClinicalTrialsAnalyzer()
        print("✅ Clinical analyzer initialized")
        
        # Run analysis
        print("🔍 Searching for open-label placebo trials...")
        results_df = analyzer.analyze_trials()
        
        if not results_df.empty:
            print(f"✅ Found {len(results_df)} relevant trials")
            
            # Show basic stats
            digital_trials = len(results_df[results_df['is_digital'] == True])
            significant_trials = len(results_df[results_df['statistical_significance'] == True])
            
            print(f"   - Digital interventions: {digital_trials}")
            print(f"   - Statistically significant: {significant_trials}")
            
            # Save results
            results_df.to_csv('clinical_trials_test_results.csv', index=False)
            print("✅ Results saved to clinical_trials_test_results.csv")
            
            # Generate report
            report = analyzer.generate_summary_report(results_df)
            with open('clinical_trials_test_report.md', 'w') as f:
                f.write(report)
            print("✅ Report saved to clinical_trials_test_report.md")
            
        else:
            print("⚠️  No relevant trials found")
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  Clinical analysis completed in {elapsed_time:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Clinical analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clinical_analysis()
    sys.exit(0 if success else 1)