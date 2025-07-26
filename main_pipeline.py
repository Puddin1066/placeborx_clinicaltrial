#!/usr/bin/env python3
"""
PlaceboRx Validation Pipeline
Solo entrepreneur execution - 2-4 hours runtime
"""

import time
import os
from datetime import datetime
from clinical_trials_analyzer import ClinicalTrialsAnalyzer
from market_analyzer import MarketAnalyzer

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease create a .env file with your Reddit API credentials:")
        print("REDDIT_CLIENT_ID=your_client_id")
        print("REDDIT_CLIENT_SECRET=your_client_secret")
        return False
    
    return True

def run_clinical_analysis():
    """Run clinical trials analysis"""
    print("\n" + "="*60)
    print("🔬 CLINICAL TRIALS ANALYSIS")
    print("="*60)
    
    start_time = time.time()
    
    analyzer = ClinicalTrialsAnalyzer()
    results_df = analyzer.analyze_trials()
    
    # Generate report
    report = analyzer.generate_summary_report(results_df)
    with open('clinical_trials_report.md', 'w') as f:
        f.write(report)
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Clinical analysis completed in {elapsed_time:.1f} seconds")
    
    return results_df

def run_market_analysis():
    """Run market validation analysis"""
    print("\n" + "="*60)
    print("📊 MARKET VALIDATION ANALYSIS")
    print("="*60)
    
    start_time = time.time()
    
    analyzer = MarketAnalyzer()
    results = analyzer.run_analysis()
    
    # Generate report
    report = analyzer.generate_market_report(results)
    with open('market_validation_report.md', 'w') as f:
        f.write(report)
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Market analysis completed in {elapsed_time:.1f} seconds")
    
    return results

def generate_final_report(clinical_results, market_results):
    """Generate consolidated validation report"""
    print("\n" + "="*60)
    print("📋 GENERATING FINAL VALIDATION REPORT")
    print("="*60)
    
    report = []
    report.append("# PlaceboRx Validation Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    
    # Clinical validation
    if not clinical_results.empty:
        digital_trials = len(clinical_results[clinical_results['is_digital'] == True])
        significant_trials = len(clinical_results[clinical_results['statistical_significance'] == True])
        high_relevance = len(clinical_results[clinical_results['clinical_relevance'] == 'High'])
        
        report.append("### Clinical Evidence")
        report.append(f"- **Total relevant trials**: {len(clinical_results)}")
        report.append(f"- **Digital interventions**: {digital_trials}")
        report.append(f"- **Statistically significant**: {significant_trials}")
        report.append(f"- **High clinical relevance**: {high_relevance}")
        
        if significant_trials > 0:
            report.append("✅ **CLINICAL VALIDATION: STRONG**")
        elif high_relevance > 0:
            report.append("✅ **CLINICAL VALIDATION: MODERATE**")
        else:
            report.append("⚠️ **CLINICAL VALIDATION: WEAK**")
    else:
        report.append("❌ **CLINICAL VALIDATION: NO EVIDENCE FOUND**")
    
    report.append("")
    
    # Market validation
    if market_results:
        market_signals = market_results['market_signals']
        desperation_pct = market_signals['high_desperation_percentage']
        openness_pct = market_signals['high_openness_percentage']
        engagement_pct = market_signals['high_engagement_percentage']
        
        report.append("### Market Demand")
        report.append(f"- **Relevant posts analyzed**: {market_signals['total_relevant_posts']}")
        report.append(f"- **High desperation**: {desperation_pct:.1f}%")
        report.append(f"- **High openness**: {openness_pct:.1f}%")
        report.append(f"- **High engagement**: {engagement_pct:.1f}%")
        
        # Market validation score
        market_score = (desperation_pct + openness_pct + engagement_pct) / 3
        
        if market_score > 40:
            report.append("✅ **MARKET VALIDATION: STRONG**")
        elif market_score > 25:
            report.append("✅ **MARKET VALIDATION: MODERATE**")
        else:
            report.append("⚠️ **MARKET VALIDATION: WEAK**")
    else:
        report.append("❌ **MARKET VALIDATION: NO DATA AVAILABLE**")
    
    report.append("")
    
    # Go-to-market recommendations
    report.append("## Go-to-Market Recommendations")
    
    if market_results and market_results['framing_results']:
        best_framing = max(market_results['framing_results'].items(), 
                          key=lambda x: x[1]['resonance_percentage'])
        report.append(f"**Recommended messaging**: {best_framing[0].replace('_', ' ').title()}")
        report.append(f"**Resonance**: {best_framing[1]['resonance_percentage']:.1f}%")
        report.append("")
    
    if market_results and market_results['market_signals']['top_subreddits']:
        report.append("**Top target communities**:")
        for subreddit, data in market_results['market_signals']['top_subreddits'][:3]:
            report.append(f"- r/{subreddit}")
        report.append("")
    
    # Next steps
    report.append("## Next Steps")
    report.append("1. **MVP Development**: Build lightweight digital placebo intervention")
    report.append("2. **User Testing**: Validate with target audience from identified subreddits")
    report.append("3. **Clinical Pilot**: Design small-scale efficacy study")
    report.append("4. **Regulatory**: Consult on digital therapeutic classification")
    
    # Save final report
    with open('placeborx_validation_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("✅ Final validation report generated!")
    print("📄 Report saved to: placeborx_validation_report.md")

def main():
    """Main pipeline execution"""
    print("🚀 PlaceboRx Validation Pipeline")
    print("Solo entrepreneur execution - 2-4 hours runtime")
    print("="*60)
    
    # Check environment
    if not check_environment():
        return
    
    start_time = time.time()
    
    try:
        # Run clinical analysis
        clinical_results = run_clinical_analysis()
        
        # Run market analysis
        market_results = run_market_analysis()
        
        # Generate final report
        generate_final_report(clinical_results, market_results)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Pipeline completed in {total_time/60:.1f} minutes!")
        print("\n📁 Generated files:")
        print("   - clinical_trials_results.csv")
        print("   - clinical_trials_report.md")
        print("   - market_analysis_results.csv")
        print("   - market_validation_report.md")
        print("   - placeborx_validation_report.md")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("Check your internet connection and API credentials.")

if __name__ == "__main__":
    main() 