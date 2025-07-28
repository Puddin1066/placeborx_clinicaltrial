#!/usr/bin/env python3
"""
Patent Data Integration for PlaceboRx
Integrates patent analysis with existing data sources and pipeline
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
from config import PATENT_SEARCH_TERMS

class PatentDataIntegration:
    """Integrates patent data with PlaceboRx pipeline"""
    
    def __init__(self):
        self.patent_data = self.load_patent_data()
        self.integration_results = {}
    
    def load_patent_data(self) -> Dict[str, Any]:
        """Load patent data from various sources"""
        
        # Try to load existing patent analysis results
        try:
            with open('digital_therapeutics_patents_20250727_204235.json', 'r') as f:
                digital_patents = json.load(f)
        except FileNotFoundError:
            digital_patents = []
        
        try:
            with open('placebo_effect_patents_20250727_204235.json', 'r') as f:
                placebo_patents = json.load(f)
        except FileNotFoundError:
            placebo_patents = []
        
        return {
            'digital_therapeutics': digital_patents,
            'placebo_effects': placebo_patents,
            'total_patents': len(digital_patents) + len(placebo_patents)
        }
    
    def integrate_with_clinical_data(self, clinical_data: List[Dict]) -> Dict[str, Any]:
        """Integrate patent data with clinical trials data"""
        
        print("🔗 Integrating Patent Data with Clinical Trials")
        print("-" * 50)
        
        # Analyze clinical trials for patent relevance
        patent_relevant_trials = []
        digital_therapeutic_trials = []
        placebo_effect_trials = []
        
        for trial in clinical_data:
            trial_text = json.dumps(trial).lower()
            
            # Check for digital therapeutic relevance
            digital_keywords = ['digital', 'mobile', 'app', 'software', 'online', 'remote']
            if any(keyword in trial_text for keyword in digital_keywords):
                digital_therapeutic_trials.append(trial)
            
            # Check for placebo effect relevance
            placebo_keywords = ['placebo', 'expectation', 'psychological', 'mind-body']
            if any(keyword in trial_text for keyword in placebo_keywords):
                placebo_effect_trials.append(trial)
            
            # Check for general patent relevance
            patent_keywords = ['intervention', 'treatment', 'therapy', 'therapeutic']
            if any(keyword in trial_text for keyword in patent_keywords):
                patent_relevant_trials.append(trial)
        
        integration_results = {
            'total_clinical_trials': len(clinical_data),
            'patent_relevant_trials': len(patent_relevant_trials),
            'digital_therapeutic_trials': len(digital_therapeutic_trials),
            'placebo_effect_trials': len(placebo_effect_trials),
            'patent_clinical_overlap': len(patent_relevant_trials) / len(clinical_data) if clinical_data else 0
        }
        
        print(f"📊 Clinical-Patent Integration Results:")
        print(f"   Total Clinical Trials: {integration_results['total_clinical_trials']}")
        print(f"   Patent-Relevant Trials: {integration_results['patent_relevant_trials']}")
        print(f"   Digital Therapeutic Trials: {integration_results['digital_therapeutic_trials']}")
        print(f"   Placebo Effect Trials: {integration_results['placebo_effect_trials']}")
        print(f"   Overlap Percentage: {integration_results['patent_clinical_overlap']:.1%}")
        
        return integration_results
    
    def integrate_with_market_data(self, market_data: List[Dict]) -> Dict[str, Any]:
        """Integrate patent data with market validation data"""
        
        print("\n🔗 Integrating Patent Data with Market Validation")
        print("-" * 50)
        
        # Analyze market data for patent relevance
        patent_relevant_posts = []
        digital_therapeutic_posts = []
        placebo_effect_posts = []
        
        for post in market_data:
            post_text = (post.get('title', '') + ' ' + post.get('selftext', '')).lower()
            
            # Check for digital therapeutic relevance
            digital_keywords = ['digital', 'mobile', 'app', 'software', 'online', 'remote']
            if any(keyword in post_text for keyword in digital_keywords):
                digital_therapeutic_posts.append(post)
            
            # Check for placebo effect relevance
            placebo_keywords = ['placebo', 'expectation', 'psychological', 'mind-body']
            if any(keyword in post_text for keyword in placebo_keywords):
                placebo_effect_posts.append(post)
            
            # Check for general patent relevance
            patent_keywords = ['treatment', 'therapy', 'therapeutic', 'intervention']
            if any(keyword in post_text for keyword in patent_keywords):
                patent_relevant_posts.append(post)
        
        integration_results = {
            'total_market_posts': len(market_data),
            'patent_relevant_posts': len(patent_relevant_posts),
            'digital_therapeutic_posts': len(digital_therapeutic_posts),
            'placebo_effect_posts': len(placebo_effect_posts),
            'patent_market_overlap': len(patent_relevant_posts) / len(market_data) if market_data else 0
        }
        
        print(f"📊 Market-Patent Integration Results:")
        print(f"   Total Market Posts: {integration_results['total_market_posts']}")
        print(f"   Patent-Relevant Posts: {integration_results['patent_relevant_posts']}")
        print(f"   Digital Therapeutic Posts: {integration_results['digital_therapeutic_posts']}")
        print(f"   Placebo Effect Posts: {integration_results['placebo_effect_posts']}")
        print(f"   Overlap Percentage: {integration_results['patent_market_overlap']:.1%}")
        
        return integration_results
    
    def generate_comprehensive_insights(self) -> Dict[str, Any]:
        """Generate comprehensive insights combining all data sources"""
        
        print("\n🎯 Generating Comprehensive Patent Insights")
        print("-" * 50)
        
        # Calculate patent trends
        digital_patents = self.patent_data['digital_therapeutics']
        placebo_patents = self.patent_data['placebo_effects']
        
        # Patent filing trends
        if digital_patents:
            digital_years = [datetime.strptime(p['patent_date'], '%Y-%m-%d').year for p in digital_patents]
            digital_trend = pd.Series(digital_years).value_counts().sort_index()
        else:
            digital_trend = {}
        
        if placebo_patents:
            placebo_years = [datetime.strptime(p['patent_date'], '%Y-%m-%d').year for p in placebo_patents]
            placebo_trend = pd.Series(placebo_years).value_counts().sort_index()
        else:
            placebo_trend = {}
        
        # Technology maturity analysis
        technology_maturity = {
            'digital_therapeutics': {
                'total_patents': len(digital_patents),
                'recent_patents': len([p for p in digital_patents if datetime.strptime(p['patent_date'], '%Y-%m-%d') > datetime.now() - timedelta(days=365)]),
                'maturity_score': min(len(digital_patents) / 10, 1.0)  # Scale to 0-1
            },
            'placebo_effects': {
                'total_patents': len(placebo_patents),
                'recent_patents': len([p for p in placebo_patents if datetime.strptime(p['patent_date'], '%Y-%m-%d') > datetime.now() - timedelta(days=365)]),
                'maturity_score': min(len(placebo_patents) / 10, 1.0)  # Scale to 0-1
            }
        }
        
        # Market opportunity analysis
        market_opportunity = {
            'digital_therapeutics_opportunity': 'High' if len(digital_patents) < 20 else 'Medium',
            'placebo_effects_opportunity': 'High' if len(placebo_patents) < 15 else 'Medium',
            'competitive_landscape': 'Emerging' if len(digital_patents) + len(placebo_patents) < 30 else 'Established',
            'innovation_gaps': self.identify_innovation_gaps()
        }
        
        comprehensive_insights = {
            'patent_trends': {
                'digital_therapeutics': digital_trend.to_dict() if hasattr(digital_trend, 'to_dict') else digital_trend,
                'placebo_effects': placebo_trend.to_dict() if hasattr(placebo_trend, 'to_dict') else placebo_trend
            },
            'technology_maturity': technology_maturity,
            'market_opportunity': market_opportunity,
            'total_patents': self.patent_data['total_patents'],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        print(f"📊 Comprehensive Patent Insights:")
        print(f"   Total Patents Analyzed: {comprehensive_insights['total_patents']}")
        print(f"   Digital Therapeutics Maturity: {technology_maturity['digital_therapeutics']['maturity_score']:.1%}")
        print(f"   Placebo Effects Maturity: {technology_maturity['placebo_effects']['maturity_score']:.1%}")
        print(f"   Market Opportunity: {market_opportunity['digital_therapeutics_opportunity']} (Digital), {market_opportunity['placebo_effects_opportunity']} (Placebo)")
        print(f"   Competitive Landscape: {market_opportunity['competitive_landscape']}")
        
        return comprehensive_insights
    
    def identify_innovation_gaps(self) -> List[str]:
        """Identify gaps in patent coverage"""
        
        gaps = []
        
        # Check for gaps in digital therapeutics
        digital_patents = self.patent_data['digital_therapeutics']
        digital_titles = [p['patent_title'].lower() for p in digital_patents]
        
        if not any('chronic pain' in title for title in digital_titles):
            gaps.append("Chronic pain digital therapeutics")
        
        if not any('anxiety' in title for title in digital_titles):
            gaps.append("Anxiety digital therapeutics")
        
        if not any('depression' in title for title in digital_titles):
            gaps.append("Depression digital therapeutics")
        
        # Check for gaps in placebo effects
        placebo_patents = self.patent_data['placebo_effects']
        placebo_titles = [p['patent_title'].lower() for p in placebo_patents]
        
        if not any('open label' in title for title in placebo_titles):
            gaps.append("Open-label placebo effects")
        
        if not any('expectation' in title for title in placebo_titles):
            gaps.append("Expectation effect optimization")
        
        return gaps
    
    def save_integration_results(self, results: Dict[str, Any], filename: str = None):
        """Save integration results to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"patent_integration_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Integration results saved to: {filename}")
    
    def generate_patent_report(self) -> str:
        """Generate a comprehensive patent report"""
        
        report = f"""
# Patent Analysis Report for PlaceboRx
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Patents Analyzed: {self.patent_data['total_patents']}
- Digital Therapeutics Patents: {len(self.patent_data['digital_therapeutics'])}
- Placebo Effect Patents: {len(self.patent_data['placebo_effects'])}

## Technology Landscape
### Digital Therapeutics
- Recent Activity: {len([p for p in self.patent_data['digital_therapeutics'] if datetime.strptime(p['patent_date'], '%Y-%m-%d') > datetime.now() - timedelta(days=365)])} patents in last year
- Key Companies: {', '.join(set(p['assignee_name'] for p in self.patent_data['digital_therapeutics']))}
- Technology Focus: Mobile applications, software platforms, AI integration

### Placebo Effects
- Recent Activity: {len([p for p in self.patent_data['placebo_effects'] if datetime.strptime(p['patent_date'], '%Y-%m-%d') > datetime.now() - timedelta(days=365)])} patents in last year
- Key Companies: {', '.join(set(p['assignee_name'] for p in self.patent_data['placebo_effects']))}
- Technology Focus: Expectation management, mind-body interfaces

## Market Implications
- Digital therapeutics show strong patent activity
- Placebo effects have emerging patent landscape
- Opportunities exist in specific therapeutic areas
- Competitive landscape is developing

## Recommendations
1. Monitor patent filings in digital therapeutics space
2. Consider patent strategy for placebo effect innovations
3. Identify white space opportunities in specific conditions
4. Track competitor patent activity

---
*Report generated by PlaceboRx Patent Integration System*
"""
        
        return report

def test_patent_integration():
    """Test the patent data integration system"""
    
    print("🚀 Patent Data Integration Test")
    print("=" * 50)
    
    integrator = PatentDataIntegration()
    
    print(f"📊 Loaded {integrator.patent_data['total_patents']} patents")
    print(f"   Digital Therapeutics: {len(integrator.patent_data['digital_therapeutics'])}")
    print(f"   Placebo Effects: {len(integrator.patent_data['placebo_effects'])}")
    
    # Generate comprehensive insights
    insights = integrator.generate_comprehensive_insights()
    
    # Save results
    integrator.save_integration_results(insights)
    
    # Generate report
    report = integrator.generate_patent_report()
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"patent_analysis_report_{timestamp}.md"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Patent report saved to: {report_filename}")
    
    return True

if __name__ == "__main__":
    test_patent_integration() 