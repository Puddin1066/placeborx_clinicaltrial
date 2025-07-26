import requests
import pandas as pd
import json
import time
from typing import List, Dict
import re

class ComprehensiveClinicalAnalyzer:
    def __init__(self):
        self.api_url = "https://clinicaltrials.gov/api/v2/studies"
        self.all_trials = []
        
    def search_comprehensive(self, search_terms: List[str], max_pages: int = 10) -> List[Dict]:
        """Comprehensive search across multiple pages"""
        all_trials = []
        
        for term in search_terms:
            print(f"🔍 Searching for: {term}")
            
            # Search across multiple pages
            for page in range(max_pages):
                params = {
                    'pageSize': 100,
                    'format': 'json'
                }
                
                try:
                    response = requests.get(self.api_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'studies' in data:
                        trials = data['studies']
                        # Filter trials that contain our search term
                        matching_trials = []
                        for trial in trials:
                            trial_text = json.dumps(trial).lower()
                            if term.lower() in trial_text:
                                matching_trials.append(trial)
                        
                        all_trials.extend(matching_trials)
                        print(f"   Page {page+1}: Found {len(matching_trials)} trials containing '{term}'")
                        
                        # If no trials found, break early
                        if len(trials) < 100:
                            break
                            
                except Exception as e:
                    print(f"   Error on page {page+1}: {e}")
                    break
                    
                time.sleep(0.5)  # Rate limiting
        
        return all_trials
    
    def extract_detailed_info(self, trial: Dict) -> Dict:
        """Extract detailed trial information"""
        protocol = trial.get('protocolSection', {})
        results = trial.get('resultsSection', {})
        
        info = {
            'nct_id': protocol.get('identificationModule', {}).get('nctId', ''),
            'title': protocol.get('identificationModule', {}).get('briefTitle', ''),
            'official_title': protocol.get('identificationModule', {}).get('officialTitle', ''),
            'condition': '',
            'intervention': '',
            'enrollment': '',
            'phase': '',
            'completion_date': '',
            'study_type': protocol.get('designModule', {}).get('studyType', ''),
            'is_digital': False,
            'has_results': trial.get('hasResults', False),
            'is_olp': False,
            'olp_indicators': [],
            'digital_indicators': []
        }
        
        # Extract condition
        conditions = protocol.get('conditionsModule', {}).get('conditions', [])
        if conditions:
            info['condition'] = conditions[0]
        
        # Extract intervention
        arms_interventions = protocol.get('armsInterventionsModule', {})
        if arms_interventions:
            interventions = arms_interventions.get('interventions', [])
            if interventions:
                info['intervention'] = interventions[0].get('name', '')
        
        # Check for OLP indicators
        title_text = (info['title'] + ' ' + info['official_title']).lower()
        olp_indicators = [
            'open-label', 'open label', 'non-blind', 'nonblind', 'unblinded',
            'open label placebo', 'open-label placebo'
        ]
        
        for indicator in olp_indicators:
            if indicator in title_text:
                info['is_olp'] = True
                info['olp_indicators'].append(indicator)
        
        # Check for digital indicators
        intervention_text = info['intervention'].lower()
        digital_indicators = ['app', 'digital', 'online', 'web', 'mobile', 'computer', 'software', 'platform']
        
        for indicator in digital_indicators:
            if indicator in intervention_text or indicator in title_text:
                info['is_digital'] = True
                info['digital_indicators'].append(indicator)
        
        # Extract enrollment
        enrollment_info = protocol.get('designModule', {}).get('enrollmentInfo', {})
        if enrollment_info:
            info['enrollment'] = enrollment_info.get('count', '')
        
        # Extract phase
        phases = protocol.get('designModule', {}).get('phases', [])
        if phases:
            info['phase'] = phases[0]
        
        # Extract completion date
        status = protocol.get('statusModule', {})
        if status:
            completion_date = status.get('completionDateStruct', {}).get('date', '')
            info['completion_date'] = completion_date
        
        return info
    
    def analyze_trials(self) -> pd.DataFrame:
        """Main analysis function"""
        print("🔬 Comprehensive Clinical Trials Analysis")
        print("="*60)
        
        # Search terms for OLP and related concepts
        search_terms = [
            'open-label placebo',
            'open label placebo',
            'non-blind placebo',
            'unblinded placebo',
            'digital placebo',
            'app placebo',
            'online placebo',
            'digital therapeutic',
            'placebo'
        ]
        
        # Search for trials
        trials = self.search_comprehensive(search_terms, max_pages=5)
        
        if not trials:
            print("No trials found.")
            return pd.DataFrame()
        
        print(f"\n📊 Found {len(trials)} total trials")
        
        # Extract detailed information
        trial_info = []
        for trial in trials:
            info = self.extract_detailed_info(trial)
            trial_info.append(info)
        
        # Convert to DataFrame
        df = pd.DataFrame(trial_info)
        
        # Remove duplicates based on NCT ID
        df = df.drop_duplicates(subset=['nct_id'])
        
        print(f"📋 Unique trials: {len(df)}")
        
        return df
    
    def generate_analysis_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive analysis report"""
        if df.empty:
            return "No trials found for analysis."
        
        report = []
        report.append("# Comprehensive Clinical Trials Analysis Report")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        total_trials = len(df)
        olp_trials = len(df[df['is_olp'] == True])
        digital_trials = len(df[df['is_digital'] == True])
        trials_with_results = len(df[df['has_results'] == True])
        
        report.append("## Executive Summary")
        report.append(f"- **Total trials analyzed**: {total_trials}")
        report.append(f"- **Open-label placebo trials**: {olp_trials}")
        report.append(f"- **Digital interventions**: {digital_trials}")
        report.append(f"- **Trials with results**: {trials_with_results}")
        report.append("")
        
        # OLP Analysis
        if olp_trials > 0:
            report.append("## Open-Label Placebo Trials")
            olp_df = df[df['is_olp'] == True]
            for _, trial in olp_df.iterrows():
                report.append(f"- **{trial['title']}**")
                report.append(f"  - Condition: {trial['condition']}")
                report.append(f"  - Intervention: {trial['intervention']}")
                report.append(f"  - Phase: {trial['phase']}")
                report.append(f"  - Has Results: {trial['has_results']}")
                report.append("")
        
        # Digital Interventions
        if digital_trials > 0:
            report.append("## Digital Interventions")
            digital_df = df[df['is_digital'] == True]
            for _, trial in digital_df.iterrows():
                report.append(f"- **{trial['title']}**")
                report.append(f"  - Condition: {trial['condition']}")
                report.append(f"  - Digital indicators: {', '.join(trial['digital_indicators'])}")
                report.append(f"  - Has Results: {trial['has_results']}")
                report.append("")
        
        # Top Conditions
        if not df['condition'].empty:
            top_conditions = df['condition'].value_counts().head(10)
            report.append("## Top Conditions")
            for condition, count in top_conditions.items():
                report.append(f"- {condition}: {count} trials")
            report.append("")
        
        # Trials with Results
        if trials_with_results > 0:
            report.append("## Trials with Results")
            results_df = df[df['has_results'] == True]
            for _, trial in results_df.head(5).iterrows():
                report.append(f"- **{trial['title']}**")
                report.append(f"  - Condition: {trial['condition']}")
                report.append(f"  - OLP: {trial['is_olp']}")
                report.append(f"  - Digital: {trial['is_digital']}")
                report.append("")
        
        # Clinical Validation Assessment
        report.append("## Clinical Validation Assessment")
        
        if olp_trials > 0:
            report.append("✅ **OLP Evidence Found**: Open-label placebo trials identified")
        else:
            report.append("⚠️ **Limited OLP Evidence**: No specific open-label placebo trials found")
        
        if digital_trials > 0:
            report.append("✅ **Digital Interventions**: Digital therapeutic trials identified")
        else:
            report.append("⚠️ **Limited Digital Evidence**: No digital intervention trials found")
        
        if trials_with_results > 0:
            report.append("✅ **Results Available**: Trials with published results found")
        else:
            report.append("⚠️ **Limited Results**: Few trials with published results")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("1. **Focus on Market Validation**: Given limited OLP clinical evidence, prioritize market demand validation")
        report.append("2. **Digital Therapeutic Approach**: Consider positioning as digital therapeutic rather than traditional placebo")
        report.append("3. **Pilot Study Design**: Design small-scale efficacy studies to generate initial clinical data")
        report.append("4. **Regulatory Strategy**: Consult on digital therapeutic classification and regulatory pathway")
        
        return "\n".join(report)

def main():
    analyzer = ComprehensiveClinicalAnalyzer()
    df = analyzer.analyze_trials()
    
    if not df.empty:
        # Save results
        df.to_csv('comprehensive_clinical_trials.csv', index=False)
        
        # Generate report
        report = analyzer.generate_analysis_report(df)
        with open('comprehensive_clinical_report.md', 'w') as f:
            f.write(report)
        
        print("\n✅ Analysis complete!")
        print(f"📁 Results saved to: comprehensive_clinical_trials.csv")
        print(f"📄 Report saved to: comprehensive_clinical_report.md")
        
        # Show summary
        print(f"\n📈 Summary:")
        print(f"- Total trials: {len(df)}")
        print(f"- OLP trials: {len(df[df['is_olp'] == True])}")
        print(f"- Digital trials: {len(df[df['is_digital'] == True])}")
        print(f"- Trials with results: {len(df[df['has_results'] == True])}")

if __name__ == "__main__":
    main() 