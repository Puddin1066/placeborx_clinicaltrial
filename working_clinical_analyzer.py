#!/usr/bin/env python3
"""
Improved Clinical Trials Analyzer that works with the actual API structure
"""

import requests
import pandas as pd
import json
import time
from typing import List, Dict
import re

class WorkingClinicalAnalyzer:
    def __init__(self):
        self.api_url = "https://clinicaltrials.gov/api/v2/studies"
        self.results = []
        
    def search_relevant_trials(self, page_size: int = 500) -> List[Dict]:
        """Search for trials and filter for relevant ones"""
        all_trials = []
        
        try:
            print(f"📡 Fetching {page_size} trials from ClinicalTrials.gov...")
            response = requests.get(f"{self.api_url}?pageSize={page_size}")
            response.raise_for_status()
            data = response.json()
            
            if 'studies' in data:
                trials = data['studies']
                print(f"✅ Retrieved {len(trials)} trials")
                
                # Filter for relevant trials
                relevant_trials = []
                for trial in trials:
                    if self.is_relevant_trial(trial):
                        relevant_trials.append(trial)
                
                print(f"🔍 Found {len(relevant_trials)} potentially relevant trials")
                return relevant_trials
                
        except Exception as e:
            print(f"❌ Error fetching trials: {e}")
            return []
        
        return all_trials
    
    def is_relevant_trial(self, trial: Dict) -> bool:
        """Check if a trial is relevant for placebo/digital intervention research"""
        trial_text = json.dumps(trial).lower()
        
        # Look for placebo mentions
        has_placebo = 'placebo' in trial_text
        
        # Look for open-label design
        has_open_label = ('open' in trial_text and 
                         ('label' in trial_text or 'labeled' in trial_text))
        
        # Look for digital/app interventions
        has_digital = any(term in trial_text for term in [
            'digital', 'app', 'mobile', 'online', 'web', 'virtual',
            'telemedicine', 'telehealth', 'smartphone', 'internet'
        ])
        
        # Look for relevant conditions
        has_relevant_condition = any(condition in trial_text for condition in [
            'pain', 'anxiety', 'depression', 'stress', 'sleep', 'fatigue',
            'chronic', 'mental health', 'psychological', 'behavioral'
        ])
        
        # Trial is relevant if it has placebo OR (digital + relevant condition)
        return has_placebo or (has_digital and has_relevant_condition)
    
    def extract_trial_data(self, trial: Dict) -> Dict:
        """Extract relevant data from a trial"""
        protocol = trial.get('protocolSection', {})
        identification = protocol.get('identificationModule', {})
        conditions_module = protocol.get('conditionsModule', {})
        interventions_module = protocol.get('armsInterventionsModule', {})
        design_module = protocol.get('designModule', {})
        status_module = protocol.get('statusModule', {})
        
        # Basic information
        nct_id = identification.get('nctId', 'Unknown')
        title = identification.get('briefTitle', 'No title')
        
        # Conditions
        conditions = conditions_module.get('conditions', [])
        condition_str = ', '.join(conditions) if conditions else 'Not specified'
        
        # Interventions
        interventions = interventions_module.get('interventions', [])
        intervention_names = []
        intervention_types = []
        for intervention in interventions:
            intervention_names.append(intervention.get('name', 'Unknown'))
            intervention_types.append(intervention.get('type', 'Unknown'))
        
        intervention_str = '; '.join(intervention_names) if intervention_names else 'Not specified'
        
        # Study details
        study_type = design_module.get('studyType', 'Unknown')
        enrollment_info = design_module.get('enrollmentInfo', {})
        enrollment = enrollment_info.get('count', 0) if enrollment_info else 0
        
        # Status
        overall_status = status_module.get('overallStatus', 'Unknown')
        
        # Analyze for placebo characteristics
        trial_text = json.dumps(trial).lower()
        
        is_placebo_trial = 'placebo' in trial_text
        is_open_label = ('open' in trial_text and 
                        ('label' in trial_text or 'labeled' in trial_text))
        is_digital = any(term in trial_text for term in [
            'digital', 'app', 'mobile', 'online', 'web', 'virtual'
        ])
        
        # Assess clinical relevance based on condition and intervention
        clinical_relevance = 'High' if any(condition in trial_text for condition in [
            'pain', 'depression', 'anxiety', 'chronic'
        ]) else 'Medium' if any(condition in trial_text for condition in [
            'stress', 'sleep', 'fatigue', 'mental'
        ]) else 'Low'
        
        # Statistical significance (placeholder - would need actual results)
        statistical_significance = enrollment > 50  # Simple heuristic
        
        # Effect size estimation (placeholder)
        effect_size = 0.3 if is_placebo_trial else 0.2  # Placeholder values
        
        return {
            'nct_id': nct_id,
            'title': title,
            'condition': condition_str,
            'intervention': intervention_str,
            'enrollment': enrollment,
            'phase': design_module.get('phases', ['Not specified'])[0] if design_module.get('phases') else 'Not specified',
            'completion_date': 'Not specified',  # Would need to extract from dates
            'outcomes': 'Not specified',  # Would need to extract from outcomes
            'results': 'Not specified',   # Would need results data
            'descriptions': title,        # Using title as description
            'is_digital': is_digital,
            'effect_size': effect_size,
            'statistical_significance': statistical_significance,
            'clinical_relevance': clinical_relevance,
            'is_placebo_trial': is_placebo_trial,
            'is_open_label': is_open_label,
            'study_type': study_type,
            'status': overall_status
        }
    
    def analyze_trials(self) -> pd.DataFrame:
        """Main analysis function"""
        print("🔬 Starting clinical trials analysis...")
        
        # Search for relevant trials
        trials = self.search_relevant_trials()
        
        if not trials:
            print("⚠️ No trials found. Creating empty DataFrame with proper columns.")
            return pd.DataFrame(columns=[
                'nct_id', 'title', 'condition', 'intervention', 'enrollment',
                'phase', 'completion_date', 'outcomes', 'results', 'descriptions',
                'is_digital', 'effect_size', 'statistical_significance', 'clinical_relevance'
            ])
        
        print(f"📊 Analyzing {len(trials)} trials for clinical significance...")
        
        # Extract data from each trial
        trial_data = []
        for trial in trials:
            data = self.extract_trial_data(trial)
            trial_data.append(data)
        
        # Create DataFrame
        df = pd.DataFrame(trial_data)
        
        # Save results
        df.to_csv('clinical_trials_working_results.csv', index=False)
        print(f"✅ Results saved to clinical_trials_working_results.csv")
        
        # Print summary statistics
        print(f"\n📈 Analysis Summary:")
        print(f"   Total trials analyzed: {len(df)}")
        print(f"   Placebo trials: {df['is_placebo_trial'].sum()}")
        print(f"   Open-label trials: {df['is_open_label'].sum()}")
        print(f"   Digital interventions: {df['is_digital'].sum()}")
        print(f"   High clinical relevance: {(df['clinical_relevance'] == 'High').sum()}")
        
        return df
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> str:
        """Generate a summary report"""
        if results_df.empty:
            return "# Clinical Trials Analysis Report\n\nNo relevant trials found."
        
        report = []
        report.append("# Clinical Trials Analysis Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        total_trials = len(results_df)
        placebo_trials = results_df['is_placebo_trial'].sum()
        digital_trials = results_df['is_digital'].sum()
        high_relevance = (results_df['clinical_relevance'] == 'High').sum()
        
        report.append("## Summary")
        report.append(f"- **Total trials analyzed**: {total_trials}")
        report.append(f"- **Placebo trials**: {placebo_trials}")
        report.append(f"- **Digital interventions**: {digital_trials}")
        report.append(f"- **High clinical relevance**: {high_relevance}")
        report.append("")
        
        # Top trials
        if not results_df.empty:
            report.append("## Top Relevant Trials")
            
            # Sort by relevance and digital features
            top_trials = results_df.nlargest(5, ['is_digital', 'is_placebo_trial'])
            
            for _, trial in top_trials.iterrows():
                report.append(f"### {trial['nct_id']}")
                report.append(f"**Title**: {trial['title']}")
                report.append(f"**Condition**: {trial['condition']}")
                report.append(f"**Intervention**: {trial['intervention']}")
                report.append(f"**Enrollment**: {trial['enrollment']}")
                report.append(f"**Digital**: {'Yes' if trial['is_digital'] else 'No'}")
                report.append(f"**Placebo**: {'Yes' if trial['is_placebo_trial'] else 'No'}")
                report.append(f"**Clinical Relevance**: {trial['clinical_relevance']}")
                report.append("")
        
        return '\n'.join(report)

def main():
    """Test the working analyzer"""
    analyzer = WorkingClinicalAnalyzer()
    results_df = analyzer.analyze_trials()
    
    # Generate report
    report = analyzer.generate_summary_report(results_df)
    with open('clinical_trials_working_report.md', 'w') as f:
        f.write(report)
    
    print("✅ Report saved to clinical_trials_working_report.md")

if __name__ == "__main__":
    main()