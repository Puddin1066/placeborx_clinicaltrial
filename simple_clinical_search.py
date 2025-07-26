import requests
import pandas as pd
import json
import time
from typing import List, Dict
import re

def search_trials_simple(search_terms: List[str]) -> List[Dict]:
    """Simple search using the working API format"""
    all_trials = []
    
    for term in search_terms:
        print(f"Searching for: {term}")
        
        # Use the basic API that we know works
        params = {
            'pageSize': 100,
            'format': 'json'
        }
        
        try:
            # Get all trials and filter locally
            response = requests.get("https://clinicaltrials.gov/api/v2/studies", params=params)
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
                print(f"Found {len(matching_trials)} trials containing '{term}'")
                
        except Exception as e:
            print(f"Error searching for {term}: {e}")
            
        time.sleep(1)  # Rate limiting
    
    return all_trials

def extract_basic_info(trial: Dict) -> Dict:
    """Extract basic trial information"""
    protocol = trial.get('protocolSection', {})
    
    info = {
        'nct_id': protocol.get('identificationModule', {}).get('nctId', ''),
        'title': protocol.get('identificationModule', {}).get('briefTitle', ''),
        'condition': '',
        'intervention': '',
        'enrollment': '',
        'phase': '',
        'completion_date': '',
        'is_digital': False,
        'has_results': trial.get('hasResults', False)
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
    
    # Check if digital
    intervention_text = info['intervention'].lower()
    if any(term in intervention_text for term in ['app', 'digital', 'online', 'web', 'mobile', 'computer']):
        info['is_digital'] = True
    
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

def main():
    print("🔬 Simple Clinical Trials Search")
    print("="*50)
    
    # Search terms for OLP
    search_terms = [
        'open-label placebo',
        'open label placebo', 
        'digital placebo',
        'app placebo',
        'online placebo',
        'digital therapeutic',
        'placebo'
    ]
    
    # Search for trials
    trials = search_trials_simple(search_terms)
    
    if not trials:
        print("No trials found.")
        return
    
    print(f"\n📊 Found {len(trials)} total trials")
    
    # Extract basic information
    trial_info = []
    for trial in trials:
        info = extract_basic_info(trial)
        trial_info.append(info)
    
    # Convert to DataFrame
    df = pd.DataFrame(trial_info)
    
    # Remove duplicates based on NCT ID
    df = df.drop_duplicates(subset=['nct_id'])
    
    print(f"📋 Unique trials: {len(df)}")
    
    # Show some examples
    print("\n📋 Sample Trials:")
    for i, (_, trial) in enumerate(df.head(10).iterrows()):
        print(f"{i+1}. {trial['title']}")
        print(f"   Condition: {trial['condition']}")
        print(f"   Intervention: {trial['intervention']}")
        print(f"   Digital: {trial['is_digital']}")
        print(f"   Has Results: {trial['has_results']}")
        print()
    
    # Save results
    df.to_csv('simple_clinical_trials.csv', index=False)
    print(f"💾 Results saved to: simple_clinical_trials.csv")
    
    # Summary statistics
    digital_trials = len(df[df['is_digital'] == True])
    trials_with_results = len(df[df['has_results'] == True])
    
    print(f"\n📈 Summary:")
    print(f"- Total trials: {len(df)}")
    print(f"- Digital interventions: {digital_trials}")
    print(f"- Trials with results: {trials_with_results}")

if __name__ == "__main__":
    main() 