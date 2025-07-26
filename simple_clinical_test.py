#!/usr/bin/env python3
"""
Simple test to examine ClinicalTrials.gov API data structure
"""

import requests
import json

def test_api_data():
    """Test what data we can get from the API"""
    print("🔍 Testing ClinicalTrials.gov API Data Structure")
    print("="*55)
    
    try:
        # Get some basic trials
        response = requests.get("https://clinicaltrials.gov/api/v2/studies?pageSize=5")
        response.raise_for_status()
        data = response.json()
        
        if 'studies' in data:
            studies = data['studies']
            print(f"✅ Retrieved {len(studies)} studies")
            
            for i, study in enumerate(studies[:2]):  # Look at first 2 studies
                print(f"\n📋 Study {i+1}:")
                print(f"   NCT ID: {study.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'N/A')}")
                
                # Title
                title = study.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'No title')
                print(f"   Title: {title[:100]}...")
                
                # Conditions
                conditions = study.get('protocolSection', {}).get('conditionsModule', {}).get('conditions', [])
                print(f"   Conditions: {conditions[:3] if conditions else 'None'}")
                
                # Interventions  
                interventions = study.get('protocolSection', {}).get('armsInterventionsModule', {}).get('interventions', [])
                if interventions:
                    intervention_types = [i.get('type', 'Unknown') for i in interventions[:3]]
                    print(f"   Interventions: {intervention_types}")
                else:
                    print(f"   Interventions: None")
                
                # Study type
                study_type = study.get('protocolSection', {}).get('designModule', {}).get('studyType', 'Unknown')
                print(f"   Study Type: {study_type}")
                
                # Status
                status = study.get('protocolSection', {}).get('statusModule', {}).get('overallStatus', 'Unknown')
                print(f"   Status: {status}")
                
                # Let's search for placebo in the full study text
                study_text = json.dumps(study).lower()
                has_placebo = 'placebo' in study_text
                has_open_label = 'open' in study_text and 'label' in study_text
                print(f"   Contains 'placebo': {has_placebo}")
                print(f"   Contains 'open' and 'label': {has_open_label}")
                
        else:
            print("❌ No studies found in response")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")

def test_search_approach():
    """Test different ways to search for relevant trials"""
    print("\n🔍 Testing Search Approaches")
    print("="*35)
    
    # Try to get more trials and search within them
    try:
        response = requests.get("https://clinicaltrials.gov/api/v2/studies?pageSize=100")
        response.raise_for_status()
        data = response.json()
        
        if 'studies' in data:
            studies = data['studies']
            print(f"✅ Retrieved {len(studies)} studies to search")
            
            placebo_count = 0
            open_label_count = 0
            digital_count = 0
            
            for study in studies:
                study_text = json.dumps(study).lower()
                
                if 'placebo' in study_text:
                    placebo_count += 1
                    
                if 'open' in study_text and ('label' in study_text or 'labeled' in study_text):
                    open_label_count += 1
                    
                if any(term in study_text for term in ['digital', 'app', 'mobile', 'online', 'web']):
                    digital_count += 1
            
            print(f"   Studies mentioning 'placebo': {placebo_count}")
            print(f"   Studies with 'open label/labeled': {open_label_count}")
            print(f"   Studies with digital terms: {digital_count}")
            
            # Find any open-label placebo studies
            olp_studies = []
            for study in studies:
                study_text = json.dumps(study).lower()
                if ('placebo' in study_text and 
                    ('open' in study_text and ('label' in study_text or 'labeled' in study_text))):
                    olp_studies.append(study)
            
            print(f"   Potential open-label placebo studies: {len(olp_studies)}")
            
            if olp_studies:
                print("\n📋 Sample OLP Study:")
                study = olp_studies[0]
                title = study.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'No title')
                print(f"   Title: {title}")
                nct_id = study.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'N/A')
                print(f"   NCT ID: {nct_id}")
                
    except Exception as e:
        print(f"❌ Search test failed: {e}")

if __name__ == "__main__":
    test_api_data()
    test_search_approach()