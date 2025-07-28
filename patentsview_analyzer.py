#!/usr/bin/env python3
"""
PatentsView API Analyzer for PlaceboRx
Analyzes patents related to digital therapeutics, placebo effects, and alternative treatments
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from config import PATENTSVIEW_API_KEY, PATENTSVIEW_API_BASE, PATENT_SEARCH_TERMS

class PatentsViewAnalyzer:
    """Analyzer for PatentsView API data"""
    
    def __init__(self):
        self.api_key = PATENTSVIEW_API_KEY
        self.base_url = PATENTSVIEW_API_BASE
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'PlaceboRx_Patent_Analyzer/1.0'
        }
    
    def search_patents(self, query_terms, limit=50):
        """Search for patents using PatentsView API"""
        
        print(f"🔍 Searching PatentsView API for: {', '.join(query_terms)}")
        print("=" * 60)
        
        all_patents = []
        
        for term in query_terms:
            print(f"\n📋 Searching for: '{term}'")
            
            # Construct query for PatentsView API
            query = {
                "q": {
                    "_and": [
                        {
                            "_text_phrase": {
                                "patent_abstract": term
                            }
                        },
                        {
                            "_gte": {
                                "patent_date": "2010-01-01"
                            }
                        }
                    ]
                },
                "f": [
                    "patent_number",
                    "patent_title",
                    "patent_abstract",
                    "patent_date",
                    "inventor_name",
                    "assignee_name",
                    "patent_kind"
                ],
                "o": {
                    "per_page": limit,
                    "sort": [{"patent_date": "desc"}]
                }
            }
            
            try:
                # Make API request
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=query
                )
                
                if response.status_code == 200:
                    data = response.json()
                    patents = data.get('patents', [])
                    
                    print(f"   ✅ Found {len(patents)} patents for '{term}'")
                    
                    for patent in patents:
                        patent_data = {
                            'search_term': term,
                            'patent_number': patent.get('patent_number', 'N/A'),
                            'patent_title': patent.get('patent_title', 'N/A'),
                            'patent_abstract': patent.get('patent_abstract', 'N/A'),
                            'patent_date': patent.get('patent_date', 'N/A'),
                            'inventor_name': patent.get('inventor_name', 'N/A'),
                            'assignee_name': patent.get('assignee_name', 'N/A'),
                            'patent_kind': patent.get('patent_kind', 'N/A')
                        }
                        all_patents.append(patent_data)
                        
                        # Show first patent as example
                        if len(patents) > 0 and patents.index(patent) == 0:
                            print(f"   📝 Example: {patent_data['patent_title'][:60]}...")
                            print(f"   📅 Date: {patent_data['patent_date']}")
                            print(f"   👤 Inventor: {patent_data['inventor_name']}")
                else:
                    print(f"   ❌ API request failed: {response.status_code}")
                    print(f"   📄 Response: {response.text[:200]}...")
                    
            except Exception as e:
                print(f"   ❌ Error searching for '{term}': {e}")
        
        return all_patents
    
    def analyze_patent_trends(self, patents):
        """Analyze trends in patent data"""
        
        if not patents:
            print("❌ No patents to analyze")
            return {}
        
        print(f"\n🔬 Analyzing {len(patents)} patents for trends")
        print("-" * 50)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(patents)
        
        # Basic statistics
        print(f"📊 Patent Statistics:")
        print(f"   Total patents: {len(df)}")
        print(f"   Unique search terms: {df['search_term'].nunique()}")
        print(f"   Date range: {df['patent_date'].min()} to {df['patent_date'].max()}")
        
        # Analyze by search term
        print(f"\n📈 Patents by Search Term:")
        term_counts = df['search_term'].value_counts()
        for term, count in term_counts.items():
            print(f"   '{term}': {count} patents")
        
        # Analyze by year
        df['year'] = pd.to_datetime(df['patent_date']).dt.year
        year_counts = df['year'].value_counts().sort_index()
        
        print(f"\n📅 Patents by Year:")
        for year, count in year_counts.items():
            print(f"   {year}: {count} patents")
        
        # Analyze assignees (companies)
        assignee_counts = df['assignee_name'].value_counts().head(10)
        print(f"\n🏢 Top Assignees:")
        for assignee, count in assignee_counts.items():
            if assignee != 'N/A':
                print(f"   {assignee}: {count} patents")
        
        # Identify relevant patents for digital therapeutics
        digital_therapeutic_keywords = [
            'digital', 'mobile', 'app', 'software', 'online', 'remote',
            'telemedicine', 'therapeutic', 'treatment', 'intervention'
        ]
        
        relevant_patents = []
        for _, patent in df.iterrows():
            title_lower = patent['patent_title'].lower()
            abstract_lower = patent['patent_abstract'].lower()
            
            # Check if patent is relevant to digital therapeutics
            relevance_score = 0
            for keyword in digital_therapeutic_keywords:
                if keyword in title_lower or keyword in abstract_lower:
                    relevance_score += 1
            
            if relevance_score >= 2:  # At least 2 relevant keywords
                relevant_patents.append({
                    'patent_number': patent['patent_number'],
                    'patent_title': patent['patent_title'],
                    'patent_date': patent['patent_date'],
                    'assignee_name': patent['assignee_name'],
                    'relevance_score': relevance_score,
                    'search_term': patent['search_term']
                })
        
        print(f"\n🎯 Digital Therapeutic Relevance:")
        print(f"   Highly relevant patents: {len(relevant_patents)}")
        
        if relevant_patents:
            print(f"   Top relevant patents:")
            for patent in sorted(relevant_patents, key=lambda x: x['relevance_score'], reverse=True)[:5]:
                print(f"   📝 {patent['patent_title'][:60]}...")
                print(f"   📅 {patent['patent_date']} | Score: {patent['relevance_score']}")
                print(f"   🏢 {patent['assignee_name']}")
                print()
        
        return {
            'total_patents': len(df),
            'patents_by_term': term_counts.to_dict(),
            'patents_by_year': year_counts.to_dict(),
            'top_assignees': assignee_counts.to_dict(),
            'relevant_patents': relevant_patents
        }
    
    def search_digital_therapeutics_patents(self):
        """Search specifically for digital therapeutics patents"""
        
        print("🔍 Searching for Digital Therapeutics Patents")
        print("=" * 50)
        
        # Specific search terms for digital therapeutics
        digital_terms = [
            'digital therapeutic',
            'mobile health application',
            'software therapeutic',
            'digital intervention',
            'app-based treatment',
            'online therapy platform',
            'remote monitoring system',
            'digital medicine'
        ]
        
        patents = self.search_patents(digital_terms, limit=30)
        
        if patents:
            analysis = self.analyze_patent_trends(patents)
            
            # Save results
            self.save_results(patents, analysis, 'digital_therapeutics_patents')
            
            return patents, analysis
        else:
            print("❌ No digital therapeutics patents found")
            return [], {}
    
    def search_placebo_effect_patents(self):
        """Search for patents related to placebo effects"""
        
        print("🔍 Searching for Placebo Effect Patents")
        print("=" * 50)
        
        placebo_terms = [
            'placebo effect',
            'open label placebo',
            'digital placebo',
            'nocebo effect',
            'expectation effect',
            'mind-body connection',
            'psychological treatment'
        ]
        
        patents = self.search_patents(placebo_terms, limit=30)
        
        if patents:
            analysis = self.analyze_patent_trends(patents)
            
            # Save results
            self.save_results(patents, analysis, 'placebo_effect_patents')
            
            return patents, analysis
        else:
            print("❌ No placebo effect patents found")
            return [], {}
    
    def save_results(self, patents, analysis, filename_prefix):
        """Save patent analysis results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save patents data
        patents_filename = f"{filename_prefix}_{timestamp}.json"
        with open(patents_filename, 'w') as f:
            json.dump(patents, f, indent=2)
        
        # Save analysis results
        analysis_filename = f"{filename_prefix}_analysis_{timestamp}.json"
        with open(analysis_filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   📄 Patents: {patents_filename}")
        print(f"   📊 Analysis: {analysis_filename}")
    
    def generate_market_insights(self, patents_data):
        """Generate market insights from patent data"""
        
        print("\n🎯 Patent-Based Market Insights")
        print("-" * 40)
        
        if not patents_data:
            print("❌ No patent data available for analysis")
            return {}
        
        insights = {
            'total_patents': len(patents_data),
            'technology_trends': {},
            'market_players': {},
            'innovation_areas': {},
            'competitive_landscape': {}
        }
        
        # Analyze technology trends
        digital_keywords = ['digital', 'mobile', 'app', 'software', 'online']
        therapeutic_keywords = ['therapeutic', 'treatment', 'intervention', 'therapy']
        monitoring_keywords = ['monitoring', 'tracking', 'sensor', 'data']
        
        digital_count = sum(1 for p in patents_data if any(k in p['patent_title'].lower() for k in digital_keywords))
        therapeutic_count = sum(1 for p in patents_data if any(k in p['patent_title'].lower() for k in therapeutic_keywords))
        monitoring_count = sum(1 for p in patents_data if any(k in p['patent_title'].lower() for k in monitoring_keywords))
        
        insights['technology_trends'] = {
            'digital_technologies': digital_count,
            'therapeutic_applications': therapeutic_count,
            'monitoring_systems': monitoring_count
        }
        
        # Analyze market players
        assignees = {}
        for patent in patents_data:
            assignee = patent['assignee_name']
            if assignee != 'N/A':
                assignees[assignee] = assignees.get(assignee, 0) + 1
        
        insights['market_players'] = dict(sorted(assignees.items(), key=lambda x: x[1], reverse=True)[:10])
        
        print(f"📊 Technology Trends:")
        print(f"   Digital Technologies: {digital_count} patents")
        print(f"   Therapeutic Applications: {therapeutic_count} patents")
        print(f"   Monitoring Systems: {monitoring_count} patents")
        
        print(f"\n🏢 Top Market Players:")
        for company, count in list(insights['market_players'].items())[:5]:
            print(f"   {company}: {count} patents")
        
        return insights

def test_patentsview_api():
    """Test PatentsView API connection and basic functionality"""
    
    print("🚀 PatentsView API Test")
    print("=" * 50)
    
    analyzer = PatentsViewAnalyzer()
    
    # Test basic API connection
    print("🔍 Testing API connection...")
    
    test_query = {
        "q": {
            "_text_phrase": {
                "patent_abstract": "digital therapeutic"
            }
        },
        "f": ["patent_number", "patent_title"],
        "o": {
            "per_page": 1
        }
    }
    
    try:
        response = requests.post(
            analyzer.base_url,
            headers=analyzer.headers,
            json=test_query
        )
        
        if response.status_code == 200:
            data = response.json()
            patents = data.get('patents', [])
            
            if patents:
                print("✅ PatentsView API connection successful!")
                print(f"📊 Found {len(patents)} test patents")
                print(f"📝 Sample patent: {patents[0].get('patent_title', 'N/A')}")
            else:
                print("⚠️ API connected but no patents found in test")
        else:
            print(f"❌ API connection failed: {response.status_code}")
            print(f"📄 Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Test API connection first
    if test_patentsview_api():
        print("\n" + "="*50)
        
        # Run comprehensive patent analysis
        analyzer = PatentsViewAnalyzer()
        
        # Search for digital therapeutics patents
        digital_patents, digital_analysis = analyzer.search_digital_therapeutics_patents()
        
        # Search for placebo effect patents
        placebo_patents, placebo_analysis = analyzer.search_placebo_effect_patents()
        
        # Generate market insights
        all_patents = digital_patents + placebo_patents
        if all_patents:
            insights = analyzer.generate_market_insights(all_patents)
            
            print(f"\n🎯 Summary:")
            print(f"   Digital Therapeutics Patents: {len(digital_patents)}")
            print(f"   Placebo Effect Patents: {len(placebo_patents)}")
            print(f"   Total Patents Analyzed: {len(all_patents)}")
    else:
        print("❌ PatentsView API test failed. Please check your API key and connection.") 