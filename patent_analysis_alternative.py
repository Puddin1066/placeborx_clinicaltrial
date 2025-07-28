#!/usr/bin/env python3
"""
Alternative Patent Analysis for PlaceboRx
Analyzes patents using alternative data sources and mock data
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import random

class PatentAnalyzer:
    """Alternative patent analyzer using mock data and public sources"""
    
    def __init__(self):
        self.mock_patents = self.generate_mock_patent_data()
    
    def generate_mock_patent_data(self):
        """Generate realistic mock patent data for digital therapeutics"""
        
        mock_patents = []
        
        # Digital therapeutics patent titles
        digital_therapeutic_titles = [
            "Digital Therapeutic System for Chronic Pain Management",
            "Mobile Application for Anxiety Treatment",
            "Software-Based Cognitive Behavioral Therapy Platform",
            "Digital Intervention System for Depression",
            "App-Based Mindfulness Training Program",
            "Remote Monitoring System for Mental Health",
            "Digital Placebo Effect Enhancement Platform",
            "Mobile Health Application for Stress Reduction",
            "Online Therapy Platform with AI Integration",
            "Digital Therapeutic for Sleep Disorders",
            "App-Based Biofeedback System",
            "Digital Intervention for PTSD Treatment",
            "Mobile Application for Chronic Disease Management",
            "Software Therapeutic for Addiction Recovery",
            "Digital Health Platform for Wellness"
        ]
        
        # Placebo effect related patent titles
        placebo_titles = [
            "Open Label Placebo Effect Enhancement System",
            "Digital Placebo Response Optimization",
            "Expectation Effect Amplification Platform",
            "Mind-Body Connection Digital Interface",
            "Psychological Treatment Enhancement System",
            "Nocebo Effect Mitigation Platform",
            "Digital Therapeutic with Placebo Component",
            "Expectation Management Digital System",
            "Placebo Response Prediction Algorithm",
            "Digital Intervention with Expectation Effects"
        ]
        
        # Company names
        companies = [
            "Digital Therapeutics Inc.",
            "Mindful Health Solutions",
            "Telemedicine Technologies",
            "Mental Health Innovations",
            "Digital Medicine Corp.",
            "Wellness Tech Solutions",
            "Mobile Health Systems",
            "Therapeutic Apps LLC",
            "Digital Intervention Labs",
            "Placebo Effect Technologies"
        ]
        
        # Inventor names
        inventors = [
            "Dr. Sarah Chen", "Dr. Michael Rodriguez", "Dr. Emily Johnson",
            "Dr. David Kim", "Dr. Lisa Thompson", "Dr. Robert Wilson",
            "Dr. Jennifer Davis", "Dr. Christopher Brown", "Dr. Amanda Garcia",
            "Dr. Kevin Lee", "Dr. Rachel Martinez", "Dr. Daniel White"
        ]
        
        # Generate digital therapeutics patents
        for i, title in enumerate(digital_therapeutic_titles):
            patent_date = datetime.now() - timedelta(days=random.randint(0, 1000))
            patent = {
                'patent_number': f"US{random.randint(10000000, 99999999)}B2",
                'patent_title': title,
                'patent_abstract': f"This invention relates to a {title.lower()} that provides therapeutic benefits through digital means. The system includes mobile applications, remote monitoring capabilities, and AI-driven interventions to improve patient outcomes.",
                'patent_date': patent_date.strftime('%Y-%m-%d'),
                'inventor_name': random.choice(inventors),
                'assignee_name': random.choice(companies),
                'patent_kind': 'B2',
                'search_term': 'digital therapeutic',
                'category': 'digital_therapeutics'
            }
            mock_patents.append(patent)
        
        # Generate placebo effect patents
        for i, title in enumerate(placebo_titles):
            patent_date = datetime.now() - timedelta(days=random.randint(0, 800))
            patent = {
                'patent_number': f"US{random.randint(10000000, 99999999)}B2",
                'patent_title': title,
                'patent_abstract': f"This invention discloses a {title.lower()} that leverages the placebo effect and expectation mechanisms to enhance therapeutic outcomes. The system utilizes digital interfaces to optimize psychological responses.",
                'patent_date': patent_date.strftime('%Y-%m-%d'),
                'inventor_name': random.choice(inventors),
                'assignee_name': random.choice(companies),
                'patent_kind': 'B2',
                'search_term': 'placebo effect',
                'category': 'placebo_effects'
            }
            mock_patents.append(patent)
        
        return mock_patents
    
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
        print(f"   Digital Therapeutics: {len(df[df['category'] == 'digital_therapeutics'])}")
        print(f"   Placebo Effects: {len(df[df['category'] == 'placebo_effects'])}")
        print(f"   Date range: {df['patent_date'].min()} to {df['patent_date'].max()}")
        
        # Analyze by category
        print(f"\n📈 Patents by Category:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"   {category.replace('_', ' ').title()}: {count} patents")
        
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
            print(f"   {assignee}: {count} patents")
        
        # Identify highly relevant patents
        relevant_keywords = [
            'digital', 'mobile', 'app', 'software', 'online', 'remote',
            'telemedicine', 'therapeutic', 'treatment', 'intervention',
            'placebo', 'expectation', 'psychological', 'mind-body'
        ]
        
        relevant_patents = []
        for _, patent in df.iterrows():
            title_lower = patent['patent_title'].lower()
            abstract_lower = patent['patent_abstract'].lower()
            
            # Check if patent is highly relevant
            relevance_score = 0
            for keyword in relevant_keywords:
                if keyword in title_lower or keyword in abstract_lower:
                    relevance_score += 1
            
            if relevance_score >= 3:  # At least 3 relevant keywords
                relevant_patents.append({
                    'patent_number': patent['patent_number'],
                    'patent_title': patent['patent_title'],
                    'patent_date': patent['patent_date'],
                    'assignee_name': patent['assignee_name'],
                    'relevance_score': relevance_score,
                    'category': patent['category']
                })
        
        print(f"\n🎯 Highly Relevant Patents:")
        print(f"   Total relevant patents: {len(relevant_patents)}")
        
        if relevant_patents:
            print(f"   Top relevant patents:")
            for patent in sorted(relevant_patents, key=lambda x: x['relevance_score'], reverse=True)[:5]:
                print(f"   📝 {patent['patent_title'][:60]}...")
                print(f"   📅 {patent['patent_date']} | Score: {patent['relevance_score']}")
                print(f"   🏢 {patent['assignee_name']}")
                print()
        
        return {
            'total_patents': len(df),
            'patents_by_category': category_counts.to_dict(),
            'patents_by_year': year_counts.to_dict(),
            'top_assignees': assignee_counts.to_dict(),
            'relevant_patents': relevant_patents
        }
    
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
        placebo_keywords = ['placebo', 'expectation', 'psychological', 'mind-body']
        
        digital_count = sum(1 for p in patents_data if any(k in p['patent_title'].lower() for k in digital_keywords))
        therapeutic_count = sum(1 for p in patents_data if any(k in p['patent_title'].lower() for k in therapeutic_keywords))
        placebo_count = sum(1 for p in patents_data if any(k in p['patent_title'].lower() for k in placebo_keywords))
        
        insights['technology_trends'] = {
            'digital_technologies': digital_count,
            'therapeutic_applications': therapeutic_count,
            'placebo_effects': placebo_count
        }
        
        # Analyze market players
        assignees = {}
        for patent in patents_data:
            assignee = patent['assignee_name']
            assignees[assignee] = assignees.get(assignee, 0) + 1
        
        insights['market_players'] = dict(sorted(assignees.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Analyze innovation areas
        innovation_areas = {
            'mobile_health': sum(1 for p in patents_data if 'mobile' in p['patent_title'].lower()),
            'digital_therapeutics': sum(1 for p in patents_data if 'digital therapeutic' in p['patent_title'].lower()),
            'placebo_effects': sum(1 for p in patents_data if 'placebo' in p['patent_title'].lower()),
            'mental_health': sum(1 for p in patents_data if any(k in p['patent_title'].lower() for k in ['anxiety', 'depression', 'mental'])),
            'remote_monitoring': sum(1 for p in patents_data if 'remote' in p['patent_title'].lower())
        }
        
        insights['innovation_areas'] = innovation_areas
        
        print(f"📊 Technology Trends:")
        print(f"   Digital Technologies: {digital_count} patents")
        print(f"   Therapeutic Applications: {therapeutic_count} patents")
        print(f"   Placebo Effects: {placebo_count} patents")
        
        print(f"\n🏢 Top Market Players:")
        for company, count in list(insights['market_players'].items())[:5]:
            print(f"   {company}: {count} patents")
        
        print(f"\n🔬 Innovation Areas:")
        for area, count in innovation_areas.items():
            if count > 0:
                print(f"   {area.replace('_', ' ').title()}: {count} patents")
        
        return insights
    
    def search_digital_therapeutics_patents(self):
        """Search for digital therapeutics patents"""
        
        print("🔍 Analyzing Digital Therapeutics Patents")
        print("=" * 50)
        
        digital_patents = [p for p in self.mock_patents if p['category'] == 'digital_therapeutics']
        
        if digital_patents:
            analysis = self.analyze_patent_trends(digital_patents)
            
            # Save results
            self.save_results(digital_patents, analysis, 'digital_therapeutics_patents')
            
            return digital_patents, analysis
        else:
            print("❌ No digital therapeutics patents found")
            return [], {}
    
    def search_placebo_effect_patents(self):
        """Search for placebo effect patents"""
        
        print("🔍 Analyzing Placebo Effect Patents")
        print("=" * 50)
        
        placebo_patents = [p for p in self.mock_patents if p['category'] == 'placebo_effects']
        
        if placebo_patents:
            analysis = self.analyze_patent_trends(placebo_patents)
            
            # Save results
            self.save_results(placebo_patents, analysis, 'placebo_effect_patents')
            
            return placebo_patents, analysis
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
    
    def generate_competitive_analysis(self):
        """Generate competitive analysis from patent data"""
        
        print("\n🏢 Competitive Analysis")
        print("-" * 30)
        
        # Analyze company patent portfolios
        company_analysis = {}
        for patent in self.mock_patents:
            company = patent['assignee_name']
            if company not in company_analysis:
                company_analysis[company] = {
                    'total_patents': 0,
                    'digital_therapeutics': 0,
                    'placebo_effects': 0,
                    'recent_patents': 0
                }
            
            company_analysis[company]['total_patents'] += 1
            
            if patent['category'] == 'digital_therapeutics':
                company_analysis[company]['digital_therapeutics'] += 1
            elif patent['category'] == 'placebo_effects':
                company_analysis[company]['placebo_effects'] += 1
            
            # Check if patent is recent (last 2 years)
            patent_date = datetime.strptime(patent['patent_date'], '%Y-%m-%d')
            if patent_date > datetime.now() - timedelta(days=730):
                company_analysis[company]['recent_patents'] += 1
        
        # Sort companies by total patents
        sorted_companies = sorted(company_analysis.items(), key=lambda x: x[1]['total_patents'], reverse=True)
        
        print("📊 Company Patent Portfolios:")
        for company, data in sorted_companies[:5]:
            print(f"   🏢 {company}:")
            print(f"      Total Patents: {data['total_patents']}")
            print(f"      Digital Therapeutics: {data['digital_therapeutics']}")
            print(f"      Placebo Effects: {data['placebo_effects']}")
            print(f"      Recent Patents (2 years): {data['recent_patents']}")
            print()
        
        return company_analysis

def test_patent_analysis():
    """Test the patent analysis system"""
    
    print("🚀 Patent Analysis Test")
    print("=" * 50)
    
    analyzer = PatentAnalyzer()
    
    print(f"📊 Generated {len(analyzer.mock_patents)} mock patents")
    print(f"   Digital Therapeutics: {len([p for p in analyzer.mock_patents if p['category'] == 'digital_therapeutics'])}")
    print(f"   Placebo Effects: {len([p for p in analyzer.mock_patents if p['category'] == 'placebo_effects'])}")
    
    # Run comprehensive patent analysis
    print("\n" + "="*50)
    
    # Search for digital therapeutics patents
    digital_patents, digital_analysis = analyzer.search_digital_therapeutics_patents()
    
    # Search for placebo effect patents
    placebo_patents, placebo_analysis = analyzer.search_placebo_effect_patents()
    
    # Generate market insights
    all_patents = digital_patents + placebo_patents
    if all_patents:
        insights = analyzer.generate_market_insights(all_patents)
        
        # Generate competitive analysis
        competitive_analysis = analyzer.generate_competitive_analysis()
        
        print(f"\n🎯 Summary:")
        print(f"   Digital Therapeutics Patents: {len(digital_patents)}")
        print(f"   Placebo Effect Patents: {len(placebo_patents)}")
        print(f"   Total Patents Analyzed: {len(all_patents)}")
        print(f"   Companies Analyzed: {len(competitive_analysis)}")
        
        return True
    else:
        print("❌ No patents to analyze")
        return False

if __name__ == "__main__":
    test_patent_analysis() 