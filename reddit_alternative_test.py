#!/usr/bin/env python3
"""
Alternative Reddit API Test
Tests Reddit API with fallback to mock data for development
"""

import os
import json
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

def generate_mock_reddit_data():
    """Generate realistic mock Reddit data for testing"""
    
    mock_posts = []
    
    # Realistic post titles and content for chronic pain, anxiety, depression
    chronic_pain_titles = [
        "Chronic pain for 3 years, nothing helps",
        "Desperate for pain relief alternatives",
        "Has anyone tried non-pharmaceutical treatments?",
        "Chronic back pain - need suggestions",
        "Alternative therapies for chronic pain?",
        "Nothing works for my chronic pain",
        "Looking for non-drug pain management",
        "Chronic pain and mental health",
        "Desperate for pain relief",
        "Chronic pain affecting my life"
    ]
    
    anxiety_titles = [
        "Anxiety getting worse, need help",
        "Alternative treatments for anxiety?",
        "Anxiety and depression together",
        "Non-medication anxiety relief",
        "Desperate for anxiety help",
        "Anxiety affecting my daily life",
        "Looking for anxiety alternatives",
        "Anxiety and chronic stress",
        "Need help with severe anxiety",
        "Anxiety treatment options"
    ]
    
    depression_titles = [
        "Depression treatment alternatives",
        "Non-pharmaceutical depression help",
        "Depression and chronic pain",
        "Looking for depression relief",
        "Depression affecting everything",
        "Alternative depression treatments",
        "Depression and anxiety together",
        "Need help with depression",
        "Depression treatment options",
        "Desperate for depression help"
    ]
    
    subreddits = [
        {'name': 'chronicpain', 'titles': chronic_pain_titles},
        {'name': 'anxiety', 'titles': anxiety_titles},
        {'name': 'depression', 'titles': depression_titles},
        {'name': 'mentalhealth', 'titles': anxiety_titles + depression_titles},
        {'name': 'wellness', 'titles': chronic_pain_titles + anxiety_titles}
    ]
    
    usernames = [
        'pain_sufferer_2024', 'anxiety_free', 'depression_warrior',
        'chronic_pain_survivor', 'mental_health_seeker', 'wellness_journey',
        'alternative_healing', 'non_pharma_seeker', 'holistic_approach',
        'mind_body_connection', 'healing_journey', 'recovery_path'
    ]
    
    for subreddit in subreddits:
        for i in range(random.randint(8, 15)):  # 8-15 posts per subreddit
            post = {
                'title': random.choice(subreddit['titles']),
                'author': random.choice(usernames),
                'score': random.randint(1, 150),
                'num_comments': random.randint(0, 25),
                'created_utc': (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S'),
                'url': f"https://reddit.com/r/{subreddit['name']}/comments/mock{i}",
                'selftext': f"This is a mock post about {subreddit['name']} and alternative treatments. Looking for help and suggestions.",
                'subreddit': subreddit['name']
            }
            mock_posts.append(post)
    
    return mock_posts

def test_reddit_api_with_fallback():
    """Test Reddit API with fallback to mock data"""
    
    print("🔍 Testing Reddit API with Fallback")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get API credentials
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT', 'PlaceboRx_Validation_Bot/1.0')
    
    print(f"Client ID: {client_id}")
    print(f"Client Secret: {'*' * len(client_secret) if client_secret else 'NOT SET'}")
    print(f"User Agent: {user_agent}")
    print()
    
    # Try real Reddit API first
    try:
        import praw
        
        print("🔄 Attempting real Reddit API connection...")
        
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Test with a public subreddit
        subreddit = reddit.subreddit('AskReddit')
        posts = []
        
        for post in subreddit.hot(limit=3):
            post_data = {
                'title': post.title,
                'author': str(post.author),
                'score': post.score,
                'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'num_comments': post.num_comments,
                'url': post.url,
                'selftext': post.selftext[:200] + '...' if len(post.selftext) > 200 else post.selftext,
                'subreddit': 'AskReddit'
            }
            posts.append(post_data)
        
        print("✅ Real Reddit API working!")
        print(f"📊 Retrieved {len(posts)} real posts")
        
        # Show example post
        if posts:
            first_post = posts[0]
            print(f"   📝 Example: {first_post['title'][:50]}...")
            print(f"   👤 Author: {first_post['author']}")
            print(f"   ⬆️ Score: {first_post['score']}")
        
        return posts, True
        
    except Exception as e:
        print(f"❌ Real Reddit API failed: {e}")
        print("🔄 Falling back to mock data...")
        
        # Generate mock data
        mock_posts = generate_mock_reddit_data()
        
        print("✅ Mock data generated successfully!")
        print(f"📊 Generated {len(mock_posts)} mock posts")
        
        # Show example posts
        for subreddit_name in ['chronicpain', 'anxiety', 'depression']:
            subreddit_posts = [p for p in mock_posts if p['subreddit'] == subreddit_name]
            if subreddit_posts:
                example_post = subreddit_posts[0]
                print(f"   📝 r/{subreddit_name}: {example_post['title'][:40]}...")
                print(f"   👤 Author: {example_post['author']}")
                print(f"   ⬆️ Score: {example_post['score']}")
        
        return mock_posts, False

def analyze_mock_data(posts):
    """Analyze the mock data for market insights"""
    
    print(f"\n🔬 Analyzing {len(posts)} posts for market insights")
    print("-" * 50)
    
    # Group by subreddit
    subreddit_data = {}
    for post in posts:
        subreddit = post['subreddit']
        if subreddit not in subreddit_data:
            subreddit_data[subreddit] = []
        subreddit_data[subreddit].append(post)
    
    # Analyze each subreddit
    for subreddit, subreddit_posts in subreddit_data.items():
        print(f"\n📊 r/{subreddit} Analysis:")
        print(f"   📝 Total posts: {len(subreddit_posts)}")
        
        # Calculate engagement metrics
        total_score = sum(p['score'] for p in subreddit_posts)
        total_comments = sum(p['num_comments'] for p in subreddit_posts)
        avg_score = total_score / len(subreddit_posts)
        avg_comments = total_comments / len(subreddit_posts)
        
        print(f"   ⬆️ Average score: {avg_score:.1f}")
        print(f"   💬 Average comments: {avg_comments:.1f}")
        
        # Keyword analysis
        keywords = ['desperate', 'nothing works', 'alternative', 'non-pharma', 'help', 'treatment']
        keyword_counts = {}
        
        for keyword in keywords:
            count = sum(1 for p in subreddit_posts if keyword.lower() in p['title'].lower())
            if count > 0:
                keyword_counts[keyword] = count
        
        if keyword_counts:
            print(f"   🔍 Key phrases found:")
            for keyword, count in keyword_counts.items():
                print(f"      - '{keyword}': {count} posts")
    
    # Overall market demand indicators
    print(f"\n🎯 Market Demand Indicators:")
    print("-" * 30)
    
    total_posts = len(posts)
    desperate_posts = sum(1 for p in posts if 'desperate' in p['title'].lower())
    alternative_posts = sum(1 for p in posts if 'alternative' in p['title'].lower())
    help_posts = sum(1 for p in posts if 'help' in p['title'].lower())
    
    print(f"   📊 Total posts analyzed: {total_posts}")
    print(f"   😰 'Desperate' mentions: {desperate_posts} ({desperate_posts/total_posts*100:.1f}%)")
    print(f"   🔄 'Alternative' mentions: {alternative_posts} ({alternative_posts/total_posts*100:.1f}%)")
    print(f"   🆘 'Help' mentions: {help_posts} ({help_posts/total_posts*100:.1f}%)")
    
    # Market opportunity score
    opportunity_score = (desperate_posts + alternative_posts + help_posts) / total_posts * 100
    print(f"   🎯 Market opportunity score: {opportunity_score:.1f}%")

def save_test_results(posts, is_real_data):
    """Save test results to file"""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_type': 'real' if is_real_data else 'mock',
        'total_posts': len(posts),
        'subreddits': list(set(p['subreddit'] for p in posts)),
        'sample_posts': posts[:5]  # Save first 5 posts as sample
    }
    
    filename = 'reddit_api_test_results.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    print("🚀 Reddit API Test with Fallback")
    print("=" * 50)
    
    # Test API with fallback
    posts, is_real_data = test_reddit_api_with_fallback()
    
    # Analyze the data
    analyze_mock_data(posts)
    
    # Save results
    save_test_results(posts, is_real_data)
    
    print(f"\n🎯 Test Summary:")
    print(f"   Data Source: {'✅ Real Reddit API' if is_real_data else '🔄 Mock Data'}")
    print(f"   Posts Analyzed: {len(posts)}")
    print(f"   Subreddits: {len(set(p['subreddit'] for p in posts))}")
    
    if not is_real_data:
        print(f"\n💡 To fix Reddit API authentication:")
        print(f"   1. Go to https://www.reddit.com/prefs/apps")
        print(f"   2. Edit your app and change type to 'script'")
        print(f"   3. Remove any redirect URI")
        print(f"   4. Update your .env file with new credentials") 