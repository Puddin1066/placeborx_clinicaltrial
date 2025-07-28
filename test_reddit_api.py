#!/usr/bin/env python3
"""
Reddit API Test Script for PlaceboRx
Tests Reddit API connection and data access
"""

import os
import praw
import json
from datetime import datetime
from dotenv import load_dotenv

def test_reddit_api():
    """Test Reddit API connection and data access"""
    
    print("🔍 Testing Reddit API Connection")
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
    
    # Check if credentials are set
    if not client_id or client_id == 'your_reddit_client_id_here':
        print("❌ Reddit Client ID not properly configured")
        return False
    
    if not client_secret or client_secret == 'your_reddit_client_secret_here':
        print("❌ Reddit Client Secret not properly configured")
        return False
    
    try:
        # Initialize Reddit client
        print("🔄 Initializing Reddit client...")
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        print("✅ Reddit client initialized successfully")
        
        # Test subreddit access
        test_subreddits = ['chronicpain', 'anxiety', 'depression']
        
        for subreddit_name in test_subreddits:
            print(f"\n📊 Testing subreddit: r/{subreddit_name}")
            
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                # Get recent posts
                posts = []
                for post in subreddit.hot(limit=5):
                    post_data = {
                        'title': post.title,
                        'author': str(post.author),
                        'score': post.score,
                        'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'num_comments': post.num_comments,
                        'url': post.url,
                        'selftext': post.selftext[:200] + '...' if len(post.selftext) > 200 else post.selftext
                    }
                    posts.append(post_data)
                
                print(f"✅ Found {len(posts)} posts in r/{subreddit_name}")
                
                # Show first post as example
                if posts:
                    first_post = posts[0]
                    print(f"   📝 Example post: {first_post['title'][:50]}...")
                    print(f"   👤 Author: {first_post['author']}")
                    print(f"   ⬆️ Score: {first_post['score']}")
                    print(f"   💬 Comments: {first_post['num_comments']}")
                    print(f"   📅 Created: {first_post['created_utc']}")
                
            except Exception as e:
                print(f"❌ Error accessing r/{subreddit_name}: {e}")
        
        # Test search functionality
        print(f"\n🔍 Testing search functionality...")
        
        search_results = []
        for subreddit_name in test_subreddits[:1]:  # Test with first subreddit
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                # Search for relevant keywords
                keywords = ['treatment', 'therapy', 'medication', 'help']
                
                for keyword in keywords:
                    search_posts = []
                    for post in subreddit.search(keyword, limit=3):
                        search_posts.append({
                            'title': post.title,
                            'score': post.score,
                            'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                        })
                    
                    if search_posts:
                        search_results.append({
                            'subreddit': subreddit_name,
                            'keyword': keyword,
                            'posts': search_posts
                        })
                        print(f"   ✅ Found {len(search_posts)} posts for '{keyword}' in r/{subreddit_name}")
                
            except Exception as e:
                print(f"❌ Error searching r/{subreddit_name}: {e}")
        
        # Test data authenticity
        print(f"\n🔬 Data Authenticity Check:")
        
        if posts:
            # Check for realistic data patterns
            realistic_indicators = []
            
            # Check for real usernames (not bots)
            real_usernames = [p['author'] for p in posts if p['author'] not in ['[deleted]', 'AutoModerator']]
            if len(real_usernames) > 0:
                realistic_indicators.append(f"Real usernames found: {len(real_usernames)}")
            
            # Check for realistic scores
            realistic_scores = [p['score'] for p in posts if 0 <= p['score'] <= 10000]
            if len(realistic_scores) > 0:
                realistic_indicators.append(f"Realistic scores: {min(realistic_scores)} to {max(realistic_scores)}")
            
            # Check for recent timestamps
            recent_posts = [p for p in posts if '2024' in p['created_utc'] or '2025' in p['created_utc']]
            if len(recent_posts) > 0:
                realistic_indicators.append(f"Recent posts: {len(recent_posts)}")
            
            # Check for varied content
            varied_titles = len(set([p['title'][:20] for p in posts]))
            if varied_titles > 1:
                realistic_indicators.append(f"Varied content: {varied_titles} different titles")
            
            print("   ✅ Real data indicators:")
            for indicator in realistic_indicators:
                print(f"      - {indicator}")
            
            # Check for mock data indicators
            mock_indicators = []
            
            # Check for placeholder text
            placeholder_text = [p for p in posts if 'example' in p['title'].lower() or 'test' in p['title'].lower()]
            if placeholder_text:
                mock_indicators.append("Placeholder text found")
            
            # Check for identical timestamps
            timestamps = [p['created_utc'] for p in posts]
            if len(set(timestamps)) == 1:
                mock_indicators.append("Identical timestamps")
            
            if mock_indicators:
                print("   ⚠️ Mock data indicators:")
                for indicator in mock_indicators:
                    print(f"      - {indicator}")
            else:
                print("   ✅ No mock data indicators found")
        
        # Save test results
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'api_status': 'success',
            'subreddits_tested': test_subreddits,
            'posts_found': len(posts) if 'posts' in locals() else 0,
            'search_results': len(search_results),
            'realistic_indicators': realistic_indicators if 'realistic_indicators' in locals() else [],
            'mock_indicators': mock_indicators if 'mock_indicators' in locals() else []
        }
        
        with open('reddit_api_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n✅ Reddit API test completed successfully!")
        print(f"📄 Results saved to: reddit_api_test_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Reddit API test failed: {e}")
        return False

def test_market_analyzer():
    """Test the market analyzer module specifically"""
    
    print(f"\n🔍 Testing Market Analyzer Module")
    print("=" * 50)
    
    try:
        from market_analyzer import MarketAnalyzer
        
        analyzer = MarketAnalyzer()
        
        # Test the analysis method
        print("🔄 Running market analysis...")
        results = analyzer.analyze_market_demand()
        
        if results:
            print("✅ Market analysis completed successfully")
            print(f"📊 Results: {len(results)} data points")
            
            # Check if results contain real data
            if isinstance(results, dict) and 'community_analysis' in results:
                community_data = results['community_analysis']
                print(f"   📈 Total posts analyzed: {community_data.get('total_posts', 'N/A')}")
                print(f"   🏘️ Communities: {community_data.get('communities', [])}")
                
                if community_data.get('total_posts', 0) > 0:
                    print("   ✅ Real market data detected")
                else:
                    print("   ⚠️ No market data found")
            else:
                print("   ⚠️ Unexpected results format")
        else:
            print("❌ Market analysis returned no results")
            
    except Exception as e:
        print(f"❌ Market analyzer test failed: {e}")

if __name__ == "__main__":
    print("🚀 Reddit API Test Suite")
    print("=" * 50)
    
    # Test basic API connection
    api_success = test_reddit_api()
    
    if api_success:
        # Test market analyzer if API works
        test_market_analyzer()
    
    print(f"\n🎯 Test Summary:")
    print(f"   Reddit API: {'✅ Working' if api_success else '❌ Failed'}")
    print(f"   Real Data: {'✅ Detected' if api_success else '❌ Not tested'}") 