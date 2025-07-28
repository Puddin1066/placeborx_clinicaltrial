#!/usr/bin/env python3
"""
Reddit API Diagnostic Script
Tests Reddit API credentials and connection
"""

import os
import praw
from dotenv import load_dotenv

def test_reddit_credentials():
    """Test Reddit API credentials and connection"""
    
    print("🔍 Reddit API Diagnostic")
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
        print("   Please set REDDIT_CLIENT_ID in your .env file")
        return False
    
    if not client_secret or client_secret == 'your_reddit_client_secret_here':
        print("❌ Reddit Client Secret not properly configured")
        print("   Please set REDDIT_CLIENT_SECRET in your .env file")
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
        
        # Test basic API access
        print("\n🔍 Testing basic API access...")
        
        # Try to access Reddit's front page
        try:
            front_page = reddit.front.hot(limit=1)
            for post in front_page:
                print(f"✅ Successfully accessed Reddit front page")
                print(f"   📝 Post: {post.title[:50]}...")
                break
        except Exception as e:
            print(f"❌ Error accessing front page: {e}")
            return False
        
        # Test subreddit access with read-only mode
        print("\n🔍 Testing subreddit access...")
        
        test_subreddits = ['AskReddit', 'science']  # Public subreddits
        
        for subreddit_name in test_subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                # Get a single post
                for post in subreddit.hot(limit=1):
                    print(f"✅ Successfully accessed r/{subreddit_name}")
                    print(f"   📝 Post: {post.title[:50]}...")
                    print(f"   👤 Author: {post.author}")
                    print(f"   ⬆️ Score: {post.score}")
                    break
                    
            except Exception as e:
                print(f"❌ Error accessing r/{subreddit_name}: {e}")
        
        print("\n✅ Reddit API diagnostic completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Reddit API diagnostic failed: {e}")
        return False

if __name__ == "__main__":
    test_reddit_credentials() 