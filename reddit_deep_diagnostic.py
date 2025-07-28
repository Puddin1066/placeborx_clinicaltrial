#!/usr/bin/env python3
"""
Deep Reddit API Diagnostic
Comprehensive testing of Reddit API authentication and access
"""

import os
import praw
import requests
from dotenv import load_dotenv

def deep_reddit_diagnostic():
    """Comprehensive Reddit API diagnostic"""
    
    print("🔍 Deep Reddit API Diagnostic")
    print("=" * 60)
    
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
    
    # Credential validation
    print("🔧 Credential Analysis:")
    print("-" * 30)
    
    if not client_id or client_id == 'your_reddit_client_id_here':
        print("❌ Client ID not configured")
        return False
    
    if not client_secret or client_secret == 'your_reddit_client_secret_here':
        print("❌ Client Secret not configured")
        return False
    
    # Check credential format
    if len(client_id) < 10:
        print(f"⚠️ Client ID seems short ({len(client_id)} chars)")
    else:
        print(f"✅ Client ID length looks good ({len(client_id)} chars)")
    
    if len(client_secret) < 20:
        print(f"⚠️ Client Secret seems short ({len(client_secret)} chars)")
    else:
        print(f"✅ Client Secret length looks good ({len(client_secret)} chars)")
    
    print()
    
    # Test direct HTTP authentication
    print("🌐 Testing Direct HTTP Authentication:")
    print("-" * 40)
    
    try:
        # Reddit OAuth2 token endpoint
        auth_url = "https://www.reddit.com/api/v1/access_token"
        
        # Prepare the request
        headers = {
            'User-Agent': user_agent
        }
        
        data = {
            'grant_type': 'client_credentials'
        }
        
        # Use basic auth with client_id and client_secret
        response = requests.post(
            auth_url,
            headers=headers,
            data=data,
            auth=(client_id, client_secret)
        )
        
        print(f"HTTP Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ Direct HTTP authentication successful")
            token_data = response.json()
            print(f"Token type: {token_data.get('token_type', 'N/A')}")
            print(f"Expires in: {token_data.get('expires_in', 'N/A')} seconds")
        else:
            print("❌ Direct HTTP authentication failed")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ HTTP authentication test failed: {e}")
    
    print()
    
    # Test PRAW with different configurations
    print("🤖 Testing PRAW Configurations:")
    print("-" * 35)
    
    # Test 1: Standard configuration
    print("\n1. Standard PRAW Configuration:")
    try:
        reddit1 = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        print("   ✅ PRAW client created")
        
        # Test basic access
        try:
            subreddit = reddit1.subreddit('test')
            print("   ✅ Subreddit object created")
        except Exception as e:
            print(f"   ❌ Subreddit access failed: {e}")
            
    except Exception as e:
        print(f"   ❌ PRAW client creation failed: {e}")
    
    # Test 2: Different user agent
    print("\n2. Alternative User Agent:")
    try:
        reddit2 = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent='MyBot/1.0 (by /u/your_username)'
        )
        print("   ✅ Alternative PRAW client created")
        
        try:
            subreddit = reddit2.subreddit('test')
            print("   ✅ Alternative subreddit object created")
        except Exception as e:
            print(f"   ❌ Alternative subreddit access failed: {e}")
            
    except Exception as e:
        print(f"   ❌ Alternative PRAW client failed: {e}")
    
    # Test 3: Minimal configuration
    print("\n3. Minimal Configuration:")
    try:
        reddit3 = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent='bot'
        )
        print("   ✅ Minimal PRAW client created")
        
        try:
            subreddit = reddit3.subreddit('test')
            print("   ✅ Minimal subreddit object created")
        except Exception as e:
            print(f"   ❌ Minimal subreddit access failed: {e}")
            
    except Exception as e:
        print(f"   ❌ Minimal PRAW client failed: {e}")
    
    print()
    
    # Troubleshooting guide
    print("🔧 Troubleshooting Guide:")
    print("-" * 25)
    print("If you're getting 401 errors, try these steps:")
    print()
    print("1. Verify your Reddit app settings:")
    print("   - Go to https://www.reddit.com/prefs/apps")
    print("   - Find your app and click 'edit'")
    print("   - App type should be 'script' (not 'web app')")
    print("   - Remove any redirect URI (leave blank)")
    print("   - Save changes")
    print()
    print("2. Check your credentials:")
    print("   - Client ID is the string under your app name")
    print("   - Client Secret is the 'secret' field")
    print("   - Both should be copied exactly")
    print()
    print("3. Create a new app if needed:")
    print("   - Go to https://www.reddit.com/prefs/apps")
    print("   - Click 'Create App' or 'Create Another App'")
    print("   - Choose 'script' as app type")
    print("   - Name: 'PlaceboRx_Validation_Bot'")
    print("   - Description: 'Market validation bot'")
    print("   - Leave redirect URI blank")
    print()
    print("4. Common issues:")
    print("   - App type must be 'script' for this use case")
    print("   - No redirect URI needed for script apps")
    print("   - User Agent should be descriptive")
    print("   - Credentials are case-sensitive")

if __name__ == "__main__":
    deep_reddit_diagnostic() 