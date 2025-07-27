#!/usr/bin/env python3
"""
Test script for PlaceboRx automation pipeline
Verifies that all components are working correctly
"""

import os
import sys
import json
import subprocess
from datetime import datetime

def test_prerequisites():
    """Test that all prerequisites are met"""
    print("🔍 Testing prerequisites...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required files
    required_files = [
        "package.json",
        "vercel.json", 
        "automated_hypothesis_pipeline.py",
        "run_hypothesis_pipeline.sh"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file missing: {file_path}")
            return False
        print(f"✅ Found: {file_path}")
    
    # Check git repository
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Not in a git repository")
            return False
        print("✅ Git repository found")
    except FileNotFoundError:
        print("❌ Git not installed")
        return False
    
    return True

def test_environment_variables():
    """Test environment variable configuration"""
    print("\n🔍 Testing environment variables...")
    
    required_vars = ["OPENAI_API_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var} is set")
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {missing_vars}")
        print("Some analysis may be limited without API access")
        return False
    
    return True

def test_python_imports():
    """Test that required Python packages can be imported"""
    print("\n🔍 Testing Python imports...")
    
    required_packages = [
        "pandas",
        "requests", 
        "numpy",
        "matplotlib",
        "seaborn"
    ]
    
    optional_packages = [
        "openai",
        "praw",
        "plotly"
    ]
    
    # Test required packages
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            return False
    
    # Test optional packages
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️ {package} not installed (optional)")
    
    return True

def test_automation_script():
    """Test that the automation script can be imported"""
    print("\n🔍 Testing automation script...")
    
    try:
        # Test import without running
        import automated_hypothesis_pipeline
        print("✅ Automation script imports successfully")
        return True
    except Exception as e:
        print(f"❌ Automation script import failed: {str(e)}")
        return False

def test_api_endpoint():
    """Test that the API endpoint file exists and is valid"""
    print("\n🔍 Testing API endpoint...")
    
    api_file = "pages/api/hypothesis-data.js"
    if not os.path.exists(api_file):
        print(f"❌ API file not found: {api_file}")
        return False
    
    try:
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Basic syntax check
        if "export default" in content and "hypothesisData" in content:
            print("✅ API endpoint file is valid")
            return True
        else:
            print("❌ API endpoint file appears invalid")
            return False
    except Exception as e:
        print(f"❌ Error reading API file: {str(e)}")
        return False

def test_landing_content():
    """Test that the landing content file exists and is valid"""
    print("\n🔍 Testing landing content...")
    
    content_file = "vercel_landing_content.js"
    if not os.path.exists(content_file):
        print(f"❌ Content file not found: {content_file}")
        return False
    
    try:
        with open(content_file, 'r') as f:
            content = f.read()
        
        # Basic syntax check
        if "landingContent" in content and "heroSection" in content:
            print("✅ Landing content file is valid")
            return True
        else:
            print("❌ Landing content file appears invalid")
            return False
    except Exception as e:
        print(f"❌ Error reading content file: {str(e)}")
        return False

def test_shell_script():
    """Test that the shell script is executable"""
    print("\n🔍 Testing shell script...")
    
    script_file = "run_hypothesis_pipeline.sh"
    if not os.path.exists(script_file):
        print(f"❌ Shell script not found: {script_file}")
        return False
    
    # Check if executable
    if not os.access(script_file, os.X_OK):
        print(f"❌ Shell script not executable: {script_file}")
        return False
    
    print("✅ Shell script is executable")
    return True

def run_quick_test():
    """Run a quick test of the automation pipeline"""
    print("\n🔍 Running quick automation test...")
    
    try:
        # Import the automation pipeline
        from automated_hypothesis_pipeline import AutomatedHypothesisPipeline
        
        # Create pipeline instance
        pipeline = AutomatedHypothesisPipeline()
        
        # Test data generation
        data = pipeline.generate_hypothesis_data()
        if data and isinstance(data, dict):
            print("✅ Hypothesis data generation works")
        else:
            print("❌ Hypothesis data generation failed")
            return False
        
        # Test API content generation
        api_content = pipeline.create_api_content(data)
        if api_content and "export default" in api_content:
            print("✅ API content generation works")
        else:
            print("❌ API content generation failed")
            return False
        
        # Test landing content generation
        landing_content = pipeline.generate_landing_content()
        if landing_content and "landingContent" in landing_content:
            print("✅ Landing content generation works")
        else:
            print("❌ Landing content generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 PlaceboRx Automation Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Prerequisites", test_prerequisites),
        ("Environment Variables", test_environment_variables),
        ("Python Imports", test_python_imports),
        ("Automation Script", test_automation_script),
        ("API Endpoint", test_api_endpoint),
        ("Landing Content", test_landing_content),
        ("Shell Script", test_shell_script),
        ("Quick Automation Test", run_quick_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The automation pipeline is ready to use.")
        print("\n🚀 To run the complete pipeline:")
        print("   ./run_hypothesis_pipeline.sh")
    else:
        print("⚠️ Some tests failed. Please fix the issues before running the pipeline.")
        print("\n🔧 To check prerequisites only:")
        print("   ./run_hypothesis_pipeline.sh --check-only")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 