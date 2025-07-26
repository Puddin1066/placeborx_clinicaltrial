import requests
import json

# Test different API formats
base_url = "https://clinicaltrials.gov/api/v2/studies"

# Test 1: Basic request
print("Test 1: Basic request")
try:
    response = requests.get(base_url)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data.get('studies', []))} studies")
except Exception as e:
    print(f"Error: {e}")

# Test 2: With pageSize
print("\nTest 2: With pageSize")
try:
    response = requests.get(f"{base_url}?pageSize=5")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data.get('studies', []))} studies")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Try different search parameters
print("\nTest 3: Try different search parameters")
search_params = [
    "?query=placebo",
    "?expression=placebo", 
    "?search=placebo",
    "?term=placebo"
]

for param in search_params:
    try:
        response = requests.get(f"{base_url}{param}")
        print(f"Status for {param}: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data.get('studies', []))} studies")
    except Exception as e:
        print(f"Error for {param}: {e}") 