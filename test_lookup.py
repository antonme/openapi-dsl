import requests
import json
import sys

def test_lookup(word, port=8001):
    """Test the lookup endpoint with a single word"""
    url = f"http://localhost:{port}/lookup/{word}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"✅ Lookup for '{word}' successful")
            data = response.json()
            print(f"Found {data['count']} entries in {data['dictionaries_searched']}")
            print(f"Response time: {data['time_taken']:.6f} seconds")
            return True
        else:
            print(f"❌ Lookup for '{word}' failed with status code {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during lookup for '{word}': {e}")
        return False

def test_multi_lookup(words, port=8001):
    """Test the multi-lookup endpoint with multiple words"""
    query = " ".join(words)
    url = f"http://localhost:{port}/multi-lookup?query={query}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"✅ Multi-lookup for '{query}' successful")
            data = response.json()
            print(f"Found {data['count']} entries in {data['dictionaries_searched']}")
            print(f"Response time: {data['time_taken']:.6f} seconds")
            return True
        else:
            print(f"❌ Multi-lookup for '{query}' failed with status code {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during multi-lookup for '{query}': {e}")
        return False

def test_search(query, port=8001):
    """Test the search endpoint"""
    url = f"http://localhost:{port}/search?query={query}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"✅ Search for '{query}' successful")
            data = response.json()
            print(f"Found {data['count']} entries in {data['dictionaries_searched']}")
            print(f"Response time: {data['time_taken']:.6f} seconds")
            return True
        else:
            print(f"❌ Search for '{query}' failed with status code {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during search for '{query}': {e}")
        return False

def main():
    port = 8001
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    # Test single word lookup
    print("\n=== Testing single word lookup ===")
    test_lookup("абажур", port)
    
    # Test multi-word lookup
    print("\n=== Testing multi-word lookup ===")
    test_multi_lookup(["стол", "стул"], port)
    
    # Test search
    print("\n=== Testing search ===")
    test_search("абажур", port)

if __name__ == "__main__":
    main() 