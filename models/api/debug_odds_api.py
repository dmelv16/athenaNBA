"""
Quick test script for Sports Game Odds API
Run: python -m models.api.test_sgo_api
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get('SGO_API_KEY') or os.environ.get('ODDSPAPI_KEY')
BASE_URL = "https://api.sportsgameodds.com/v2"

def test_api():
    print("=" * 60)
    print("SPORTS GAME ODDS API DIAGNOSTIC")
    print("=" * 60)
    
    if not API_KEY:
        print("❌ No API key found! Set SGO_API_KEY in .env")
        return
    
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    
    # Test 1: Basic events endpoint (no filters)
    print("\n📋 Test 1: Get NBA events (no filters)")
    print("-" * 40)
    
    url = f"{BASE_URL}/events"
    params = {
        'apiKey': API_KEY,
        'leagueID': 'NBA',
        'limit': 10
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"Response keys: {list(data.keys())}")
            if 'events' in data:
                print(f"Events count: {len(data['events'])}")
                if data['events']:
                    event = data['events'][0]
                    print(f"First event keys: {list(event.keys())}")
                    print(f"First event: {event.get('homeTeam', {}).get('name', '?')} vs {event.get('awayTeam', {}).get('name', '?')}")
            else:
                print(f"Full response: {str(data)[:500]}")
        else:
            print(f"Error response: {resp.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n⏳ Waiting 7 seconds for rate limit...")
    import time
    time.sleep(7)
    
    # Test 2: Events with odds
    print("\n📋 Test 2: Get NBA events with odds available")
    print("-" * 40)
    
    params['marketOddsAvailable'] = 'true'
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            if 'events' in data:
                print(f"Events with odds: {len(data['events'])}")
                if data['events']:
                    event = data['events'][0]
                    print(f"Event ID: {event.get('eventID')}")
                    print(f"Start time: {event.get('startTime')}")
            else:
                print(f"Response: {str(data)[:500]}")
        else:
            print(f"Error: {resp.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n⏳ Waiting 7 seconds for rate limit...")
    time.sleep(7)
    
    # Test 3: Check account/usage
    print("\n📋 Test 3: Check account usage")
    print("-" * 40)
    
    url = f"{BASE_URL}/account/usage"
    params = {'apiKey': API_KEY}
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Usage info: {data}")
        else:
            print(f"Response: {resp.text[:300]}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    test_api()