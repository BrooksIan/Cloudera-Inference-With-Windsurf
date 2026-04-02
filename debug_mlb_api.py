#!/usr/bin/env python3
"""
Debug MLB API response to understand why values are showing as 0s.
This will help us fix the table data extraction.
"""

import requests
import json
from datetime import datetime

def debug_mlb_api_response():
    """Debug the MLB API response structure."""
    print("🔍 Debugging MLB API Response")
    print("🔒 Using Cloudera AI infrastructure for debugging")
    print("=" * 60)
    
    base_url = "https://statsapi.mlb.com/api/v1"
    current_year = datetime.now().year
    
    print(f"📊 Testing MLB API for {current_year} season...")
    
    # Test different API endpoints
    endpoints_to_test = [
        f"{base_url}/standings?leagueId=103&season={current_year}",
        f"{base_url}/standings?leagueId=104&season={current_year}",
        f"{base_url}/standings",
        f"{base_url}/standings?season={current_year}",
        f"{base_url}/standings?hydrate=team,league,division"
    ]
    
    for i, endpoint in enumerate(endpoints_to_test, 1):
        print(f"\n🔍 Testing Endpoint {i}:")
        print(f"URL: {endpoint}")
        print("-" * 60)
        
        try:
            response = requests.get(endpoint, timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Show structure overview
                print("✅ Response Structure:")
                print(f"   Top-level keys: {list(data.keys())}")
                
                # Check for records
                if 'records' in data:
                    records = data['records']
                    print(f"   Number of records: {len(records)}")
                    
                    if records:
                        first_record = records[0]
                        print(f"   First record keys: {list(first_record.keys())}")
                        
                        # Check division info
                        if 'division' in first_record:
                            division = first_record['division']
                            print(f"   Division keys: {list(division.keys())}")
                            if 'name' in division:
                                print(f"   Division name: {division['name']}")
                        
                        # Check team records
                        if 'teamRecords' in first_record:
                            team_records = first_record['teamRecords']
                            print(f"   Number of team records: {len(team_records)}")
                            
                            if team_records:
                                first_team = team_records[0]
                                print(f"   First team keys: {list(first_team.keys())}")
                                
                                # Check team info
                                if 'team' in first_team:
                                    team = first_team['team']
                                    print(f"   Team keys: {list(team.keys())}")
                                    if 'name' in team:
                                        print(f"   Team name: {team['name']}")
                                
                                # Check for different possible stat locations
                                stat_keys_to_check = ['record', 'stats', 'teamRecord', 'divisionRecord']
                                for stat_key in stat_keys_to_check:
                                    if stat_key in first_team:
                                        stats = first_team[stat_key]
                                        print(f"   Stats found in '{stat_key}': {list(stats.keys())}")
                                        
                                        # Look for wins/losses
                                        if 'wins' in stats:
                                            print(f"   ✅ Found wins: {stats['wins']}")
                                        if 'losses' in stats:
                                            print(f"   ✅ Found losses: {stats['losses']}")
                                        if 'gamesBack' in stats:
                                            print(f"   ✅ Found gamesBack: {stats['gamesBack']}")
                                        if 'streak' in stats:
                                            print(f"   ✅ Found streak: {stats['streak']}")
                                
                                # Show raw first team data
                                print(f"   Raw first team data: {json.dumps(first_team, indent=2)[:500]}...")
                
                # Show raw response (truncated)
                print(f"   Raw response (first 500 chars): {json.dumps(data, indent=2)[:500]}...")
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def test_current_season_data():
    """Test with a known good season (2023) to see structure."""
    print("\n" + "=" * 60)
    print("🔍 Testing with 2023 Season (known good data)")
    print("=" * 60)
    
    try:
        # Try 2023 season which should have real data
        endpoint = "https://statsapi.mlb.com/api/v1/standings?leagueId=103&season=2023"
        response = requests.get(endpoint, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'records' in data and data['records']:
                first_record = data['records'][0]
                
                if 'teamRecords' in first_record and first_record['teamRecords']:
                    first_team = first_record['teamRecords'][0]
                    
                    print("📊 2023 Season Sample Data:")
                    print(f"   Team: {first_team.get('team', {}).get('name', 'Unknown')}")
                    
                    # Check all possible stat locations
                    for key in first_team:
                        if isinstance(first_team[key], dict) and any(k in first_team[key] for k in ['wins', 'losses', 'gamesBack']):
                            stats = first_team[key]
                            print(f"   Stats in '{key}': W={stats.get('wins', 'N/A')}, L={stats.get('losses', 'N/A')}")
                            print(f"   Full stats: {stats}")
                            
        else:
            print("❌ No records found in 2023 data")
            
    except Exception as e:
        print(f"❌ Error testing 2023 data: {e}")

def main():
    """Main debugging function."""
    print("🎯 MLB API Debugging Tool")
    print("🔒 Using Cloudera AI infrastructure for debugging")
    print("🚫 NO WINDSURF MODELS - Cloudera ONLY")
    print("=" * 70)
    
    # Debug current API response
    debug_mlb_api_response()
    
    # Test with known good season
    test_current_season_data()
    
    print("\n" + "=" * 70)
    print("🎉 Debugging Complete!")
    print("📊 Check the output above to understand the API structure")
    print("🔒 This will help fix the table data extraction")

if __name__ == "__main__":
    main()
