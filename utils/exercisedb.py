import requests
import json
import os
from pathlib import Path

api_base_url = "http://192.168.3.21:5002"
CACHE_FILE = Path(__file__).parent.parent / "cache" / "exercise_cache.json"

class ExerciseDBSearch:
    def __init__(self):
        self.cache = None
        self.load_cache()
    
    def load_cache(self):
        """Load exercise data from cache file, or API if not available"""
        if self.cache is None:
            # Try loading from file first
            if CACHE_FILE.exists():
                try:
                    print(f"Loading cache from file: {CACHE_FILE}")
                    with open(CACHE_FILE, 'r') as f:
                        self.cache = json.load(f)
                    print(f"Successfully loaded {len(self.cache)} exercises from cache")
                    return
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Failed to load cache file: {e}")
            
            # Fall back to API
            self._fetch_from_api()
    
    def _fetch_from_api(self):
        """Fetch exercises from API and save to cache file"""
        print("Loading Cache from ExerciseDB API...")
        try:
            response = requests.get(f"{api_base_url}/v1/exercises", timeout=10)
            response.raise_for_status()
            self.cache = response.json()
            self._save_cache()
            print(f"Successfully loaded {len(self.cache)} exercises from API")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch exercises from API: {e}")
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f, indent=2)
            print(f"Cache saved to {CACHE_FILE}")
        except IOError as e:
            print(f"Warning: Could not save cache file: {e}")

    def search(self, exercise_id: str):
        """Search for exercise by ID or name"""
        if not exercise_id:
            raise ValueError("Exercise ID cannot be empty")
        
        if self.cache is None:
            raise RuntimeError("Cache not loaded. Please check API connection.")
        
        # Search by ID first
        for exercise in self.cache:
            if exercise.get('id') == exercise_id:
                return exercise
        
        # Then search by name (case-insensitive)
        for exercise in self.cache:
            if exercise.get('name', '').lower() == exercise_id.lower():
                return exercise
        
        raise ValueError(f"Exercise '{exercise_id}' not found")