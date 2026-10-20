"""Check that ANTHROPIC_API_KEY is accepted by the API, without spending anything.
Exits 0 if the key works, 1 if not. Never prints the key."""
import os
import sys

import requests

key = os.environ.get("ANTHROPIC_API_KEY", "")
if not key:
    print("No API key is set.")
    sys.exit(1)
try:
    r = requests.get("https://api.anthropic.com/v1/models", headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                     timeout=20)
except requests.RequestException as e:
    print(f"Could not reach the API: {e}")
    sys.exit(1)
if r.status_code == 200:
    print("API key accepted.")
    sys.exit(0)
if r.status_code in (401, 403):
    print(f"The API rejected this key (HTTP {r.status_code}). Check you copied the whole key and that it is not deleted.")
else:
    print(f"The API returned HTTP {r.status_code}: {r.text[:200]}")
sys.exit(1)
