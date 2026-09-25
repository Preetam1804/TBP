import urllib.request
import re
import json

url = 'https://www.youtube.com/watch?v=IW5n-Tjy6hQ'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    html = resp.read().decode('utf-8', errors='ignore')

# Storyboards
m_spec = re.findall(r'"storyboards":\{"playerStoryboardSpecRenderer":\{"spec":"([^"]*)"', html)
print('Storyboard spec:', m_spec)

# Check caption tracks in player_data
player_json_m = re.search(r'ytInitialPlayerResponse\s*=\s*({.+?});(?:var\s|window\[|,|\n)', html)
if player_json_m:
    try:
        p = json.loads(player_json_m.group(1))
        captions_info = p.get('captions', {})
        print('Captions info:', json.dumps(captions_info, indent=2))
    except Exception as e:
        print('JSON error:', e)

# Also check comments or description links in ytInitialData
initial_data_m = re.search(r'ytInitialData\s*=\s*({.+?});</script>', html)
if initial_data_m:
    print('Found ytInitialData (length %d)' % len(initial_data_m.group(1)))
    # find links / github / software mentioned
    text = initial_data_m.group(1)
    links = re.findall(r'https?://[^\s",]+', text)
    print('Links found in ytInitialData:', set(links))
