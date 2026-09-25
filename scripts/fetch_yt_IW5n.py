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

owner = re.findall(r'"ownerChannelName":"([^"]+)"', html)
author = re.findall(r'"author":"([^"]+)"', html)
keywords = re.findall(r'"keywords":\[(.*?)\]', html)
length = re.findall(r'"lengthSeconds":"([^"]+)"', html)
view_count = re.findall(r'"viewCount":"([^"]+)"', html)

# Let's extract any player response JSON
player_json_m = re.search(r'ytInitialPlayerResponse\s*=\s*({.+?});(?:var\s|window\[|,|\n)', html)
player_data = {}
if player_json_m:
    try:
        player_data = json.loads(player_json_m.group(1))
    except Exception as e:
        pass

out = {
    'owner': owner,
    'author': author,
    'length_seconds': length,
    'keywords': keywords,
    'videoDetails': player_data.get('videoDetails', {}),
}

with open('data/yt_details_IW5n.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)

print("Saved data/yt_details_IW5n.json")
