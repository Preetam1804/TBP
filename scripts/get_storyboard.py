import urllib.request
import re

url = 'https://www.youtube.com/watch?v=IW5n-Tjy6hQ'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    html = resp.read().decode('utf-8', errors='ignore')

m_spec = re.findall(r'"playerStoryboardSpecRenderer":\{"spec":"([^"]*)"', html)
if m_spec:
    print('Found spec:', m_spec[0])
    parts = m_spec[0].split('|')
    print('Spec parts:', parts)
else:
    print('No storyboard spec found')
