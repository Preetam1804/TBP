import urllib.request, re

req = urllib.request.Request('https://www.youtube.com/watch?v=xOmMjNfuus4', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
try:
    html = urllib.request.urlopen(req).read().decode('utf-8')
    m = re.search(r'"shortDescription":"(.*?)"', html)
    if m:
        desc = m.group(1).encode('utf-8').decode('unicode_escape')
        with open('data/yt_desc.txt', 'w', encoding='utf-8') as f:
            f.write(desc)
        print('Wrote data/yt_desc.txt')
    m2 = re.search(r'"lengthSeconds":"(.*?)"', html)
    if m2:
        print('Length (seconds):', m2.group(1))
    
    # Check for caption tracks
    m_cap = re.search(r'"captionTracks":\[(.*?)\]', html)
    if m_cap:
        m_url = re.search(r'"baseUrl":"(.*?)"', m_cap.group(1))
        if m_url:
            import xml.etree.ElementTree as ET
            cap_url = m_url.group(1).encode('utf-8').decode('unicode_escape')
            xml_data = urllib.request.urlopen(cap_url).read().decode('utf-8')
            root = ET.fromstring(xml_data)
            lines = [elem.text for elem in root.findall('.//text') if elem.text]
            with open('data/yt_transcript.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f'Wrote data/yt_transcript.txt with {len(lines)} lines')
except Exception as e:
    print('Error:', e)
