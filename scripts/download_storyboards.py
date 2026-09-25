import urllib.request
from pathlib import Path

out_dir = Path("debug/yt_IW5n")
out_dir.mkdir(parents=True, exist_ok=True)

# 320x180, 3x3 = 9 frames per sheet, 119 frames total -> 119 / 9 = ~14 sheets (M0 to M13)
base_url = "https://i.ytimg.com/sb/IW5n-Tjy6hQ/storyboard3_L3/M{}.jpg?sqp=-oaymwENSDfyq4qpAwVwAcABBqLzl_8DBgjpns_VBg==&rs=AOn4CLDFEE6dF_xQfelv1s_65Uz_ZJSIrA"

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
for i in range(14):
    url = base_url.format(i)
    req = urllib.request.Request(url, headers=headers)
    try:
        data = urllib.request.urlopen(req).read()
        fpath = out_dir / f"sheet_{i:02d}.jpg"
        fpath.write_bytes(data)
        print(f"Downloaded sheet {i} ({len(data)} bytes)")
    except Exception as e:
        print(f"Failed sheet {i}: {e}")
