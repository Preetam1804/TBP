import re
path = r'C:\Users\Preetam\.gemini\antigravity-ide\brain\0280f994-1ca4-4294-972c-67d314205298\.system_generated\steps\1440\content.md'
with open(path, encoding='utf-8', errors='ignore') as f:
    text = f.read()

m = re.findall(r'https://i\.ytimg\.com/[^"\'\s]*', text)
print('i.ytimg URLs:', set([x.split('\\')[0] for x in m if 'sb' in x or 'storyboard' in x or 'vi' in x]))

m_spec = re.findall(r'"storyboards":\{"playerStoryboardSpecRenderer":\{"spec":"([^"]*)"', text)
if m_spec:
    print('Storyboard spec:', m_spec[0])
