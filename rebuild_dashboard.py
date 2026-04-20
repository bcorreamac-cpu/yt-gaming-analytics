#!/usr/bin/env python3
"""Helper: reconstruye strategic_dashboard.html inyectando strategic_data.json."""
import re, json, subprocess

with open('strategic_dashboard.html') as f:
    html = f.read()
with open('strategic_data.json') as f:
    new_json = f.read()

all_opens = [(m.start(), m.end()) for m in re.finditer(r'<script[^>]*>', html)]
all_closes = [m.start() for m in re.finditer(r'</script>', html)]
app_js = html[all_opens[-1][1]:all_closes[-1]]
de = app_js.find(';\nvar fmt')
pure_code = app_js[de:]
header = html[:all_opens[-1][0]]

final = header + '<script>\nvar D=' + new_json + pure_code + '\n</script>\n</body>\n</html>'

# Syntax verify
s2 = [(m.start(), m.end()) for m in re.finditer(r'<script[^>]*>', final)]
c2 = [m.start() for m in re.finditer(r'</script>', final)]
js = final[s2[-1][1]:c2[-1]]
with open('/tmp/check.js','w') as f:
    f.write(js[:js.find(';\nvar fmt')+2])
    f.write('\nvar Chart=function(){};Chart.defaults={color:"",borderColor:"",font:{family:""}};')
    f.write('\nvar document={getElementById:function(){return null},querySelectorAll:function(){return []},createElement:function(){return {innerHTML:"",style:{}}}};')
    f.write('\nvar localStorage={getItem:function(){return "{}"},setItem:function(){}};\nvar navigator={clipboard:{writeText:function(){}}};\nvar window={};\n')
    f.write(js[js.find(';\nvar fmt')+2:])

r = subprocess.run(['node','--check','/tmp/check.js'], capture_output=True, text=True)
if r.returncode != 0:
    print(f"SYNTAX FAIL: {r.stderr[:200]}")
    exit(1)

with open('strategic_dashboard.html','w') as f:
    f.write(final)
print(f"Dashboard rebuilt: {len(final):,} bytes")
