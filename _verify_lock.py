import subprocess
import sys

# 1) Try a real syntax check if node exists
try:
    r = subprocess.run(['node', '--check', 'static/js/app.js'],
                       capture_output=True, text=True, timeout=15)
    js_syntax = ('NODE OK' if r.returncode == 0 else 'NODE FAIL: ' + r.stderr[:300])
except FileNotFoundError:
    # Fallback: crude delimiter balance (ignores strings/comments well enough for smoke)
    src = open('static/js/app.js', encoding='utf-8').read()
    pairs = {'{': '}', '(': ')', '[': ']'}
    counts = {k: src.count(k) - src.count(v) for k, v in pairs.items()}
    js_syntax = f"NO NODE — balance deltas {{:{counts['{']}, (:{counts['(']}, [:{counts['[']} (0 = balanced)"

print(js_syntax)

# 2) Static assertions on the new logic
src = open('static/js/app.js', encoding='utf-8').read()
checks = {
    "removeItem('wallpaper_unlocked')": "sessionStorage.removeItem('wallpaper_unlocked')" in src,
    'instant hide display=none': "wallpaperSection.style.display = 'none'" in src,
    'lock button handler bound': "getElementById('lockSettingsBtn')" in src,
    'confirmation message': 'Advanced settings locked' in src,
}
for name, ok in checks.items():
    print(('PASS' if ok else 'FAIL'), name)

html = open('templates/settings.html', encoding='utf-8').read()
print(('PASS' if 'lockSettingsBtn' in html else 'FAIL'), 'settings.html has lock button')

sys.exit(0 if all(checks.values()) else 1)