"""make_corpus3.py TEXT OUT: the closing KLD corpus (11 documents) from a plain-text file of project documents (1,020,004 characters in
the run). Eight short documents of about 12,000 characters (3,300 to 4,400 tokens) at 47,000-character spacing from offset 120,000, then three
long ones of about 48,000 characters (10,800 to 14,500 scored positions) at 95,000-character spacing; each starts and ends at a line boundary.
These regions were not used by the first two-document corpus."""
import json, sys
t = open(sys.argv[1], encoding='utf-8', errors='ignore').read()
def cut(start, n):
    a = t.find('\n', start) + 1; b = t.find('\n', a + n); return t[a:b if b > 0 else a + n]
docs = []; pos = 120000
for i in range(8): docs.append((f'set3-short-{i:02d}', cut(pos, 12000))); pos += 47000
for i in range(3): docs.append((f'set3-long-{i:02d}', cut(pos, 48000))); pos += 95000
with open(sys.argv[2], 'w') as f:
    for s, p in docs: f.write(json.dumps({'prompt': p, 'score_from': 0, 'source': s}) + '\n')
