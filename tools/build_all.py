"""Rebuild the whole site: python tools/build_all.py (needs Python 3 with 'markdown' and 'pygments', and Node; run `npm install` once inside tools/)."""
import subprocess, sys, os
T = os.path.dirname(os.path.abspath(__file__))
for s in ('build_notes.py', 'build_prep.py'):
    print('building', s); subprocess.run([sys.executable, os.path.join(T, s)], check=True, cwd=T)
print('done: open docs/index.html')
import datetime
h=open(os.path.join(T,'home.html')).read().replace('<!--DATE-->',datetime.date.today().strftime('%b %d, %Y').replace(' 0',' '))
open(os.path.join(T,'..','docs','index.html'),'w').write(h)
open(os.path.join(T,'..','docs','.nojekyll'),'w').write('')
