# -*- coding: utf-8 -*-
import re, json, subprocess, html, sys
import os
T=os.path.dirname(os.path.abspath(__file__))+'/'
sys.path.insert(0,T+'prep')
from content import *
B=T; P=T+'prep/'
maths=[]
def stash(s):
    def d(m): maths.append((m.group(1).strip(),True)); return 'MATHPH%dX'%(len(maths)-1)
    def i(m): maths.append((m.group(1),False)); return 'MATHPH%dX'%(len(maths)-1)
    s=re.sub(r'\$\$(.+?)\$\$',d,s,flags=re.S)
    return re.sub(r'\$([^$\n]+?)\$',i,s)
def M(s): return stash(s)   # call on every content string

def li(items): return '<ul>'+''.join('<li>%s</li>'%M(x) for x in items)+'</ul>'

# ---- plan
plan=''
for n,(day,dur,title,tasks) in enumerate(DAYS):
    items=''.join('<li><label><input type="checkbox" data-k="plan-%d-%d"><span>%s</span></label></li>'%(n,k,M(t)) for k,t in enumerate(tasks))
    plan+='<article class="day"><header><span class="when">%s</span><span class="dur">%s</span></header><h3>%s</h3><ul class="tasks">%s</ul></article>'%(day,dur,title,items)

# ---- steps
steps=''.join('<li><strong>%s.</strong> %s</li>'%(t,M(b)) for t,b in TEMPLATE_STEPS)

# ---- topics
lik={'High':'high','Medium':'med','Low–Medium':'low'}
tops=''
for n,(i,t,lec,L,why,must,fm,traps,read,drill) in enumerate(TOPICS,1):
    tops+='''<details class="topic" id="%s"><summary><span class="tn">%d</span><span class="tt">%s</span><span class="tl">%s</span><span class="chip %s">%s</span></summary>
<div class="tb"><p class="why">%s</p>
<div class="lesson"><h4>The lesson</h4>%s</div>
<div class="cols"><div><h4>You must be able to</h4>%s</div><div><h4>Formulas to know</h4><div class="fm">%s</div><h4>Common mistakes</h4>%s</div></div>
<p class="read"><strong>Read:</strong> %s &nbsp; <strong>Drill:</strong> <a href="#practice">%s</a></p></div></details>'''%(
        i,n,t,lec,lik[L],L+' priority',M(why),M(LESSONS[i]),li(must),''.join('<div>%s</div>'%M('$'+f+'$') if not f.startswith('$') else '<div>%s</div>'%M(f) for f in fm),li(traps),M(read),drill)

# ---- practice
nav=''.join('<a href="#g-%s">%s</a>'%(c,c) for c,_,_ in GROUPS)
prac=''
dif={1:'warm-up',2:'exam level',3:'stretch'}
for c,title,intro in GROUPS:
    prac+='<section class="grp" id="g-%s"><h3><span class="gc">%s</span>%s</h3><p class="gi">%s</p>'%(c,c,title,intro)
    for (i,g,d,stem,ans) in Q:
        if g!=c: continue
        prac+='''<article class="q" data-id="%s" id="q-%s"><div class="qh"><span class="qid">%s</span><span class="lvl l%d">%s</span><span class="st"></span></div>
%s<div class="stem">%s</div><details><summary>Show the worked answer</summary><div class="ans">%s</div>
<div class="mk"><button type="button" data-v="ok">Got it</button><button type="button" data-v="miss">Review again</button></div></details></article>'''%(i,i,i,d,dif[d],('<p class="bonus">Gradient part is bonus: you will not be asked to derive gradients.</p>' if i in ('A2','A3','A4','A5') else ''),M(stem),M(ans))
    prac+='</section>'

# ---- mock
mc=''
for k,(qq,opts,right,why) in enumerate(MOCK_MC):
    o=''.join('<button type="button" class="opt" data-c="%d">%s</button>'%(1 if j==right else 0,M(x)) for j,x in enumerate(opts))
    mc+='<div class="mc"><p><strong>%d.</strong> %s</p><div class="opts">%s</div><p class="why" hidden>%s</p></div>'%(k+1,M(qq),o,M(why))
lg=''
for (t,stem,ans) in MOCK_LONG:
    lg+='<article class="long"><h4>%s</h4><div class="stem">%s</div><details><summary>Show solution and marking notes</summary><div class="ans">%s</div></details></article>'%(t,M(stem),M(ans))
mist=li(MISTAKES)
mem=''.join('<h4 style="margin:10px 0 4px;font-size:1rem">%s</h4>%s'%(h,li(x)) for h,x in MEMORIZE)

src=M(open(P+'page.html').read())
rep={'<!--PLAN-->':plan,'<!--STEPS-->':steps,'<!--WORKED-->':M(WORKED),'<!--TOPICS-->':tops,'<!--PNAV-->':nav,'<!--PRACTICE-->':prac,
     '<!--MOCKINTRO-->':M(MOCK_INTRO),'<!--MC-->':mc,'<!--LONG-->':lg,'<!--MISTAKES-->':mist,'<!--MEMORIZE-->':mem,
     '<!--KATEX-->':open(T+'katex_inline.css').read().split('@font-face{font-family:KaTeX',1)[0] and '@font-face{font-family:KaTeX'+open(T+'katex_inline.css').read().split('@font-face{font-family:KaTeX',1)[1]}
page=src
for k,v in rep.items(): page=page.replace(k,v)
json.dump(maths,open(P+'maths_prep.json','w'))
subprocess.run(['node',T+'render_math.js',P+'maths_prep.json',P+'maths_prep_out.json'],check=True,cwd=T)
rendered=json.load(open(P+'maths_prep_out.json'))
page=re.sub(r'MATHPH(\d+)X',lambda m:rendered[int(m.group(1))],page)
assert '$' not in re.sub(r'<style.*?</style>|<script.*?</script>','',page,flags=re.S).replace('\\$',''), 'stray $'
page=page.replace('<a href="#plan">Plan</a>','<a href="index.html">Home</a><a href="#plan">Plan</a>',1)
open(T+'../docs/prep.html','w').write(page)
print('bytes',len(page),'questions',len(Q))
