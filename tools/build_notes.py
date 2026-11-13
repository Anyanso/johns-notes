import re,json,subprocess,html,sys
import os
T=os.path.dirname(os.path.abspath(__file__))+'/'; R=T+'../'; S=T
def combine():
    full=open(R+'content/study_notes.md').read(); easy=open(R+'content/easy_edition.md').read()
    E={}
    for m in re.finditer(r'^## Lecture (\d+)[^\n]*\n(.*?)(?=^## |\Z)',easy,flags=re.S|re.M):
        body=m.group(2); cut=body.find('#### 📖 Textbook deep dive')
        if cut>=0: body=body[:cut]
        body=re.sub(r'^---\s*$','',body,flags=re.M)
        body=re.sub(r'^#{3,4} (.*)$',r'**\1**',body,flags=re.M)
        E[m.group(1)]=body.strip()
    def ins(m):
        head,rest=m.group(1),m.group(2)
        n=re.match(r'## Lecture (\d+)',head).group(1)
        if n not in E: return m.group(0)
        k=rest.find('\n### ')
        if k<0: return m.group(0)
        return head+rest[:k]+'\n\n#### 🧒 Explain it simply\n\n'+E[n]+'\n'+rest[k:]
    return re.sub(r'^(## Lecture \d+[^\n]*\n)(.*?)(?=^## |\Z)',ins,full,flags=re.S|re.M)
md_src=combine()
import markdown
# ---- 1. protect code, then math
codes=[]
def stash_code(m): codes.append(m.group(0)); return 'CODEPH%dX'%(len(codes)-1)
t=re.sub(r'```.*?```',stash_code,md_src,flags=re.S)
t=re.sub(r'`[^`\n]+`',stash_code,t)
t=t.replace('\\$','ESCDOLLARX')
maths=[]
def stash_math(disp):
    def f(m): maths.append((m.group(1).strip() if disp else m.group(1),disp)); return 'MATHPH%dX'%(len(maths)-1)
    return f
t=re.sub(r'\$\$(.+?)\$\$',stash_math(True),t,flags=re.S)
t=re.sub(r'\$([^$\n]+?)\$',stash_math(False),t)
assert '$' not in t.replace('ESCDOLLARX',''), [l for l in t.split('\n') if '$' in l][:3]
json.dump(maths,open(S+'maths_notes.json','w'))
subprocess.run(['node',S+'render_math.js',S+'maths_notes.json',S+'maths_notes_out.json'],check=True,cwd=S)
rendered=json.load(open(S+'maths_notes_out.json'))
def restore(s,tex_only=False):
    def rm(m):
        i=int(m.group(1)); return maths[i][0] if tex_only else rendered[i]
    s=re.sub(r'MATHPH(\d+)X',rm,s)
    return s
def restore_code(s):
    for _ in range(3): s=re.sub(r'CODEPH(\d+)X',lambda m:codes[int(m.group(1))],s)
    return s
# ---- 2. markdown helpers
def mkmd(): return markdown.Markdown(extensions=['tables','fenced_code','sane_lists','codehilite'],extension_configs={'codehilite':{'css_class':'codehilite','guess_lang':False}})
def md_block(text):
    # restore code for markdown, keep math placeholders
    text=restore_code(text)
    h=mkmd().convert(text)
    h=re.sub(r'<table>',r'<div class="tablewrap"><table>',h); h=h.replace('</table>','</table></div>')
    return finish(h)
def finish(h):
    h=restore(h); h=h.replace('ESCDOLLARX','$'); return h
def inline(text):
    text=restore_code(text); h=mkmd().convert(text)
    h=re.sub(r'^<p>(.*)</p>$',r'\1',h.strip(),flags=re.S); return finish(h)
def plain(text):  # for slugs/labels: strip math, code marks, emphasis
    s=restore(re.sub(r'MATHPH\d+X','',text)); s=restore_code(s)
    s=s.replace('`','').replace('*','').replace('ESCDOLLARX','$'); return s
def slug(text):
    s=re.sub(r'[^a-z0-9]+','-',plain(text).lower()).strip('-'); return s[:48].strip('-')
used={}
def uniq(base):
    if base not in used: used[base]=1; return base
    used[base]+=1; return '%s-%d'%(base,used[base])
EMO=re.compile(r'^[^\w]+')
# ---- 3. split into heading chunks (outside code fences)
lines=t.split('\n'); chunks=[]; cur=None; infence=False
for ln in lines:
    if ln.startswith('CODEPH'): pass
    m=re.match(r'^(#{1,4}) (.*)$',ln)
    if m:
        cur={'lvl':len(m.group(1)),'title':m.group(2).strip(),'body':[]}; chunks.append(cur)
    elif cur is not None: cur['body'].append(ln)
    else: pass
for c in chunks: c['body']='\n'.join(c['body']).strip('\n')
out=[]; toc=[]; sec_open=False; sub_open=False; cur_sub_toc=None
def close_sub():
    global sub_open
    if sub_open: out.append('</div>\n'); sub_open=False
def close_sec():
    global sec_open
    close_sub()
    if sec_open: out.append('</section>\n'); sec_open=False
for c in chunks:
    lv=c['lvl']; title=c['title']; body=md_block(c['body']) if c['body'] else ''
    if lv==1:
        out.append('<header class="hero" id="%s"><h1>%s</h1>%s</header>\n'%(uniq(slug(title)),inline(title),body))
    elif lv==2:
        close_sec(); pt=plain(title)
        m=re.match(r'Lecture (\d+)',pt); m2=re.match(r'Part ([A-G])\.',pt)
        if m: sid='lecture-'+m.group(1); cls='lec'; label='L%s · %s'%(m.group(1),inline(re.sub(r'^Lecture \d+ — ','',title)))
        elif m2: sid='part-'+m2.group(1).lower(); cls='part'; label='%s · %s'%(m2.group(1),inline(re.sub(r'^Part [A-G]\. ','',title)))
        else: sid=uniq(slug(title)); cls='part'; label=inline(title)
        used[sid]=1
        out.append('<section class="%s" id="%s"><h2>%s</h2>%s'%(cls,sid,inline(title),body)); sec_open=True
        toc.append([sid,label,[]])
    elif lv==3:
        close_sub(); sid=uniq(slug(title))
        out.append('<div class="sub" id="%s"><h3>%s</h3>%s'%(sid,inline(title),body)); sub_open=True
        toc[-1][2].append((sid,inline(title)))
    elif lv==4:
        pt=plain(title); core=EMO.sub('',pt).strip()
        if core.startswith('Textbook deep dive'): kind,label,ttl='deep','Textbook deep dive',''
        elif core.startswith('Textbook extra'): kind,label='extra','Textbook extra'; ttl=re.sub(r'^Textbook extra\s*[—-]?\s*','',core)
        elif core.startswith('Key takeaways'): kind,label,ttl='key','Key takeaways',''
        elif core.startswith('Explain it simply'): kind,label,ttl='easy','Explain it simply',''
        else: kind,label,ttl='note',pt,''
        if kind=='extra':
            ttl_html=inline(re.sub(r'^[^\w]*Textbook extra\s*[—-]?\s*','',restore_code(title)) ) if ttl else ''
            pid=uniq('p-'+slug('textbook extra '+ttl)) if ttl else uniq('p-textbook-extra')
            head='<h4>%s</h4>'%ttl_html if ttl else ''
            ph=label
        elif kind=='note': pid=uniq('p-'+slug(title)); head=''; ph=inline(title)
        else: pid=uniq('p-'+slug(label)); head=''; ph=label
        if kind=='easy':
            out.append('<aside class="panel easy" id="%s" data-label="Explain it simply"><details><summary>Explain it simply: the plain-language version, with a self-check</summary>%s</details></aside>\n'%(pid,body))
        else:
            out.append('<aside class="panel %s" id="%s" data-label="%s"><div class="ph">%s</div>%s%s</aside>\n'%(kind,pid,html.escape(label,quote=True),ph,head,body))
close_sec()
# ---- 4. TOC and template
def tocli(e):
    sid,label,subs=e
    s='<li class="lecitem" data-id="%s"><a class="lecl" href="#%s" data-id="%s">%s</a><ul>'%(sid,sid,sid,label)
    s+=''.join('<li><a href="#%s" data-id="%s">%s</a></li>'%(i,i,l) for i,l in subs)
    return s+'</ul></li>'
tpl=open(S+'template.html').read()
ANN=open(T+'annotate/annotate.css').read(),open(T+'annotate/annotate.js').read()
page=tpl.replace('<!--__TOC__-->','\n'.join(tocli(e) for e in toc)+'\n').replace('<!--__BODY__-->',''.join(out))
page+='\n<style>\n'+ANN[0]+'</style>\n<script>\n'+ANN[1]+'</script>\n'
page=page.replace('</body>','<a class="homelink" href="index.html">← Home</a></body>',1) if '</body>' in page else page+'<a class="homelink" href="index.html">← Home</a>'
page+='<style>.panel.easy{background:var(--accent-soft);border-color:var(--accent)}.panel.easy summary{cursor:pointer;font-family:var(--font-ui);font-size:14px;font-weight:600;color:var(--accent);padding-bottom:10px}.homelink{position:fixed;left:max(12px,env(safe-area-inset-left,0px));bottom:max(12px,env(safe-area-inset-bottom,0px));z-index:50;font:500 13px var(--font-ui);background:var(--surface);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:6px 10px;text-decoration:none}body.no-tb .panel.easy{display:block}</style>'
open(R+'docs/notes.html','w').write(page)
print('sections',len(toc),'subs',sum(len(e[2]) for e in toc),'bytes',len(page))
assert 'MATHPH' not in page and 'CODEPH' not in page and 'ESCDOLLAR' not in page
