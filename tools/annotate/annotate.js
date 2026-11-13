/* Annotation layer: highlights, notes and freehand ink. Saved privately per viewer (db) with a localStorage mirror. */
(function () {
  'use strict';
  var main = document.getElementById('main'); if (!main) return;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var BLOCK = 'p, li, td, th, h2, h3, h4, h5, blockquote, pre';
  var SKIP = '.katex, button, .copy, .annink';
  var LS = 'ann1:' + (document.title || 'site') + ':';
  var COLORS = { y: 'Yellow', g: 'Green', p: 'Pink', b: 'Blue' };
  var INKS = [['auto', 'Text colour', 'var(--ink)'], ['#e5484d', 'Red', '#e5484d'], ['#3b82f6', 'Blue', '#3b82f6'], ['#16a34a', 'Green', '#16a34a'], ['#f59e0b', 'Orange', '#f59e0b']];
  var S = {};            // sec -> {a: groups, ink: strokes}
  var cloud = { col: null, ok: false, dl: null };
  var dirty = {}, writing = {}, nChunks = {};
  var undo = [], tool = 'pen', inkColor = 'auto', inkW = 3, drawing = false, curStroke = null;
  var tbEl = document.getElementById('tb');

  function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function uid() { return 'g' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function sec(id) { return S[id] || (S[id] = { a: [], ink: [], t: 0 }); }
  function variant() { return (tbEl && tbEl.checked ? 't' : 'n') + (window.innerWidth < 900 ? 'm' : 'd'); }
  var toastEl = el('div', 'anntoast'); toastEl.setAttribute('role', 'status'); document.body.appendChild(toastEl);
  var toastT; function toast(m) { toastEl.textContent = m; toastEl.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(function () { toastEl.classList.remove('show'); }, 2600); }

  /* ---------- text model ---------- */
  function blocksOf(s) {
    if (s._bl) return s._bl;
    return (s._bl = $$(BLOCK, s).filter(function (b) { return !b.closest(SKIP); }));
  }
  function ownNodes(b) {
    var w = document.createTreeWalker(b, NodeFilter.SHOW_TEXT, { acceptNode: function (n) {
      var p = n.parentNode; if (p.closest(SKIP)) return NodeFilter.FILTER_REJECT;
      return p.closest(BLOCK) === b ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    } }), out = [], n, pos = 0;
    while ((n = w.nextNode())) { out.push({ n: n, s: pos }); pos += n.nodeValue.length; }
    out.total = pos; return out;
  }
  function textOf(b) { return ownNodes(b).map(function (o) { return o.n.nodeValue; }).join(''); }
  function pointOff(b, cont, off, list) {
    var pr = document.createRange(); pr.setStart(cont, off); pr.collapse(true);
    var total = 0;
    for (var i = 0; i < list.length; i++) {
      var o = list[i]; if (o.n === cont) return o.s + off;
      if (pr.comparePoint(o.n, o.n.nodeValue.length) <= 0) total = o.s + o.n.nodeValue.length; else break;
    }
    return total;
  }

  /* ---------- rendering highlights ---------- */
  function unwrap(s) {
    $$('.ah', s).forEach(function (sp) { var p = sp.parentNode; while (sp.firstChild) p.insertBefore(sp.firstChild, sp); p.removeChild(sp); });
    $$('.ahm', s).forEach(function (m) { m.classList.remove('ahm', 'cy', 'cg', 'cp', 'cb', 'pulse'); m.removeAttribute('data-g'); });
    s.normalize(); s._bl = null;
  }
  function wrapRange(b, st, en, g) {
    var list = ownNodes(b);
    list.forEach(function (o) {
      var a = Math.max(st, o.s), z = Math.min(en, o.s + o.n.nodeValue.length);
      if (z <= a) return;
      var n = o.n; if (z - o.s < n.nodeValue.length) n.splitText(z - o.s);
      if (a - o.s > 0) n = n.splitText(a - o.s);
      var sp = el('span', 'ah c' + g.c); sp.setAttribute('data-g', g.id);
      n.parentNode.insertBefore(sp, n); sp.appendChild(n);
    });
    // atomic math inside the range
    $$('.katex', b).forEach(function (k) {
      if (k.parentNode.closest('.katex') || k.closest(BLOCK) !== b) return;
      var p = pointOff(b, k.parentNode, Array.prototype.indexOf.call(k.parentNode.childNodes, k), ownNodes(b));
      if (p > st && p < en) { k.classList.add('ahm', 'c' + g.c); k.setAttribute('data-g', g.id); }
    });
  }
  function renderHL(id) {
    var s = document.getElementById(id); if (!s) return;
    unwrap(s); var d = sec(id), bl = blocksOf(s);
    d.a.forEach(function (g) {
      (g.segs || []).forEach(function (sg) {
        var b = bl[sg.b], tx = b ? textOf(b) : '';
        if (!b || tx.slice(sg.s, sg.e) !== sg.q) {            // moved? find the quote again
          b = null; for (var i = 0; i < bl.length; i++) { var t = textOf(bl[i]), k = t.indexOf(sg.q); if (k >= 0) { b = bl[i]; sg.b = i; sg.s = k; sg.e = k + sg.q.length; break; } }
        }
        if (b) wrapRange(b, sg.s, sg.e, g);
      });
      if (g.n) { var sps = $$('.ah[data-g="' + g.id + '"]', s); if (sps.length) sps[sps.length - 1].classList.add('hasnote'); }
    });
  }

  /* ---------- ink ---------- */
  var inkSvg = {};
  function svgFor(id) {
    var s = document.getElementById(id); if (!s) return null;
    var v = inkSvg[id];
    if (!v) {
      v = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); v.setAttribute('class', 'annink'); v.setAttribute('preserveAspectRatio', 'none'); v.dataset.sec = id;
      s.appendChild(v); inkSvg[id] = v;
      if (window.ResizeObserver) new ResizeObserver(function () { sizeSvg(id); }).observe(s);
      attachDraw(v);
    }
    sizeSvg(id); return v;
  }
  function sizeSvg(id) { var s = document.getElementById(id), v = inkSvg[id]; if (!s || !v) return; v.setAttribute('viewBox', '0 0 1000 ' + Math.max(1, s.offsetHeight)); }
  function pathD(p) {
    if (p.length < 4) return 'M' + p[0] + ' ' + p[1] + ' l0.01 0';
    var d = 'M' + p[0] + ' ' + p[1];
    for (var i = 2; i < p.length - 2; i += 2) d += ' Q' + p[i] + ' ' + p[i + 1] + ' ' + ((p[i] + p[i + 2]) / 2).toFixed(1) + ' ' + ((p[i + 1] + p[i + 3]) / 2).toFixed(1);
    return d + ' L' + p[p.length - 2] + ' ' + p[p.length - 1];
  }
  function mkPath(st) {
    var pa = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pa.setAttribute('d', pathD(st.p)); pa.setAttribute('stroke', st.k === 'auto' ? 'var(--ink)' : st.k); pa.setAttribute('stroke-width', st.w);
    if (st.m) pa.setAttribute('opacity', '.38'); pa.__st = st; return pa;
  }
  function renderInk(id) {
    var d = sec(id); var v = d.ink.length ? svgFor(id) : inkSvg[id]; if (!v) return;
    while (v.firstChild) v.removeChild(v.firstChild);
    var vr = variant();
    d.ink.forEach(function (st) { if (st.v === vr) v.appendChild(mkPath(st)); });
  }
  function hiddenInk() { var n = 0, vr = variant(); Object.keys(S).forEach(function (k) { S[k].ink.forEach(function (st) { if (st.v !== vr) n++; }); }); return n; }
  function attachDraw(v) {
    v.addEventListener('pointerdown', function (e) {
      if (!document.body.classList.contains('anndraw') || document.body.classList.contains('annscroll')) return;
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      e.preventDefault(); v.setPointerCapture(e.pointerId); drawing = true;
      var id = v.dataset.sec;
      if (tool === 'eraser') { eraseAt(v, e); return; }
      curStroke = { id: id, st: { k: inkColor, w: tool === 'marker' ? 16 : inkW, m: tool === 'marker' ? 1 : 0, v: variant(), p: [] }, el: null };
      addPt(v, e);
      curStroke.el = mkPath(curStroke.st); v.appendChild(curStroke.el);
    });
    v.addEventListener('pointermove', function (e) {
      if (!drawing) return; e.preventDefault();
      if (tool === 'eraser') { eraseAt(v, e); return; }
      if (!curStroke) return;
      var evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e]; if (!evs.length) evs = [e];
      evs.forEach(function (ev) { addPt(v, ev); });
      curStroke.el.setAttribute('d', pathD(curStroke.st.p));
    });
    function end(e) {
      if (!drawing) return; drawing = false;
      if (curStroke) {
        var c = curStroke; curStroke = null;
        sec(c.id).ink.push(c.st); undo.push({ t: 'add', id: c.id, st: c.st }); c.el.__st = c.st; touch(c.id, 'ink'); updateBar();
      }
    }
    v.addEventListener('pointerup', end); v.addEventListener('pointercancel', end);
  }
  function addPt(v, e) {
    var r = v.getBoundingClientRect(), p = curStroke.st.p;
    var x = Math.round((e.clientX - r.left) / r.width * 10000) / 10, y = Math.round((e.clientY - r.top) * 10) / 10;
    var n = p.length;
    if (n >= 2) { var dx = (x - p[n - 2]) * r.width / 1000, dy = y - p[n - 1]; if (dx * dx + dy * dy < 2.25) return; }
    p.push(x, y);
  }
  function eraseAt(v, e) {
    var r = v.getBoundingClientRect(), px = e.clientX, py = e.clientY, id = v.dataset.sec, d = S[id]; if (!d) return;
    var vr = variant(), hit = [];
    d.ink.forEach(function (st) {
      if (st.v !== vr) return;
      var p = st.p, tol = 12 + st.w / 2;
      for (var i = 0; i < p.length; i += 2) {
        var x = r.left + p[i] * r.width / 1000, y = r.top + p[i + 1];
        var nx = i + 2 < p.length ? r.left + p[i + 2] * r.width / 1000 : x, ny = i + 2 < p.length ? r.top + p[i + 3] : y;
        if (distSeg(px, py, x, y, nx, ny) < tol) { hit.push(st); break; }
      }
    });
    if (!hit.length) return;
    hit.forEach(function (st) { d.ink.splice(d.ink.indexOf(st), 1); undo.push({ t: 'del', id: id, st: st }); });
    renderInk(id); touch(id, 'ink'); updateBar();
  }
  function distSeg(px, py, x1, y1, x2, y2) {
    var dx = x2 - x1, dy = y2 - y1, l = dx * dx + dy * dy, t = l ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / l)) : 0;
    var cx = x1 + t * dx, cy = y1 + t * dy; return Math.hypot(px - cx, py - cy);
  }
  function doUndo() {
    var u = undo.pop(); if (!u) return; var d = sec(u.id);
    if (u.t === 'add') { var i = d.ink.indexOf(u.st); if (i >= 0) d.ink.splice(i, 1); } else d.ink.push(u.st);
    renderInk(u.id); touch(u.id, 'ink'); updateBar();
  }
  function centerSec() {
    var y = innerHeight / 2, best = null;
    $$('#main > section').forEach(function (s) { var r = s.getBoundingClientRect(); if (r.top <= y && r.bottom >= y) best = s; });
    return best || $('#main > section');
  }

  /* ---------- persistence ---------- */
  function touch(id, what) { sec(id).t = Date.now(); lsSet(LS + id, JSON.stringify(S[id])); dirty[id] = true; setStatus('saving'); schedule(id); updateFab(); }
  var timers = {};
  function schedule(id) { clearTimeout(timers[id]); timers[id] = setTimeout(function () { flush(id); }, 1200); }
  function chunkInk(ink) {
    var out = [], cur = [], size = 0;
    ink.forEach(function (st) { var l = JSON.stringify(st).length; if (size + l > 150000 && cur.length) { out.push(cur); cur = []; size = 0; } cur.push(st); size += l; });
    out.push(cur); return out;
  }
  function flush(id) {
    if (!cloud.ok) { setStatus('local'); return; }
    if (writing[id]) { dirty[id] = true; return; }
    writing[id] = true; dirty[id] = false;
    var d = sec(id), chunks = chunkInk(d.ink), prev = nChunks[id] || 0, col = cloud.col;
    var jobs = [col.doc('a_' + id).set({ a: d.a, t: d.t, nink: chunks.length })];
    chunks.forEach(function (c, i) { jobs.push(col.doc('i_' + id + '_' + i).set({ ink: c, t: d.t })); });
    for (var j = chunks.length; j < prev; j++) jobs.push(col.doc('i_' + id + '_' + j).delete());
    nChunks[id] = chunks.length;
    Promise.all(jobs).then(function () { writing[id] = false; if (dirty[id]) flush(id); else setStatus('cloud'); }, function (err) {
      writing[id] = false; cloud.ok = false; setStatus('local'); toast('Could not save to your account here. Marks are kept on this device.');
    });
  }
  function loadLocal() {
    $$('#main > section').forEach(function (s) { var raw = lsGet(LS + s.id); if (raw) { try { var d = JSON.parse(raw); if (d && d.a) S[s.id] = { a: d.a || [], ink: d.ink || [], t: d.t || 0 }; } catch (e) {} } });
  }
  var remote = {};   // id -> {a, t, nink, chunks:{}}
  function applyRemote(id) {
    var r = remote[id]; if (!r || !r.a) return;
    var have = r.nink || 0, ink = [], ok = true;
    for (var i = 0; i < have; i++) { if (!r.chunks[i]) { ok = false; break; } ink = ink.concat(r.chunks[i]); }
    if (!ok) return;
    nChunks[id] = have;
    var cur = S[id];
    if (cur && (cur.t || 0) > (r.t || 0)) { dirty[id] = true; schedule(id); return; }   // local is newer: push it up
    if (cur && JSON.stringify(cur.a) === JSON.stringify(r.a) && JSON.stringify(cur.ink) === JSON.stringify(ink)) return;
    S[id] = { a: r.a, ink: ink, t: r.t || 0 };
    lsSet(LS + id, JSON.stringify(S[id]));
    if (!(pop.open && pop.g && pop.g.secId === id)) renderHL(id);
    renderInk(id); renderPanel(); updateFab();
  }
  function onSnap(snap) {
    snap.docChanges().forEach(function (ch) {
      var did = ch.doc.id, data = ch.doc.data(); if (!data) return;
      var m;
      if ((m = /^a_(.+)$/.exec(did))) { var id = m[1]; remote[id] = remote[id] || { chunks: {} }; remote[id].a = data.a || []; remote[id].t = data.t; remote[id].nink = data.nink || 0; if (ch.type !== 'removed') applyRemote(id); }
      else if ((m = /^i_(.+)_(\d+)$/.exec(did))) { var id2 = m[1]; remote[id2] = remote[id2] || { chunks: {} }; remote[id2].chunks[+m[2]] = data.ink || []; if (remote[id2].a) applyRemote(id2); }
    });
    // push local-only sections up (first time on this account)
    Object.keys(S).forEach(function (id) { if ((S[id].a.length || S[id].ink.length) && !remote[id]) { dirty[id] = true; schedule(id); } });
    setStatus(cloud.ok ? 'cloud' : 'local');
  }
  async function initCloud() {
    try {
      if (!window.claude || !claude.use) throw 0;
      var db = await claude.use('db'), user = await claude.use('user');
      cloud.dl = await claude.use('downloads');
      if (!db || !user) throw 0;
      var id = await user.id(); if (!id) throw 0;
      cloud.col = db.collection('data/users/' + id); cloud.ok = true;
      var tm = setTimeout(function () { if (cloud.ok) setStatus('slow'); }, 10000);
      cloud.col.onSnapshot(function (s) { clearTimeout(tm); onSnap(s); }, function () { clearTimeout(tm); cloud.ok = false; setStatus('local'); });
    } catch (e) { cloud.ok = false; setStatus('local'); }
  }

  /* ---------- UI: floating buttons, bar, popover, panel ---------- */
  var fab = el('div', 'annfab');
  var bPen = el('button', null, '<span aria-hidden="true">✏️</span><span class="lbl">Draw</span>'); bPen.type = 'button'; bPen.title = 'Scribble on the notes'; bPen.setAttribute('aria-pressed', 'false');
  var bNotes = el('button', null, '<span aria-hidden="true">📝</span><span class="lbl">My marks</span><span class="cnt">0</span>'); bNotes.type = 'button'; bNotes.title = 'Your highlights and notes';
  fab.appendChild(bPen); fab.appendChild(bNotes); document.body.appendChild(fab);
  var saveEl = el('div', 'annsave', ''); saveEl.setAttribute('aria-live', 'polite'); document.body.appendChild(saveEl);
  function setStatus(s) {
    var t = { cloud: '☁ saved to your account', saving: 'saving…', local: '💾 saved on this device', slow: '☁ connecting…' }[s] || '';
    saveEl.textContent = t; saveEl.title = s === 'local' ? 'Your marks are stored in this browser only. Open the notes inside Claude to keep them on your account.' : '';
  }
  function countMarks() { var n = 0; Object.keys(S).forEach(function (k) { n += S[k].a.length; }); return n; }
  function updateFab() { $('.cnt', bNotes).textContent = countMarks(); }

  var bar = el('div', 'annbar'); bar.setAttribute('role', 'toolbar'); bar.setAttribute('aria-label', 'Drawing tools');
  document.body.appendChild(bar);
  function btn(label, title, fn, pressedKey) { var b = el('button', null, label); b.type = 'button'; b.title = title; b.addEventListener('click', fn); return b; }
  var barBtns = {};
  function buildBar() {
    bar.innerHTML = '';
    barBtns.pen = btn('✏️<span class="t"> Pen</span>', 'Pen', function () { tool = 'pen'; updateBar(); });
    barBtns.marker = btn('🖍<span class="t"> Marker</span>', 'Wide see-through marker', function () { tool = 'marker'; updateBar(); });
    barBtns.eraser = btn('🧽<span class="t"> Erase</span>', 'Touch a stroke to remove it', function () { tool = 'eraser'; updateBar(); });
    [barBtns.pen, barBtns.marker, barBtns.eraser].forEach(function (b) { bar.appendChild(b); });
    bar.appendChild(el('span', 'sep'));
    barBtns.sw = INKS.map(function (c) { var b = el('button', 'sw'); b.type = 'button'; b.title = c[1]; b.style.background = c[0] === 'auto' ? 'linear-gradient(135deg,#fff 50%,#111 50%)' : c[0]; b.addEventListener('click', function () { inkColor = c[0]; if (tool === 'eraser') tool = 'pen'; updateBar(); }); bar.appendChild(b); return b; });
    bar.appendChild(el('span', 'sep'));
    barBtns.w = [[2, 'S'], [4, 'M'], [8, 'L']].map(function (w) { var b = btn(w[1], 'Pen width', function () { inkW = w[0]; if (tool !== 'pen') tool = 'pen'; updateBar(); }); b.style.minWidth = '34px'; b.style.padding = '0 8px'; bar.appendChild(b); return b; });
    bar.appendChild(el('span', 'sep'));
    barBtns.undo = btn('↶<span class="t"> Undo</span>', 'Undo last stroke', doUndo); bar.appendChild(barBtns.undo);
    barBtns.scroll = btn('⇅<span class="t"> Scroll</span>', 'Pause drawing so you can scroll the page', function () { document.body.classList.toggle('annscroll'); updateBar(); }); bar.appendChild(barBtns.scroll);
    var clr = btn('🗑<span class="t"> Clear lecture</span>', 'Remove all ink in the lecture you are looking at', function () {
      if (clr.__t) { var s = centerSec(); if (s) { var d = sec(s.id); d.ink = d.ink.filter(function (st) { return st.v !== variant(); }); renderInk(s.id); touch(s.id, 'ink'); } clr.__t = 0; clr.innerHTML = '🗑<span class="t"> Clear lecture</span>'; updateBar(); }
      else { clr.__t = 1; clr.innerHTML = '<span>Sure?</span>'; setTimeout(function () { clr.__t = 0; clr.innerHTML = '🗑<span class="t"> Clear lecture</span>'; }, 2800); }
    });
    bar.appendChild(clr);
    var done = btn('✓<span class="t"> Done</span>', 'Finish drawing', function () { setDraw(false); }); done.style.fontWeight = '600'; bar.appendChild(done);
    barBtns.hint = el('div', 'hint'); bar.appendChild(barBtns.hint);
  }
  function updateBar() {
    barBtns.pen.setAttribute('aria-pressed', tool === 'pen'); barBtns.marker.setAttribute('aria-pressed', tool === 'marker'); barBtns.eraser.setAttribute('aria-pressed', tool === 'eraser');
    barBtns.sw.forEach(function (b, i) { b.setAttribute('aria-pressed', INKS[i][0] === inkColor && tool === 'pen'); });
    barBtns.w.forEach(function (b, i) { b.setAttribute('aria-pressed', [2, 4, 8][i] === inkW && tool === 'pen'); });
    barBtns.undo.disabled = !undo.length; barBtns.undo.style.opacity = undo.length ? 1 : .45;
    var sc = document.body.classList.contains('annscroll'); barBtns.scroll.setAttribute('aria-pressed', sc);
    document.body.classList.remove('tool-pen', 'tool-eraser'); document.body.classList.add(tool === 'eraser' ? 'tool-eraser' : 'tool-pen');
    var h = hiddenInk(); barBtns.hint.textContent = sc ? 'Scrolling. Tap “Scroll” again to draw.' : 'Ink stays put as long as the Textbook switch and screen size match when you drew.' + (h ? ' ' + h + ' stroke' + (h > 1 ? 's are' : ' is') + ' hidden: drawn in a different layout.' : '');
  }
  function setDraw(on) {
    if (on) { closePop(); closePanel(); $$('#main > section').forEach(function (s) { svgFor(s.id); renderInk(s.id); }); window.getSelection().removeAllRanges(); }
    document.body.classList.toggle('anndraw', on); document.body.classList.remove('annscroll'); bPen.setAttribute('aria-pressed', on); updateBar();
  }
  buildBar(); updateBar();
  bPen.addEventListener('click', function () { setDraw(true); });

  /* popover (colours + note) */
  var pop = { open: false, g: null, mode: 'sel', rng: null };
  var popEl = el('div', 'annpop'); popEl.setAttribute('role', 'dialog'); popEl.setAttribute('aria-label', 'Highlight options'); document.body.appendChild(popEl);
  function closePop() { popEl.classList.remove('open'); pop.open = false; pop.g = null; pop.rng = null; $$('.pulse').forEach(function (x) { x.classList.remove('pulse'); }); }
  function dots(cur, fn) {
    return Object.keys(COLORS).map(function (c) {
      var b = el('button', 'dot c-' + c); b.type = 'button'; b.title = COLORS[c]; b.setAttribute('aria-label', COLORS[c]); b.setAttribute('aria-pressed', c === cur);
      b.addEventListener('mousedown', function (e) { e.preventDefault(); }); b.addEventListener('click', function () { fn(c); }); return b;
    });
  }
  function placePop(rect) {
    popEl.style.left = '0px'; popEl.style.top = '0px'; popEl.classList.add('open');
    var w = popEl.offsetWidth, h = popEl.offsetHeight, coarse = window.matchMedia && matchMedia('(pointer:coarse)').matches;
    var x = Math.min(Math.max(8, rect.left + rect.width / 2 - w / 2), innerWidth - w - 8);
    var y = coarse ? rect.bottom + 46 : rect.top - h - 10; if (y < 8 || y + h > innerHeight - 8) y = coarse ? rect.top - h - 10 : rect.bottom + 10;
    y = Math.min(Math.max(8, y), innerHeight - h - 8);
    popEl.style.left = x + 'px'; popEl.style.top = y + 'px'; pop.open = true;
  }
  function showSel(range) {
    pop.mode = 'sel'; pop.rng = range; pop.g = null; popEl.innerHTML = '';
    dots(null, function (c) { makeGroup(range, c, false); }).forEach(function (d) { popEl.appendChild(d); });
    var nb = el('button', null, '📝 Note'); nb.type = 'button'; nb.addEventListener('mousedown', function (e) { e.preventDefault(); }); nb.addEventListener('click', function () { makeGroup(range, 'y', true); }); popEl.appendChild(nb);
    placePop(range.getBoundingClientRect());
  }
  function findGroup(gid) { for (var k in S) { for (var i = 0; i < S[k].a.length; i++) if (S[k].a[i].id === gid) { var g = S[k].a[i]; g.secId = k; return g; } } return null; }
  function showEdit(g, anchor) {
    pop.mode = 'edit'; pop.g = g; popEl.innerHTML = '';
    if (g.segs && g.segs.length) { var q = el('div', 'q'); q.textContent = '“' + g.segs.map(function (s) { return s.q; }).join(' … ').slice(0, 160) + '”'; popEl.appendChild(q); dots(g.c, function (c) { g.c = c; saveG(g); showEdit(g, anchor); }).forEach(function (d) { popEl.appendChild(d); }); }
    var gr = el('span', 'grow'); popEl.appendChild(gr);
    var del = el('button', 'danger', '🗑'); del.type = 'button'; del.title = 'Remove this mark'; del.addEventListener('click', function () { removeG(g); closePop(); }); popEl.appendChild(del);
    var ta = el('textarea'); ta.placeholder = 'Your note…'; ta.value = g.n || ''; ta.setAttribute('aria-label', 'Note'); popEl.appendChild(ta);
    var t; ta.addEventListener('input', function () { clearTimeout(t); t = setTimeout(function () { g.n = ta.value.trim(); saveG(g, true); }, 500); });
    ta.addEventListener('blur', function () { clearTimeout(t); if ((g.n || '') !== ta.value.trim()) { g.n = ta.value.trim(); saveG(g, true); } });
    var ok = el('button', null, '✓ Done'); ok.type = 'button'; ok.addEventListener('click', function () { ta.blur(); closePop(); }); popEl.appendChild(ok);
    placePop(anchor.getBoundingClientRect ? anchor.getBoundingClientRect() : anchor);
  }
  function saveG(g, norender) { var id = g.secId; var d = sec(id); if (!norender) renderHL(id); else { var s = document.getElementById(id); $$('.hasnote', s).forEach(function (x) { x.classList.remove('hasnote'); }); d.a.forEach(function (gg) { if (gg.n) { var sps = $$('.ah[data-g="' + gg.id + '"]', s); if (sps.length) sps[sps.length - 1].classList.add('hasnote'); } }); }
    touch(id, 'a'); renderPanel(); }
  function removeG(g) { var id = g.secId, d = sec(id); d.a = d.a.filter(function (x) { return x.id !== g.id; }); renderHL(id); touch(id, 'a'); renderPanel(); }
  function makeGroup(range, color, withNote) {
    var s = (range.startContainer.nodeType === 1 ? range.startContainer : range.startContainer.parentNode).closest('#main > section'); if (!s) return;
    var bl = blocksOf(s), segs = [];
    bl.forEach(function (b, i) {
      if (!range.intersectsNode(b)) return;
      var list = ownNodes(b), whole = document.createRange(); whole.selectNodeContents(b);
      var st = whole.comparePoint(range.startContainer, range.startOffset) === 0 ? pointOff(b, range.startContainer, range.startOffset, list) : 0;
      var en = whole.comparePoint(range.endContainer, range.endOffset) === 0 ? pointOff(b, range.endContainer, range.endOffset, list) : list.total;
      if (en <= st) return; var t = list.map(function (o) { return o.n.nodeValue; }).join('').slice(st, en);
      if (!t.trim()) return; segs.push({ b: i, s: st, e: en, q: t });
    });
    if (!segs.length) { closePop(); return; }
    var g = { id: uid(), c: color, n: '', t: Date.now(), segs: segs, secId: s.id };
    sec(s.id).a.push(g); window.getSelection().removeAllRanges(); renderHL(s.id); touch(s.id, 'a'); renderPanel();
    var first = $('.ah[data-g="' + g.id + '"]', s);
    if (withNote && first) showEdit(g, first); else closePop();
    updateFab();
  }
  var selT;
  function checkSel() {
    clearTimeout(selT);
    selT = setTimeout(function () {
      if (document.body.classList.contains('anndraw')) return;
      var sel = window.getSelection();
      if (!sel || sel.isCollapsed || !sel.rangeCount) { if (pop.mode === 'sel') closePop(); return; }
      var r = sel.getRangeAt(0), a = r.commonAncestorContainer; a = a.nodeType === 1 ? a : a.parentNode;
      if (!a.closest || !a.closest('#main > section')) { if (pop.mode === 'sel') closePop(); return; }
      if (a.closest('.annpop, .annpanel, .annbar')) return;
      if (!r.toString().trim()) return;
      showSel(r.cloneRange());
    }, 240);
  }
  document.addEventListener('selectionchange', checkSel);
  main.addEventListener('click', function (e) {
    var h = e.target.closest('.ah, .ahm'); if (!h) return;
    var sel = window.getSelection(); if (sel && !sel.isCollapsed) return;
    var g = findGroup(h.getAttribute('data-g')); if (g) { e.preventDefault(); showEdit(g, h); }
  });
  document.addEventListener('mousedown', function (e) { if (pop.open && !e.target.closest('.annpop') && !e.target.closest('.ah, .ahm')) { if (pop.mode === 'edit') closePop(); } });
  document.addEventListener('touchstart', function (e) { if (pop.open && pop.mode === 'edit' && !e.target.closest('.annpop') && !e.target.closest('.ah, .ahm')) closePop(); }, { passive: true });
  var scT; window.addEventListener('scroll', function () { if (pop.open && pop.mode === 'sel') { closePop(); } clearTimeout(scT); scT = setTimeout(checkSel, 180); }, { passive: true });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') { if (pop.open) closePop(); else if (document.body.classList.contains('anndraw')) setDraw(false); else closePanel(); }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && document.body.classList.contains('anndraw')) { e.preventDefault(); doUndo(); }
  });

  /* panel */
  var panel = el('aside', 'annpanel'); panel.setAttribute('aria-label', 'My highlights and notes'); document.body.appendChild(panel);
  var onlyNotes = false;
  function closePanel() { panel.classList.remove('open'); bNotes.setAttribute('aria-expanded', 'false'); }
  function openPanel() { closePop(); renderPanel(); panel.classList.add('open'); bNotes.setAttribute('aria-expanded', 'true'); }
  bNotes.addEventListener('click', function () { panel.classList.contains('open') ? closePanel() : openPanel(); });
  function secLabel(id) { var s = document.getElementById(id), h = s && s.querySelector('h2'); return h ? h.textContent.replace(/\s+/g, ' ').trim() : id; }
  function renderPanel() {
    var keep = $('.list', panel) ? $('.list', panel).scrollTop : 0;
    panel.innerHTML = '';
    var hd = el('header'); hd.innerHTML = '<h2>My highlights and notes</h2>';
    var ex = el('button', null, '⬇ Export'); ex.type = 'button'; ex.title = 'Save all your highlights and notes as a Markdown file'; ex.addEventListener('click', exportMd); hd.appendChild(ex);
    var cl = el('button', null, '✕'); cl.type = 'button'; cl.setAttribute('aria-label', 'Close'); cl.addEventListener('click', closePanel); hd.appendChild(cl); panel.appendChild(hd);
    var tl = el('div', 'tools'); var lab = el('label'); var cb = el('input'); cb.type = 'checkbox'; cb.checked = onlyNotes; cb.addEventListener('change', function () { onlyNotes = cb.checked; renderPanel(); });
    lab.appendChild(cb); lab.appendChild(document.createTextNode(' Only marks with a note')); tl.appendChild(lab);
    var add = el('button', null, '＋ Note on this lecture'); add.type = 'button'; add.title = 'Add a note about the whole lecture you are reading'; add.addEventListener('click', function () {
      var s = centerSec(); if (!s) return; var g = { id: uid(), c: 'y', n: '', t: Date.now(), segs: [], secId: s.id }; sec(s.id).a.push(g); touch(s.id, 'a'); renderPanel();
      showEdit(g, { left: innerWidth / 2 - 50, top: innerHeight / 3, width: 100, height: 10, bottom: innerHeight / 3 + 10 });
    }); tl.appendChild(add); panel.appendChild(tl);
    var list = el('div', 'list'), any = false;
    $$('#main > section').forEach(function (s) {
      var d = S[s.id]; var items = d ? d.a.filter(function (g) { return !onlyNotes || g.n; }) : [];
      var grp = el('div', 'grp'), h3 = el('h3'); var sp = el('span'); sp.textContent = secLabel(s.id); h3.appendChild(sp);
      grp.appendChild(h3);
      items.forEach(function (g) {
        any = true; var it = el('div', 'it'); it.setAttribute('role', 'button'); it.tabIndex = 0;
        var cbar = el('div', 'bar'); cbar.style.background = g.segs && g.segs.length ? 'var(--ah-' + g.c + ')' : 'var(--accent)'; cbar.style.border = '1px solid var(--line)'; it.appendChild(cbar);
        var bd = el('div', 'body'); var q = el('div', 'qt'); q.textContent = g.segs && g.segs.length ? g.segs.map(function (x) { return x.q; }).join(' … ').slice(0, 220) : '(note on the whole section)'; bd.appendChild(q);
        if (g.n) { var nt = el('div', 'nt'); nt.textContent = g.n; bd.appendChild(nt); } it.appendChild(bd);
        var x = el('button', 'x', '✕'); x.type = 'button'; x.setAttribute('aria-label', 'Remove'); x.addEventListener('click', function (ev) { ev.stopPropagation(); removeG(findGroup(g.id) || g); }); it.appendChild(x);
        function go() {
          var tgt = $('.ah[data-g="' + g.id + '"], .ahm[data-g="' + g.id + '"]', s) || s.querySelector('h2');
          if (window.innerWidth < 700) closePanel();
          var top = tgt.getBoundingClientRect().top + scrollY - 140; window.scrollTo({ top: top, behavior: 'smooth' });
          $$('.ah[data-g="' + g.id + '"]', s).forEach(function (n) { n.classList.add('pulse'); setTimeout(function () { n.classList.remove('pulse'); }, 1800); });
          if (!(g.segs && g.segs.length)) setTimeout(function () { showEdit(findGroup(g.id) || g, tgt); }, 500);
        }
        it.addEventListener('click', go); it.addEventListener('keydown', function (ev) { if (ev.key === 'Enter') go(); });
        grp.appendChild(it);
      });
      if (items.length) list.appendChild(grp);
    });
    if (!any) list.insertBefore(el('div', 'empty', onlyNotes ? 'No notes yet.' : 'Select any text in the notes to highlight it or add a note. Use ✏️ Draw to scribble.'), list.firstChild);
    panel.appendChild(list); list.scrollTop = keep;
  }
  function exportMd() {
    var out = '# My highlights and notes\n\n', n = 0;
    $$('#main > section').forEach(function (s) {
      var d = S[s.id]; if (!d || !d.a.length) return; out += '## ' + secLabel(s.id) + '\n\n';
      d.a.forEach(function (g) { n++; var q = g.segs && g.segs.length ? g.segs.map(function (x) { return x.q.replace(/\s+/g, ' ').trim(); }).join(' … ') : ''; if (q) out += '> ' + q + ' *(' + COLORS[g.c].toLowerCase() + ')*\n'; if (g.n) out += (q ? '\n' : '') + g.n + '\n'; out += '\n'; });
    });
    if (!n) { toast('Nothing to export yet.'); return; }
    if (cloud.dl) cloud.dl.save({ filename: 'CMPT423-my-notes.md', data: out }).then(function () { toast('Exported.'); }, function (e) { if (!e || e.code !== 'declined') copyOut(out); });
    else copyOut(out);
  }
  function copyOut(t) { try { navigator.clipboard.writeText(t).then(function () { toast('Copied to your clipboard as Markdown.'); }, function () { toast('Could not export here.'); }); } catch (e) { toast('Could not export here.'); } }

  /* ---------- boot ---------- */
  loadLocal();
  Object.keys(S).forEach(function (id) { renderHL(id); if (S[id].ink.length) renderInk(id); });
  updateFab(); setStatus('local');
  var lastV = variant();
  function relayout() { var v = variant(); if (v === lastV) return; lastV = v; Object.keys(inkSvg).forEach(renderInk); updateBar(); }
  window.addEventListener('resize', relayout); if (tbEl) tbEl.addEventListener('change', function () { setTimeout(function () { relayout(); Object.keys(inkSvg).forEach(sizeSvg); }, 50); });
  initCloud();
})();
