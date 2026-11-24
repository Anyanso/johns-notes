# -*- coding: utf-8 -*-
# Small, dependency-free SVG plots for the "Read the graph" questions.
# Colours use the page's CSS variables so the plots follow light/dark mode.
import math

W, H = 440, 270
L, R, T, B = 52, 14, 14, 42          # margins
COLS = ["var(--accent)", "var(--ok)", "#6b5bd2"]

def _nice_ticks(lo, hi, n=5):
    span = hi - lo
    raw = span / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks

def _cover(top):
    t = _nice_ticks(0, top)
    if len(t) > 1 and t[-1] < top:
        t.append(round(t[-1] + (t[1] - t[0]), 10))
    return t

def _fmt(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return ('%.2f' % v).rstrip('0').rstrip('.')

def _frame(xlo, xhi, ylo, yhi, xlabel, ylabel, xticks=None, yticks=None):
    sx = lambda x: L + (x - xlo) / (xhi - xlo) * (W - L - R)
    sy = lambda y: H - B - (y - ylo) / (yhi - ylo) * (H - T - B)
    g = []
    g.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="var(--muted)" stroke-width="1"/>' % (L, H - B, W - R, H - B))
    g.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="var(--muted)" stroke-width="1"/>' % (L, T, L, H - B))
    for t in (xticks if xticks is not None else _nice_ticks(xlo, xhi)):
        x = sx(t)
        g.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="var(--muted)"/>' % (x, H - B, x, H - B + 4))
        g.append('<text x="%.1f" y="%d" text-anchor="middle" font-size="11" fill="var(--muted)">%s</text>' % (x, H - B + 16, _fmt(t)))
    for t in (yticks if yticks is not None else _nice_ticks(ylo, yhi)):
        y = sy(t)
        g.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="var(--line)"/>' % (L, y, W - R, y))
        g.append('<text x="%d" y="%.1f" text-anchor="end" font-size="11" fill="var(--muted)">%s</text>' % (L - 6, y + 4, _fmt(t)))
    g.append('<text x="%.1f" y="%d" text-anchor="middle" font-size="12" fill="var(--fg)">%s</text>' % ((L + W - R) / 2, H - 6, xlabel))
    g.append('<text x="14" y="%.1f" text-anchor="middle" font-size="12" fill="var(--fg)" transform="rotate(-90 14 %.1f)">%s</text>' % ((T + H - B) / 2, (T + H - B) / 2, ylabel))
    return sx, sy, g

def _wrap(g, title):
    return ('<figure class="plot"><svg viewBox="0 0 %d %d" role="img" aria-label="%s" xmlns="http://www.w3.org/2000/svg">' % (W, H, title)
            + ''.join(g) + '</svg><figcaption>%s</figcaption></figure>' % title)

def scatter(xs, ys, xlabel, ylabel, title, xlim, ylim, groups=None, xticks=None, yticks=None, r=3.2):
    sx, sy, g = _frame(xlim[0], xlim[1], ylim[0], ylim[1], xlabel, ylabel, xticks, yticks)
    for i, (x, y) in enumerate(zip(xs, ys)):
        c = COLS[groups[i]] if groups else COLS[0]
        g.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" fill-opacity="0.7"/>' % (sx(x), sy(y), r, c))
    return _wrap(g, title)

def scatter_classes(pts, labels, xlabel, ylabel, title, xlim, ylim, names, legend='tr'):
    sx, sy, g = _frame(xlim[0], xlim[1], ylim[0], ylim[1], xlabel, ylabel)
    for (x, y), c in zip(pts, labels):
        X, Y = sx(x), sy(y)
        if c == 0:
            g.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s" fill-opacity="0.75"/>' % (X, Y, COLS[0]))
        elif c == 1:
            g.append('<rect x="%.1f" y="%.1f" width="7" height="7" fill="%s" fill-opacity="0.75"/>' % (X - 3.5, Y - 3.5, COLS[1]))
        else:
            g.append('<path d="M%.1f %.1f l4 7 h-8 z" fill="%s" fill-opacity="0.8"/>' % (X, Y - 4.5, COLS[2]))
    # legend
    lx, ly = (W - R - 110, T + 6) if legend == 'tr' else (W - R - 110, H - B - 16 * len(names) - 2)
    for k, nm in enumerate(names):
        y = ly + 16 * k
        if k == 0:
            g.append('<circle cx="%d" cy="%d" r="4" fill="%s"/>' % (lx, y, COLS[0]))
        elif k == 1:
            g.append('<rect x="%d" y="%d" width="7" height="7" fill="%s"/>' % (lx - 3, y - 3, COLS[1]))
        else:
            g.append('<path d="M%d %d l4 7 h-8 z" fill="%s"/>' % (lx, y - 4, COLS[2]))
        g.append('<text x="%d" y="%d" font-size="11" fill="var(--fg)">%s</text>' % (lx + 10, y + 4, nm))
    return _wrap(g, title)

def histogram(values, edges, xlabel, title, ylabel="count", xticks=None):
    counts = [0] * (len(edges) - 1)
    for v in values:
        for k in range(len(edges) - 1):
            if edges[k] <= v < edges[k + 1] or (k == len(edges) - 2 and v == edges[-1]):
                counts[k] += 1
                break
    yt = _cover(max(counts))
    sx, sy, g = _frame(edges[0], edges[-1], 0, yt[-1], xlabel, ylabel, xticks, yt)
    for k, c in enumerate(counts):
        x0, x1 = sx(edges[k]), sx(edges[k + 1])
        g.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="var(--accent)" fill-opacity="0.55" stroke="var(--surface)"/>' % (x0, sy(c), x1 - x0, sy(0) - sy(c)))
    return _wrap(g, title)

def bars(cats, counts, xlabel, title, ylabel="count"):
    n = len(cats)
    yt = _cover(max(counts))
    sx, sy, g = _frame(-0.5, n - 0.5, 0, yt[-1], xlabel, ylabel, xticks=[], yticks=yt)
    w = (W - L - R) / n * 0.7
    for k, (c, v) in enumerate(zip(cats, counts)):
        x = sx(k)
        g.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="var(--accent)" fill-opacity="0.55"/>' % (x - w / 2, sy(v), w, sy(0) - sy(v)))
        g.append('<text x="%.1f" y="%d" text-anchor="middle" font-size="11" fill="var(--muted)">%s</text>' % (x, H - B + 16, c))
    return _wrap(g, title)
