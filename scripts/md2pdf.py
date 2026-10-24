# -*- coding: utf-8 -*-
"""极简 Markdown -> HTML -> PDF(Edge headless)。只为交付打印,不追求通用。"""
import io, os, re, subprocess, sys, html as H

CSS = """
@page { size: A4; margin: 16mm 16mm 14mm; }
body { font-family: "Microsoft YaHei","DengXian",sans-serif; font-size: 10.6pt; line-height: 1.75;
       color: #111; max-width: 100%; }
h1 { font-size: 17pt; margin: 22px 0 10px; border-bottom: 2px solid #111; padding-bottom: 5px; }
h2 { font-size: 13.5pt; margin: 20px 0 8px; border-left: 4px solid #2f6fb5; padding-left: 8px; }
h3 { font-size: 11.6pt; margin: 15px 0 6px; color: #222; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.6pt; }
th, td { border: 1px solid #999; padding: 5px 7px; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-weight: 700; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-family: Consolas,monospace; font-size: 9.4pt; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 9px 11px; overflow-x: auto; font-size: 9pt; line-height: 1.5; }
pre code { background: none; padding: 0; }
blockquote { margin: 10px 0; padding: 7px 12px; background: #f7f9fb; border-left: 3px solid #bbb; color: #333; }
ul, ol { margin: 7px 0 7px 20px; padding: 0; }
li { margin: 3px 0; }
hr { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
strong { font-weight: 700; }
img { max-width: 100%; }
"""


def inline(s):
    s = H.escape(s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', s)
    s = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img alt="\1" src="\2">', s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', s)
    return s


def convert(md_path, out_html):
    lines = io.open(md_path, encoding='utf-8').read().split('\n')
    out, i = [], 0
    in_code = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            if in_code:
                out.append('</code></pre>')
                in_code = False
            else:
                out.append('<pre><code>')
                in_code = True
            i += 1
            continue
        if in_code:
            out.append(H.escape(ln))
            i += 1
            continue

        # 表格:当前行含 | 且下一行是分隔行
        if '|' in ln and i + 1 < len(lines) and re.match(r'^\s*\|?[\s:|-]+\|[\s:|-]*$', lines[i + 1]):
            header = [c.strip() for c in ln.strip().strip('|').split('|')]
            out.append('<table><thead><tr>' + ''.join('<th>%s</th>' % inline(c) for c in header) + '</tr></thead><tbody>')
            i += 2
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                out.append('<tr>' + ''.join('<td>%s</td>' % inline(c) for c in cells) + '</tr>')
                i += 1
            out.append('</tbody></table>')
            continue

        m = re.match(r'^(#{1,4})\s+(.*)$', ln)
        if m:
            lv = len(m.group(1))
            out.append('<h%d>%s</h%d>' % (lv, inline(m.group(2)), lv))
            i += 1
            continue
        if re.match(r'^\s*---+\s*$', ln):
            out.append('<hr>')
            i += 1
            continue
        if ln.startswith('>'):
            buf = []
            while i < len(lines) and lines[i].startswith('>'):
                buf.append(inline(lines[i].lstrip('> ').rstrip()))
                i += 1
            out.append('<blockquote>%s</blockquote>' % '<br>'.join(buf))
            continue
        m = re.match(r'^\s*[-*]\s+(.*)$', ln)
        if m:
            out.append('<ul>')
            while i < len(lines) and re.match(r'^\s*[-*]\s+', lines[i]):
                out.append('<li>%s</li>' % inline(re.sub(r'^\s*[-*]\s+', '', lines[i])))
                i += 1
            out.append('</ul>')
            continue
        m = re.match(r'^\s*\d+\.\s+(.*)$', ln)
        if m:
            out.append('<ol>')
            while i < len(lines) and re.match(r'^\s*\d+\.\s+', lines[i]):
                out.append('<li>%s</li>' % inline(re.sub(r'^\s*\d+\.\s+', '', lines[i])))
                i += 1
            out.append('</ol>')
            continue
        if ln.strip():
            out.append('<p>%s</p>' % inline(ln))
        i += 1

    doc = ('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><style>%s</style></head><body>%s</body></html>'
           % (CSS, '\n'.join(out)))
    io.open(out_html, 'w', encoding='utf-8').write(doc)


if __name__ == '__main__':
    src, pdf = sys.argv[1], sys.argv[2]
    html_path = os.path.splitext(pdf)[0] + '.html'
    convert(src, html_path)
    if os.path.exists(pdf):
        os.remove(pdf)
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    subprocess.run([edge, '--headless', '--disable-gpu', '--no-pdf-header-footer',
                    '--virtual-time-budget=20000', '--print-to-pdf=' + pdf,
                    'file:///' + html_path.replace('\\', '/')],
                   capture_output=True, timeout=180)
    print('PDF' if os.path.exists(pdf) else 'FAILED', pdf, os.path.getsize(pdf) if os.path.exists(pdf) else '')
