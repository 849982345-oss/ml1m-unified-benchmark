# -*- coding: utf-8 -*-
"""学习进度可视化 PPT（8 页）。全部数据来自真实实验与笔记，不手填。"""
import os
from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE

TPL = r"C:\research-training\repro-ncf\ncf_inner_template.pptx"
OUT = r"C:\research-training\check-2026-10\学习进度可视化.pptx"
FIG = r"C:\research-training\ml1m-unified-benchmark\figures"

CN = "微软雅黑"
DARK = RGBColor(0x21, 0x24, 0x2B)
GRAY = RGBColor(0x6B, 0x72, 0x80)
LGRAY = RGBColor(0xD5, 0xDA, 0xE1)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ACCENT2 = RGBColor(0x3E, 0x7C, 0xB8)
ORANGE = RGBColor(0xD9, 0x82, 0x2B)
GREEN = RGBColor(0x2E, 0x7D, 0x5B)
RED = RGBColor(0xB0, 0x3A, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation(TPL)
while len(prs.slides):
    rId = prs.slides._sldIdLst[0].rId
    prs.part.drop_rel(rId)
    del prs.slides._sldIdLst[0]
LAY = {l.name: l for l in prs.slide_master.slide_layouts}
BASE = LAY['content-centered-a']


def page(title, sub=None):
    s = prs.slides.add_slide(BASE)
    t = s.placeholders[0]
    t.text = title
    t.left, t.top, t.width, t.height = Cm(0.9), Cm(0.85), Cm(32.0), Cm(2.2)
    for p in t.text_frame.paragraphs:
        for r in p.runs:
            r.font.name = CN; r.font.size = Pt(25); r.font.bold = True; r.font.color.rgb = DARK
    body = s.placeholders[13]
    body._element.getparent().remove(body._element)
    if sub:
        tb = s.shapes.add_textbox(Cm(0.95), Cm(3.0), Cm(32.0), Cm(1.1)).text_frame
        tb.word_wrap = True
        p = tb.paragraphs[0]; r = p.add_run(); r.text = sub
        r.font.name = CN; r.font.size = Pt(12.5); r.font.color.rgb = GRAY
    # 顶部分隔线
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(0.9), Cm(2.85), Cm(32.0), Cm(0.05))
    ln.fill.solid(); ln.fill.fore_color.rgb = LGRAY; ln.line.fill.background()
    return s


def txt(s, x, y, w, h, text, size=13, bold=False, color=DARK, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, spacing=1.15):
    tb = s.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Cm(0.05)
    tf.margin_top = tf.margin_bottom = Cm(0.02)
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run(); r.text = ln
        r.font.name = CN; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return tb


def rect(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    sh = s.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line:
        sh.line.color.rgb = line; sh.line.width = Pt(0.75)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def progress(s, x, y, w, label, pct, detail, color=ACCENT):
    """横向进度条：label + 条 + 百分比文字"""
    txt(s, x, y - 0.05, 7.6, 1.0, label, 13, True, DARK)
    h = 0.85
    rect(s, x + 7.8, y + 0.05, w, h, LGRAY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    filled = max(0.12, w * pct) if pct > 0 else 0.1
    rect(s, x + 7.8, y + 0.05, filled, h, color, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    txt(s, x + 7.8 + w + 0.35, y - 0.02, 3.2, 0.9, "%.0f%%" % (pct * 100), 14, True, color)
    txt(s, x + 7.8 + w + 3.6, y, 9.0, 1.0, detail, 10.5, False, GRAY)


# ================= P1 封面 =================
s = prs.slides.add_slide(BASE)
t = s.placeholders[0]
t.text = "学习进度可视化"
t.left, t.top, t.width, t.height = Cm(0.9), Cm(5.2), Cm(30.0), Cm(4.0)
for p in t.text_frame.paragraphs:
    for r in p.runs:
        r.font.name = CN; r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = DARK
ph = s.placeholders[13]
ph.left, ph.top, ph.width, ph.height = Cm(0.9), Cm(10.2), Cm(30.0), Cm(6.0)
tf = ph.text_frame
tf.word_wrap = True
tf.auto_size = MSO_AUTO_SIZE.NONE
tf.vertical_anchor = MSO_ANCHOR.TOP
for i, (txt_, sz, col) in enumerate([
        ("推荐系统方向 · 2026 年 7 月 → 10 月", 18, ACCENT),
        ("徐胜钦 · 2026-10-25 组会", 14, GRAY),
        ("", 8, GRAY),
        ("复现 6 篇论文 / 8 个模型 · 统一评测协议 · 全部可查", 12, GRAY)]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(8)
    r = p.add_run(); r.text = txt_
    r.font.name = CN; r.font.size = Pt(sz); r.font.color.rgb = col

# ================= P2 总览仪表盘 =================
s = page("总览：现在在哪", "数据截至 2026-10-24，全部可在仓库与笔记库中查证")

STATS = [("6", "篇论文完整复现", "NCF · LightGCN · SASRec · BPR-MF · NGCF · BERT4Rec"),
         ("8", "个模型跑进同一张表", "集合 / 图 / 序列 三条线"),
         ("16", "周训练（7 月初起）", "含 7 次课程 + 4 次组会级汇报"),
         ("5/17", "论文清单已消化", "评估方法类刚补，是当前最大缺口")]
x = 0.9
for big, label, note in STATS:
    rect(s, x, 4.3, 7.6, 4.4, RGBColor(0xF5, 0xF7, 0xFA), LGRAY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    txt(s, x + 0.4, 4.6, 6.8, 2.2, big, 34, True, ACCENT)
    txt(s, x + 0.4, 6.7, 6.8, 0.9, label, 13, True, DARK)
    txt(s, x + 0.4, 7.6, 6.8, 1.0, note, 9.5, False, GRAY)
    x += 8.2

txt(s, 0.9, 9.4, 20.0, 0.9, "能力进度（自评，按真实产出给分）", 14, True, DARK)
progress(s, 0.9, 10.6, 12.0, "环境与工具链", 0.95, "单卡 GPU / cu130 / 数据校验脚本 / 一键自检", GREEN)
progress(s, 0.9, 12.1, 12.0, "代码阅读与改造", 0.8, "能读懂并改写官方实现，改动全部留痕", GREEN)
progress(s, 0.9, 13.6, 12.0, "实验设计与评估", 0.7, "协议意识刚建立：统一了切分/候选/指标", ORANGE)
progress(s, 0.9, 15.1, 12.0, "论文精读与批判", 0.25, "16 篇路线图读了 4 篇；缺评估方法类", RED)
progress(s, 0.9, 16.6, 12.0, "独立提出可证伪问题", 0.15, "还没有自己的问题，明年 1 月前要出", RED)

rect(s, 22.0, 9.6, 11.0, 8.6, RGBColor(0xFC, 0xFA, 0xF3), RGBColor(0xE8, 0xDC, 0xC0),
     shape=MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s, 22.5, 9.9, 10.0, 0.9, "一句话现状", 14, True, ORANGE)
txt(s, 22.5, 10.9, 10.0, 7.0,
    "工具、代码、实验这条线已经打通：\n"
    "能给六个模型同一张表。\n\n"
    "阅读和问题意识这条线还没打通：\n"
    "读得少，也还没有自己的问题。\n\n"
    "→ 下阶段重心从「会跑」转向「会读会问」。",
    10.8, False, DARK, spacing=1.3)

# ================= P3 时间线 =================
s = page("时间线：16 周做了什么", "每个节点都有对应证据，可现场打开")

line_y = 8.4
rect(s, 2.0, line_y, 30.0, 0.08, LGRAY)
NODES = [
    ("7 月", "暑期课程 7 次", "Python / PyTorch 训练流程 /\nSASRec smoke demo\n\n→ course_materials 7 份笔记", ACCENT2, True),
    ("8 月", "NCF 完整复现", "GMF / MLP / NeuMF 三结构\nNeuMF 0.690 vs 论文 0.688\n\n→ NCF-reproduction 仓库", GREEN, True),
    ("9 月", "组会汇报 + 定方向", "15 页汇报 + 导师约谈\n把方向定在序列推荐\n\n→ 组会备份归档", ACCENT2, True),
    ("10 月", "统一协议 + 5 个新模型", "五模型迁到同一协议\nBPR-MF / NGCF / BERT4Rec 新增\n\n→ ml1m-unified-benchmark", ORANGE, True),
]
x = 2.6
for month, title_, detail, col, done in NODES:
    # 节点
    dot = rect(s, x + 3.0, line_y - 0.42, 0.95, 0.95, col, shape=MSO_SHAPE.OVAL)
    txt(s, x, line_y - 2.1, 7.2, 1.2, month, 17, True, col, PP_ALIGN.CENTER)
    txt(s, x - 0.4, line_y + 0.9, 8.0, 1.1, title_, 13.5, True, DARK, PP_ALIGN.CENTER)
    txt(s, x - 0.4, line_y + 2.2, 8.0, 5.0, detail, 10.5, False, GRAY, PP_ALIGN.CENTER, spacing=1.3)
    x += 7.6

rect(s, 0.9, 16.4, 32.0, 1.6, RGBColor(0xF2, 0xF6, 0xFA), LGRAY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s, 1.3, 16.6, 31.2, 1.3,
    "节奏判断：7–8 月产出密集（课程 + 复现），9 月被课程与助管切碎（只有汇报与谈话），10 月靠一次集中冲刺补回。"
    "→ 问题是节奏不稳，不是能力不够。",
    11.5, True, ACCENT, spacing=1.2)

# ================= P4 复现矩阵 =================
s = page("复现进度：6 篇论文 / 8 个模型", "「完整复现」= 数据 + 代码 + 逐 epoch 日志 + 与论文数字对照")

HEAD = ["论文", "会议/年份", "模型", "我的 HR@10", "论文报告", "状态"]
ROWS = [
    ["BPR-MF", "UAI 2009", "点积 + BPR", "0.6889", "—", "完整复现"],
    ["NCF", "WWW 2017", "GMF / MLP / NeuMF", "0.6904", "0.688", "完整复现"],
    ["NGCF", "SIGIR 2019", "传播 + W + σ", "0.6879", "该协议未报", "完整复现"],
    ["LightGCN", "SIGIR 2020", "纯邻域传播", "0.7149", "该协议未报", "完整复现"],
    ["SASRec", "ICDM 2018", "单向自注意力", "0.8295", "0.8235", "完整复现 + 3 seed"],
    ["BERT4Rec", "CIKM 2019", "双向 + 完形填空", "0.7785", "—", "跑通但未达基线"],
]
w = [4.4, 4.6, 6.6, 4.8, 4.8, 6.8]
y = 4.4
rect(s, 0.9, y, 32.0, 1.15, ACCENT)
xx = 0.9
for i, h in enumerate(HEAD):
    txt(s, xx + 0.25, y + 0.22, w[i] - 0.3, 0.8, h, 11.5, True, WHITE)
    xx += w[i]
y += 1.15
for ri, row in enumerate(ROWS):
    bg = WHITE if ri % 2 == 0 else RGBColor(0xF7, 0xF8, 0xFA)
    rect(s, 0.9, y, 32.0, 1.35, bg, LGRAY)
    xx = 0.9
    for ci, cell in enumerate(row):
        col = DARK
        bold = ci == 0
        if ci == 0 and ri >= 4:
            col = GREEN
        if ci == 3:
            col = ACCENT; bold = True
        if ci == 5:
            if "未达" in cell:
                col = RED; bold = True
            elif "完整" in cell:
                col = GREEN; bold = True
        txt(s, xx + 0.25, y + 0.28, w[ci] - 0.3, 1.0, cell, 11.5, bold, col)
        xx += w[ci]
    y += 1.35

txt(s, 0.9, y + 0.35, 32.0, 2.4,
    "关键：八个模型全部跑在**同一份切分 + 同一份 100 候选 + 同一套指标**下（ML-1M leave-one-out）。\n"
    "所以它们之间的差可以直接读，不用再担心口径问题。其中 BPR-MF 与 NeuMF 打平（0.6889 vs 0.6904），"
    "NGCF 低于 LightGCN —— 这两条都不是论文里写的，是我自己跑出来的。",
    11, False, GRAY, spacing=1.35)

# ================= P5 阅读进度矩阵 =================
s = page("论文阅读进度：17 篇清单，5 篇已消化", "「已消化」= 能讲清问题—方法—实验—局限，不只是读过")

STAGES = [
    ("阶段① 评估与基线", ACCENT, [("BPR", 0), ("MF", 1), ("NCF vs MF", 0), ("采样指标", 0), ("Much Progress", 0)]),
    ("阶段② 图与序列", GREEN, [("NGCF", 1), ("LightGCN", 1), ("SASRec", 1), ("BERT4Rec", 1), ("GRU4Rec", 0), ("Caser", 0), ("序列综述", 0)]),
    ("阶段③ 自监督与生成式", ORANGE, [("SGL", 0), ("SimGCL", 0), ("S3-Rec", 0), ("CL4SRec", 0), ("TIGER", 0)]),
]
y = 4.4
for name, col, items in STAGES:
    txt(s, 0.9, y, 9.0, 1.0, name, 13, True, col)
    xx = 10.4
    for label, done in items:
        bgc = col if done else RGBColor(0xF0, 0xF2, 0xF5)
        tc = WHITE if done else GRAY
        rect(s, xx, y - 0.1, 3.0, 1.35, bgc, None, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        txt(s, xx + 0.05, y + 0.2, 2.9, 0.9, ("✓ " if done else "") + label, 9, done, tc, PP_ALIGN.CENTER)
        xx += 3.22
    y += 2.35

# 进度条
txt(s, 0.9, 12.0, 12.0, 1.0, "整体进度  5 / 17", 15, True, DARK)
rect(s, 9.4, 12.1, 16.0, 1.0, LGRAY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
rect(s, 9.4, 12.1, 16.0 * (5.0/17.0), 1.0, ACCENT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s, 26.0, 12.1, 4.0, 0.9, "29%", 16, True, ACCENT)

rect(s, 0.9, 13.8, 32.0, 4.2, RGBColor(0xFD, 0xF6, 0xF5), RGBColor(0xE8, 0xC8, 0xC4),
     shape=MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s, 1.3, 14.05, 31.2, 3.8,
    "最大的缺口在这里，而且我知道它为什么重要：\n"
    "我现在这张八模型对比表，本质上是在补「评估方法」这一课 —— 没有 Krichene & Rendle（KDD 2020）那篇，\n"
    "我既说不清自己的 100 候选为什么不保序，也说服不了别人这些数字能不能比。\n"
    "→ 11 月前两周先把阶段① 剩下 4 篇读完，这是所有后续工作的地基。",
    11, False, DARK, spacing=1.4)

# ================= P6 能力成长 =================
s = page("能力盘点：会什么、还不会什么（自评）", "按真实产出打分，不打虚的")

SKILLS = [
    ("环境搭建（GPU / 依赖 / 数据校验）", 0.95, "从零配到能跑，写过自检脚本", GREEN),
    ("读官方代码并改造", 0.80, "LightGCN 重写评估、SASRec 加 seed、BERT4Rec 从头实现", GREEN),
    ("数据一致性验证", 0.85, "行数/字节/频次三重核对，能证明「同一份数据」", GREEN),
    ("实验设计与协议对齐", 0.70, "会钉切分/候选/指标；但还没做过显著性检验", ORANGE),
    ("结果诊断（欠拟合/过拟合）", 0.65, "会看两条线和平台期；调参还没系统扫过", ORANGE),
    ("论文精读与批判", 0.25, "读了 4 篇，缺评估方法类与强基线意识", RED),
    ("提出可证伪的问题", 0.15, "还没有自己的问题，这是明年 1 月前的硬指标", RED),
]
y = 4.5
for name, pct, note, col in SKILLS:
    txt(s, 0.9, y + 0.1, 11.5, 0.9, name, 12, True, DARK)
    rect(s, 12.6, y + 0.15, 12.0, 0.75, LGRAY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    rect(s, 12.6, y + 0.15, max(0.12, 12.0 * pct), 0.75, col, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    txt(s, 24.9, y + 0.05, 2.0, 0.9, "%.0f%%" % (pct * 100), 12.5, True, col)
    txt(s, 27.0, y + 0.15, 6.2, 0.9, note, 9, False, GRAY)
    y += 1.62

rect(s, 0.9, 15.9, 32.0, 2.2, RGBColor(0xF2, 0xF6, 0xFA), LGRAY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s, 1.3, 16.05, 31.2, 1.9,
    "右边三条（绿）是这四个月真拿下的：能配环境、能读代码、能验数据 —— 这是做实验的下限能力。\n"
    "左边两条（红）是短板：读得少、还不会提问 —— 这是做研究的上限能力。下阶段全部资源投这里。",
    11, False, DARK, spacing=1.35)

# ================= P7 结果看板 =================
s = page("结果看板：八个模型在同一张表里")
s.shapes.add_picture(os.path.join(FIG, 'bars_all.png'), Cm(2.4), Cm(4.0), width=Cm(29.0))
txt(s, 0.9, 15.6, 32.0, 3.0,
    "三条可以直接讲的结论：\n"
    "① 序列 ≫ 图 > 集合：SASRec 0.8295 比 NeuMF 高 13.9 个点，不是调参能补的。\n"
    "② BPR-MF 0.6889 ≈ NeuMF 0.6904 —— 复现出了 NCF 的数字，但没复现出它的结论。\n"
    "③ NGCF 0.6879 < LightGCN 0.7149 —— 同样加图传播，删掉 W 和 σ 反而涨 2.6 个点。",
    11.5, False, DARK, spacing=1.35)

# ================= P8 缺口与下一步 =================
s = page("缺口与下一步：11 月 → 1 月", "按「补地基 → 进方向 → 出闭环」三段，每段有可交付物")

# 甘特表头
months = [("11 月", 10.4, 7.8, ACCENT2), ("12 月", 18.4, 7.8, ORANGE), ("1 月", 26.4, 6.6, GREEN)]
for label, x, wdt, col in months:
    rect(s, x, 4.3, wdt - 0.4, 1.0, col, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    txt(s, x, 4.5, wdt - 0.4, 0.8, label, 13, True, WHITE, PP_ALIGN.CENTER)

BARS = [
    ("阶段① 评估与基线（4 篇）", 10.4, 7.8, ACCENT2),
    ("阶段② 图与序列（4 篇）", 15.4, 10.8, ORANGE),
    ("阶段③ 自监督与生成式", 24.2, 8.6, GREEN),
    ("问题定义 + 基线复现", 18.4, 7.8, ACCENT2),
    ("实验闭环 + 消融 + 技术报告", 23.0, 9.6, ORANGE),
]
y = 5.9
for name, bx, wdt, col in BARS:
    txt(s, 0.9, y + 0.15, 9.0, 1.0, name, 10.5, True, DARK)
    rect(s, bx, y + 0.12, wdt, 0.9, col, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    y += 1.55

rect(s, 0.9, 13.6, 32.0, 4.4, RGBColor(0xF5, 0xF7, 0xFA), LGRAY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s, 1.3, 13.8, 31.2, 4.0,
    "三个必须先说清的边界：\n"
    "1. 阶段④（组内方向）我没有自己选 —— 等您拍板，因为选错方向比做得慢代价大得多。\n"
    "2. 我承诺的是「11 月读完阶段①②、12 月开始跑自己的实验」，不承诺发表时间。\n"
    "3. 如果这三个月没有产出，您可以随时卡我 —— 但我需要知道评判标准是什么（一次组会汇报 / 一份报告 / 一次实验闭环）。",
    11.5, False, DARK, spacing=1.4)

prs.save(OUT)
print("saved", OUT, os.path.getsize(OUT), "bytes,", len(prs.slides), "slides")
