# -*- coding: utf-8 -*-
"""组会汇报 PPT(8 页):ML-1M 统一评测协议下的三线复现。基于 inner-chapter 设计模板。"""
import os
from pptx import Presentation
from pptx.util import Cm, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE

TPL = r"C:\research-training\repro-ncf\ncf_inner_template.pptx"
OUT = r"C:\research-training\check-2026-10\统一协议复现_组会汇报.pptx"
FIG = r"C:\research-training\check-2026-10\figures"

CN = "微软雅黑"
DARK = RGBColor(0x21, 0x24, 0x2B)
GRAY = RGBColor(0x6B, 0x72, 0x80)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
RED = RGBColor(0xB0, 0x3A, 0x2E)

prs = Presentation(TPL)
# 删掉模板自带的那页
while len(prs.slides):
    rId = prs.slides._sldIdLst[0].rId
    prs.part.drop_rel(rId)
    del prs.slides._sldIdLst[0]

LAY = {l.name: l for l in prs.slide_master.slide_layouts}
BODY_LAYOUT = LAY['content-centered-a']


def add_page(title, kicker=None):
    s = prs.slides.add_slide(BODY_LAYOUT)
    t = s.placeholders[0]
    t.text = title
    t.top = Cm(0.9); t.left = Cm(0.9); t.width = Cm(32.0); t.height = Cm(2.6)
    for p in t.text_frame.paragraphs:
        for r in p.runs:
            r.font.name = CN
            r.font.size = Pt(26)
            r.font.color.rgb = DARK
            r.font.bold = True
    body = s.placeholders[13]
    body.left = Cm(0.9); body.top = Cm(4.2); body.width = Cm(32.0); body.height = Cm(13.4)
    tf = body.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = MSO_ANCHOR.TOP
    return s, tf


def para(tf, text, size=15, bold=False, color=DARK, space=6, first=False, level=0, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.level = level
    p.alignment = align
    p.space_after = Pt(space)
    r = p.add_run()
    r.text = text
    r.font.name = CN
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return p


def clear_ph(s, idx):
    """模板里未使用的占位符留着会显示空框,删掉。"""
    try:
        sp = s.placeholders[idx]
        sp._element.getparent().remove(sp._element)
    except Exception:
        pass


# ---------------- P1 封面 ----------------
s = prs.slides.add_slide(BODY_LAYOUT)
t = s.placeholders[0]
t.text = "ML-1M 统一评测协议下的三线复现"
t.left = Cm(0.9); t.top = Cm(5.6); t.width = Cm(30.0); t.height = Cm(4.2)
for p in t.text_frame.paragraphs:
    for r in p.runs:
        r.font.name = CN; r.font.size = Pt(38); r.font.bold = True; r.font.color.rgb = DARK
ph = s.placeholders[13]
ph.left = Cm(0.9); ph.top = Cm(10.9); ph.width = Cm(30.0); ph.height = Cm(6.4)
tf = ph.text_frame
tf.word_wrap = True
tf.auto_size = MSO_AUTO_SIZE.NONE
tf.vertical_anchor = MSO_ANCHOR.TOP
para(tf, "NCF / LightGCN / SASRec —— 同一份切分、同一份候选集、同一套指标", 20, False, ACCENT, 14, first=True)
para(tf, "徐胜钦 · 2026-10-25 组会", 15, False, GRAY, 4)
para(tf, "", 8, False, GRAY, 4)
para(tf, "仓库：github.com/849982345-oss/ml1m-unified-benchmark", 13, False, GRAY, 4)

# ---------------- P2 结论 ----------------
s, tf = add_page("一句话结论")
para(tf, "序列建模  ≫  图传播  >  集合融合", 27, True, ACCENT, 6, first=True)
para(tf, "同一份数据、同一份 100 候选集、同一套指标下，三个家族的排序首次可以直接比较。", 14, False, GRAY, 14)
para(tf, "SASRec   0.8295 ± 0.0014      （论文 0.8235）", 15, True, DARK, 3)
para(tf, "LightGCN  0.7149   ·   NGCF  见下页", 15, True, DARK, 3)
para(tf, "NeuMF    0.6904   ·   BPR-MF 0.6889", 15, True, DARK, 3)
para(tf, "MLP      0.6750   ·   GMF    0.6396", 15, True, GRAY, 10)
para(tf, "序列比集合高 13.9 个百分点——不是调参能补的。", 14, True, RED, 5)
para(tf, "而 BPR-MF 与 NeuMF 打平：神经网络并没有赢。", 14, True, RED, 4)
body = s.placeholders[13]
body.width = Cm(15.8)
s.shapes.add_picture(os.path.join(FIG, 'bars_unified.png'), Cm(17.2), Cm(4.4), width=Cm(15.8))

# ---------------- P3 为什么先统一协议 ----------------
s, tf = add_page("为什么必须先统一协议")
para(tf, "8 月复现 NCF 时发现的一个坑", 16, True, ACCENT, 10, first=True)
para(tf, "NCF 报告的 HR@10，是在「1 正 + 99 负 = 100 个候选」里排出来的。", 15)
para(tf, "LightGCN 官方代码报告的 HR，是对全部物品排序。", 15, False, DARK, 14)
para(tf, "候选集越小，HR 天然越高 —— 同协议的 0.69 与全量排序的 0.30 说的是两个世界。", 15, True, RED, 14)
para(tf, "所以「各自都复现到论文数字」≠「它们可以互相比较」。要能比，必须钉死三件事：", 15, False, DARK, 10)
para(tf, "①  切分 —— leave-one-out，每用户时间戳最新一条为测试", 15, False, ACCENT, 5, level=1)
para(tf, "②  候选集 —— 100 个（1 正 + 99 负），三个模型复用同一份候选文件", 15, False, ACCENT, 5, level=1)
para(tf, "③  指标 —— HR@10 / NDCG@10，6,040 用户取均值", 15, False, ACCENT, 5, level=1)

# ---------------- P4 数据对齐 ----------------
s, tf = add_page("数据对齐：这张表可信的地基")
para(tf, "问题：两份数据的物品空间不一样", 16, True, ACCENT, 10, first=True)
para(tf, "SASRec 官方 ml-1m.txt    999,611 行 / 3,416 物品   ← 剔除了出现 <5 次的 290 个长尾", 14)
para(tf, "NCF 官方切分              1,000,209 行 / 3,706 物品", 14, False, DARK, 12)
para(tf, "物品空间不同 → 候选池不同 → 结果不可比。所以从 NCF 官方切分重建序列数据，保留全量物品。", 15, True, RED, 12)
para(tf, "校验（脚本实际输出）", 15, True, ACCENT, 6)
para(tf, "前5物品频次   [3428, 2991, 2990, 2883, 2672]   ← 与官方文件逐个相同", 13, False, GRAY, 3)
para(tf, "剔除 <5 频次物品后行数   999,611   ← 与官方文件精确相等", 13, False, GRAY, 3)
para(tf, "序列长度  min/max/median = 20 / 2314 / 96", 13, False, GRAY, 12)
para(tf, "「剔掉长尾后行数精确相等」= 同一份原始数据、同一套切分。", 15, True, ACCENT, 4)

# ---------------- P5 结果 ----------------
s, tf = add_page("结果：同一协议下的八个模型")
rows = [
    ("模型", "建模假设", "HR@10", "NDCG@10", "论文报告"),
    ("BPR-MF", "集合 · 纯点积（强基线）", "0.6889", "0.4130", "—"),
    ("GMF", "集合 · 点积", "0.6396", "0.3676", "图读数 ≈0.645"),
    ("MLP", "集合 · 非线性", "0.6750", "0.3978", "图读数 ≈0.640"),
    ("NeuMF", "集合 · 融合", "0.6904", "0.4086", "0.688 / 0.410"),
    ("NGCF", "图 · 传播 + W + σ", "0.6879", "0.4150", "该协议未报告"),
    ("LightGCN", "图 · 纯邻域传播", "0.7149", "0.4265", "该协议未报告"),
    ("SASRec", "序列 · 自注意力", "0.8295", "0.5976", "0.8235 / 0.5958"),
    ("BERT4Rec", "序列 · 双向（未达 SASRec）", "0.7785", "0.5669", "—"),
]
tb = s.shapes.add_table(len(rows), len(rows[0]), Cm(0.8), Cm(3.9), Cm(32.2), Cm(9.4)).table
widths = [Cm(4.6), Cm(6.6), Cm(4.6), Cm(4.8), Cm(11.6)]
for i, w in enumerate(widths):
    tb.columns[i].width = w
for ri, row in enumerate(rows):
    for ci, val in enumerate(row):
        cell = tb.cell(ri, ci)
        cell.text = val
        cell.margin_left = Cm(0.25); cell.margin_right = Cm(0.15)
        cell.margin_top = Cm(0.06); cell.margin_bottom = Cm(0.06)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if ci in (0, 1, 4) else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.name = CN
                r.font.size = Pt(13 if ri else 12.5)
                r.font.bold = (ri == 0) or (ri == 5 and ci in (2, 3))
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if ri == 0 else DARK
        if ri == 0:
            cell.fill.solid(); cell.fill.fore_color.rgb = ACCENT
        elif ri == 5:
            cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF0, 0xF8)
        else:
            cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
tf2 = s.placeholders[13]
tf2._element.getparent().remove(tf2._element)
para_note = s.shapes.add_textbox(Cm(0.8), Cm(14.4), Cm(32.2), Cm(2.4)).text_frame
para_note.word_wrap = True
para(para_note, "SASRec 为 3 个随机种子的均值 ± 标准差（seed 42 / 2024 / 7）；HR 极差 0.0033，远小于模型间差距 0.115 → 结论不是种子运气。", 12, False, GRAY, 3, first=True)
para(para_note, "训练曲线见 figures/curves_all.png；每个模型的逐 epoch 指标在仓库 logs/ 下，可逐行核对。", 12, False, GRAY, 3)

# ---------------- P6 两类局限 ----------------
s, tf = add_page("两类局限，必须分开讲")
para(tf, "复现局限（我的问题，不是方法的问题）", 16, True, ACCENT, 8, first=True)
para(tf, "1. LightGCN 只跑 300 epoch（220 已进平台期），未做超参网格搜索", 14, False, DARK, 4, level=1)
para(tf, "2. SASRec 用倒数第二条做验证集选 epoch，比 NCF 协议每用户少 1 条训练数据（0.6%）", 14, False, DARK, 4, level=1)
para(tf, "3. LightGCN 与 NCF 为单次运行；只有一个数据集，不能外推到稀疏场景", 14, False, DARK, 12, level=1)
para(tf, "方法缺点（它们的问题，不是我的问题）", 16, True, RED, 8)
para(tf, "SASRec —— 只用本人序列，用不到他人高阶协同；maxlen=200 硬截断", 14, False, DARK, 3, level=1)
para(tf, "绝对位置编码不辨真实时间间隔；冷启动失效", 14, False, DARK, 6, level=1)
para(tf, "LightGCN —— 只用 ID 交互，冷启动失效；静态图不增量更新", 14, False, DARK, 3, level=1)
para(tf, "过平滑限制在 2–4 层；均匀负采样大量假负例；推理需全量算分", 14, False, DARK, 6, level=1)
para(tf, "NCF —— 后人复盘（RecSys 2020）显示同等调参下简单点积显著优于 MLP，核心主张未被支持", 14, False, DARK, 4, level=1)

# ---------------- P7 协议自身的局限 ----------------
s, tf = add_page("协议本身的问题：主动交代")
para(tf, "100 候选的采样式评估，已被证明不保序", 17, True, RED, 10, first=True)
para(tf, "Krichene & Rendle, KDD 2020《On Sampled Metrics for Item Recommendation》：", 14, False, GRAY, 5)
para(tf, "采样指标不保持排序 —— A 在采样指标上优于 B，全量上未必；取期望也不成立；采样量越小，各指标越趋同。", 15, False, DARK, 12, level=1)
para(tf, "所以这张表的正确读法是：", 16, True, ACCENT, 8)
para(tf, "同协议横向比较   有效   ← 这是它的价值", 15, False, DARK, 5, level=1)
para(tf, "拿去和全量排序口径的论文数字对表   不可   ← 这是它的边界", 15, False, DARK, 12, level=1)
para(tf, "下一步补法：加一版全量排序评估（3,706 物品全打分）作交叉验证，两条口径都报。", 15, True, GRAY, 4)

# ---------------- P8 下一步 ----------------
s, tf = add_page("下一步")
para(tf, "P0   全量排序评估交叉验证 —— 确认「SASRec > LightGCN」在两种口径下都成立", 15, True, DARK, 8, first=True)
para(tf, "P0   换更稀疏的数据（Amazon-Beauty / Gowalla）复测 —— 检验「序列 > 图」是否只是 ML-1M 的中等密度效应", 15, True, DARK, 8)
para(tf, "P1   图 + 序列融合：用 LightGCN 预训练嵌入注入序列模型，看能否兼得两条信号", 15, True, DARK, 8)
para(tf, "P2   难负采样替掉均匀负采样，量化「假负例」的实际损失", 15, True, DARK, 16)
para(tf, "自己的判断：在 ML-1M 这种中等密度数据上，顺序信号比高阶协同更值钱；这个判断的边界在哪里，是我下一步真正想验证的问题。", 15, False, ACCENT, 6)

prs.save(OUT)
print("saved", OUT, os.path.getsize(OUT), "bytes,", len(prs.slides), "slides")
