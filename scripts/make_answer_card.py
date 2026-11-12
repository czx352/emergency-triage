"""生成一页 A4 的答辩速答卡（docs/答辩速答卡.docx）。

数字与代码路径均与报告一致；排版目标是一页装下，字号因此偏小（9.5pt）。
用法：py scripts/make_answer_card.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "答辩速答卡.docx"

NAVY = RGBColor(0x12, 0x32, 0x4F)
ACCENT = RGBColor(0xC0, 0x50, 0x4D)
GREY = RGBColor(0x55, 0x60, 0x6B)

QUESTIONS = [
    (
        "Q1  数据集来源是什么？",
        "训练用自建的合成数据；真实数据只用来验证管线，不参与训练。",
        [
            "40,000 条合成样本，按公开文献的 ESI 分布生成（1–5 级占 2% / 9% / 39% / 35% / 15%），"
            "注入 12% 标签噪声与随机缺失，避免指标虚高",
            "MIMIC-IV-ED Demo（PhysioNet 开放获取，207 条 / 56 患者）跑通完整 ETL 与推理；"
            "因其 ESI 5 为 0 条、ESI 4 仅 2 条，无法训练",
            "全量需 CITI 培训 + credentialed 审批，周期内未获批；报告中如实写为方法验证，不外推临床性能",
        ],
    ),
    (
        "Q2  双塔融合和序数回归是什么？算深度学习吗？",
        "两种模态各用一个神经网络编码再融合；把“分级有序”写进输出层。都是课程内容。",
        [
            "结构化塔：9 维生命体征 → 标准化 → FC(128) → FC(64)；"
            "文本塔：主诉 → 词嵌入 128 维 → 掩码均值池化",
            "门控融合 g = σ(W[z_t; z_v])，z = g⊙z_t + (1−g)⊙z_v，权重由数据学习，非人工加权",
            "CORAL：输出 4 个累积 logit P(y>k)，sigmoid 后差分成 5 类概率，天然单调；损失为 4 个 BCE 之和",
            "对照保留 HGB 与 TF-IDF+LR 基线，能说明深度模型赢在哪里",
        ],
    ),
    (
        "Q3  主要代码是哪些？",
        "模型两个文件是核心，其余按“数据—训练—评估—服务”分层。",
        [
            "模型：models/fusion.py（双塔+门控+输出头）、models/text.py、models/losses.py（CORAL / 代价敏感）",
            "数据：data/features.py（标准化、缺失处理、按 subject_id 划分）",
            "训练：train.py（单一入口，消融只换命令行开关）；评估：metrics.py（QWK / 危重降级 / Bootstrap CI）",
            "服务：serve/app.py（FastAPI）、serve/demo.py（Gradio 演示页）",
        ],
    ),
    (
        "Q4  怎么训练的模型？",
        "按患者划分、AdamW 优化、CORAL 损失、以验证集 QWK 早停。",
        [
            "划分 70 / 15 / 15，按 subject_id，同一患者不跨集，避免泄漏",
            "AdamW，lr 2e-4，weight decay 0.01，batch 64，最多 8 个 epoch，梯度裁剪 5.0",
            "每轮算验证集 QWK，patience=2 早停并保存最优权重；停滞时学习率减半",
            "测试集 n=6,035：QWK 0.9547，macro-F1 0.9220，危重降级率 3.84%（结构化基线 11.37%）",
        ],
    ),
]

FOLLOWUPS = [
    ("“简单拼接 QWK 0.9575，比你的 0.9547 还高？”",
     "是，但危重降级率 4.58% 高于本模型 3.84%。主指标是“分得安全”而非“分得准”，这一点已如实写入报告。"),
    ("“合成数据训练出来的结论能信吗？”",
     "不能外推到临床。它证明的是模型设计与评估口径有效；换真实数据重训，代码与流程不用改。"),
]

NUMBERS = [("QWK", "0.9547"), ("macro-F1", "0.9220"), ("危重降级率", "3.84%"),
           ("测试集", "6,035"), ("单条延迟", "48.7 ms")]


def styled(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal.font.size = Pt(9.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.22


def para(doc, text="", size=9.5, bold=False, color=None, space_before=0, space_after=0,
         align=None, indent=0.0, font="宋体"):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.22
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if color is not None:
        run.font.color.rgb = color
    return p


def rule(doc) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    pPr = p._p.get_or_add_pPr()
    border = pPr.makeelement(qn("w:pBdr"), {})
    bottom = border.makeelement(qn("w:bottom"), {
        qn("w:val"): "single", qn("w:sz"): "6",
        qn("w:space"): "1", qn("w:color"): "C0504D",
    })
    border.append(bottom)
    pPr.append(border)


def main() -> None:
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.3)
    section.left_margin = section.right_margin = Cm(1.5)
    styled(doc)

    para(doc, "答辩速答卡", size=19, bold=True, color=NAVY, font="微软雅黑")
    para(doc, "基于双塔融合与序数回归的急诊分诊分级模型 ｜ 每问先给一句话结论，再补证据",
         size=8.5, color=GREY)
    rule(doc)

    for title, one_line, points in QUESTIONS:
        para(doc, title, size=11, bold=True, color=NAVY, space_before=9, font="微软雅黑")
        para(doc, one_line, size=9.5, bold=True, color=ACCENT, space_before=1)
        for point in points:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.45)
            p.paragraph_format.space_before = Pt(0.5)
            p.paragraph_format.line_spacing = 1.22
            run = p.add_run("• " + point)
            run.font.size = Pt(9.5)
            run.font.name = "宋体"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    rule(doc)

    para(doc, "必背的两个追问", size=11, bold=True, color=NAVY, space_before=8, font="微软雅黑")
    for question, answer in FOLLOWUPS:
        para(doc, question, size=9, bold=True, color=ACCENT, space_before=1)
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.45)
        p.paragraph_format.line_spacing = 1.22
        run = p.add_run("→ " + answer)
        run.font.size = Pt(9.5)
        run.font.name = "宋体"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    table = doc.add_table(rows=2, cols=len(NUMBERS))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = True
    for i, (label, value) in enumerate(NUMBERS):
        head = table.cell(0, i).paragraphs[0]
        head.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = head.add_run(label)
        run.font.size = Pt(8)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.name = "微软雅黑"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        cell = table.cell(0, i)
        shd = cell._tc.get_or_add_tcPr().makeelement(qn("w:shd"), {
            qn("w:val"): "clear", qn("w:fill"): "12324F"})
        cell._tc.get_or_add_tcPr().append(shd)

        body = table.cell(1, i).paragraphs[0]
        body.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = body.add_run(value)
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = NAVY
        run.font.name = "微软雅黑"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    para(doc, "提示：数字均来自 artifacts/ 与 report.md，若重跑实验请同步更新本页。",
         size=8, color=GREY, space_before=6)

    doc.save(OUT)
    print(f"已生成 {OUT}")


if __name__ == "__main__":
    main()
