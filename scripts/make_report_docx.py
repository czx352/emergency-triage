"""把 docs/report.md 转成 docs/报告.docx（A4、中文正文、图片与表格保留）。

只支持报告里实际用到的 Markdown 子集：标题、段落、无序列表、引用、代码块、
表格、图片、分隔线、**加粗**、`行内代码`。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG_SRC = ROOT / "artifacts" / "figures"
FIG_DST = DOCS / "figures"

BODY_FONT = "宋体"
HEAD_FONT = "微软雅黑"
CODE_FONT = "Consolas"


def set_font(run, name: str, size: float | None = None, bold: bool | None = None,
             color: RGBColor | None = None) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def add_inline(par, text: str, size: float = 10.5) -> None:
    """处理 **加粗** 与 `代码`。"""
    tokens = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text)
    for token in tokens:
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = par.add_run(token[2:-2])
            set_font(run, BODY_FONT, size, bold=True)
        elif token.startswith("`") and token.endswith("`"):
            run = par.add_run(token[1:-1])
            set_font(run, CODE_FONT, size - 1)
            run.font.color.rgb = RGBColor(0xB0, 0x3A, 0x2E)
        else:
            run = par.add_run(token)
            set_font(run, BODY_FONT, size)


def style_document(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(4)
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(1.0)
        section.top_margin = section.bottom_margin = Inches(0.9)


def main() -> None:
    FIG_DST.mkdir(parents=True, exist_ok=True)
    if FIG_SRC.exists():
        for png in FIG_SRC.glob("*.png"):
            shutil.copy2(png, FIG_DST / png.name)

    lines = (DOCS / "report.md").read_text(encoding="utf-8").splitlines()
    doc = Document()
    style_document(doc)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # 代码块
        if line.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            par = doc.add_paragraph()
            par.paragraph_format.line_spacing = 1.1
            par.paragraph_format.left_indent = Inches(0.25)
            run = par.add_run("\n".join(block))
            set_font(run, CODE_FONT, 9)
            run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
            i += 1
            continue

        # 表格
        if line.startswith("|") and i + 1 < len(lines) and set(lines[i + 1]) <= set("|-: "):
            header = [c.strip() for c in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            table = doc.add_table(rows=1 + len(rows), cols=len(header))
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for c, text in enumerate(header):
                cell = table.rows[0].cells[c]
                cell.text = ""
                run = cell.paragraphs[0].add_run(text)
                set_font(run, HEAD_FONT, 9.5, bold=True)
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r, row in enumerate(rows, start=1):
                for c in range(len(header)):
                    text = row[c] if c < len(row) else ""
                    cell = table.rows[r].cells[c]
                    cell.text = ""
                    add_inline(cell.paragraphs[0], text, size=9)
            doc.add_paragraph()
            continue

        # 图片
        m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if m:
            caption, rel = m.group(1), m.group(2)
            path = (DOCS / rel).resolve()
            if path.exists():
                doc.add_picture(str(path), width=Inches(6.1))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = cap.add_run(caption)
                set_font(run, BODY_FONT, 9, color=RGBColor(0x66, 0x66, 0x66))
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if level == 1:
                par = doc.add_paragraph()
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = par.add_run(text)
                set_font(run, HEAD_FONT, 20, bold=True)
            elif level == 2:
                par = doc.add_heading(level=1)
                run = par.add_run(text)
                set_font(run, HEAD_FONT, 15, bold=True, color=RGBColor(0x12, 0x32, 0x4F))
            elif level == 3:
                par = doc.add_heading(level=2)
                run = par.add_run(text)
                set_font(run, HEAD_FONT, 12.5, bold=True)
            else:
                par = doc.add_paragraph()
                run = par.add_run(text)
                set_font(run, HEAD_FONT, 11, bold=True)
            i += 1
            continue

        # 分隔线
        if line.strip() == "---":
            i += 1
            continue

        # 引用
        if line.startswith(">"):
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Inches(0.3)
            add_inline(par, line.lstrip("> ").strip(), size=9.5)
            for run in par.runs:
                run.font.color.rgb = RGBColor(0x60, 0x6A, 0x75)
            i += 1
            continue

        # 列表
        if re.match(r"^\s*[-*]\s+", line):
            text = re.sub(r"^\s*[-*]\s+", "", line)
            par = doc.add_paragraph(style="List Bullet")
            add_inline(par, text)
            i += 1
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            text = re.sub(r"^\s*\d+\.\s+", "", line)
            par = doc.add_paragraph(style="List Number")
            add_inline(par, text)
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        par = doc.add_paragraph()
        add_inline(par, line)
        i += 1

    out = DOCS / "报告.docx"
    doc.save(str(out))
    print(f"已生成 {out}")


if __name__ == "__main__":
    main()
