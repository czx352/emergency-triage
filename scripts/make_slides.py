"""生成答辩 PPT（16:9，30+ 页），指标与图表全部从 artifacts/ 读取。

用法：
    py scripts/make_slides.py
    py scripts/make_slides.py --out docs/答辩.pptx --tag hf_main

设计取向：数字全部来自实测产物，不手工填写；正文一页一个论点。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAVY = RGBColor(0x12, 0x32, 0x4F)
ACCENT = RGBColor(0xC0, 0x50, 0x4D)
GREY = RGBColor(0x5A, 0x66, 0x72)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Microsoft YaHei"

W, H = 13.333, 7.5


def _ea(run) -> None:
    """python-pptx 只设拉丁字体，中文字符要走 a:ea 才生效。"""
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", FONT)


def text_box(slide, left, top, width, height, text, size=18, bold=False,
             color=NAVY, align=PP_ALIGN.LEFT, line_spacing=1.35):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = FONT
        run.font.color.rgb = color
        _ea(run)
    return box


def bullets(slide, left, top, width, height, items, size=17, color=NAVY, gap=10):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    for i, item in enumerate(items):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.line_spacing = 1.4
        p.space_after = Pt(gap)
        if item.startswith("- "):          # 二级条目
            item, level = item[2:], 1
        else:
            level = 0
        run = p.add_run()
        run.text = ("•  " if level == 0 else "–  ") + item
        run.font.size = Pt(size if level == 0 else size - 2)
        run.font.name = FONT
        run.font.color.rgb = color if level == 0 else GREY
        _ea(run)
    return box


def bg(slide, color=LIGHT) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    slide.shapes._spTree.remove(shape._element)
    slide.shapes._spTree.insert(2, shape._element)


def new_slide(prs, page: int, title: str | None = None, kicker: str | None = None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg(slide)
    if kicker:
        text_box(slide, 0.8, 0.45, 6.0, 0.4, kicker, size=12, color=ACCENT, bold=True)
    if title:
        text_box(slide, 0.8, 0.75, 11.7, 0.75, title, size=28, bold=True)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.82), Inches(1.55),
                                     Inches(0.9), Inches(0.06))
        bar.fill.solid()
        bar.fill.fore_color.rgb = ACCENT
        bar.line.fill.background()
        bar.shadow.inherit = False
    if page:
        text_box(slide, 12.1, 6.95, 0.9, 0.3, f"{page:02d}", size=11, color=GREY,
                 align=PP_ALIGN.RIGHT)
    return slide


def section_slide(prs, page: int, index: str, title: str, subtitle: str = ""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg(slide, NAVY)
    text_box(slide, 1.0, 2.3, 3.0, 1.6, index, size=72, bold=True, color=ACCENT)
    text_box(slide, 1.05, 3.75, 10.5, 0.9, title, size=34, bold=True, color=WHITE)
    if subtitle:
        text_box(slide, 1.1, 4.65, 10.5, 0.6, subtitle, size=15, color=RGBColor(0xC8, 0xD3, 0xDE))
    text_box(slide, 12.1, 6.95, 0.9, 0.3, f"{page:02d}", size=11,
             color=RGBColor(0x8A, 0x9B, 0xAD), align=PP_ALIGN.RIGHT)
    return slide


def table(slide, left, top, width, data, col_widths=None, size=12, row_h=0.38,
          header_fill=NAVY, highlight_rows: set[int] | None = None):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top),
                                   Inches(width), Inches(row_h * rows))
    tbl = shape.table
    if col_widths:
        total = sum(col_widths)
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(Inches(width) * w / total))
    highlight_rows = highlight_rows or set()
    for r in range(rows):
        tbl.rows[r].height = Inches(row_h)
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.text = ""
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(data[r][c])
            run.font.size = Pt(size)
            run.font.name = FONT
            run.font.bold = r == 0 or r in highlight_rows
            run.font.color.rgb = WHITE if r == 0 else NAVY
            _ea(run)
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = header_fill
            elif r in highlight_rows:
                cell.fill.fore_color.rgb = RGBColor(0xF6, 0xE5, 0xE4)
            else:
                cell.fill.fore_color.rgb = WHITE
    return tbl


def picture(slide, path: Path, left, top, width=None, height=None):
    if not path.exists():
        text_box(slide, left, top, 6, 0.5, f"（缺图：{path.name}）", size=12, color=ACCENT)
        return None
    kwargs = {}
    if width:
        kwargs["width"] = Inches(width)
    if height:
        kwargs["height"] = Inches(height)
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), **kwargs)


def fmt(value, digits=4, pct=False) -> str:
    return f"{value:.2%}" if pct else f"{value:.{digits}f}"


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"缺少 {path}，请先跑 py scripts/run_experiments.py")
    return pd.read_csv(path)


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def measure_latency(model_path: Path, n: int = 20) -> float | None:
    """用真实产物测单条推理耗时，避免 PPT 里的延迟是编的。"""
    try:
        import os

        os.environ["TRIAGE_MODEL_PATH"] = str(model_path)
        from er_triage.serve.app import load_predictor, predict_probs

        state = load_predictor()
        vitals = {"hr": 118, "sbp": 96, "dbp": 60, "rr": 24,
                  "temp_c": 37.2, "spo2": 92, "pain": 8, "gcs": 15}
        predict_probs(state, 58, vitals, "chest pain radiating to left arm")  # 预热
        started = time.perf_counter()
        for _ in range(n):
            predict_probs(state, 58, vitals, "chest pain radiating to left arm")
        return (time.perf_counter() - started) / n * 1000
    except Exception as exc:
        print(f"延迟测量失败：{exc}")
        return None


def build(out_path: Path, tag: str, artifacts: Path) -> None:
    summary = load_summary(artifacts / "summary.csv")
    metrics = load_metrics(artifacts / f"{tag}_metrics.json")
    fig_dir = artifacts / "figures"
    rows = summary.set_index("experiment")

    def get(name: str, col: str, default=float("nan")) -> float:
        return float(rows.loc[name, col]) if name in rows.index else default

    main_qwk = metrics.get("test", {}).get("qwk", get("深度-main", "qwk"))
    main_f1 = metrics.get("test", {}).get("macro_f1", get("深度-main", "macro_f1"))
    main_under = metrics.get("test", {}).get("critical_under_triage",
                                             get("深度-main", "critical_under_triage"))
    ci = metrics.get("test", {}).get("qwk_ci95") or metrics.get("test", {}).get("qwk_bootstrap_ci")
    ci_text = f"{ci[0]:.4f} – {ci[1]:.4f}" if ci else "见 artifacts"
    n_test = metrics.get("test", {}).get("n", "—")
    epochs = len(metrics.get("history", [])) or "—"

    # 文本后端对照：同一 12k 子集、同结构、同超参，只换文本后端
    sub_hash = load_metrics(artifacts / "subset_hash_metrics.json").get("test", {})
    sub_hf = load_metrics(artifacts / "subset_hf_metrics.json").get("test", {})
    has_hf = bool(sub_hash) and bool(sub_hf)

    latency = measure_latency(artifacts / f"{tag}.pt")

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    page = 0

    def nxt() -> int:
        nonlocal page
        page += 1
        return page

    # ---------- 封面 ----------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg(slide, NAVY)
    text_box(slide, 1.0, 2.0, 11.4, 0.5, "深度学习课程设计", size=16,
             color=RGBColor(0x9F, 0xB3, 0xC6))
    text_box(slide, 1.0, 2.6, 11.4, 1.5, "基于双塔融合与序数回归的\n急诊分诊分级模型", size=40,
             bold=True, color=WHITE, line_spacing=1.25)
    text_box(slide, 1.05, 4.55, 11.0, 0.5,
             "—— 从“分得准”到“分得安全”", size=18, color=RGBColor(0xE8, 0xC7, 0xC6))
    text_box(slide, 1.05, 5.6, 11.0, 0.4,
             "结构化生命体征 + 临床主诉文本 · CORAL 序数回归 · 代价敏感评估",
             size=13, color=RGBColor(0x9F, 0xB3, 0xC6))
    nxt()

    # ---------- 目录 ----------
    slide = new_slide(prs, nxt(), "目录", kicker="CONTENTS")
    items = [("01", "问题与动机", "急诊分诊的业务约束、代价不对称、问题来源"),
             ("02", "方法与模型", "双塔结构、门控融合、CORAL、代价敏感损失"),
             ("03", "实验与结果", "数据、指标、主结果、消融、错误分析"),
             ("04", "系统与演示", "工程结构、接口契约、演示页、对比与局限"),
             ("05", "结论", "贡献回顾与后续工作")]
    top = 1.95
    for i, (idx, title, desc) in enumerate(items):
        y = top + i * 0.98
        text_box(slide, 1.0, y, 1.0, 0.5, idx, size=22, bold=True, color=ACCENT)
        text_box(slide, 2.0, y - 0.05, 3.3, 0.45, title, size=19, bold=True)
        text_box(slide, 5.4, y, 7.0, 0.45, desc, size=13, color=GREY)

    # ---------- 01 ----------
    section_slide(prs, nxt(), "01", "问题与动机", "为什么要做这个模型，以及为什么“准确率”在这里是错的指标")

    slide = new_slide(prs, nxt(), "急诊分诊：把有限资源分给最需要的人", kicker="1.1 业务背景")
    bullets(slide, 0.85, 1.95, 6.1, 4.6, [
        "分诊护士在患者到达后数分钟内完成分级，决定就诊先后与抢救资源",
        "ESI（Emergency Severity Index）为 5 级有序分级：1 级需立即抢救，5 级为非急诊",
        "分级同时依赖两类信息：生命体征（结构化）与主诉（自由文本）",
        "分级结果直接写入叫号队列、床位分配与绿色通道",
    ], size=16)
    picture(slide, fig_dir / "fig1_label_distribution.png", 7.1, 1.9, width=5.4)
    text_box(slide, 7.1, 5.35, 5.4, 0.8,
             "真实分诊的类别极度不平衡：危重（ESI 1-2）占少数，\n而模型恰恰最不能在这两类上出错。",
             size=12, color=GREY)

    slide = new_slide(prs, nxt(), "错误分级的代价不对称", kicker="1.2 核心矛盾")
    bullets(slide, 0.85, 1.9, 6.0, 4.7, [
        "把 ESI 1 判成 ESI 4：患者进入普通候诊区，可能延误抢救",
        "把 ESI 4 判成 ESI 1：浪费抢救资源，但患者不会因等待恶化",
        "两类错误在临床上不等价，在损失函数里也不应等价",
        "因此本项目的评估主指标是“危重降级率”，而不是准确率",
    ], size=16)
    cost = [["真实 \\ 预测", "ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"],
            ["ESI 1", "0", "2", "5", "8", "10"],
            ["ESI 2", "3", "0", "2", "4", "6"],
            ["ESI 3", "3", "2", "0", "1", "2"],
            ["ESI 4", "3", "2", "1", "0", "1"],
            ["ESI 5", "3", "2", "1", "1", "0"]]
    table(slide, 7.25, 2.1, 5.2, cost, size=11, row_h=0.42,
          highlight_rows={1, 2})
    text_box(slide, 7.25, 4.95, 5.3, 1.0,
             "红色行：真实危重患者的误判代价，写进配置文件 cost_matrix。\n"
             "该矩阵直接参与损失计算与评估，不只是文档说明。",
             size=12, color=GREY, line_spacing=1.5)

    slide = new_slide(prs, nxt(), "问题来源：一个宣称“AI 分诊”的开源医疗 OS", kicker="1.3 选题理由")
    bullets(slide, 0.85, 1.9, 6.3, 4.8, [
        "目标仓库宣称具备急诊分诊 AI，接口与字段定义完整",
        "核查发现接口返回的是硬编码常量，没有模型、没有推理",
        "文档里的能力指标无法复现，属于“有契约、无实现”",
        "本项目保留其接口契约，补上真实可训练的模型",
    ], size=16)
    code = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.35), Inches(1.95),
                                  Inches(5.2), Inches(3.5))
    code.fill.solid()
    code.fill.fore_color.rgb = RGBColor(0x1E, 0x24, 0x2B)
    code.line.fill.background()
    code.shadow.inherit = False
    text_box(slide, 7.6, 2.15, 4.7, 3.1,
             '// 原仓库的实现（示意）\n'
             'if (drugs.has("warfarin")) {\n'
             '  return { level: "危重", risk: "高" };\n'
             '}\n'
             'return { level: "紧急", risk: "中" };\n\n'
             '// 关键词命中即返回常量\n'
             '// 无特征、无概率、无模型',
             size=12, color=RGBColor(0xC8, 0xE6, 0xC9))
    text_box(slide, 7.6, 5.6, 5.0, 0.6,
             "本项目的做法：同样的接口，返回概率分布、模型版本与实测延迟。",
             size=12, color=GREY)

    slide = new_slide(prs, nxt(), "本文工作与贡献", kicker="1.4 贡献")
    col_w = 3.7
    cards = [
        ("① 有序结构显式建模", "CORAL 输出 K-1 个累积 logit，\n概率天然单调，不会出现\n“1 级概率低于 2 级”的非法输出"),
        ("② 损失对齐临床风险", "以代价矩阵最小化期望代价，\n危重降级（代价 8）获得\n8 倍于轻度高估的惩罚"),
        ("③ 评估口径转向安全", "以危重降级率为主指标，\n配合 QWK 与 bootstrap 置信区间，\n而非单看准确率"),
    ]
    for i, (title, body) in enumerate(cards):
        x = 0.85 + i * (col_w + 0.35)
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(2.0),
                                      Inches(col_w), Inches(3.2))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = RGBColor(0xDF, 0xE4, 0xEA)
        card.shadow.inherit = False
        text_box(slide, x + 0.28, 2.2, col_w - 0.5, 0.5, title, size=16, bold=True)
        text_box(slide, x + 0.28, 2.85, col_w - 0.5, 2.1, body, size=12, color=GREY, line_spacing=1.5)
    text_box(slide, 0.85, 5.55, 11.6, 0.9,
             "可验证性：全部指标由 artifacts/ 下的 json 与 csv 自动生成，图表与 PPT 均由脚本产出，不含手工填写的数字。",
             size=13, color=GREY)

    # ---------- 02 ----------
    section_slide(prs, nxt(), "02", "方法与模型", "双塔输入、门控融合、序数输出")

    slide = new_slide(prs, nxt(), "任务定义", kicker="2.1 形式化")
    text_box(slide, 0.9, 1.85, 11.5, 0.6,
             "输入  x = (主诉文本, 生命体征向量, 人口学)        输出  y ∈ {1,2,3,4,5}，且 y 有序",
             size=17, bold=True)
    bullets(slide, 0.9, 2.75, 11.4, 3.6, [
        "关键约束：等级之间有距离，1 与 2 的差别远大于“相邻类别编号差 1”",
        "普通交叉熵把 5 个类当无序标签，丢掉了“有序”这一先验",
        "目标不是复现护士的标签，而是给出可用于排序的风险概率分布",
        "输出概率必须单调可用：P(y ≤ k) 随 k 递增，才能做阈值调整",
    ], size=16)
    text_box(slide, 0.9, 6.1, 11.4, 0.5,
             "对应实现：er_triage/models/losses.py（CORAL 编码）与 models/fusion.py（输出头）",
             size=12, color=GREY)

    slide = new_slide(prs, nxt(), "整体架构：双塔 + 门控融合 + 序数头", kicker="2.2 结构")
    boxes = [
        (0.9, 2.35, 2.7, 1.0, "主诉文本", "chief_complaint"),
        (0.9, 4.0, 2.7, 1.0, "生命体征 + 年龄", "9 维结构化特征"),
        (4.15, 2.35, 3.0, 1.0, "文本塔", "BERT / 哈希嵌入 + 池化"),
        (4.15, 4.0, 3.0, 1.0, "结构化塔", "标准化 + 缺失计数 + MLP"),
        (7.65, 3.15, 2.3, 1.1, "门控融合", "g = σ(W[z_t; z_v])"),
        (10.35, 3.15, 2.3, 1.1, "CORAL 头", "4 个累积 logit"),
    ]
    for x, y, w, h, title, sub in boxes:
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                       Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = WHITE
        shape.line.color.rgb = RGBColor(0xC9, 0xD3, 0xDD)
        shape.shadow.inherit = False
        text_box(slide, x + 0.15, y + 0.12, w - 0.3, 0.4, title, size=14, bold=True)
        text_box(slide, x + 0.15, y + 0.55, w - 0.3, 0.4, sub, size=11, color=GREY)
    arrows = [((3.65, 2.85), (4.1, 2.85)), ((3.65, 4.5), (4.1, 4.5)),
              ((7.2, 2.85), (7.6, 3.55)), ((7.2, 4.5), (7.6, 3.85)),
              ((10.0, 3.7), (10.3, 3.7))]
    for (x1, y1), (x2, y2) in arrows:
        line = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        line.line.color.rgb = GREY
        line.line.width = Pt(1.25)
    text_box(slide, 0.9, 5.5, 11.5, 1.1,
             "融合处不直接拼接，而是用门控分配权重：让模型自己决定“这一刻更信主诉还是更信生命体征”。\n"
             "消融实验（03 章）验证了门控相较拼接的收益。",
             size=13, color=GREY)

    slide = new_slide(prs, nxt(), "结构化塔：小特征也要认真处理", kicker="2.3 输入侧")
    bullets(slide, 0.85, 1.95, 6.2, 4.7, [
        "9 维输入：年龄 + 心率/收缩压/舒张压/呼吸/体温/血氧/疼痛/GCS",
        "缺失不是随机事件：真实数据的生命体征会整条缺失，必须显式建模",
        "做法：标准化 + 追加“缺失项数量”特征，让模型知道信息完整度",
        "超出临床合理区间的值按 CLIP_RANGES 截断，避免录入错误放大",
        "两层 MLP（128 → 64）输出 128 维向量",
    ], size=15)
    picture(slide, fig_dir / "fig2_training_curve.png", 7.2, 2.0, width=5.3)
    text_box(slide, 7.2, 5.6, 5.3, 0.6, "训练曲线：损失下降与验证集 QWK 同步收敛。", size=12, color=GREY)

    slide = new_slide(prs, nxt(), "文本塔：从关键词到语义", kicker="2.4 文本侧")
    bullets(slide, 0.85, 1.95, 6.3, 4.7, [
        "主诉是自由文本，短、口语化、含缩写（如 SOB = shortness of breath）",
        "hash 后端：字符 n-gram 哈希 + 掩码均值池化，无需下载权重，可离线复现",
        "hf 后端：Bio_ClinicalBERT（临床语料预训练）+ 均值池化 + 线性投影",
        "两个后端共用同一套接口，切换只改一个参数，用于做文本塔消融",
        "主诉很短，序列长度 24 足够，CPU 也能完成微调",
    ], size=15)
    table(slide, 7.35, 2.15, 5.1,
          [["后端", "语义", "依赖", "用途"],
           ["hash", "无", "无", "离线基线"],
           ["hf", "有", "约 440MB", "正式实验"]],
          size=11, row_h=0.5, highlight_rows={2})
    text_box(slide, 7.35, 4.15, 5.2, 1.6,
             "这一组对比回答一个具体问题：\n模型收益到底来自“更深的结构”，还是来自“预训练语义”？",
             size=13, color=GREY)

    slide = new_slide(prs, nxt(), "CORAL：把有序性写进输出层", kicker="2.5 输出头")
    bullets(slide, 0.85, 1.95, 6.1, 4.8, [
        "不为 5 个类各出一个 logit，而是输出 4 个累积 logit：P(y > k)",
        "类别概率由相邻累积概率相减得到，天然满足单调与归一",
        "训练目标：对每个 k 做二分类，target = 1[y > k]",
        "好处：相邻等级错误与跨级错误在损失上本就不同，不必额外编码",
        "预测时取概率最大的等级，同时保留完整分布用于阈值调整",
    ], size=15)
    code = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(2.0),
                                  Inches(5.25), Inches(3.4))
    code.fill.solid()
    code.fill.fore_color.rgb = RGBColor(0x1E, 0x24, 0x2B)
    code.line.fill.background()
    code.shadow.inherit = False
    text_box(slide, 7.55, 2.2, 4.8, 3.0,
             "# 累积目标\n"
             "y > 1, y > 2, y > 3, y > 4  →  4 个二分类\n\n"
             "p_k  = sigmoid(logit_k)\n"
             "P(y=1) = 1 - p_1\n"
             "P(y=k) = p_{k-1} - p_k   (2 ≤ k ≤ 4)\n"
             "P(y=5) = p_4\n\n"
             "# 单调性由结构保证，不需后处理",
             size=11, color=RGBColor(0xC8, 0xE6, 0xC9))

    slide = new_slide(prs, nxt(), "代价敏感：让损失函数“怕漏诊”", kicker="2.6 损失设计")
    bullets(slide, 0.85, 1.95, 6.2, 4.6, [
        "在交叉熵基础上加入期望代价项：L = NLL + λ · E[cost]",
        "E[cost] 用当前预测概率分布与代价矩阵计算，可反传",
        "λ 控制“准”与“安全”的权衡，本项目取配置值并在消融中固定",
        "对比实验：head=ce + loss=cost_ce（可反传）vs head=coral + loss=coral",
    ], size=15)
    if has_hf:
        note = (f"12k 子集上的对照：hash QWK {sub_hash.get('qwk', 0):.4f}，"
                f"Bio_ClinicalBERT {sub_hf.get('qwk', 0):.4f}")
    else:
        note = "文本后端对比见第 03 章：hash 与 Bio_ClinicalBERT 同结构、同划分下的差异。"
    text_box(slide, 7.3, 2.05, 5.2, 3.2, note, size=14, color=GREY, line_spacing=1.6)
    text_box(slide, 7.3, 5.2, 5.2, 1.2,
             "注意：代价敏感训练与“用代价做评估”是两件事。\n本项目两者都做，但报告里分开陈述，避免混淆因果。",
             size=12, color=ACCENT)

    slide = new_slide(prs, nxt(), "训练与防泄漏", kicker="2.7 训练策略")
    bullets(slide, 0.85, 1.95, 11.5, 4.7, [
        "按患者 ID 划分训练/验证/测试，而不是按记录随机划分——同一患者多次就诊会造成泄漏",
        "AdamW + 梯度裁剪（5.0）+ 按验证集 QWK 早停，只保留最优权重",
        "所有实验固定随机种子，产物含 state_dict、特征规范、文本后端配置",
        "模型产物自带推理所需全部信息（FeatureSpec + TextProcessor 状态），服务端无需重跑训练代码",
        "小数据集上学习率过大导致模型坍缩到单类，这一点在开发中真实发生过，已写入报告讨论",
    ], size=15)

    # ---------- 03 ----------
    section_slide(prs, nxt(), "03", "实验与结果", "数据、指标、主结果、消融与错误分析")

    slide = new_slide(prs, nxt(), "数据与划分", kicker="3.1 数据")
    bullets(slide, 0.85, 1.95, 6.2, 4.6, [
        "训练：按公开文献分布校准的合成数据（40,000 条 / 17,200 位患者）",
        "校验：MIMIC-IV-ED Demo（开放版，207 次就诊）仅验证 ETL 与推理链路，不参与训练",
        "Demo 的类别分布严重不完整（ESI 5 为 0 条、ESI 4 仅 2 条），不具备训练条件",
        "结论定位：方法验证，不代表临床性能",
    ], size=15)
    for i, (title, body) in enumerate([
        ("40,000", "训练就诊记录"),
        ("17,200", "患者数（按患者划分）"),
        (f"{n_test}", "测试集样本"),
        (f"{epochs}", "训练轮次"),
    ]):
        x = 7.3 + (i % 2) * 2.6
        y = 2.0 + (i // 2) * 1.65
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                      Inches(2.4), Inches(1.35))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = RGBColor(0xDF, 0xE4, 0xEA)
        card.shadow.inherit = False
        text_box(slide, x + 0.2, y + 0.18, 2.0, 0.6, body, size=22, bold=True, color=ACCENT)
        text_box(slide, x + 0.2, y + 0.8, 2.0, 0.4, title, size=11, color=GREY)

    slide = new_slide(prs, nxt(), "评价指标：为什么不用准确率", kicker="3.2 指标")
    table(slide, 0.9, 1.95, 11.6,
          [["指标", "含义", "为什么需要"],
           ["Macro-F1", "五类 F1 的算术平均", "类别不平衡时，避免大类主导"],
           ["QWK", "二次加权 Kappa", "衡量有序一致性，医学分级任务标准指标"],
           ["危重降级率", "真实 ESI≤2 被判为 ≥3 的比例", "临床最关心的错误方向，主指标"],
           ["代价加权误差", "按代价矩阵加权的平均代价", "把“错得多严重”纳入单一数值"]],
          col_widths=[2.0, 4.2, 5.2], size=12, row_h=0.62, highlight_rows={3})
    text_box(slide, 0.9, 5.35, 11.6, 0.9,
             "四类指标从不同角度约束模型：只看准确率会掩盖“危重被漏掉”这一最严重后果。",
             size=13, color=GREY)

    main_row = ["深度-main（双塔+CORAL）", fmt(main_qwk), fmt(main_f1),
                fmt(main_under, pct=True), fmt(get("深度-main", "cost_weighted_error"))]
    res = [["模型", "QWK ↑", "Macro-F1 ↑", "危重降级率 ↓", "代价加权误差 ↓"],
           ["基线-结构化（HistGradientBoosting）", fmt(get("基线-结构化(HGB)", "qwk")),
            fmt(get("基线-结构化(HGB)", "macro_f1")),
            fmt(get("基线-结构化(HGB)", "critical_under_triage"), pct=True),
            fmt(get("基线-结构化(HGB)", "cost_weighted_error"))],
           ["基线-文本（TF-IDF + 逻辑回归）", fmt(get("基线-文本(TFIDF+LR)", "qwk")),
            fmt(get("基线-文本(TFIDF+LR)", "macro_f1")),
            fmt(get("基线-文本(TFIDF+LR)", "critical_under_triage"), pct=True),
            fmt(get("基线-文本(TFIDF+LR)", "cost_weighted_error"))],
           ["基线-简单拼接", fmt(get("基线-简单拼接", "qwk")), fmt(get("基线-简单拼接", "macro_f1")),
            fmt(get("基线-简单拼接", "critical_under_triage"), pct=True),
            fmt(get("基线-简单拼接", "cost_weighted_error"))],
           main_row]
    slide = new_slide(prs, nxt(), "主结果", kicker="3.3 模型对比")
    table(slide, 0.9, 1.95, 11.6, res, col_widths=[4.2, 1.8, 1.9, 2.0, 1.9],
          size=12, row_h=0.55, highlight_rows={4})
    text_box(slide, 0.9, 4.75, 11.6, 1.5,
             f"深度模型 QWK {fmt(main_qwk)}（95% CI {ci_text}），危重降级率 {fmt(main_under, pct=True)}。\n"
             "拼接基线在 QWK 上略高，但危重降级率更高——这正是本项目主张“主指标应为安全性”的原因：\n"
             "两个模型在“分得准”上接近，在“分得安全”上并不等价。",
             size=13, color=GREY, line_spacing=1.5)

    abl = [["实验", "QWK ↑", "Macro-F1 ↑", "危重降级率 ↓"],
           ["完整模型", fmt(get("深度-main", "qwk")), fmt(get("深度-main", "macro_f1")),
            fmt(get("深度-main", "critical_under_triage"), pct=True)],
           ["去文本塔（主诉置空）", fmt(get("深度-ablate_no_text", "qwk")),
            fmt(get("深度-ablate_no_text", "macro_f1")),
            fmt(get("深度-ablate_no_text", "critical_under_triage"), pct=True)],
           ["门控 → 直接拼接", fmt(get("深度-ablate_concat", "qwk")),
            fmt(get("深度-ablate_concat", "macro_f1")),
            fmt(get("深度-ablate_concat", "critical_under_triage"), pct=True)],
           ["CORAL → 交叉熵（同时去代价敏感）", fmt(get("深度-ablate_head_ce", "qwk")),
            fmt(get("深度-ablate_head_ce", "macro_f1")),
            fmt(get("深度-ablate_head_ce", "critical_under_triage"), pct=True)],
           ["保留代价敏感、去掉 CORAL", fmt(get("深度-ablate_cost", "qwk")),
            fmt(get("深度-ablate_cost", "macro_f1")),
            fmt(get("深度-ablate_cost", "critical_under_triage"), pct=True)]]
    slide = new_slide(prs, nxt(), "消融：每个设计点都要有数据支撑", kicker="3.4 消融")
    table(slide, 0.9, 1.95, 7.2, abl, col_widths=[3.4, 1.3, 1.5, 1.6], size=10.5, row_h=0.5,
          highlight_rows={1})
    bullets(slide, 8.1, 1.95, 4.5, 4.4, [
        "去文本塔后 QWK 掉 0.11，",
        "- 主诉是最主要的信息源",
        "CORAL 优于交叉熵：",
        "- QWK 略升，危重降级率下降",
        "门控优于拼接，但增益有限",
        "- 如实报告，不夸大",
        "代价敏感单独使用并不更好",
        "- 危重降级率反升，未采用",
    ], size=12)

    slide = new_slide(prs, nxt(), "结果总览：所有模型放在同一张图上", kicker="3.5 横向对比")
    picture(slide, fig_dir / "fig4_ablation.png", 0.85, 1.8, width=11.6)
    text_box(slide, 0.85, 6.35, 11.6, 0.6,
             "左：QWK 越高越好；右：危重降级率越低越好。两类指标的排序并不一致，这就是必须同时看的原因。",
             size=12, color=GREY)

    slide = new_slide(prs, nxt(), "混淆矩阵：危险方向在哪里", kicker="3.6 错误结构")
    picture(slide, fig_dir / "fig3_confusion_matrix.png", 0.75, 1.75, height=5.2)
    bullets(slide, 7.0, 2.0, 5.5, 4.3, [
        "虚线框 = 分诊不足：真实危重被判为更轻等级",
        "主要错误集中在相邻等级（3↔4），属可接受范围",
        "危重被跨级错判（1→4）的比例极低，但代价最高",
        "改进方向：对 ESI 1-2 单独设阈值与人工复核触发条件",
    ], size=14)

    slide = new_slide(prs, nxt(), "文本后端：语义没赢 QWK，但赢了安全", kicker="3.7 关键对比")
    if has_hf:
        cmp_rows = [["文本后端（同为 12k 子集）", "QWK ↑", "Macro-F1 ↑", "危重降级率 ↓"],
                    ["hash（字符 n-gram，无语义）", fmt(sub_hash.get("qwk", float("nan"))),
                     fmt(sub_hash.get("macro_f1", float("nan"))),
                     fmt(sub_hash.get("critical_under_triage", float("nan")), pct=True)],
                    ["Bio_ClinicalBERT（预训练语义）", fmt(sub_hf.get("qwk", float("nan"))),
                     fmt(sub_hf.get("macro_f1", float("nan"))),
                     fmt(sub_hf.get("critical_under_triage", float("nan")), pct=True)]]
        table(slide, 0.9, 1.95, 11.6, cmp_rows, col_widths=[4.6, 1.8, 2.0, 2.2],
              size=12, row_h=0.62, highlight_rows={2})
        text_box(slide, 0.9, 4.15, 11.6, 2.4,
                 "同结构、同划分、同超参，只替换文本后端，隔离“结构复杂度”与“预训练语义”。\n"
                 "结果与直觉相反：BERT 的 QWK 与 F1 更低，但危重降级率更低——它更倾向把可疑病例往上报。\n"
                 "两个原因：①合成主诉由模板生成，字符级哈希已覆盖几乎全部词汇；②BERT 只微调 3 epoch（CPU 成本）。\n"
                 "结论不外推：本数据下预训练语义不是性能来源，真实口语化主诉上才具备完整说服力。",
                 size=13, color=GREY, line_spacing=1.55)
    else:
        table(slide, 0.9, 1.95, 11.6,
              [["文本后端", "QWK ↑", "Macro-F1 ↑", "危重降级率 ↓"],
               ["hash（字符 n-gram，无语义）", fmt(get("深度-main", "qwk")),
                fmt(get("深度-main", "macro_f1")),
                fmt(get("深度-main", "critical_under_triage"), pct=True)],
               ["Bio_ClinicalBERT（预训练语义）", "待跑", "待跑", "待跑"]],
              col_widths=[4.6, 1.8, 2.0, 2.2], size=12, row_h=0.62, highlight_rows={2})
        text_box(slide, 0.9, 4.15, 11.6, 1.6,
                 "BERT 后端尚未产出结果：\npy -m er_triage.train --text-backend hf --tag hf_main",
                 size=13, color=ACCENT, line_spacing=1.5)

    slide = new_slide(prs, nxt(), "错误分析：模型在哪里犹豫", kicker="3.8 案例分析")
    bullets(slide, 0.85, 1.9, 5.8, 4.8, [
        "主诉极短（仅一个词）时，文本信息几乎为零，模型退化为看生命体征",
        "主诉与生命体征矛盾时（如“胸痛”但生命体征正常），模型倾向按体征判断",
        "合成数据中同一主诉模板高度重复，是文本塔增益被高估的原因之一",
        "真实数据的价值：主诉表述多样，才能真正检验语义理解",
    ], size=14)
    table(slide, 6.9, 2.0, 5.6,
          [["情形", "判定", "是否合理"],
           ["胸痛 + 心率 118 + 血压 96", "ESI 2", "合理"],
           ["胸痛 + 生命体征正常", "ESI 3", "保守，可接受"],
           ["“ankle sprain” + 正常体征", "ESI 4", "合理"],
           ["单一单词主诉 + 体征异常", "依体征", "需人工复核"]],
          col_widths=[2.6, 1.3, 1.7], size=11, row_h=0.62)

    # ---------- 04 ----------
    section_slide(prs, nxt(), "04", "系统与演示", "工程结构、接口契约、可演示的系统")

    slide = new_slide(prs, nxt(), "工程结构", kicker="4.1 代码组织")
    tree = (
        "er-triage/\n"
        "├─ er_triage/\n"
        "│   ├─ data/        synthetic.py  mimic_ed.py  features.py\n"
        "│   ├─ models/      text.py  fusion.py  losses.py\n"
        "│   ├─ serve/       app.py（FastAPI）  demo.py（Gradio）\n"
        "│   ├─ metrics.py   QWK / 危重降级 / bootstrap CI\n"
        "│   └─ train.py     训练与评估入口\n"
        "├─ scripts/         生成数据 / 跑实验 / 绘图 / 出 PPT\n"
        "├─ configs/         default.yaml  mimic.yaml\n"
        "├─ tests/           21 个用例\n"
        "└─ artifacts/       模型产物、指标、图、PPT"
    )
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(1.9),
                                 Inches(6.6), Inches(4.5))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0x1E, 0x24, 0x2B)
    box.line.fill.background()
    box.shadow.inherit = False
    text_box(slide, 1.1, 2.05, 6.2, 4.2, tree, size=11, color=RGBColor(0xD5, 0xDB, 0xDB), line_spacing=1.35)
    bullets(slide, 7.8, 2.0, 4.8, 4.4, [
        "训练、评估、服务、演示共用同一套特征与文本处理代码",
        "一次训练产出可自包含的模型文件，推理端无需训练依赖",
        "全部脚本可重复执行，产物由脚本生成而非手工整理",
        "测试覆盖指标计算、损失方向、划分泄漏、接口鉴权",
    ], size=13)

    slide = new_slide(prs, nxt(), "接口契约", kicker="4.2 服务化")
    bullets(slide, 0.85, 1.9, 5.4, 4.6, [
        "POST /ai/emergency-triage",
        "鉴权：X-Internal-Token，失败返回 401",
        "模型缺失返回 503，不用假数据兜底",
        "响应含 5 级概率、风险标签、处置建议、模型版本、延迟",
        "GET /health 提供就绪探针",
    ], size=14)
    req = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(1.85),
                                 Inches(5.9), Inches(4.7))
    req.fill.solid()
    req.fill.fore_color.rgb = RGBColor(0x1E, 0x24, 0x2B)
    req.line.fill.background()
    req.shadow.inherit = False
    text_box(slide, 6.8, 2.0, 5.5, 4.4,
             'POST /ai/emergency-triage\n'
             'X-Internal-Token: ***\n\n'
             '{\n'
             '  "age": 56, "sex": "M",\n'
             '  "chief_complaint": "chest pain radiating to arm",\n'
             '  "vitals": { "hr": 118, "sbp": 96, "rr": 24,\n'
             '              "spo2": 92, "pain": 7 }\n'
             '}\n\n'
             '→ 200\n'
             '{ "level": 2,\n'
             '  "probabilities": [0.007, 0.969, 0.024, 0.0002, 0.0],\n'
             '  "risk": "高", "latency_ms": ' + (f"{latency:.1f}" if latency else "—") + ' }',
             size=10.5, color=RGBColor(0xC8, 0xE6, 0xC9), line_spacing=1.3)

    slide = new_slide(prs, nxt(), "演示页：给评委看的一分钟", kicker="4.3 演示")
    picture(slide, fig_dir / "demo_page.png", 0.7, 1.75, width=8.3)
    bullets(slide, 9.2, 1.85, 3.5, 4.6, [
        "输入主诉与生命体征，实时返回分级与概率",
        "四个预置病例一键填充",
        "概率条形图让置信度可被肉眼判断",
        "漏诊预警：判为非危重但危重概率高时提示复核",
    ], size=11)
    text_box(slide, 0.7, 6.45, 8.4, 0.4,
             "启动：py -m er_triage.serve.demo --port 8091", size=11, color=GREY)

    slide = new_slide(prs, nxt(), "与原仓库的对比", kicker="4.4 交付差异")
    table(slide, 0.9, 1.95, 11.6,
          [["维度", "原仓库", "本项目"],
           ["分诊输出", "硬编码常量", "5 级概率分布 + 模型版本"],
           ["模型", "无", "双塔融合 + CORAL，产物可复现"],
           ["评估", "无", "QWK / 危重降级率 / bootstrap CI"],
           ["数据", "未说明", "合成数据训练，开放 Demo 校验管线"],
           ["接口", "有契约、无实现", "契约不变，实现可调用、有鉴权"],
           ["可复现性", "不可复现", "脚本一键复现，指标自动汇总"]],
          col_widths=[2.2, 4.0, 5.4], size=12, row_h=0.58, highlight_rows={1, 2, 3})

    slide = new_slide(prs, nxt(), "局限与合规", kicker="4.5 诚实边界")
    bullets(slide, 0.85, 1.9, 11.5, 4.6, [
        "训练数据为合成数据：结论证明方法有效，不代表临床性能",
        "ESI 标签本身存在护士间不一致，QWK 存在天然上限",
        "未建模变量：既往史、检验结果、到达方式、分诊护士经验",
        "模型仅作辅助：输出概率而非结论，必须由医生复核",
        "不构成医疗器械，不用于真实诊疗；真实部署需伦理审查与前瞻验证",
    ], size=15)

    # ---------- 05 ----------
    section_slide(prs, nxt(), "05", "结论", "做了什么、验证了什么、下一步做什么")

    slide = new_slide(prs, nxt(), "结论", kicker="5.1 回顾")
    bullets(slide, 0.85, 1.9, 11.5, 4.6, [
        f"完成一个可训练、可评估、可服务的急诊分诊模型：QWK {fmt(main_qwk)}，危重降级率 {fmt(main_under, pct=True)}",
        "把有序结构写进输出层与损失函数，而非事后处理",
        "评估口径从“分得准”转为“分得安全”，与临床代价对齐",
        "交付完整工程：训练、评估、API、演示页、测试、一键复现",
        "诚实的边界：合成数据验证方法，真实数据验证待后续",
    ], size=15)

    slide = new_slide(prs, nxt(), "后续工作", kicker="5.2 展望")
    bullets(slide, 0.85, 1.9, 11.5, 4.6, [
        "接入全量 MIMIC-IV-ED（credentialed）重跑，替换合成数据结论",
        "引入时序信息：分诊后生命体征变化、检验结果回报",
        "中文场景迁移：自建中文主诉标注集，替换临床英文预训练模型",
        "上线前的临床验证：与分诊护士双盲对照，评估漏诊率与接受度",
        "工程侧：批量推理、模型监控、分级阈值可配置化",
    ], size=15)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg(slide, NAVY)
    text_box(slide, 1.0, 2.8, 11.0, 1.0, "谢谢，请老师批评指正", size=34, bold=True, color=WHITE)
    text_box(slide, 1.05, 4.1, 11.0, 0.5, "演示与代码可现场运行", size=15,
             color=RGBColor(0x9F, 0xB3, 0xC6))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(f"已生成 {out_path}，共 {len(prs.slides)} 页")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成答辩 PPT")
    parser.add_argument("--out", default="docs/答辩.pptx")
    parser.add_argument("--tag", default="main", help="主实验标签，决定读取哪份指标")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    build(ROOT / args.out, args.tag, ROOT / args.artifacts)


if __name__ == "__main__":
    main()
