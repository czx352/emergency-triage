"""Gradio 演示页：单条病例分诊，含概率分布与漏诊预警。

启动：
    py -m er_triage.serve.demo
    py -m er_triage.serve.demo --port 8091 --model artifacts/hf_main.pt

说明：这是课程设计的研究原型，不是医疗器械，输出不构成临床决策依据。
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gradio as gr

from ..schema import ESI_LABELS, ESI_RISK, ESI_SUGGESTIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LEVEL_COLORS = {1: "#b03a2e", 2: "#d35400", 3: "#b7950b", 4: "#2e86c1", 5: "#5d6d7e"}

# 演示病例：(年龄, 性别, 到达方式, 主诉, hr, sbp, dbp, rr, temp_c, spo2, pain, gcs)
EXAMPLES = [
    [58, "M", "ambulance", "chest pain radiating to left arm with sweating", 118, 96, 60, 24, 37.2, 92, 8, 15],
    [72, "F", "ambulance", "sudden onset weakness on right side and slurred speech", 96, 178, 98, 20, 36.9, 96, 2, 13],
    [24, "M", "walk_in", "ankle sprain after playing basketball", 76, 126, 78, 16, 36.8, 99, 3, 15],
    [33, "F", "walk_in", "fever and cough for two days", 102, 118, 72, 22, 38.7, 97, 4, 15],
]

_STATE: dict = {"state": None, "error": None}


def get_state() -> dict:
    """懒加载模型，失败时把原因留给界面显示，不静默兜底。"""
    if _STATE["state"] is not None or _STATE["error"] is not None:
        return _STATE
    try:
        from .app import load_predictor
        _STATE["state"] = load_predictor()
    except Exception as exc:  # 模型缺失或依赖不全
        _STATE["error"] = str(exc)
    return _STATE


def _empty_figure(message: str):
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#a33")
    ax.axis("off")
    return fig


def _prob_figure(probs: np.ndarray, level: int):
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    y = np.arange(5)
    colors = [LEVEL_COLORS[i + 1] if (i + 1) == level else "#d5d8dc" for i in range(5)]
    ax.barh(y, probs, color=colors, edgecolor=["black" if (i + 1) == level else "none" for i in range(5)])
    for i, p in enumerate(probs):
        ax.text(p + 0.01, i, f"{p:.1%}", va="center", fontsize=9)
    ax.set_yticks(y, [f"ESI {i + 1} {ESI_LABELS[i + 1].split('（')[0]}" for i in range(5)])
    ax.set_xlim(0, max(probs.max() * 1.25, 0.12))
    ax.set_xlabel("模型预测概率")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def _card(level: int, risk: str, suggestion: str, warn: str, version: str, latency: float) -> str:
    color = LEVEL_COLORS[level]
    warn_html = f'<div class="warn">{warn}</div>' if warn else ""
    return f"""
    <div class="triage-card">
      <div class="badge" style="background:{color}">ESI {level}</div>
      <div class="meta">
        <div class="label">{ESI_LABELS[level]}</div>
        <div class="sub">风险等级：{risk} ｜ 耗时 {latency:.1f} ms</div>
      </div>
      <div class="suggest">{suggestion}</div>
      {warn_html}
      <div class="ver">模型版本：{version}</div>
    </div>
    """


def triage(age, sex, arrival, complaint, hr, sbp, dbp, rr, temp_c, spo2, pain, gcs):
    state = get_state()
    if state["state"] is None:
        return (f'<div class="triage-card"><div class="label">模型未加载</div>'
                f'<div class="sub">{state["error"]}</div></div>'), _empty_figure("模型未加载")

    vitals = {"hr": hr, "sbp": sbp, "dbp": dbp, "rr": rr,
              "temp_c": temp_c, "spo2": spo2, "pain": pain, "gcs": gcs}
    started = time.perf_counter()
    from .app import predict_probs
    probs = predict_probs(state["state"], age, vitals, complaint)
    latency = (time.perf_counter() - started) * 1000

    level = int(np.argmax(probs)) + 1
    # 漏诊预警：判为非危重，但仍有可观概率落在 ESI 1-2
    critical_mass = float(probs[0] + probs[1])
    warn = ""
    if level >= 3 and critical_mass >= 0.30:
        warn = f"⚠ 漏诊预警：模型判定 ESI {level}，但危重概率合计 {critical_mass:.0%}，建议人工复核"

    return (_card(level, ESI_RISK[level], ESI_SUGGESTIONS[level], warn,
                  state["state"]["version"], latency),
            _prob_figure(probs, level))


CSS = """
body, .gradio-container {background:#f4f6f9 !important;}
.gradio-container {max-width: 1180px !important;}

/* 系统开深色模式时 Gradio 会把文字变量切成浅色，而卡片底色是固定白，
   结果浅字压白底。这里强制一套浅色变量，卡片选择器再加 !important 兜底。 */
.dark, .dark .gradio-container {
  --body-text-color:#1b2631; --body-text-color-subdued:#5a6672;
  --background-fill-primary:#ffffff; --background-fill-secondary:#f4f6f9;
  --block-background-fill:#ffffff; --block-label-text-color:#1b2631;
  --block-title-text-color:#1b2631; --border-color-primary:#e3e6ea;
  --input-background-fill:#ffffff; --input-border-color:#cfd6dd;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container .prose p, .gradio-container label span {color:#1b2631 !important;}

.triage-card {border:1px solid #dfe4ea; border-radius:12px; padding:18px 20px;
              background:#fff !important;}
.triage-card .badge {display:inline-block; color:#fff !important; font-size:26px; font-weight:700;
        padding:10px 18px; border-radius:10px; margin-right:14px; vertical-align:middle;}
.triage-card .meta {display:inline-block; vertical-align:middle;}
.triage-card .label {font-size:21px; font-weight:700; color:#12232e !important;}
.triage-card .sub {font-size:14px; color:#465563 !important; margin-top:5px;}
.triage-card .suggest {margin-top:16px; font-size:16px; color:#1b2a36 !important; line-height:1.6;}
.triage-card .warn {margin-top:12px; padding:10px 12px; border-radius:8px;
       background:#fdf2e9; color:#8a3500 !important; font-size:14px; border-left:4px solid #d35400;}
.triage-card .ver {margin-top:14px; font-size:12.5px; color:#7b8794 !important;}
.note {font-size:12px; color:#7f8c8d !important; line-height:1.7;}

/* 选中文字时默认是深蓝底 + 深色字，几乎看不清，这里换成浅蓝底 + 近黑字 */
::selection {background:#cfe0f7; color:#12232e;}
"""


# 页面固定用浅色：投影与截图更清楚。Gradio 跟随系统深色模式给容器加 .dark 类，
# 这里在加载时去掉，并持续监听，避免系统主题切换后又变深。
FORCE_LIGHT_JS = """
() => {
  const strip = () => document.querySelectorAll('.dark').forEach((el) => el.classList.remove('dark'));
  strip();
  new MutationObserver(strip).observe(document.documentElement,
    {attributes: true, subtree: true, attributeFilter: ['class']});
}
"""


def build_demo():
    # 固定浅色主题：投影和截图都更清楚，也不受系统深色模式影响
    with gr.Blocks(title="急诊分诊辅助（课程设计）") as demo:
        gr.Markdown(
            "## 急诊分诊辅助模型\n"
            "双塔融合（结构化生命体征 + 主诉文本）· CORAL 序数回归 · 输出 ESI 1-5 级及概率分布"
        )
        with gr.Row():
            with gr.Column(scale=5):
                complaint = gr.Textbox(label="主诉（英文，MIMIC 口径）", lines=2,
                                       value=EXAMPLES[0][3])
                with gr.Row():
                    age = gr.Number(label="年龄", value=58, precision=0)
                    sex = gr.Dropdown(["M", "F"], label="性别", value="M")
                    arrival = gr.Dropdown(["ambulance", "walk_in", "other"], label="到达方式", value="ambulance")
                with gr.Row():
                    hr = gr.Slider(20, 250, value=118, step=1, label="心率 bpm")
                    sbp = gr.Slider(40, 260, value=96, step=1, label="收缩压 mmHg")
                    dbp = gr.Slider(20, 160, value=60, step=1, label="舒张压 mmHg")
                with gr.Row():
                    rr = gr.Slider(4, 60, value=24, step=1, label="呼吸")
                    temp_c = gr.Slider(32, 43, value=37.2, step=0.1, label="体温℃")
                    spo2 = gr.Slider(50, 100, value=92, step=1, label="SpO2")
                with gr.Row():
                    pain = gr.Slider(0, 10, value=8, step=1, label="疼痛评分")
                    gcs = gr.Slider(3, 15, value=15, step=1, label="GCS")
                run = gr.Button("开始分诊", variant="primary")
            with gr.Column(scale=6):
                card = gr.HTML()
                chart = gr.Plot(label="各级概率")
                gr.HTML('<div class="note">说明：本页面为深度学习课程设计的研究原型，训练数据为合成数据，'
                        '输出不代表临床性能，不构成诊疗依据。</div>')

        inputs = [age, sex, arrival, complaint, hr, sbp, dbp, rr, temp_c, spo2, pain, gcs]

        # 用四个按钮代替表格型 Examples：表格会把 12 个输入横着铺开，投影时看不清
        gr.Markdown("**演示病例**（点击填充后再点“开始分诊”）")
        with gr.Row():
            for label, case in zip(["① 胸痛", "② 卒中", "③ 踝扭伤", "④ 发热"], EXAMPLES):
                gr.Button(label, size="sm").click(
                    (lambda c=case: tuple(c)), outputs=inputs
                )

        run.click(triage, inputs=inputs, outputs=[card, chart])
        demo.load(triage, inputs=inputs, outputs=[card, chart])
    return demo


def _pick_port(preferred: int, span: int = 20) -> int:
    """端口被占就往后顺延：答辩现场同时开着旧窗口时，不该直接崩掉。"""
    import socket

    for port in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:  # 连不上即视为空闲
                return port
    return preferred


def main() -> None:
    parser = argparse.ArgumentParser(description="急诊分诊演示页")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", default=None, help="模型产物路径，默认取 TRIAGE_MODEL_PATH 或配置")
    parser.add_argument("--share", action="store_true", help="生成公网临时链接")
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = parser.parse_args()

    if args.model:
        os.environ["TRIAGE_MODEL_PATH"] = args.model

    # 本机若开着系统代理（Clash/V2Ray 等），127.0.0.1 也会被代理转发，
    # Gradio 启动自检会拿到 502。这里把回环地址排除在代理之外。
    no_proxy = os.environ.get("NO_PROXY", "")
    for host in ("127.0.0.1", "localhost"):
        if host not in no_proxy:
            no_proxy = f"{no_proxy},{host}" if no_proxy else host
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = no_proxy

    port = _pick_port(args.port)
    if port != args.port:
        print(f"端口 {args.port} 已被占用，改用 {port}")

    build_demo().launch(
        server_name="127.0.0.1", server_port=port, share=args.share,
        css=CSS, quiet=True,
        # v6 起 theme 与 css 都在 launch() 传入，写在 Blocks 里会告警
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        inbrowser=args.open,
        # v6 起 js 也归 launch()：加载时去掉 .dark 并持续监听，保证始终是浅色
        js=FORCE_LIGHT_JS,
        # 关闭 SSR：v6 默认走 node 服务端渲染，无 node 环境时启动自检会失败
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
