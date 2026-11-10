"""PyInstaller 打包入口：把 Gradio 演示页冻结成 exe。

与 `py -m er_triage.serve.demo` 行为一致，区别只是模型与配置从 exe 自带的资源目录读取，
因此双击即可运行，不依赖目标机器的 Python 环境。

构建：
    py -m PyInstaller --noconfirm --clean packaging/demo.spec
产物：
    dist/er-triage-demo/er-triage-demo.exe
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _base_dir() -> Path:
    """冻结后资源在 sys._MEIPASS（onedir 下即 _internal），源码运行时是项目根。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def main() -> None:
    base = _base_dir()
    os.environ.setdefault("TRIAGE_CONFIG", str(base / "configs" / "default.yaml"))
    os.environ.setdefault("TRIAGE_MODEL_PATH", str(base / "artifacts" / "main.pt"))

    argv = sys.argv[1:]
    if not argv:
        argv = ["--open"]  # 双击启动（无参数）时自动拉起浏览器；带参数时按命令行语义走
    sys.argv = [sys.argv[0], *argv]

    from er_triage.serve.demo import main as demo_main

    demo_main()


if __name__ == "__main__":
    main()
