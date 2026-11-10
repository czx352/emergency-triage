# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：急诊分诊演示页（onedir）。

为什么用 onedir 而不是 onefile：torch + gradio 合计数百 MB，onefile 每次启动
都要解压到临时目录，答辩现场要等几十秒；onedir 启动更快，也不会被杀毒软件拦。

构建：
    py -m PyInstaller --noconfirm --clean packaging/demo.spec
产物：
    dist/er-triage-demo/er-triage-demo.exe
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = Path(SPECPATH).parent

datas, binaries, hiddenimports = [], [], []

# gradio 的前端静态资源、模板、组件元数据不在 Python 包里，必须整包收集
for pkg in ("gradio", "gradio_client", "safehttpx", "groovy"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

datas += collect_data_files("matplotlib")
datas += [
    (str(ROOT / "configs" / "default.yaml"), "configs"),
    (str(ROOT / "artifacts" / "main.pt"), "artifacts"),
]

# uvicorn 按字符串动态加载协议与事件循环实现，静态分析看不到
hiddenimports += [
    "er_triage.serve.demo",
    "er_triage.serve.app",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

# 演示页只用 hash 文本后端与 CPU 推理，以下依赖都可以不带，能省下近 1 GB
# 注意：gradio_client 会 import huggingface_hub，不能把后者排除掉
excludes = [
    "transformers", "tokenizers", "safetensors",
    "scipy", "sklearn", "onnxruntime", "torchvision", "torchaudio",
    "IPython", "jupyter", "notebook", "nbformat",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "tkinter",
]

a = Analysis(
    [str(ROOT / "scripts" / "demo_exe.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="er-triage-demo",
    debug=False,
    strip=False,
    upx=False,
    console=True,  # 保留控制台：双击时能看到访问地址与报错
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="er-triage-demo",
)
