# 急诊分诊分级模型（ESI 1-5）

课程设计项目：**用双塔融合 + 序数回归，把开源医疗 OS 「JLAIharnessos」里声明却未实现的急诊分诊 AI 补上**。

- 输入：主诉文本 + 生命体征 + 人口学
- 输出：ESI 1-5 分级、5 类概率、处置建议
- 主指标：QWK（有序一致性）、危重降级率（critical under-triage）
- 接口：`POST /ai/emergency-triage`，与仓库 BFF 的契约一致

## 目录

```
configs/default.yaml        训练/服务配置（含代价矩阵）
er_triage/
  data/synthetic.py         合成数据（MIMIC 审批前先跑通链路）
  data/mimic_ed.py          MIMIC-IV-ED -> 训练表 的 ETL
  data/features.py          清洗/缺失/标准化/按患者划分
  models/text.py            文本塔（hash / HuggingFace 两种后端）
  models/fusion.py          双塔 + 门控融合 + 输出头
  models/losses.py          CORAL 序数损失 / 代价敏感 CE
  metrics.py                QWK、macro-F1、危重降级率、代价误差、Bootstrap CI
  train.py                  训练入口（消融靠命令行开关）
  serve/app.py              FastAPI 推理服务
scripts/make_synthetic.py   生成合成数据
scripts/prepare_mimic_ed.py MIMIC ETL 入口
scripts/train_baseline.py   三个基线：结构化 / 文本 / 简单拼接
tests/                      pytest 用例
```

## 快速开始

```bash
# 1) 环境（Python 3.10+）
pip install -r requirements.txt

# 2) 生成合成数据并训练一个小模型（CPU 也能跑）
py scripts/make_synthetic.py --n 40000
py -m er_triage.train --epochs 3 --tag smoke

# 3) 启动服务
uvicorn er_triage.serve.app:app --port 8090

# 4) 或者直接起演示页（含表单、概率条形图、漏诊预警）
py -m er_triage.serve.demo --port 8091
```

演示页默认读取配置里的模型路径，也可以指定某次实验的产物：

```powershell
py -m er_triage.serve.demo --port 8091 --model artifacts/hf_main.pt
```

> 若本机开着系统代理（Clash/V2Ray），演示页启动自检可能报 502；脚本已自动把
> `127.0.0.1` 与 `localhost` 加入 `NO_PROXY`，无需手动处理。

调用示例（PowerShell：把请求体写入文件再传，避免 curl 参数被 shell 吞掉引号）：

```powershell
'{"age":56,"sex":"M","chief_complaint":"chest pain radiating to arm",
  "vitals":{"hr":118,"sbp":96,"rr":24,"spo2":92,"pain":7}}' | Set-Content req.json -Encoding UTF8

curl.exe -s -X POST http://127.0.0.1:8090/ai/emergency-triage `
  -H "Content-Type: application/json" `
  -H "X-Internal-Token: CHANGE_ME" `
  --data-binary "@req.json"
```

返回：

```json
{
  "level": 2,
  "probabilities": [0.0069, 0.9693, 0.0236, 0.0002, 0.0],
  "risk": "高",
  "suggestion": "10 分钟内接诊，持续监护，优先安排检查与床位",
  "model_version": "er-triage-0.1.0-main",
  "latency_ms": 58.95
}
```

## 数据来源（两种，都不影响代码结构）

本项目的 ETL 同时兼容 `.csv` 与 `.csv.gz`，两种来源都能直接跑。

### A. 免授权：MIMIC-IV-ED Demo（推荐先做这个）

PhysioNet 上该版本 Access 为 **Open**：只需登录并同意 Data Use Agreement，**不需要 CITI
培训、不走 credentialed 审批**，下载即可用。

```bash
# 把 demo 的 triage / edstays 解压到 data/raw/（.csv 或 .csv.gz 均可）
py scripts/prepare_mimic_ed.py
```

Demo 只有百级患者，**用于校验真实数据格式与管线正确性，不足以训练模型**。
对应的报告写法：

- 训练数据为按公开文献分布校准的合成数据；
- Demo 结果仅证明 ETL 与推理链路在真实数据上可用；
- 结论不代表临床性能，全量数据验证列为后续工作。

这是数据受限时的标准做法（方法验证型研究）。会被扣分的是把合成数据当作真实数据，而不是诚实使用它。

### B. 全量：MIMIC-IV-ED（credentialed，需审批）

1. 在 PhysioNet 完成 CITI「Data or Specimens Only Research」培训并申请审批；
2. 把 `triage`、`edstays`（可选 `patients`）解压到 `data/raw/`；
3. 转换并跑完整实验矩阵：

```bash
py scripts/prepare_mimic_ed.py --patients patients.csv.gz
py scripts/run_experiments.py --config configs/mimic.yaml   # 已指向 data/processed/mimic_ed.csv
```

两种来源的数据都不得再分发，`data/` 已在 `.gitignore` 中排除。

## 实验矩阵（写进报告的核心表格）

```bash
# 一键跑完 3 个基线 + 主模型 + 4 组消融，并生成 artifacts/summary.md / summary.csv
py scripts/run_experiments.py

# 先验证流程（小样本 + 2 轮）
py scripts/run_experiments.py --quick

# 正式数据
py scripts/run_experiments.py --config configs/mimic.yaml
```

## 报告配图

```bash
py scripts/make_figures.py                       # 读 artifacts/ 与合成数据
py scripts/make_figures.py --data data/processed/mimic_ed.csv --tag main
```

产物在 `artifacts/figures/`，对应报告大纲的图 2-5：

| 文件 | 内容 |
|---|---|
| `fig1_label_distribution.png` | ESI 类别分布（说明不平衡） |
| `fig2_training_curve.png` | 训练损失与验证集 QWK 曲线 |
| `fig3_confusion_matrix.png` | 混淆矩阵热力图，虚线框标出分诊不足区域 |
| `fig4_ablation.png` | 模型与消融对比（QWK / 危重降级率双面板，黑框为本方法） |
| `fig5_text_backend.png` | hash 与 Bio_ClinicalBERT 文本后端对照（同 12k 子集） |

也可以单跑某一组：

```bash
py scripts/train_baseline.py                       # 基线：结构化 / 文本 / 简单拼接
py -m er_triage.train --tag main --epochs 8        # 主模型：双塔 + 门控 + CORAL
py -m er_triage.train --tag ablate_head_ce --head ce --loss ce
py -m er_triage.train --tag ablate_concat --fusion concat
py -m er_triage.train --tag ablate_cost --head ce --loss cost_ce
py -m er_triage.train --tag ablate_no_text --no-text   # 主诉置空：严格的去文本塔消融
py -m er_triage.train --tag hf --config configs/mimic.yaml
```

每个实验都会产出 `artifacts/<tag>.pt`、`<tag>_metrics.json`、`<tag>_confusion.csv`；
`summary.md` 把 QWK、macro-F1、危重降级率、代价加权误差汇总成一张表，可直接粘进报告。

## 答辩材料

```bash
py scripts/make_slides.py                 # 34 页，指标与图表自动读取 artifacts/
py scripts/make_slides.py --tag hf_main   # 换成 BERT 主实验的数字出图出表
```

产物：`docs/答辩.pptx`。所有数字来自 `artifacts/` 下的 json/csv，不手工填写；
先跑 `make_figures.py` 再跑 `make_slides.py`，否则会嵌入旧图。

## 免 Python 演示包

`dist/er-triage-demo/er-triage-demo.exe` 已把模型、依赖和界面打进同一个文件夹，
拷到另一台 Windows 10/11 x64 机器上双击即可运行，不需要装 Python：

- 双击 → 启动后自动打开浏览器（默认 http://127.0.0.1:8091）
- 指定端口：`er-triage-demo.exe --port 8092`（带参数启动时不会自动开浏览器）
- 必须整个 `er-triage-demo` 文件夹一起拷贝，exe 不能单独拎出来

重新打包：

```bash
py -m pip install pyinstaller
py -m PyInstaller --noconfirm --clean packaging/demo.spec
```

打包版只带 hash 文本后端（不含 transformers），体积约 800 MB；要用 BERT 后端仍需源码启动。

## 测试

```bash
py -m pytest -q
```

覆盖：指标计算、CORAL 目标编码与梯度、代价敏感损失的惩罚方向、患者级划分无泄漏、缺失列/缺失值的特征处理、API 鉴权与契约字段、模型缺失时返回 503。

## 设计约定

- **模型缺失时返回 503，绝不用假数据兜底**——原仓库正是用硬编码常量冒充 AI 结果。
- 标准化统计量只在训练集拟合，随模型一起保存，保证线上线下一致。
- 按 `subject_id` 划分，避免同一患者跨训练/测试泄漏。
- ESI 是有序等级，默认用 CORAL 序数损失，而不是普通交叉熵。

## 已知局限

- 合成数据只用于验证管线，指标不代表临床性能；
- `sex`、`arrival_mode` 已进入接口契约但未参与建模，可作为后续改进点；
- 单中心数据存在选择偏倚，跨院迁移需要外部验证。
