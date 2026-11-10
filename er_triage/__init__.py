"""急救分诊分级模型（ESI 1-5）。

目录约定
- er_triage/data     数据生成、MIMIC-ED ETL、特征工程
- er_triage/models   文本塔 / 生命体征塔 / 融合模型 / 损失函数
- er_triage/train    训练与评估入口
- er_triage/serve    FastAPI 推理服务（对齐 JLAIharnessos 的接口契约）
"""

__version__ = "0.1.0"
