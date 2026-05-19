# SoulNPC 中文训练工作台 v1.0

这是一个面向可信游戏角色的轻量认知-情感 Agent 原型，重点支持：

- 连续情感向量，而非单一离散 mood
- 非线性关系向量，而非线性关系阶段
- 抽象事件原语 + 权重组合，描述复杂玩家事件
- 加权记忆库：重要度、情绪显著性、关系影响、随机遗忘
- 自动生成 SFT / DPO 数据
- 人工覆盖增强：不审阅也能训练，审阅后覆盖对应样本
- Baseline 训练测试
- LoRA / SFT / DPO 训练脚本模板，可在 AutoDL 或本地 CUDA 环境运行

## Windows 本地运行

```powershell
cd E:\Desktop\SoulNPC-Agent-v10
python -m venv myagent
.\myagent\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
python app\main.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

## 快速流程

1. 控制台：刷新项目状态
2. 数据工厂：生成基础训练样本
3. 训练中心：一键生成最终数据并训练 Baseline
4. 审阅覆盖：选择少量样本改写，再重新训练
5. 训练中心：保存 LoRA 配置并生成 AutoDL 命令

## LoRA 训练说明

LoRA 不是训练完整大模型，而是在冻结基础模型大部分参数的情况下，训练少量低秩适配矩阵。推理时通常需要：

- 基础模型权重
- LoRA adapter 权重

可以选择：

- 本地加载基础模型 + LoRA adapter
- AutoDL 训练并导出 adapter
- 之后在本地或服务器部署

API 模型通常不能直接拿来本地训练 LoRA，除非平台提供微调接口。因此本项目的 LoRA 主线默认使用开源模型，例如 Qwen 1.5B 级别模型。
