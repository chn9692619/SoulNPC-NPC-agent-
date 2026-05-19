# SoulNPC AutoDL 云端 LoRA 训练说明

## 1. 基本思路

本地 SoulNPC 工作台通过 SSH/SFTP 控制已开机的 AutoDL 实例：

1. 本地生成最终训练数据。
2. 打包项目与数据。
3. 上传到 AutoDL。
4. 在 AutoDL 安装 LoRA 训练依赖。
5. 使用 tmux 后台启动 SFT LoRA 训练。
6. 本地工作台读取训练日志与状态。
7. 训练完成后可通过 SSH 推理或启动 6006 端口推理服务。

## 2. AutoDL 信息

当前默认预填：

- Host: `connect.bjb2.seetacloud.com`
- Port: `24024`
- User: `root`
- Remote Path: `/root/autodl-tmp/SoulNPC-Agent`
- 6006 映射地址：`https://u815652-9973-5b0ca6e7.bjb2.seetacloud.com:8443`
- 6008 映射地址：`https://uu815652-9973-5b0ca6e7.bjb2.seetacloud.com:8443`

密码只在网页会话中输入，不会写入配置文件。

## 3. 推荐操作顺序

1. 在本地工作台：数据工厂 -> 生成基础训练样本。
2. 训练中心 -> 生成最终 SFT/DPO 训练文件。
3. LoRA 配置 -> 保存基础模型、训练轮数、学习率等参数。
4. 云端训练 -> 检查云端环境。
5. 云端训练 -> 生成本地训练包。
6. 云端训练 -> 上传项目与数据。
7. 云端训练 -> 安装云端训练依赖。
8. 云端训练 -> 启动 SFT LoRA 训练。
9. 云端训练 -> 读取训练日志 / 检查训练状态。
10. 训练完成后，云端推理测试。

## 4. LoRA 训练结果

默认输出目录：

```text
outputs/lora_sft
```

推理时需要：

```text
基础模型 base model + LoRA adapter
```

## 5. 注意事项

- 如果训练中断，先检查 `outputs/sft_train.log`。
- 如果 SSH 断开，tmux 后台训练一般不会中断。
- 如果显存不足，降低 batch size、max_seq_length，或启用 4bit。
- 如果 Hugging Face 下载模型很慢，可考虑提前在 AutoDL 中配置镜像或手动上传模型。

## v1.8 基础模型准备流程

在 AutoDL 训练前，必须先保证基础模型已经存在于云端本地目录。推荐流程：

1. 上传项目与数据。
2. 安装云端训练依赖。
3. 在“基础模型准备”区域选择模型来源。
4. 推荐使用 ModelScope 下载 `Qwen/Qwen2.5-1.5B-Instruct` 到 `/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`。
5. 下载成功后，工作台会自动把远程训练配置改为 `model_source=local`，训练时将只从本地模型目录加载，避免 HuggingFace 网络不可达导致失败。

如果模型已经手动下载好，可以选择“仅检查本地路径”。
