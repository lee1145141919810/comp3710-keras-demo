> 历史文档：当前 Windows/GAN/DAWNBench 状态请先看 [CURRENT_STATUS_zh.md](CURRENT_STATUS_zh.md)。下文“未选 GAN”“尚未连接”等描述不再代表最新进度。

# COMP3710 Lab 2 评分标准核对与 Medium 完成路线

核对起点：GitHub `lee1145141919810/comp3710-keras-demo`，提交 `3bb2379`。
依据：用户提供的 `COMP3710_Lab_2_2026_v2.01.pdf` 和 `COMP3710_Demo_rubric_v2_final.pdf`。
用户选择：Medium（VAE + UNet），目前没有 Rangpur 模型、日志或 SSH 连接资料。

## 结论与计分边界

原仓库覆盖各任务的代码，但“代码存在”不等于“实验达标”。原有 RF 0.6118、LFW CNN 0.9410 是仓库中的历史记录，本次没有复现这两项，不能称为本次实测。原有合成数据 UNet 成绩不能证明真实 OASIS 达标。

实验表总分是 15 分。Medium 的 recognition tasks 上限是 5/7；另有 Git 课程 1 分，因此 Part 4 上限是 6/8，整份实验的任务分上限是 13/15。这是档位上限，不是预计得分或保证分数。

Demo rubric 另按功能完成度 20%、问答及理解 40%、编程/文档/合理 AI 使用 20%、总结与所有权展示 20%考核。不能简单把代码文件数或测试通过率换成最终分数。

| 任务 | 实验分值 | 原仓库证据 | 本次及待完成证据 |
|---|---:|---|---|
| Part 1 DFT | 1 | 方波、谐波、CPU DFT/FFT、tensor DFT、图与 CPU 记录 | 单元验证通过；GPU 多规模实测仍需 Rangpur |
| Part 2 Eigenfaces | 1 | SVD、训练均值、compactness、RF、历史测试结果 | 结构符合；原结果未在本次重跑 |
| Part 3.1 LFW CNN | 1 | 两个 3×3/32 卷积层、dense、Adam/CE | 结构符合；历史测试结果未在本次重跑 |
| Part 3.2 DAWNBench | 4 | 自写 ResNet-18、AMP、训练/推理脚本 | >90%、94%/约360秒均待实测；Rangpur 现场推理和一轮训练待完成 |
| Part 4.1 Advanced Git Course | 1 | 未发现完成证明 | 自行完成 Version Control for Teams using Git 并准备证明 |
| Part 4 Medium | 最多5/7 | VAE/UNet 代码，无真实数据训练证据 | 真实数据审计及本次训练见 `MEDIUM_RESULTS_zh.md`；现场 UNet 推理仍需演示 |
| Hard GAN | 不选 | 原代码保留 | 用户不选此档，不追加 GAN 训练 |

实验第10页说明除最低难度外需要 Rangpur；本次 CPU 基线不能替代相应集群要求。

## 本次发现并修正的问题

1. **Mask 配对错误风险**：原实现找不到一个 mask 时，退化为按排序配对，可能将不同病例/切片错误地当作真值。现在仅接受可核实的名称或病例/切片编号配对；缺失、歧义和重复匹配直接报错。
2. **标签校验**：原实现将未知灰度舍入成合法类别，可能掩盖损坏数据。现在仅接受 0/85/170/255 或 0/1/2/3。
3. **训练统计与小数据**：VAE/UNet 原 `drop_last=True` 配合按完整数据集长度除，低估训练 loss，且可能产生空训练轮。现在保留末尾批次。
4. **模型与评估隔离**：新 `--output_dir` 防止演示覆盖正式 checkpoint；UNet 的 `--validation_only` 在调参阶段不加载测试集，`--init_checkpoint` 是明确标注的权重热启动（不冒充恢复优化器）。
5. **Categorical 输出**：softmax 概率不等于 hard one-hot。现在提供实际 hard one-hot 推理及小型 NPZ 示例；逐类别 DSC 与整体像素 accuracy 分开报告。
6. **演示证据**：UNet 推理记录 checkpoint SHA256、设备、样本数、分辨率、逐类 DSC 和是否全部严格超过 0.9。
7. **SLURM 日志**：SLURM 在脚本运行前打开日志，原脚本内的 `mkdir logs` 太晚。输出改到提交目录，避免目录不存在导致作业启动失败。
8. **DAWNBench 时间**：同时报告纯训练时间与包含评估时间的耗时，单列最终 TTA 达标情形；`--eval_only` 必须提供 checkpoint。没有实际 GPU 跑分前不宣称 94%/360秒达标。
9. **可视化边界**：VAE 小测试集不会因固定取10张图而越界；二维 latent 已有直接流形图，不再额外强制跑 UMAP。
10. **Medium 工作流**：本地脚本默认仅训练 VAE/UNet，GAN 需 `RUN_GAN=1` 显式启用。

## 数据检查

执行 `python scripts/audit_oasis.py --data_root <path>`，审计全部原始 PNG 的可解码性、图像/mask尺寸、标签和病例级数据分离。机器可读证据：`docs/results/oasis_audit.json`。

训练/验证/测试为 9664/1120/544 张，原始尺寸 256×256，mask 灰度仅为 0/85/170/255，三个 split 的病例编号互不交叉。这里验证的是文件中的 case ID；没有额外受试者元数据时，不能据此证明不同 case ID 一定对应不同自然人。

## 仍需要本人完成的部分

- 完成并展示 Advanced Git Course 证明。
- 在 Rangpur 使用实际可用分区和个人 Python 环境跑 Part 1 GPU 与 Part 3.2；默认 `a100` 是模板值，先用 `sinfo` 核实。OASIS 的具体子目录也要实际确认。
- 通过最终 UNet checkpoint 在测试集现场推理，讲清楚每类 DSC、one-hot 与 argmax、skip connection、分辨率变更对边界的影响。
- 约3分钟总结，随后回答问题；解释 `mu/log_var`、reparameterisation、重建项/KL项，不能只背代码。
- 保存本次和此前 AI 工具的真实对话证据；本仓库的交互摘要不能替代缺失的历史完整对话。


## 数据来源复核：已排除异常疑点

发现302个训练病例的slice_0背景mask完全相同后，从实验PDF中的Preprocessed OASIS链接重新下载官方压缩包进行核实。
官方压缩包也只有一种slice_0背景图案；用户提供的22656个PNG（11328图像+11328 mask）逐文件SHA-256与官方文件全部相同。
因此不能把这一背景现象解释为用户文件被伪造或替换；其预处理原因未由本次核对确定。
来源比较证据在 `docs/results/oasis_source_comparison.json`，可用 `scripts/verify_oasis_archive.py` 对独立参考ZIP重新核实。


## 最终本地结果

VAE：15轮全量训练完成，已有checkpoint、重建图和二维流形图。
UNet：固定模型在完整544张测试图（128×128）上的DSC为背景0.9982、CSF0.9267、灰质0.9401、白质0.9665，四类均严格超过0.9；同时提供hard one-hot导出、模型哈希、推理计时和分割图。
因此Medium的本地模型与结果证据已补齐；它不等于已经取得5分，现场问答/推理、课程证明和集群要求仍须完成。
21项回归测试通过。完整细节见 `MEDIUM_RESULTS_zh.md`。
