# GAN 改进结果与复现

## 本次结果

Windows RTX 5080，PyTorch 2.11.0+cu128。使用已核验的课程 OASIS 训练图 9,664 张，128×128，batch size 64，seed 42，60 轮，约 176 秒。

原模型在训练和推理模式下均对不同随机输入输出近似图像，排除了仅由 BatchNorm 推理状态造成的显示问题。本次在原有对抗损失中加入权重 10 的粗尺度空间统计约束：先将真实与生成图池化到 16×16，再匹配批次内各位置的均值及标准差。这是本项目的实验性正则项；不是额外训练数据，也不是评分指标。默认权重仍为 0，可复现原始配置。

同时冻结生成器更新期间的判别器参数梯度，保存训练进度、优化器状态，并新增独立推理和潜空间插值脚本。生成器曲线中的 loss_G 为加入正则项后的总损失，不能与原始 BCE 数值直接比较。

## 独立对比

两个模型均使用 seed 2026 的 256 个新随机输入。真实参考为验证集随机抽取的 256 张图，抽样 seed 123；图像范围统一为 [0,1]。测试集未用于此对比。

| 指标 | 原始模型 | 改进模型 |
|---|---:|---:|
| 生成/真实平均像素 L2 距离比 | 0.0537 | 0.8930 |
| 16×16 池化后的距离比 | 0.0616 | 0.9455 |
| 前景面积占比标准差 | 0.00182 | 0.02240 |

真实验证图的前景面积占比标准差为 0.02172（灰度阈值 0.15）。对比图依次为真实验证图、原始模型、改进模型，每行 16 张；两模型使用相同随机向量。

可视化显示改进后脑室形状、图像大小和内部结构有明显变化，原先严重的重复输出已得到缓解。部分纹理仍偏平滑或存在伪影。距离比接近 1 不能证明解剖真实性、没有记忆训练图、覆盖全部模式，亦不能直接换算为课程成绩。没有执行 FID 或完整训练集最近邻审计。

## 验证

23 项单元测试通过，修改文件 Ruff 检查通过。重新加载第 60 轮模型后成功生成新样本和 8 组潜空间插值，输出均为有限数值。

## 使用交付包

压缩包包含当前源代码、模型、日志、样图、对比指标及环境清单；不包含原始数据或 Python 环境。

在解压目录中，使用已有环境的 Python 执行以下命令。若换电脑，先安装兼容该显卡的 GPU 版 PyTorch，再安装 requirements-dev.txt；windows_environment.txt 记录本次确切版本。

生成新样本（无需 OASIS 原始数据）：

```powershell
& 'C:/Users/Hi/Documents/Codex/gan-env/Scripts/python.exe' part4_recognition/gan/sample.py --checkpoint results/gan_improved/dcgan.pt --output_dir results/new_samples --device cuda --seed 2026
```

重新训练（把数据路径替换为含各数据划分子目录的解压目录；使用新输出目录）：

```powershell
& 'C:/Users/Hi/Documents/Codex/gan-env/Scripts/python.exe' part4_recognition/gan/train.py --data_root 'D:/你的数据目录/keras_png_slices_data' --device cuda --image_size 128 --epochs 60 --batch_size 64 --moment_weight 10 --output_dir results/gan_new_run
```

检查：

```powershell
& 'C:/Users/Hi/Documents/Codex/gan-env/Scripts/python.exe' -m pytest tests -q
```

保存了优化器状态，但本版训练命令尚未实现 --resume；重复运行训练命令会从头训练。

## 课程后续

本次完成的是本机 GAN 训练与对比，不等同于整份作业全部达标。仍需结合原始评分表核对 Hard 档要求、在 Rangpur 上验证适用的演示流程、保留 Git 课程证明，并准备本人答辩。学校 CIFAR 模型和日志尚未下载到本机。代码未推送 GitHub。

AI 协助范围：本次环境配置、GAN 正则项实现、测试、训练运行、对比和说明整理由 Codex 协助完成；提交时应按课程规定披露。
