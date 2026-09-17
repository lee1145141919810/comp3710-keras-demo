> 历史文档：当前 Windows/GAN/DAWNBench 状态请先看 [CURRENT_STATUS_zh.md](CURRENT_STATUS_zh.md)。下文“未选 GAN”“尚未连接”等描述不再代表最新进度。

# COMP3710 Lab 2：Medium 演示与答辩指南

本指南对应当前实际状态，详细数值见 [Medium结果](MEDIUM_RESULTS_zh.md)，评分项见 [核对报告](RUBRIC_AUDIT_zh.md)，登录与后续操作见 [下一阶段](NEXT_STEPS_zh.md)。

## 先准确说明完成范围

- VAE：已在全部9664张训练切片上训练15轮，64×64、二维latent，已有模型、重建与流形图。
- UNet：128×128、base_channels=8、depth=3；固定模型在全部544张测试切片上的DSC为背景0.9982、CSF0.9267、灰质0.9401、白质0.9665。
- 数据：用户PNG与从实验PDF链接下载的官方归档逐文件SHA-256完全一致，病例编号在训练/验证/测试之间没有交叉。
- Parts 1–3.1已有代码和历史CPU结果；历史RF 0.6118和LFW CNN 0.9410不是本次重新训练的数值。
- 待完成：Rangpur GPU/DAWNBench实测与现场演示、Advanced Git Course、本人问答。GAN属于未选择的Hard档，不声称已经训练完成。

## 展示材料

1. `docs/figures/medium_vae_manifold.png` 和 `medium_vae_reconstructions.png`。
2. `docs/figures/medium_unet_predictions.png`、`docs/results/medium_unet_test.json`。
3. `git log --oneline`、`docs/AI_USAGE.md`、实际AI对话记录。
4. 之后取得的Rangpur日志、DAWNBench `results.json` 和Git课程完成证明。

先取得当前分区列表，再用 `bash slurm/interactive_gpu.sh ACTUAL_GPU_PARTITION` 申请GPU。激活环境后执行 `bash scripts/cluster_preflight.sh gpu`。真实Rangpur连接方法须来自学校说明，不能将模板值当成已验证配置。

GPU现场演示（模型和数据已准备好之后）：

```bash
bash scripts/run_cluster_demo.sh \
  results/part3_dawnbench/resnet18_cifar10.pt \
  results/medium_unet128_refine/unet_best.pt \
  /actual/path/to/keras_png_slices_data
```

## 三分钟讲述提纲

- **前35秒**：说明任务链条和选择Medium；明确本地已经完成哪些结果、集群还有哪些证据。
- **35–70秒**：DFT用固定频率基表示信号，PCA学习数据方差方向；CNN则通过分类损失联合学习非线性空间特征。
- **70–110秒**：展示VAE重建与二维流形，解释重建/KL之间的权衡。图中形态连续变化，但细节模糊，不能声称完整恢复原图。
- **110–155秒**：展示UNet真值、预测、错误图及四类DSC。说明128×128评价、验证集选模、测试集最终评价与one-hot导出。
- **155–180秒**：说明Git提交、发现并修复mask配对风险、与官方归档比对数据、AI辅助及本人检查过程。

讲述时用自己的话，不把这份提纲当作已证明本人理解的材料。

## 必须能解释的内容

### DFT

- 50个奇次谐波的频率为1、3、…、99倍基频，幅度4/(πn)。增加谐波使边沿更陡，但Gibbs过冲约占跳变量9%，不会完全消失。
- 朴素DFT为O(N²)，FFT为O(N log N)。GPU并行不改变朴素DFT的复杂度；实际快慢依赖规模、数据类型和设备，不能背一个未经实测的固定GPU排名。
- GPU运算异步，计时需要同步并说明预热方法。N翻倍时，朴素DFT通常接近4倍计算量，FFT约2倍多一点。
- 理想方波的高频成分会混叠；但偶数N下奇谐波的混叠不会凭空变成偶频项。`sign(0)`和浮点`sin(pi)`对断点采样的处理也会影响小的偶频项。

### PCA 与 LFW CNN

- 中心化训练矩阵的SVD为X=USVᵀ；V的列（Vᵀ的行）是协方差的特征向量，特征值为S²/(n_train−1)。代码的第三个返回值通常命名V或Vt，但其实际含义是Vᵀ。
- 测试数据使用训练均值和训练主成分，避免把测试信息引入模型。
- 累计解释方差高不等于身份识别信息全部保留；方差大的光照/姿态未必利于分类。
- 两个3×3、32滤波器卷积层之后接dense分类；输入是[N,C,H,W]，整数标签用于cross_entropy。
- 卷积具有平移等变性；池化等操作可带来一定平移鲁棒性，不应宣称普通CNN对任意平移严格不变。

### ResNet 与 DAWNBench

- BasicBlock输出包含F(x)+shortcut(x)。当尺寸/通道变化时shortcut需要投影，并非每个block都严格是恒等映射。
- 混合精度可在支持的GPU上加速部分运算；fp16通常需要GradScaler，bf16一般不需要。速度和准确率增益应实测，不承诺固定倍数。
- 报告训练样本数、精度、设备、目标准确率首次达成时间及计时范围。纯训练时间与含评价的耗时要分开。
- 单轮演示使用独立目录，防止覆盖完整训练权重。推理必须载入训练checkpoint。

### VAE

- 编码器输出mu和log_var，sigma=exp(0.5*log_var)，z=mu+sigma*epsilon，epsilon来自标准正态。
- 重参数化使随机性来自epsilon，让梯度通过mu/sigma传播。损失为重建项+beta乘KL项；beta=1时对应当前模型的负ELBO目标。
- KL鼓励后验接近先验，但不保证所有潜变量解耦。增大beta一般加强先验约束，可能损害重建。
- 二维latent方便直接画采样网格，但信息瓶颈强，重建会丢细节；高维latent可以另用UMAP展示。

### UNet

- 当前模型有3个下采样阶段、base=8；与原模板默认4阶段、base=32不同。说明实际checkpoint参数，不背默认配置。
- skip connection将编码器同尺度特征拼接到解码器，帮助利用空间细节。
- logits是4通道未归一化分数；softmax是概率；argmax是整数类别；`predict_one_hot`是每像素只有一个1的四通道硬分类输出。
- DSC=2|P∩G|/(|P|+|G|)。整体像素accuracy可能被背景支配，因此必须查看各组织DSC。
- 汇总DSC先累加全部切片的交集与面积，再求比值；逐图DSC均值是另一种统计量。四类汇总DSC都>0.9不意味着每张图的每类DSC都>0.9。
- 先看验证集最弱类别，均值用于打破平局；第二阶段在所有验证类别超过0.92时停止，再固定模型测完整测试集。热启动重建了优化器，不等于恢复完整训练状态。
- 当前分数在128×128计算，不能宣称原始256×256分数相同。

## Git 与 AI 使用

本人完成「Version Control for Teams using Git」课程，准备课程要求的完成证据。能够解释branch、commit、merge、冲突解决及为何使用独立功能分支。

说明AI做了哪些代码/文档工作，自己如何检查、理解和修改；保留真实对话。不要把助手生成的学习总结当作课程完成或本人理解的证明。
