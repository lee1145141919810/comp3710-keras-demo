# COMP3710 Lab 2 演示（Demo）准备指南

本文件面向作者本人，用中文梳理：演示前必须完成的事项、每个部分需要展示什么、演示时大概率会被问到的问题及参考答案。评分细则（Demo rubric v2.0）的四项权重是：

| 评分项 | 权重 | 对应准备 |
|--------|------|----------|
| I. 代码能运行、任务完成比例 | 20 % | 所有脚本能跑通，GPU 部分在 Rangpur 上跑出结果 |
| II. 回答问题、体现理解与"所有权" | 40 % | 熟读下方问答；能逐行解释代码 |
| III. 优秀理解 + 编程规范 + 文档 + 合理使用 AI | 20 % | README、注释、commit 记录、`docs/AI_USAGE.md`、prompt 历史 |
| IV. 3 分钟总结（做了什么、用了什么工作流、意义） | 20 % | 准备一段口头总结（见文末） |

> 演示只能被评一次，务必在全部准备好后再申请。

---

## 0. 演示前必做清单

1. **GPU 运行**（本仓库在无 GPU 的机器上开发，只做了 CPU 冒烟测试，以下必须在 Rangpur 上真正跑出结果）：
   - `sbatch slurm/dawnbench.slurm`（Part 3.2，30 epochs），把 `results/part3_dawnbench/results.json` 中的准确率、时间填进 README 表格。
   - `sbatch slurm/unet.slurm`（Task 2），把 `results/part4_unet/dice_test.json` 的每类 DSC 填进 README；确认每类都 > 0.9。
   - `sbatch slurm/vae.slurm`（Task 1，latent 2），可再跑一次 `--latent_dim 32 --epochs 30` 得到 UMAP 图。
   - `sbatch slurm/gan.slurm`（Task 3），检查 `results/part4_gan/final_samples.png` 是否像脑子、`diversity.json` 的 `fake_over_real_ratio` 是否接近 1。
   - 把关键图复制到 `docs/figures/` 并在 README 中引用，`git commit` + `git push`。
2. **本地数据路径**：Mac 上数据在 `~/Downloads/keras_png_slices_data`，脚本会自动搜索该路径，也可 `--data_root` 指定。Mac 会自动用 MPS 加速。一键训练：
   ```bash
   bash scripts/run_oasis_local.sh
   ```
   CPU 部分（1 / 2 / 3.1）一键复现：
   ```bash
   bash scripts/run_cpu_parts.sh
   ```
3. **edX 短课程** "Version Control for Teams using Git"（Part 4.1，1 分）完成并保留证书截图。
4. **GitHub**：仓库必须在自己的账号下，commit 信息有意义（本仓库已按"每个 Part 一个 commit"组织，后续修改也保持这个习惯）。演示时可能要求登录账号证明所有权。
5. **AI 使用证据**：导出与 AI 的对话记录（分享链接或 `prompt_history.pdf`）。rubric 明确说"把题目复制给 AI 再复制答案"不算合理使用，要能说明你如何**迭代提示、验证并改进**了 AI 生成的代码（见 `docs/AI_USAGE.md`）。
6. **现场演示环境**：提前用 `bash slurm/interactive_gpu.sh` 申请交互式 GPU 节点，激活环境，确认下面三条命令都能在几分钟内跑完：
   ```bash
   python part3_cnn/dawnbench/train_cifar10.py --eval_only --checkpoint results/part3_dawnbench/resnet18_cifar10.pt --no_download
   python part3_cnn/dawnbench/train_cifar10.py --epochs 1 --no_download
   python part4_recognition/unet/predict.py --checkpoint results/part4_unet/unet_best.pt --data_root /home/groups/comp3710/OASIS --indices 0 100 250
   ```

---

## 1. Part 1 —— 离散傅里叶变换（1 分）

**运行**：`python part1_dft/square_wave_numpy.py`，`python part1_dft/dft_torch.py`

**展示**：`square_wave_reconstruction.png`（1/3/5/20/50 次谐波重建）、`dft_spectrum_*.png`、`dft_timings.png` 和终端里的计时表。

**可能的问题与答案**

- *增加谐波（20、50）有什么影响？* 边沿更陡、平台更平，平均误差从 0.34 降到 0.017；但跳变处的过冲不会消失，它收敛到跳变幅度的约 9 %（吉布斯现象），只是变窄。
- *DFT 得到的频率分量和构造时用的一样吗？有何差别，为什么？* 对 50 次谐波构造的信号，DFT 精确恢复奇次谐波 1,3,…,99 Hz，幅度 4/(πn)，误差 ~1e-16，偶次为 0——因为窗口内恰好整数个周期（`endpoint=False`）且带宽低于奈奎斯特频率，没有频谱泄漏。对理想方波 `np.sign`，其谐波无穷多，高于奈奎斯特频率（N/2T = 1024 Hz）的分量会**混叠**到低频，导致幅度出现 ~1e-6 的偏差、偶次 bin 不再为零。
- *三种方法的快慢顺序及原因？* FFT < GPU 张量 DFT < CPU 张量 DFT ≪ Python 双循环 DFT。FFT 是 O(N log N) 算法（分治利用旋转因子的对称性），其他都是 O(N²)；张量版把 N² 次乘加写成一个大矩阵乘（`cos/sin(2πkn/N) @ x`），在 CPU 上由 BLAS 向量化执行，在 GPU 上由上千核心并行；纯 Python 循环每次迭代都有解释器开销（约 0.5 µs），4 百万次迭代就需数秒。
- *改变数据规模时的变化？* N 翻倍，O(N²) 方法时间约 ×4（log-log 图斜率 2），FFT 只约 ×2。GPU 版在 N 很小时不一定比 CPU 快——每次 kernel 启动 / 同步有几十微秒固定开销，只有 N 足够大、计算量压过开销后优势才显现。
- *为什么 GPU 计时前要 `synchronize`？* CUDA kernel 是异步执行的，不同步测到的只是"提交任务"的时间；第一次调用还包含 CUDA 上下文初始化，因此要先做 warm-up。
- *代码里为什么用 `(k·n) mod N` 再算角度？* 在 float32 下 k·n 可达 1e7 以上会丢失精度，先用整数取模再转浮点，相位准确，且 Apple MPS 不支持 float64 / 复数，所以用两次实数矩阵乘（cos、sin）代替复数运算。

---

## 2. Part 2 —— Eigenfaces（1 分）

**运行**：`python part2_eigenfaces/eigenfaces.py`

**结果**：150 个主成分解释 94.7 % 方差；随机森林准确率 **0.612**（197/322）。

**可能的问题与答案**

- *PCA 和 SVD 的关系？* 对中心化数据矩阵 X 做 SVD：X = U S Vᵀ，V 的行就是协方差矩阵 XᵀX/(n−1) 的特征向量（主成分），特征值 λᵢ = Sᵢ²/(n−1)。`components = V[:150]`，reshape 回 50×37 就是 eigenfaces。
- *为什么用训练集的均值去中心化测试集？* 避免测试信息泄漏到模型中；模型（均值 + 基）只能从训练集学习。
- *compactness 图的含义？* 累计解释方差比；前 10 个分量就有 63 %，说明人脸数据高度冗余，可以从 1850 维压到 150 维几乎不丢信息。
- *为什么准确率只有 0.61？* PCA 是无监督的，保留的是像素方差最大的方向（光照、姿态），不一定是最能区分身份的方向；且类别极度不平衡（Bush 占 146/322），少数类如 Ariel Sharon 完全没被识别。
- *与 Part 1 的联系？* 都是"换一组基表示数据"：傅里叶变换用固定的正弦/余弦基，PCA 用从数据方差中学出的基。

---

## 3. Part 3.1 —— LFW CNN（1 分）

**运行**：`python part3_cnn/lfw_cnn.py --epochs 40`（CPU 16 秒）

**结果**：测试准确率 **0.941**，宏平均 F1 0.90，远高于 PCA+RF 的 0.612。

**可能的问题与答案**

- *网络结构？* Conv3×3(32)-BN-ReLU-MaxPool → Conv3×3(32)-BN-ReLU-MaxPool → Flatten → FC(128)-ReLU-Dropout(0.5) → FC(7)。两次池化把 50×37 变为 12×9，展平 32×12×9 = 3456 维。
- *为什么输入要变成 4D 张量？* 卷积层期望 `[N, C, H, W]`；灰度图 C=1，所以 `X[:, np.newaxis]`。LFW 像素已经在 [0,1]，无需再归一化。
- *sparse categorical cross-entropy 是什么？* 标签是整数类别索引而不是 one-hot，PyTorch 的 `F.cross_entropy` 直接接受整数标签，内部做 log-softmax + NLL。
- *为什么 CNN 比 PCA+RF 好？* CNN 端到端联合学习特征和分类器、保留 2D 空间结构、有平移不变性和层次化非线性特征；PCA 是与任务无关的线性投影，先展平丢掉了空间信息。
- *有过拟合吗？* 训练 acc 0.97 vs 测试 0.94，差距小；Dropout、BatchNorm 起到了正则化作用。

---

## 4. Part 3.2 —— DAWNBench：ResNet-18 + CIFAR-10（4 分）

**运行**：`sbatch slurm/dawnbench.slurm`；演示时 `--eval_only` 推理和 `--epochs 1` 单 epoch 训练必须在 Rangpur 上现场跑。

**要点**（`part3_cnn/dawnbench/`）

- `resnet.py`：自己实现的 ResNet-18（BasicBlock ×[2,2,2,2]，通道 64→512），CIFAR 版 stem 是单个 3×3 卷积、无 max-pool，保持 32×32 分辨率；残差分支最后一个 BN 的 γ 初始化为 0，让每个 block 起始为恒等映射，高学习率更稳定。
- `data.py`：整个数据集常驻 GPU（fp16 约 0.5 GB），随机裁剪（reflect pad 4）、水平翻转、8×8 cutout 全部用向量化张量操作对整个数据集一次完成，避免 DataLoader 成为瓶颈。
- `train_cifar10.py`：SGD + Nesterov 0.9，batch 512，one-cycle 分段线性学习率（0→0.4 用 5 个 epoch，然后线性降到 0），weight decay 5e-4（BN 和 bias 不加），label smoothing 0.1，混合精度（A100 用 bf16 autocast；V100 用 fp16 + GradScaler），channels-last 内存格式，可选翻转 TTA。每个 epoch 记录测试准确率和耗时，并报告首次达到 94 % 的时间。

**可能的问题与答案**

- *残差连接为什么有用？* y = F(x) + x，让网络学习残差；梯度可以通过恒等路径直接回传，解决深层网络退化/梯度消失问题。
- *混合精度为什么快、为什么安全？* Tensor Core 上 fp16/bf16 矩阵乘吞吐是 fp32 的数倍，显存带宽占用也减半；autocast 只把矩阵乘/卷积放到低精度，损失、BN 统计、权重更新仍用 fp32。fp16 动态范围小容易梯度下溢，需要 GradScaler 把损失放大再缩回；bf16 指数位与 fp32 相同，不需要缩放。
- *为什么 batch 512 配 lr 0.4？* 线性缩放法则：batch 128 的常用 lr 0.1 ×4。warm-up 让 BN 统计与动量先稳定，然后线性衰减到 0 有利于收敛到平坦极小值。
- *label smoothing / cutout 的作用？* 都是正则化：前者防止 logits 过度自信，后者随机遮挡迫使网络利用更多上下文，二者对 CIFAR 短训练各贡献约 0.3–0.5 %。
- *DAWNBench 衡量什么？* 达到目标准确率（94 %）所需的训练时间/成本，而不是最终精度。
- *为什么 CIFAR-10 的 ResNet 不用 7×7 stride-2 + maxpool 的 stem？* 32×32 图像经过那样的 stem 只剩 8×8，空间信息几乎丢光；3×3 stride-1 保留分辨率是 CIFAR 上的标准做法。

---

## 5. Part 4 —— OASIS 识别任务（8 分）

### 数据（`part4_recognition/oasis_data.py`）

- 9664 / 1120 / 544 张 256×256 灰度切片，文件名 `case_XXX_slice_Y.nii.png`，对应掩膜 `seg_XXX_slice_Y.nii.png`。
- 掩膜以灰度 0 / 85 / 170 / 255 存储四个类别 = 背景 / 脑脊液 CSF / 灰质 / 白质；用 `round(pixel/85)` 映射为 0–3，再用 `F.one_hot` 得到 one-hot。
- 一次性解码到内存（uint8），避免集群共享文件系统上每个 epoch 反复读 PNG。

### Task 1 —— VAE（最多 3 分）

- *VAE 与普通自编码器的区别？* 编码器输出的是分布参数 (μ, log σ²) 而非确定的编码；通过重参数化 z = μ + σ·ε 采样以保持可微；损失 = 重建项 + KL(q(z|x) ‖ N(0,I))，KL 项把后验拉向标准正态，使潜空间连续、可采样、可插值。
- *为什么损失叫 −ELBO？* 最大化证据下界 log p(x) ≥ E_q[log p(x|z)] − KL(q(z|x)‖p(z))；取负即为最小化的损失。BCE 重建项对应像素服从伯努利分布的假设（图像在 [0,1]）。
- *流形如何可视化？* latent_dim=2 时，在 N(0,1) 的分位数上取 15×15 网格解码（`manifold_latent2.png`），并把测试集编码后按切片序号着色（`latent_scatter_latent2.png`）：切片位置是主要变化因素，在潜空间形成连续轨迹。高维潜空间用 UMAP 降到 2D（`umap_*.png`），并沿两个主方向解码出一张 2D 切面（`manifold_pca_*.png`）。
- *β 的作用？* β>1 加强 KL、潜变量更解耦但重建更模糊；β<1 反之。
- *为什么重建偏模糊？* 像素级 BCE/MSE 是对所有可能输出取平均的最优解，加上 2 维瓶颈信息量极小；增大 latent_dim（如 32）可显著改善。

### Task 2 —— UNet（最多再加 2 分）

- *UNet 结构与跳跃连接的作用？* 编码器 4 次下采样（DoubleConv + MaxPool，通道 32→512）提取语义，解码器用转置卷积上采样并与同分辩率的编码特征拼接（skip connection），恢复被池化丢掉的精细边界信息，这是能画出锐利组织边界的关键。
- *输出为何是 4 通道？one-hot 在哪里？* 最后 1×1 卷积输出 4 个通道，softmax 后每个像素是 4 类的概率（categorical 输出），argmax 得到标签图；训练目标 mask 用 `labels_to_one_hot` 转成 one-hot 计算 Dice。
- *损失函数？* 交叉熵 + soft Dice。CE 逐像素优化但受类别不平衡影响（背景像素占大多数）；Dice 直接优化重叠度且对每类等权，小类（CSF）不会被淹没。
- *DSC 是什么，怎么算的？* DSC = 2|P∩G|/(|P|+|G|)。`DiceAccumulator` 在整个测试集上累加交集与面积再求比值（数据集级 DSC），避免某些切片不含某类导致的 0/0；同时也给出逐图平均。
- *如何证明 > 0.9？* `predict.py` 打印每类的数据集级 DSC 和逐图平均 DSC 并标注是否 > 0.9，`predictions_test.png` 展示 MRI / 真值 / 预测 / 误差图（误差集中在组织边界 1–2 像素处）。
- *如何防止过拟合、选模型？* 用验证集平均 DSC 选最佳 checkpoint，测试集只在最后评估一次；可选翻转与强度扰动增广；cosine 学习率衰减。
- *为什么用 bf16 混合精度？* 256×256 的 UNet 计算量大，A100 上 bf16 提速约 2 倍且不需要 GradScaler。

### Task 3 —— GAN（最多再加 2 分）

- *GAN 的原理？* 生成器 G 把噪声 z 映射为图像，判别器 D 判断真假；二者对抗（极小极大博弈），最终 D(x) ≈ 0.5 表示无法区分。代码用非饱和损失（G 最大化 log D(G(z))），梯度在训练初期更强。
- *DCGAN 设计要点？* 全卷积、转置卷积上采样、G 用 BN + ReLU、D 用 LeakyReLU、Adam β1=0.5、权重 N(0, 0.02) 初始化。
- *怎样稳定训练 / 解决 mode collapse？* 判别器每层加谱归一化（Spectral Norm，约束 Lipschitz 常数）；单侧标签平滑（真=0.9）；固定噪声每 epoch 出图观察演化；`diversity.json` 计算生成图之间的平均两两距离与真实图之间的比值（接近 1 说明多样性正常，≪ 1 说明塌缩），以及到最近训练图的距离（说明不是记忆训练集）。
- *如何判断结果"真实"？* 看 `final_samples.png`：脑室、灰白质轮廓、颅骨边缘是否合理，切片位置是否多样；`losses.png` 中 D(x)、D(G(z)) 应在 0.5 附近波动而非 D 一边倒。
- *如果结果不够好怎么办？* 先用 `--image_size 64` 训练验证流程，再升到 128；增加 epoch；适当调 `--lr_d`（TTUR：D 学习率略高于 G）。

---

## 6. 3 分钟口头总结（模板）

1. **做了什么**：完成 Lab 2 全部四部分——用 NumPy/PyTorch 实现方波傅里叶级数与 DFT 并做 CPU/GPU 计时对比；用 SVD 实现 Eigenfaces + 随机森林（0.61）；两层卷积 CNN 把 LFW 准确率提高到 0.94；从零实现 ResNet-18 + GPU 数据管线 + 混合精度做 DAWNBench（在 A100 上 __ 秒达到 __ %）；在 OASIS 上完成 VAE（潜空间流形/UMAP 可视化）、UNet（每类 DSC __ / __ / __ / __，均 > 0.9）和 DCGAN（生成脑切片，多样性比值 __）。
2. **工作流**：GitHub 仓库、每个 Part 一次有意义的 commit、`common/` 共享模块、argparse 脚本、结果统一写入 `results/`、SLURM 脚本提交 Rangpur。
3. **AI 使用**：用 AI 辅助起草代码和文档，随后逐个模块运行验证（形状/参数量检查、与 NumPy FFT 对照、合成数据冒烟测试），修正问题并整理成 `docs/AI_USAGE.md`，可提供对话记录。
4. **意义**：从"固定基（傅里叶）→ 数据驱动线性基（PCA）→ 端到端学习的非线性特征（CNN/ResNet）→ 生成式建模（VAE/GAN）与稠密预测（UNet）"，这正是模式识别从经典方法到深度学习的脉络。
