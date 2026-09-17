> 历史文档：当前 Windows/GAN/DAWNBench 状态请先看 [CURRENT_STATUS_zh.md](CURRENT_STATUS_zh.md)。下文“未选 GAN”“尚未连接”等描述不再代表最新进度。

# 下一阶段：连接 Rangpur、完成 Git 课程、练习答辩

## 当前第一步：确认登录方法

你已经有UQ账号，但尚未连接Rangpur。实验PDF指定的入口是：
[EAIT Compute 官方说明](https://student.eait.uq.edu.au/infrastructure/compute/)。
已依据用户贴出的官方说明确认：登录节点是 `rangpur.compute.eait.uq.edu.au`，只接受校内网络SSH连接；校外可通过 `remote.labs.eait.uq.edu.au` 跳转或使用学校规定的VPN。

Mac终端命令（将两处 `YOUR_UQ_USERNAME` 替换为你的UQ登录用户名）：

```bash
# 校外或希望使用跳板机时：
ssh -J YOUR_UQ_USERNAME@remote.labs.eait.uq.edu.au YOUR_UQ_USERNAME@rangpur.compute.eait.uq.edu.au

# 已在校内网络、可以直连时：
ssh YOUR_UQ_USERNAME@rangpur.compute.eait.uq.edu.au
```

官方提供的Rangpur主机指纹：`SHA256:yEOt0cJWMOC5rlpiEtzKO+kJ+EZHB1lvuTVXIT62bUw`。
这个指纹仅用于Rangpur，不适用于跳板机。跳板机首次连接的指纹须按学校的远程访问说明核对。

官方区分 `a100-test`（最长20分钟，`--gres=shard:N`，每GPU共4个shard）与 `a100`（`--gres=gpu:1`）。本项目的交互脚本在测试分区请求全部4个shard以容纳ML负载；正式DAWNBench训练使用 `a100`。登录后仍用 `sinfo` 核对即时可用情况。

在Mac上使用系统“终端”执行文档中的SSH命令。密码输入时通常不会显示字符。首次连接出现主机指纹时，应与官方说明或IT提供的信息核对。不要通过关闭主机验证来绕过指纹不匹配。

连接成功后先运行这些只读命令，把输出提供给助手：

```bash
hostname
pwd
command -v sinfo
sinfo -o '%P %a %l %G'
```

若连接失败，提供完整报错即可，不提供密码、验证码或私钥。先区分主机无法解析、连接超时、认证失败和无集群权限，再处理对应问题。

## 连接确认后：下载项目与检查环境

从集群文档允许的工作目录执行：

```bash
git clone --branch codex/medium-rubric-review https://github.com/lee1145141919810/comp3710-keras-demo.git
cd comp3710-keras-demo
bash scripts/cluster_preflight.sh login
```

先确认集群提供的Python/CUDA模块或课程环境，再安装依赖。`docs/results/local_environment.txt`仅记录本地CPU实验环境，不是集群CUDA安装方案。
将本次交付的 `medium_models_and_results.zip` 通过学校允许的SFTP/scp路径上传并在仓库根目录解压。不要将完整MRI数据或模型推入Git历史。

按 `sinfo` 的实际输出选择GPU分区后：

```bash
# 将下一行的实际分区名替换为 sinfo 中核实的值。
bash slurm/interactive_gpu.sh ACTUAL_GPU_PARTITION
# 激活课程提供的环境或自己的已安装环境后：
bash scripts/cluster_preflight.sh gpu
```

接着验证CIFAR数据是否完整。在文档允许联网的位置下载数据，再在计算节点使用 `--no_download`。确认GPU环境与数据后，先单轮测试，再完整训练：

```bash
python part3_cnn/dawnbench/train_cifar10.py --device cuda --no_download --epochs 1 --output_dir results/cifar_cluster_smoke
python part3_cnn/dawnbench/train_cifar10.py --device cuda --no_download --tta --output_dir results/part3_dawnbench
```

完整训练如超过交互作业时限，使用 `sbatch --partition=ACTUAL_GPU_PARTITION slurm/dawnbench.slurm`；先激活/配置该脚本中的实际环境。不要在登录节点直接开始训练。

核对 `results.json` 中准确率、设备、样本数和计时范围。目标为>90%，高分目标94%及约360秒参考时间；未达到就记录实际结果，通过后再整理现场演示。不得把小样本、随机权重或单轮准确率作为完整挑战结果。

## Git 短课程：需要你亲自学习完成

实验表指定的是 **Version Control for Teams using Git**，不是另一个入门Git课程。使用本学期Blackboard或Ed Discussion发布的入口，按课程指引关联UQ账号；先核对校内访问方式，不要因为找不到入口就购买证书。

完成学习与测验后，保留能显示课程名称、本人账号和完成状态的页面或课程要求的证明。报告暂标“未完成”，直到取得实际证据。助手可以解释Git概念、陪你在本项目的练习分支演练冲突解决，但不会把未完成的课程写成完成。

## 答辩练习：先用自己的话回答

1. UNet四类DSC都超过0.9，是否意味着每张图每个类别都超过0.9？为什么？
2. 为什么用验证集最弱类别选模型，而不是挑测试集表现最好的一轮？
3. VAE的mu、log_var和重参数化分别做什么？去掉KL项后会怎样？
4. 为什么朴素GPU DFT仍是O(N²)，不能保证总比CPU FFT快？
5. 现场单轮训练为什么要使用单独的输出目录？

每次回答一个问题，再根据你的答案追问；以实际代码和实测结果说明，不背诵保证满分的话术。
