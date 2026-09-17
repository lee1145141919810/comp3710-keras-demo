#!/bin/bash
# Grab an interactive GPU shell on Rangpur for the live demonstration
# (inference + one epoch of training must be shown running on the cluster).
#
#   bash slurm/interactive_gpu.sh
#
# Then, inside the allocated shell:
#   source .venv/bin/activate        # or: conda activate comp3710
#   python part3_cnn/dawnbench/train_cifar10.py --eval_only --checkpoint results/part3_dawnbench/resnet18_cifar10.pt --no_download
#   python part3_cnn/dawnbench/train_cifar10.py --epochs 1 --no_download
#   python part4_recognition/unet/predict.py --checkpoint results/part4_unet/unet_best.pt --data_root /home/groups/comp3710/OASIS

srun --partition=a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --pty bash -l
