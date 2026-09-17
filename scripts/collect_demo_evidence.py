"""Run DFT, VAE, UNet and GAN checks; optionally add Rangpur CIFAR inference/training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--gan_checkpoint', type=Path, required=True)
    p.add_argument('--vae_checkpoint', type=Path, default=ROOT / 'results/medium_vae64/vae_latent2.pt')
    p.add_argument('--unet_checkpoint', type=Path, default=ROOT / 'results/medium_unet128_refine/unet_best.pt')
    p.add_argument('--cluster', action='store_true')
    p.add_argument('--cifar_checkpoint', type=Path)
    p.add_argument('--cifar_data', type=Path)
    a = p.parse_args()
    for name in ['data_root', 'output_dir', 'gan_checkpoint', 'vae_checkpoint', 'unet_checkpoint', 'cifar_checkpoint', 'cifar_data']:
        value = getattr(a, name)
        if value is not None:
            setattr(a, name, value.expanduser().resolve())
    if not torch.cuda.is_available():
        p.error('An actual CUDA GPU is required')
    if a.cluster and not os.environ.get('SLURM_JOB_ID'):
        p.error('--cluster must run inside an allocated Slurm job')
    if a.cluster and (not a.cifar_checkpoint or not a.cifar_data):
        p.error('--cluster requires --cifar_checkpoint and --cifar_data')
    checkpoints = {name: getattr(a, name) for name in ['vae_checkpoint', 'unet_checkpoint', 'gan_checkpoint']}
    if a.cifar_checkpoint:
        checkpoints['cifar_checkpoint'] = a.cifar_checkpoint
    for path in checkpoints.values():
        if not path.is_file():
            p.error(f'Missing checkpoint: {path}')
    if not a.data_root.is_dir():
        p.error('Missing OASIS data directory')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(a.output_dir / 'matplotlib_cache')
    report = {'status': 'running', 'scope': 'cluster_requested' if a.cluster else 'local',
              'hostname': socket.getfqdn(), 'platform': platform.platform(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'python': sys.version, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'checkpoints': {k: {'path': str(v), 'sha256': hashlib.sha256(v.read_bytes()).hexdigest()} for k, v in checkpoints.items()},
              'stages': []}
    def save():
        (a.output_dir / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    def run(name, script, args):
        command = [sys.executable, '-u', str(ROOT / script), *map(str, args)]
        start = time.perf_counter()
        print(f'Running {name}', flush=True)
        with (a.output_dir / f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['stages'].append({'name': name, 'command': command, 'exit_code': result.returncode,
                                 'wall_seconds': time.perf_counter() - start})
        save()
        if result.returncode:
            report['status'] = 'failed'
            save()
            raise SystemExit(f'{name} failed; inspect {a.output_dir / (name + ".log")}')
    run('dft', 'part1_dft/dft_torch.py', ['--device', 'cuda', '--sizes', 256, 512, 1024, 2048, 4096, 8192,
        '--repeats', 3, '--max_naive_n', 1024, '--output_dir', a.output_dir / 'dft'])
    run('vae', 'part4_recognition/vae/visualise.py', ['--device', 'cuda', '--checkpoint', a.vae_checkpoint,
        '--data_root', a.data_root, '--output_dir', a.output_dir / 'vae'])
    run('unet', 'part4_recognition/unet/predict.py', ['--device', 'cuda', '--checkpoint', a.unet_checkpoint,
        '--data_root', a.data_root, '--output_dir', a.output_dir / 'unet'])
    run('gan', 'part4_recognition/gan/sample.py', ['--device', 'cuda', '--checkpoint', a.gan_checkpoint,
        '--output_dir', a.output_dir / 'gan'])
    if a.cluster:
        common = ['--device', 'cuda', '--no_download', '--data_dir', a.cifar_data, '--checkpoint', a.cifar_checkpoint]
        run('cifar_inference', 'part3_cnn/dawnbench/train_cifar10.py', common + ['--eval_only', '--tta', '--output_dir', a.output_dir / 'cifar_inference'])
        run('cifar_epoch', 'part3_cnn/dawnbench/train_cifar10.py', common + ['--epochs', 1, '--output_dir', a.output_dir / 'cifar_epoch'])
    report['status'] = 'completed'
    save()
    print(f'Completed; evidence saved to {a.output_dir}')


if __name__ == '__main__':
    main()
