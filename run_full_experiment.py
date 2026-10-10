import os
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
import ml_collections
import time
import json
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import utils as mutils
from models import ncsnpp
from models.ema import ExponentialMovingAverage
from sde_lib import VESDE, VPSDE, subVPSDE
from losses import get_optimizer, get_sde_loss_fn, optimization_manager


def get_config(sde_type='vesde', lr=2e-4, batch_size=4, n_iters=2000,
               model_nf=16, warmup=500, grad_clip=1.0):
    config = ml_collections.ConfigDict()
    config.training = ml_collections.ConfigDict()
    config.training.sde = sde_type
    config.training.continuous = True
    config.training.batch_size = batch_size
    config.training.log_freq = 50
    config.training.eval_freq = 200
    config.training.snapshot_freq = 1000
    config.training.n_iters = n_iters
    config.training.likelihood_weighting = False
    config.training.reduce_mean = False

    config.model = ml_collections.ConfigDict()
    config.model.name = 'ncsnpp'
    config.model.scale_by_sigma = True
    config.model.ema_rate = 0.999
    config.model.normalization = 'GroupNorm'
    config.model.nonlinearity = 'swish'
    config.model.nf = model_nf
    config.model.ch_mult = (1, 2)
    config.model.num_res_blocks = 1
    config.model.attn_resolutions = ()
    config.model.resamp_with_conv = True
    config.model.conditional = True
    config.model.fir = True
    config.model.fir_kernel = [1, 3, 3, 1]
    config.model.skip_rescale = True
    config.model.resblock_type = 'biggan'
    config.model.progressive = 'none'
    config.model.progressive_input = 'residual'
    config.model.progressive_combine = 'sum'
    config.model.attention_type = 'ddpm'
    config.model.init_scale = 0.
    config.model.fourier_scale = 16
    config.model.conv_size = 3
    config.model.sigma_min = 0.01
    config.model.sigma_max = 50
    config.model.num_scales = 1000
    config.model.beta_min = 0.1
    config.model.beta_max = 20
    config.model.dropout = 0.1
    config.model.embedding_type = 'fourier'

    config.data = ml_collections.ConfigDict()
    config.data.dataset = 'CIFAR10'
    config.data.image_size = 32
    config.data.random_flip = True
    config.data.uniform_dequantization = True
    config.data.num_channels = 3
    config.data.centered = False

    config.optim = ml_collections.ConfigDict()
    config.optim.weight_decay = 0
    config.optim.optimizer = 'Adam'
    config.optim.lr = lr
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.warmup = warmup
    config.optim.grad_clip = grad_clip

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return config


def get_cifar10_dataloader(config, train=True, subset_size=None):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (torch.rand_like(x) + x * 255.) / 256. if config.data.uniform_dequantization else x),
        transforms.Lambda(lambda x: x * 2. - 1. if config.data.centered else x)
    ])

    dataset = datasets.CIFAR10(
        root='./data', train=train, download=False, transform=transform
    )

    if subset_size is not None and subset_size < len(dataset):
        indices = np.random.RandomState(config.seed).permutation(len(dataset))[:subset_size]
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=train,
        num_workers=0, pin_memory=True, drop_last=True
    )
    return dataloader


def train(config, workdir):
    os.makedirs(workdir, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    score_model = mutils.create_model(config)
    score_model = score_model.to(config.device)
    score_model.train()

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = get_optimizer(config, score_model.parameters())
    optimize_fn = optimization_manager(config)

    if config.training.sde.lower() == 'vpsde':
        sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'subvpsde':
        sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'vesde':
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    loss_fn = get_sde_loss_fn(sde, train=True, reduce_mean=config.training.reduce_mean,
                              continuous=config.training.continuous, likelihood_weighting=config.training.likelihood_weighting)

    train_loader = get_cifar10_dataloader(config, train=True, subset_size=5000)
    eval_loader = get_cifar10_dataloader(config, train=False, subset_size=1000)

    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)

    print(f"\n{'='*60}")
    print(f"Training: SDE={config.training.sde}, LR={config.optim.lr}, BS={config.training.batch_size}")
    print(f"Model: nf={config.model.nf}, params={sum(p.numel() for p in score_model.parameters()):,}")
    print(f"Device: {config.device}")
    print(f"{'='*60}")

    start_time = time.time()
    results = {'train_losses': [], 'eval_losses': [], 'steps': []}

    for step in range(config.training.n_iters):
        try:
            batch, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch, _ = next(train_iter)

        batch = batch.to(config.device)

        optimizer.zero_grad()
        loss = loss_fn(score_model, batch)
        loss.backward()
        optimize_fn(optimizer, score_model.parameters(), step=step)
        ema.update(score_model.parameters())

        if (step + 1) % config.training.log_freq == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print(f"Step [{step+1}/{config.training.n_iters}] | Loss: {loss.item():.4f} | LR: {lr:.6f} | Time: {elapsed:.2f}s")
            results['train_losses'].append(loss.item())
            results['steps'].append(step)

        if (step + 1) % config.training.eval_freq == 0:
            try:
                eval_batch, _ = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_loader)
                eval_batch, _ = next(eval_iter)

            eval_batch = eval_batch.to(config.device)
            score_model.eval()
            with torch.no_grad():
                eval_loss = loss_fn(score_model, eval_batch)
            score_model.train()

            print(f"Step [{step+1}] Eval Loss: {eval_loss.item():.4f}")
            results['eval_losses'].append(eval_loss.item())

        if (step + 1) % config.training.snapshot_freq == 0:
            ckpt_dir = os.path.join(workdir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'model_{step+1}.pth')
            torch.save({
                'model': score_model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step + 1,
                'config': config
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    results['final_train_loss'] = loss.item()
    results['total_time'] = time.time() - start_time

    result_path = os.path.join(workdir, 'results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Training complete! Total time: {results['total_time']:.2f}s")
    return results


def run_experiments():
    experiments = [
        {
            'name': 'vesde_optimized_full',
            'sde_type': 'vesde',
            'lr': 5e-4,
            'batch_size': 32,
            'n_iters': 50000,
            'model_nf': 32,
            'warmup': 500
        }
    ]

    all_results = {}

    for exp in experiments:
        print(f"\n{'#'*70}")
        print(f"Running experiment: {exp['name']}")
        print(f"{'#'*70}")

        warmup = exp.get('warmup', 500)
        config = get_config(
            sde_type=exp['sde_type'],
            lr=exp['lr'],
            batch_size=exp['batch_size'],
            n_iters=exp['n_iters'],
            model_nf=exp['model_nf'],
            warmup=warmup
        )

        workdir = os.path.join('experiments', exp['name'])
        try:
            results = train(config, workdir)
            all_results[exp['name']] = {
                'config': {k: v for k, v in exp.items()},
                'final_train_loss': results['final_train_loss'],
                'total_time': results['total_time'],
                'avg_eval_loss': np.mean(results['eval_losses']) if results['eval_losses'] else None
            }
        except Exception as e:
            print(f"Experiment {exp['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp['name']] = {'error': str(e)}

    summary_path = os.path.join('experiments', 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<25} {'SDE':<10} {'LR':<8} {'NF':<4} {'Loss':<10} {'Time':<8}")
    print(f"{'':-<70}")
    for name, result in all_results.items():
        if 'error' in result:
            print(f"{name:<25} {'ERROR':<10} {'':<8} {'':<4} {result['error']}")
        else:
            print(f"{name:<25} {result['config']['sde_type']:<10} {result['config']['lr']:<8.1e} {result['config']['model_nf']:<4} {result['final_train_loss']:<10.4f} {result['total_time']:<8.1f}s")

    return all_results


if __name__ == "__main__":
    run_experiments()
