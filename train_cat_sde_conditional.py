"""
策略C: Class-Conditional SDE (Classifier-Free Guidance)
使用完整CIFAR-10数据集训练，生成时指定"猫"类别。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import torch
import torch.nn as nn
import ml_collections
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils as torchvision_utils
from tqdm import tqdm
from PIL import Image

from models import utils as mutils
from models import ncsnpp
from models.ema import ExponentialMovingAverage
from sde_lib import VESDE
from losses import get_optimizer, optimization_manager


class CIFAR10FullDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        if train:
            for i in range(1, 6):
                with open(os.path.join(root, f'data_batch_{i}'), 'rb') as f:
                    batch = pickle.load(f, encoding='latin1')
                    self.images.extend(batch['data'])
                    self.labels.extend(batch['labels'])
        else:
            with open(os.path.join(root, 'test_batch'), 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                self.images.extend(batch['data'])
                self.labels.extend(batch['labels'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


def get_cfg_loss_fn(sde, train=True, reduce_mean=False, eps=1e-5, num_classes=10):
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, labels):
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        sigma = std

        if train:
            mask = torch.rand(labels.shape[0], device=labels.device) < 0.1
            labels = torch.where(mask, torch.full_like(labels, num_classes), labels)

        score = model(perturbed_data, sigma, labels)

        losses = torch.square(score * std[:, None, None, None] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_config():
    config = ml_collections.ConfigDict()

    config.training = ml_collections.ConfigDict()
    config.training.sde = 'vesde'
    config.training.continuous = True
    config.training.batch_size = 8
    config.training.log_freq = 100
    config.training.eval_freq = 2000
    config.training.snapshot_freq = 2000
    config.training.n_iters = 30000
    config.training.likelihood_weighting = False
    config.training.reduce_mean = False

    config.model = ml_collections.ConfigDict()
    config.model.name = 'ncsnpp'
    config.model.scale_by_sigma = True
    config.model.ema_rate = 0.999
    config.model.normalization = 'GroupNorm'
    config.model.nonlinearity = 'swish'
    config.model.nf = 16
    config.model.ch_mult = (1, 2, 2, 2)
    config.model.num_res_blocks = 2
    config.model.attn_resolutions = (16,)
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
    config.data.dataset = 'CIFAR10_Full'
    config.data.image_size = 32
    config.data.random_flip = True
    config.data.uniform_dequantization = True
    config.data.num_channels = 3
    config.data.centered = True
    config.data.num_classes = 10

    config.optim = ml_collections.ConfigDict()
    config.optim.weight_decay = 0
    config.optim.optimizer = 'Adam'
    config.optim.lr = 5e-4
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.warmup = 500
    config.optim.grad_clip = 1.0

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return config


def train_conditional_sde():
    exp_name = 'cifar10_cat_sde_conditional'
    workdir = os.path.join('experiments', exp_name)
    os.makedirs(workdir, exist_ok=True)

    config = get_config()
    print(f'=== CIFAR-10 Class-Conditional SDE Training (Strategy C) ===')
    print(f'Config: SDE={config.training.sde}, LR={config.optim.lr}, BS={config.training.batch_size}, Iters={config.training.n_iters}, NF={config.model.nf}')
    print(f'CFG dropout: 10%')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (torch.rand_like(x) + x * 255.) / 256. if config.data.uniform_dequantization else x),
        transforms.Lambda(lambda x: x * 2. - 1. if config.data.centered else x)
    ])

    dataset = CIFAR10FullDataset(
        root=os.path.join(os.path.dirname(__file__), 'data/cifar-10-batches-py'),
        train=True, transform=transform
    )
    train_loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)

    print(f'Dataset size: {len(dataset)} images (ALL 10 classes)')

    # 直接使用修改后的NCSN++模型
    score_model = ncsnpp.NCSNpp(config)
    score_model = score_model.to(config.device)
    score_model.train()

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = get_optimizer(config, score_model.parameters())
    optimize_fn = optimization_manager(config)

    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    loss_fn = get_cfg_loss_fn(sde, train=True, reduce_mean=config.training.reduce_mean,
                               num_classes=config.data.num_classes)

    train_iter = iter(train_loader)
    start_time = time.time()
    results = {'train_losses': [], 'eval_losses': [], 'steps': []}
    pbar = tqdm(total=config.training.n_iters, desc=f'Training {exp_name}')

    for step in range(config.training.n_iters):
        try:
            batch, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch, labels = next(train_iter)

        batch = batch.to(config.device)
        labels = labels.to(config.device)

        optimizer.zero_grad()
        loss = loss_fn(score_model, batch, labels)
        loss = loss.mean()
        loss.backward()
        optimize_fn(optimizer, score_model.parameters(), step=step)
        ema.update(score_model.parameters())

        if (step + 1) % config.training.log_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            mem_used = torch.cuda.memory_allocated(config.device) / 1024**2
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{lr:.6f}', 'Mem': f'{mem_used:.0f}MB'})
            results['train_losses'].append(loss.item())
            results['steps'].append(step)

        if (step + 1) % config.training.eval_freq == 0:
            score_model.eval()
            with torch.no_grad():
                eval_loss = loss_fn(score_model, batch, labels)
            score_model.train()
            print(f'\nStep [{step+1}] Eval Loss: {eval_loss.mean().item():.4f}')
            results['eval_losses'].append(eval_loss.mean().item())

        if (step + 1) % config.training.snapshot_freq == 0:
            ckpt_dir = os.path.join(workdir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'model_{step+1}.pth')
            torch.save({
                'model': score_model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step + 1,
                'config': config
            }, ckpt_path)
            print(f'Checkpoint saved to {ckpt_path}')

        pbar.update(1)

    pbar.close()
    results['final_train_loss'] = loss.item()
    results['total_time'] = time.time() - start_time

    with open(os.path.join(workdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    summary_path = os.path.join('experiments', 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}

    summary[exp_name] = {
        'config': {
            'sde_type': 'vesde',
            'lr': config.optim.lr,
            'batch_size': config.training.batch_size,
            'n_iters': config.training.n_iters,
            'model_nf': config.model.nf,
            'warmup': config.optim.warmup,
            'strategy': 'C_class_conditional',
            'data_size': 50000,
            'cfg_dropout': 0.1
        },
        'final_train_loss': results['final_train_loss'],
        'avg_eval_loss': np.mean(results['eval_losses']) if results['eval_losses'] else None,
        'total_time': results['total_time']
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nTraining complete! Final Loss: {results["final_train_loss"]:.4f}, Time: {results["total_time"]:.2f}s')
    return results


def generate_cat_samples_cfg(ckpt_step=30000, num_samples=36, guidance_scale=3.0):
    exp_name = 'cifar10_cat_sde_conditional'
    workdir = os.path.join('experiments', exp_name)
    ckpt_path = os.path.join(workdir, 'checkpoints', f'model_{ckpt_step}.pth')

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    num_classes = config.data.num_classes
    cat_label = 3

    score_model = ncsnpp.NCSNpp(config)
    score_model = score_model.to(config.device)
    score_model.load_state_dict(ckpt['model'])
    score_model.eval()

    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)

    def inverse_scaler(x):
        if config.data.centered:
            x = (x + 1.) / 2.
        return torch.clamp(x, 0., 1.)

    print(f"Generating {num_samples} cat samples with CFG (guidance_scale={guidance_scale})...")

    shape = (num_samples, 3, config.data.image_size, config.data.image_size)
    device = config.device

    with torch.no_grad():
        x = sde.prior_sampling(shape).to(device) * sde.sigma_max

        cond_labels = torch.full((num_samples,), cat_label, device=device, dtype=torch.long)
        uncond_labels = torch.full((num_samples,), num_classes, device=device, dtype=torch.long)

        sigmas = sde.discrete_sigmas.to(device)

        for i in tqdm(range(sde.N - 1, 0, -1), desc='Sampling'):
            sigma = sigmas[i]
            adjacent_sigma = sigmas[i - 1]
            vec_sigma = torch.full((num_samples,), sigma, device=device)

            score_cond = score_model(x, vec_sigma, cond_labels)
            score_uncond = score_model(x, vec_sigma, uncond_labels)
            score = score_uncond + guidance_scale * (score_cond - score_uncond)

            dt = sigma ** 2 - adjacent_sigma ** 2
            x_mean = x + score * dt
            std = torch.sqrt((adjacent_sigma ** 2 * dt) / (sigma ** 2))
            noise = torch.randn_like(x)
            x = x_mean + std * noise

        vec_sigma = torch.full((num_samples,), sigmas[0], device=device)
        score_cond = score_model(x, vec_sigma, cond_labels)
        score_uncond = score_model(x, vec_sigma, uncond_labels)
        score = score_uncond + guidance_scale * (score_cond - score_uncond)
        x = x + score * sigmas[0] ** 2

    samples = inverse_scaler(x)
    grid = torchvision_utils.make_grid(samples, nrow=6, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)

    save_path = os.path.join(workdir, f'cat_cfg_samples_{ckpt_step}_gs{guidance_scale}.png')
    Image.fromarray(grid_np).save(save_path)
    print(f"Samples saved to {save_path}")
    return save_path


if __name__ == '__main__':
    train_conditional_sde()
    generate_cat_samples_cfg(ckpt_step=30000, num_samples=36, guidance_scale=3.0)