import os
import sys
sys.path.insert(0, '.')

import json
import time
import torch
import ml_collections
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils as torchvision_utils
from tqdm import tqdm
from PIL import Image

from models import utils as mutils
from models import ncsnpp
from models.ema import ExponentialMovingAverage
from sde_lib import VESDE
from losses import get_optimizer, get_sde_loss_fn, optimization_manager


class CatDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.OxfordIIITPet(root=root, download=True, transform=transform)
        cat_breeds = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 
                      'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
                      'Siamese', 'Sphynx']
        cat_name_to_idx = {name: idx for idx, name in enumerate(self.dataset.classes)}
        self.cat_indices = [i for i in range(len(self.dataset)) 
                           if self.dataset.classes[self.dataset._labels[i]] in cat_breeds]
    
    def __len__(self):
        return len(self.cat_indices)
    
    def __getitem__(self, idx):
        original_idx = self.cat_indices[idx]
        image, _ = self.dataset[original_idx]
        return image, 0


def get_config():
    config = ml_collections.ConfigDict()
    
    config.training = ml_collections.ConfigDict()
    config.training.sde = 'vesde'
    config.training.continuous = True
    config.training.batch_size = 4
    config.training.log_freq = 100
    config.training.eval_freq = 2000
    config.training.snapshot_freq = 2000
    config.training.n_iters = 20000
    config.training.likelihood_weighting = False
    config.training.reduce_mean = False

    config.model = ml_collections.ConfigDict()
    config.model.name = 'ncsnpp'
    config.model.scale_by_sigma = True
    config.model.ema_rate = 0.999
    config.model.normalization = 'GroupNorm'
    config.model.nonlinearity = 'swish'
    config.model.nf = 32
    config.model.ch_mult = (1, 2, 4)
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
    config.data.dataset = 'Cat'
    config.data.image_size = 64
    config.data.random_flip = True
    config.data.uniform_dequantization = True
    config.data.num_channels = 3
    config.data.centered = False

    config.optim = ml_collections.ConfigDict()
    config.optim.weight_decay = 0
    config.optim.optimizer = 'Adam'
    config.optim.lr = 2e-4
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.warmup = 500
    config.optim.grad_clip = 1.0

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return config


def train_cat_generator():
    exp_name = 'cat_generator_vesde'
    workdir = os.path.join('experiments', exp_name)
    os.makedirs(workdir, exist_ok=True)
    
    config = get_config()
    print(f'=== Cat Generator Training ===')
    print(f'Config: SDE={config.training.sde}, LR={config.optim.lr}, BS={config.training.batch_size}, Iters={config.training.n_iters}, NF={config.model.nf}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Image Size: {config.data.image_size}x{config.data.image_size}')

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (torch.rand_like(x) + x * 255.) / 256. if config.data.uniform_dequantization else x),
        transforms.Lambda(lambda x: x * 2. - 1. if config.data.centered else x)
    ])
    
    dataset = CatDataset(root='./data', transform=transform)
    train_loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    print(f'Dataset size: {len(dataset)} cat images')

    score_model = mutils.create_model(config)
    score_model = score_model.to(config.device)
    score_model.train()

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = get_optimizer(config, score_model.parameters())
    optimize_fn = optimization_manager(config)

    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    loss_fn = get_sde_loss_fn(sde, train=True, reduce_mean=config.training.reduce_mean, continuous=config.training.continuous, likelihood_weighting=config.training.likelihood_weighting)

    train_iter = iter(train_loader)

    start_time = time.time()
    results = {'train_losses': [], 'eval_losses': [], 'steps': []}
    pbar = tqdm(total=config.training.n_iters, desc=f'Training {exp_name}')

    for step in range(config.training.n_iters):
        try:
            batch, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch, _ = next(train_iter)

        batch = batch.to(config.device)
        optimizer.zero_grad()
        loss = loss_fn(score_model, batch)
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
            eval_batch = batch
            score_model.eval()
            with torch.no_grad():
                eval_loss = loss_fn(score_model, eval_batch)
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
            'image_size': config.data.image_size,
            'warmup': config.optim.warmup
        },
        'final_train_loss': results['final_train_loss'],
        'avg_eval_loss': np.mean(results['eval_losses']) if results['eval_losses'] else None,
        'total_time': results['total_time']
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nTraining complete! Final Loss: {results["final_train_loss"]:.4f}, Time: {results["total_time"]:.2f}s')
    return results


def generate_cat_samples(ckpt_step=20000, num_samples=16):
    exp_name = 'cat_generator_vesde'
    workdir = os.path.join('experiments', exp_name)
    ckpt_path = os.path.join(workdir, 'checkpoints', f'model_{ckpt_step}.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    
    config.sampling = ml_collections.ConfigDict()
    config.sampling.method = 'pc'
    config.sampling.predictor = 'euler_maruyama'
    config.sampling.corrector = 'langevin'
    config.sampling.noise_removal = True
    config.sampling.snr = 0.16
    config.sampling.n_steps_each = 1
    config.sampling.probability_flow = False

    score_model = mutils.create_model(config)
    score_model = score_model.to(config.device)
    score_model.load_state_dict(ckpt['model'])
    score_model.eval()

    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)

    def inverse_scaler(x):
        if config.data.centered:
            x = (x + 1.) / 2.
        return torch.clamp(x, 0., 1.)

    from sampling import get_sampling_fn, get_pc_sampler, _PREDICTORS, _CORRECTORS
    
    predictor = _PREDICTORS['euler_maruyama']
    corrector = _CORRECTORS['langevin']
    
    shape = (num_samples, 3, config.data.image_size, config.data.image_size)
    
    sampler_fn = get_pc_sampler(
        sde, shape, predictor, corrector, inverse_scaler,
        snr=config.sampling.snr, n_steps=config.sampling.n_steps_each,
        probability_flow=config.sampling.probability_flow,
        continuous=config.training.continuous,
        denoise=config.sampling.noise_removal,
        eps=1e-5,
        device=config.device
    )

    print(f"Generating {num_samples} cat samples...")
    with torch.no_grad():
        samples, _ = sampler_fn(score_model)

    grid = torchvision_utils.make_grid(samples, nrow=4, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    
    save_path = os.path.join(workdir, f'cat_samples_{ckpt_step}.png')
    Image.fromarray(grid_np).save(save_path)
    print(f"Samples saved to {save_path}")
    return save_path


if __name__ == '__main__':
    train_cat_generator()
    generate_cat_samples(ckpt_step=20000, num_samples=16)
