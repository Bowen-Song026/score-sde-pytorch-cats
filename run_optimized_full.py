import os
import sys
sys.path.insert(0, '.')

import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
import ml_collections

from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE, VPSDE, subVPSDE
from losses import get_loss_fn


def create_config(sde_type='vesde', lr=5e-4, batch_size=32, n_iters=50000, 
                  model_nf=32, warmup=500, eval_freq=5000):
    config = ml_collections.ConfigDict()
    
    config.training = ml_collections.ConfigDict()
    config.training.continuous = True
    config.training.sde = sde_type
    config.training.lr = lr
    config.training.batch_size = batch_size
    config.training.n_iters = n_iters
    config.training.eval_freq = eval_freq
    config.training.snapshot_freq = eval_freq
    config.training.warmup = warmup
    config.training.log_freq = 100
    
    config.model = ml_collections.ConfigDict()
    config.model.name = 'ncsnpp'
    config.model.scale_by_sigma = True
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
    config.data.image_size = 32
    config.data.num_channels = 3
    config.data.centered = False
    config.data.dataset = 'CIFAR10'
    
    config.device = torch.device('cuda:0')
    
    return config


def train_optimized():
    exp_name = 'vesde_optimized_full'
    
    print(f"\n{'='*70}")
    print(f"Training {exp_name} with MAX iterations")
    print(f"{'='*70}")
    
    config = create_config(
        sde_type='vesde',
        lr=5e-4,
        batch_size=32,
        n_iters=50000,
        model_nf=32,
        warmup=500,
        eval_freq=5000
    )
    
    print(f"Config:\n  SDE: {config.training.sde}")
    print(f"  LR: {config.training.lr}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Iterations: {config.training.n_iters}")
    print(f"  Model NF: {config.model.nf}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    exp_dir = os.path.join('experiments', exp_name)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    transform = Compose([ToTensor()])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, 
                             shuffle=True, num_workers=4, drop_last=True)
    eval_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, 
                            shuffle=False, num_workers=4)
    
    score_model = mutils.create_model(config)
    score_model = score_model.to(config.device)
    
    optimizer = torch.optim.Adam(score_model.parameters(), lr=config.training.lr)
    
    if config.training.sde.lower() == 'vpsde':
        sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'subvpsde':
        sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'vesde':
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    loss_fn = get_loss_fn(sde, train=True, reduce_mean=False, continuous=True)
    
    def data_transform(x):
        x = 2 * x - 1.
        return x
    
    def inverse_data_transform(x):
        x = (x + 1.) / 2.
        return x
    
    train_losses = []
    eval_losses = []
    
    start_time = time.time()
    pbar = tqdm(total=config.training.n_iters, desc=f"Training {exp_name}")
    
    data_iter = iter(train_loader)
    
    for step in range(config.training.n_iters):
        try:
            batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch, _ = next(data_iter)
        
        batch = data_transform(batch).to(config.device)
        
        if step < config.training.warmup:
            lr = config.training.lr * (step + 1) / config.training.warmup
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        loss = loss_fn(score_model, batch)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (step + 1) % config.training.log_freq == 0:
            avg_train_loss = sum(train_losses[-100:]) / min(100, len(train_losses))
            mem_used = torch.cuda.memory_allocated(config.device) / 1024**2
            pbar.set_postfix({'Loss': f'{avg_train_loss:.4f}', 'Mem': f'{mem_used:.0f}MB'})
        
        if (step + 1) % config.training.eval_freq == 0:
            score_model.eval()
            eval_loss = 0.0
            eval_count = 0
            
            with torch.no_grad():
                for eval_batch, _ in eval_loader:
                    if eval_count >= 10:
                        break
                    eval_batch = data_transform(eval_batch).to(config.device)
                    eloss = loss_fn(score_model, eval_batch)
                    eval_loss += eloss.mean().item()
                    eval_count += 1
            
            avg_eval_loss = eval_loss / eval_count
            eval_losses.append(avg_eval_loss)
            
            print(f"\nStep {step+1}: Eval Loss = {avg_eval_loss:.4f}")
            
            ckpt_path = os.path.join(ckpt_dir, f'model_{step+1}.pth')
            torch.save({
                'model': score_model.state_dict(),
                'config': config,
                'step': step + 1,
                'optimizer': optimizer.state_dict()
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            
            score_model.train()
        
        pbar.update(1)
    
    pbar.close()
    
    total_time = time.time() - start_time
    
    final_train_loss = sum(train_losses[-100:]) / min(100, len(train_losses))
    avg_eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else float('inf')
    
    result = {
        'config': {
            'sde_type': config.training.sde,
            'lr': config.training.lr,
            'batch_size': config.training.batch_size,
            'n_iters': config.training.n_iters,
            'model_nf': config.model.nf,
            'warmup': config.training.warmup
        },
        'final_train_loss': final_train_loss,
        'avg_eval_loss': avg_eval_loss,
        'total_time': total_time,
        'train_losses': train_losses,
        'eval_losses': eval_losses
    }
    
    summary_path = os.path.join('experiments', 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}
    
    summary[exp_name] = result
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Final Train Loss: {final_train_loss:.4f}")
    print(f"Avg Eval Loss: {avg_eval_loss:.4f}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"{'='*70}")
    
    return result


if __name__ == "__main__":
    train_optimized()
