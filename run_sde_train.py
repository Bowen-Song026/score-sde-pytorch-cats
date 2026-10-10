import os
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import ml_collections
import time

from models import utils as mutils
from models import ncsnpp
from models.ema import ExponentialMovingAverage
from sde_lib import VESDE, VPSDE, subVPSDE
from losses import get_optimizer, get_sde_loss_fn, optimization_manager


def get_default_configs():
    config = ml_collections.ConfigDict()
    config.training = ml_collections.ConfigDict()
    config.training.sde = 'vesde'
    config.training.continuous = True
    config.training.batch_size = 2
    config.training.log_freq = 10
    config.training.eval_freq = 100
    config.training.snapshot_freq = 5000
    config.training.n_iters = 1000
    config.training.likelihood_weighting = False
    config.training.reduce_mean = False
    
    config.model = ml_collections.ConfigDict()
    config.model.name = 'ncsnpp'
    config.model.scale_by_sigma = True
    config.model.ema_rate = 0.999
    config.model.normalization = 'GroupNorm'
    config.model.nonlinearity = 'swish'
    config.model.nf = 32
    config.model.ch_mult = (1, 2, 2)
    config.model.num_res_blocks = 1
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
    config.data.dataset = 'CIFAR10'
    config.data.image_size = 32
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
    config.optim.warmup = 5000
    config.optim.grad_clip = 1.
    
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return config


def main():
    config = get_default_configs()
    print(f"========== SDE Training Configuration ==========")
    print(f"Device: {config.device}")
    print(f"SDE Type: {config.training.sde}")
    print(f"Model: {config.model.name}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.optim.lr}")
    print(f"Warmup Steps: {config.optim.warmup}")
    print(f"Total Iterations: {config.training.n_iters}")
    print(f"===============================================\n")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    score_model = mutils.create_model(config)
    score_model = score_model.to(config.device)
    score_model.train()
    print(f"Model created with {sum(p.numel() for p in score_model.parameters()):,} parameters")
    
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
    
    print(f"\nStarting training...\n")
    
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    start_time = time.time()
    
    for step in range(config.training.n_iters):
        fake_data = torch.randn(config.training.batch_size, config.data.num_channels, 
                                config.data.image_size, config.data.image_size).to(config.device)
        
        optimizer.zero_grad()
        loss = loss_fn(score_model, fake_data)
        loss.backward()
        optimize_fn(optimizer, score_model.parameters(), step=step)
        ema.update(score_model.parameters())
        
        if (step + 1) % config.training.log_freq == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print(f"Step [{step+1}/{config.training.n_iters}] | Loss: {loss.item():.4f} | LR: {lr:.6f} | Time: {elapsed:.2f}s")
        
        if (step + 1) % config.training.snapshot_freq == 0:
            ckpt_dir = os.path.join('checkpoints', config.training.sde)
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
    
    print(f"\n✅ Training complete! Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
