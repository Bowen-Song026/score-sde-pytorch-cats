import os
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from ml_collections import ConfigDict

from models import utils as mutils
from models.ema import ExponentialMovingAverage
from sde_lib import VESDE, VPSDE, subVPSDE
from losses import get_optimizer
from datasets import get_dataset, get_data_scaler, get_data_inverse_scaler


def get_default_configs():
    config = ConfigDict()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.training = ConfigDict()
    config.training.sde = 'vesde'
    config.training.continuous = True
    config.training.batch_size = 32
    config.training.log_freq = 10
    config.training.eval_freq = 100
    config.training.snapshot_freq = 50000
    config.training.n_iters = 500000
    config.eval = ConfigDict()
    config.eval.enable = True
    config.eval.batch_size = 1024
    config.eval.enable_sampling = True
    config.eval.num_samples = 50000
    config.eval.enable_bpd = False
    config.model = ConfigDict()
    config.model.name = 'ncsnpp'
    config.model.scale_by_sigma = True
    config.model.ema_rate = 0.999
    config.model.normalization = 'GroupNorm'
    config.model.nonlinearity = 'swish'
    config.model.nf = 128
    config.model.ch_mult = (1, 2, 2, 2)
    config.model.num_res_blocks = 4
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
    config.data = ConfigDict()
    config.data.dataset = 'CIFAR10'
    config.data.image_size = 32
    config.data.random_flip = True
    config.data.uniform_dequantization = True
    config.data.num_channels = 3
    config.optim = ConfigDict()
    config.optim.weight_decay = 0
    config.optim.optimizer = 'Adam'
    config.optim.lr = 2e-4
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.sampling = ConfigDict()
    config.sampling.method = 'pc'
    config.sampling.predictor = 'reverse_diffusion'
    config.sampling.corrector = 'langevin'
    config.sampling.n_steps_each = 1
    config.sampling.noise_removal = True
    config.sampling.probability_flow = False
    config.sampling.snr = 0.16
    return config


def main():
    config = get_default_configs()
    print(f"Using device: {config.device}")
    
    score_model = mutils.create_model(config)
    score_model = score_model.to(config.device)
    print(f"Model created: {config.model.name}")
    print(f"Model params: {sum(p.numel() for p in score_model.parameters()):,}")
    
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    
    train_ds, eval_ds, _ = get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)
    
    if config.training.sde.lower() == 'vpsde':
        sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    print(f"SDE: {config.training.sde}")
    
    from losses import get_loss_fn
    loss_fn = get_loss_fn(sde, train=True, reduce_mean=False, continuous=config.training.continuous)
    
    print("\nStarting training (5 iterations for test)...")
    for step in range(5):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ds)
            batch = next(train_iter)
        
        batch = batch[0].to(config.device)
        batch = scaler(batch)
        
        optimizer.zero_grad()
        loss = loss_fn(score_model, batch)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        ema.update(score_model.parameters())
        
        if (step + 1) % 1 == 0:
            print(f"Step {step+1}: loss = {loss.item():.4f}")
    
    print("\n✅ Training test completed successfully!")


if __name__ == "__main__":
    main()
