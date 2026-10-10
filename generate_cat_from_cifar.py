import os
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import ml_collections
from PIL import Image
from torchvision.utils import make_grid

from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE
from sampling import get_pc_sampler, _PREDICTORS, _CORRECTORS


def generate_cat_like_images(exp_name='vesde_nf_32', ckpt_step=2000, num_samples=64):
    exp_dir = os.path.join('experiments', exp_name)
    ckpt_path = os.path.join(exp_dir, 'checkpoints', f'model_{ckpt_step}.pth')
    
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

    print(f"Generating {num_samples} images from {exp_name}...")
    with torch.no_grad():
        samples, _ = sampler_fn(score_model)

    grid = make_grid(samples, nrow=8, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    
    save_path = os.path.join(exp_dir, f'cat_like_samples_{ckpt_step}.png')
    Image.fromarray(grid_np).save(save_path)
    print(f"Generated {num_samples} images saved to {save_path}")
    
    return save_path


def generate_with_optimized_model(num_samples=64):
    exp_dir = os.path.join('experiments', 'vesde_optimized_full')
    ckpt_path = os.path.join(exp_dir, 'checkpoints', 'model_50000.pth')
    
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

    print(f"Generating {num_samples} images from optimized model (50000 steps)...")
    with torch.no_grad():
        samples, _ = sampler_fn(score_model)

    grid = make_grid(samples, nrow=8, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    
    save_path = os.path.join(exp_dir, f'cat_like_samples_50000.png')
    Image.fromarray(grid_np).save(save_path)
    print(f"Generated {num_samples} images saved to {save_path}")
    
    return save_path


if __name__ == '__main__':
    print("=" * 70)
    print("Generating images from VESDE NF=32 model (best quality)")
    print("=" * 70)
    generate_cat_like_images(exp_name='vesde_nf_32', ckpt_step=2000, num_samples=64)
    
    print("\n" + "=" * 70)
    print("Generating images from optimized model (50000 steps)")
    print("=" * 70)
    generate_with_optimized_model(num_samples=64)
    
    print("\n✅ Done! Check the experiment directories for cat-like images.")
