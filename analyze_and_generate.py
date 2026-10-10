import os
import sys
sys.path.insert(0, '.')

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ml_collections
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm

from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import get_sampling_fn


def create_excel_summary(summary_path, output_path='experiments_summary.xlsx'):
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for exp_name, exp_data in data.items():
        if 'error' not in exp_data:
            rows.append({
                'Experiment': exp_name,
                'SDE Type': exp_data['config']['sde_type'],
                'Learning Rate': exp_data['config']['lr'],
                'Model NF': exp_data['config']['model_nf'],
                'Batch Size': exp_data['config']['batch_size'],
                'Iterations': exp_data['config']['n_iters'],
                'Warmup': exp_data['config'].get('warmup', 500),
                'Final Train Loss': exp_data['final_train_loss'],
                'Avg Eval Loss': exp_data['avg_eval_loss'],
                'Total Time (s)': exp_data['total_time']
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Final Train Loss')
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Experiment Summary', index=False)
        
        df_style = df.style.format({
            'Learning Rate': '{:.1e}',
            'Final Train Loss': '{:.4f}',
            'Avg Eval Loss': '{:.4f}',
            'Total Time (s)': '{:.1f}'
        }).highlight_min(subset=['Final Train Loss', 'Avg Eval Loss'], color='#90EE90')
        
        summary_stats = pd.DataFrame({
            'Metric': ['Min Train Loss', 'Max Train Loss', 'Avg Train Loss', 
                       'Min Eval Loss', 'Max Eval Loss', 'Avg Eval Loss',
                       'Fastest Time', 'Slowest Time'],
            'Value': [
                df['Final Train Loss'].min(),
                df['Final Train Loss'].max(),
                df['Final Train Loss'].mean(),
                df['Avg Eval Loss'].min(),
                df['Avg Eval Loss'].max(),
                df['Avg Eval Loss'].mean(),
                df['Total Time (s)'].min(),
                df['Total Time (s)'].max()
            ]
        })
        summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
    
    print(f"✅ Excel summary saved to {output_path}")
    return df


def generate_samples(exp_name, checkpoint_step=2000, num_samples=16, image_size=32):
    exp_dir = os.path.join('experiments', exp_name)
    ckpt_path = os.path.join(exp_dir, 'checkpoints', f'model_{checkpoint_step}.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint not found: {ckpt_path}")
        return None
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    if not hasattr(config, 'sampling'):
        config.sampling = ml_collections.ConfigDict()
        if config.training.sde.lower() == 'subvpsde':
            config.sampling.method = 'ode'
            config.sampling.noise_removal = True
        else:
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
    
    if config.training.sde.lower() == 'vpsde':
        sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'subvpsde':
        sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'vesde':
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    def inverse_scaler(x):
        if config.data.centered:
            x = (x + 1.) / 2.
        return torch.clamp(x, 0., 1.)
    
    sampling_fn = get_sampling_fn(config, sde, shape=(num_samples, 3, image_size, image_size), 
                                  inverse_scaler=inverse_scaler, eps=1e-5)
    
    print(f"🔄 Generating {num_samples} samples for {exp_name}...")
    with torch.no_grad():
        samples, _ = sampling_fn(score_model)
    
    grid = make_grid(samples, nrow=6, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    
    save_path = os.path.join(exp_dir, f'samples_{checkpoint_step}.png')
    Image.fromarray(grid_np).save(save_path)
    
    print(f"✅ Samples saved to {save_path}")
    return save_path


def generate_all_samples():
    summary_path = os.path.join('experiments', 'summary.json')
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    for exp_name in data.keys():
        if 'error' not in data[exp_name]:
            generate_samples(exp_name)


def test_gpu_memory(max_batch_size=32, image_size=32, model_nf=32):
    import ml_collections
    from models import ncsnpp
    
    config = ml_collections.ConfigDict()
    config.training = ml_collections.ConfigDict()
    config.training.continuous = True
    config.training.sde = 'vesde'
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
    config.data.image_size = image_size
    config.data.num_channels = 3
    config.data.centered = False
    config.device = torch.device('cuda:0')
    
    device = torch.device('cuda:0')
    
    score_model = mutils.create_model(config)
    score_model = score_model.to(device)
    score_model.eval()
    
    params = sum(p.numel() for p in score_model.parameters())
    print(f"Model params: {params:,}")
    
    torch.cuda.empty_cache()
    
    max_bs = 0
    for bs in [2, 4, 8, 16, 24, 32]:
        try:
            x = torch.randn(bs, 3, image_size, image_size).to(device)
            t = torch.rand(bs).to(device)
            
            with torch.no_grad():
                score_model(x, t)
            
            mem_used = torch.cuda.memory_allocated(device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
            
            print(f"✓ Batch size {bs}: Allocated={mem_used:.1f}MB, Reserved={mem_reserved:.1f}MB")
            max_bs = bs
            
            del x, t
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"✗ Batch size {bs}: Failed - {e}")
            break
    
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**2
    print(f"\nGPU: {torch.cuda.get_device_name(device)}")
    print(f"Total memory: {total_mem:.1f}MB")
    print(f"Max batch size: {max_bs}")
    
    return max_bs


if __name__ == "__main__":
    print("=" * 70)
    print("Step 1: Creating Excel summary...")
    print("=" * 70)
    df = create_excel_summary('experiments/summary.json')
    
    print("\n" + "=" * 70)
    print("Step 2: Generating image samples...")
    print("=" * 70)
    generate_all_samples()
    
    print("\n" + "=" * 70)
    print("Step 3: Testing GPU memory...")
    print("=" * 70)
    max_bs = test_gpu_memory()
    
    print(f"\n✅ Done! Max batch size: {max_bs}")
