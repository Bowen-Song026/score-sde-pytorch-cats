import os
import sys
sys.path.insert(0, '.')

import torch
import ml_collections
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("=== 测试环境 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n=== 测试模型导入 ===")
from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE, VPSDE, subVPSDE
from losses import get_optimizer, get_sde_loss_fn, optimization_manager
print("模型导入成功!")

print("\n=== 创建测试配置 ===")
config = ml_collections.ConfigDict()
config.training = ml_collections.ConfigDict()
config.training.sde = 'vesde'
config.training.continuous = True
config.training.batch_size = 2
config.training.n_iters = 10
config.training.reduce_mean = False
config.training.likelihood_weighting = False

config.model = ml_collections.ConfigDict()
config.model.name = 'ncsnpp'
config.model.scale_by_sigma = True
config.model.ema_rate = 0.999
config.model.normalization = 'GroupNorm'
config.model.nonlinearity = 'swish'
config.model.nf = 16
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
config.optim.lr = 2e-4
config.optim.beta1 = 0.9
config.optim.eps = 1e-8
config.optim.warmup = 100
config.optim.grad_clip = 1.0

config.seed = 42
config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print("\n=== 创建模型 ===")
score_model = mutils.create_model(config)
score_model = score_model.to(config.device)
print(f"模型参数: {sum(p.numel() for p in score_model.parameters()):,}")

print("\n=== 创建SDE ===")
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)

print("\n=== 创建损失函数 ===")
loss_fn = get_sde_loss_fn(sde, train=True, reduce_mean=config.training.reduce_mean,
                          continuous=config.training.continuous, likelihood_weighting=config.training.likelihood_weighting)

print("\n=== 创建优化器 ===")
optimizer = get_optimizer(config, score_model.parameters())
optimize_fn = optimization_manager(config)

print("\n=== 测试数据加载 ===")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (torch.rand_like(x) + x * 255.) / 256.),
    transforms.Lambda(lambda x: x * 2. - 1.)
])

dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
batch, _ = next(iter(dataloader))
print(f"数据批次形状: {batch.shape}")
batch = batch.to(config.device)

print("\n=== 测试前向传播 ===")
score_model.train()
loss = loss_fn(score_model, batch)
print(f"初始损失: {loss.item():.4f}")

print("\n=== 测试反向传播 ===")
optimizer.zero_grad()
loss.backward()
optimize_fn(optimizer, score_model.parameters(), step=0)
print("反向传播成功!")

print("\n=== 运行10步训练 ===")
for step in range(10):
    batch, _ = next(iter(dataloader))
    batch = batch.to(config.device)
    optimizer.zero_grad()
    loss = loss_fn(score_model, batch)
    loss.backward()
    optimize_fn(optimizer, score_model.parameters(), step=step)
    if (step + 1) % 5 == 0:
        print(f"Step {step+1}/10 | Loss: {loss.item():.4f}")

print("\n✅ 所有测试通过!")
