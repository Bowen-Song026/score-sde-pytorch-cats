"""
SDE网格搜索实验脚本
===================
系统化搜索最优超参数组合

使用方法:
    python grid_search.py --phase 1   # 粗搜索（24组）
    python grid_search.py --phase 2   # 细搜索（基于阶段一结果）
    python grid_search.py --phase 1 --quick  # 快速模式（减少训练步数）
"""

import os
import sys
import json
import time
import itertools
import argparse
import traceback

# 使用脚本所在目录作为搜索路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import torch
import torch.nn as nn
import numpy as np
import ml_collections
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import utils as mutils
from models import ncsnpp
from models.ema import ExponentialMovingAverage
from sde_lib import VESDE, VPSDE, subVPSDE
from losses import get_optimizer, get_sde_loss_fn, optimization_manager


# ============================================================
# 1. 参数网格定义
# ============================================================

# 阶段一：粗搜索网格
GRID_PHASE1 = {
    'sde_type': ['vesde', 'vpsde'],
    'lr': [1e-4, 2e-4, 5e-4],
    'model_nf': [16, 32],
    'warmup': [0, 500],
}

# 阶段二：细搜索网格（需基于阶段一结果自动填充）
GRID_PHASE2 = {
    'lr': [],          # 动态填充
    'ema_rate': [0.99, 0.999, 0.9999],
    'dropout': [0.0, 0.1, 0.2],
    'beta1': [0.9, 0.95],
}


# ============================================================
# 2. 配置生成函数
# ============================================================

def create_config(sde_type='vesde', lr=2e-4, batch_size=4, n_iters=2000,
                  model_nf=16, warmup=500, grad_clip=1.0,
                  ema_rate=0.999, dropout=0.1, beta1=0.9,
                  sigma_min=0.01, sigma_max=50,
                  beta_min=0.1, beta_max=20,
                  fourier_scale=16, num_res_blocks=1):
    """创建SDE训练配置"""
    config = ml_collections.ConfigDict()

    # 训练配置
    config.training = ml_collections.ConfigDict()
    config.training.sde = sde_type
    config.training.continuous = True
    config.training.batch_size = batch_size
    config.training.log_freq = 50
    config.training.eval_freq = max(n_iters // 5, 10)  # 每训练1/5评估一次
    config.training.snapshot_freq = n_iters  # 只在最后保存
    config.training.n_iters = n_iters
    config.training.likelihood_weighting = False
    config.training.reduce_mean = False

    # 模型配置
    config.model = ml_collections.ConfigDict()
    config.model.name = 'ncsnpp'
    config.model.scale_by_sigma = True
    config.model.ema_rate = ema_rate
    config.model.normalization = 'GroupNorm'
    config.model.nonlinearity = 'swish'
    config.model.nf = model_nf
    # 根据NF调整通道倍数和注意力分辨率
    if model_nf <= 16:
        config.model.ch_mult = (1, 2)
        config.model.attn_resolutions = ()
    else:
        config.model.ch_mult = (1, 2, 2)
        config.model.attn_resolutions = (16,)
    config.model.num_res_blocks = num_res_blocks
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
    config.model.fourier_scale = fourier_scale
    config.model.conv_size = 3
    config.model.sigma_min = sigma_min
    config.model.sigma_max = sigma_max
    config.model.num_scales = 1000
    config.model.beta_min = beta_min
    config.model.beta_max = beta_max
    config.model.dropout = dropout
    config.model.embedding_type = 'fourier'

    # 数据配置
    config.data = ml_collections.ConfigDict()
    config.data.dataset = 'CIFAR10'
    config.data.image_size = 32
    config.data.random_flip = True
    config.data.uniform_dequantization = True
    config.data.num_channels = 3
    config.data.centered = False

    # 优化器配置
    config.optim = ml_collections.ConfigDict()
    config.optim.weight_decay = 0
    config.optim.optimizer = 'Adam'
    config.optim.lr = lr
    config.optim.beta1 = beta1
    config.optim.eps = 1e-8
    config.optim.warmup = warmup
    config.optim.grad_clip = grad_clip

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return config


# ============================================================
# 3. 数据加载
# ============================================================

def get_dataloaders(batch_size, eval_batch_size=None):
    """获取CIFAR-10数据加载器"""
    if eval_batch_size is None:
        eval_batch_size = batch_size

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform
    )
    eval_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=0, drop_last=True
    )
    return train_loader, eval_loader


# ============================================================
# 4. 单次训练函数
# ============================================================

def train_single_config(config, experiment_name, n_iters=None):
    """
    用给定配置训练一次模型
    返回: 训练损失历史, 评估损失历史, 最终模型
    """
    if n_iters is None:
        n_iters = config.training.n_iters

    device = config.device
    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    print(f"SDE: {config.training.sde} | LR: {config.optim.lr} | NF: {config.model.nf}")
    print(f"Warmup: {config.optim.warmup} | Batch: {config.training.batch_size}")
    print(f"EMA: {config.model.ema_rate} | Dropout: {config.model.dropout}")
    print(f"{'='*60}\n")

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 创建模型（不使用DataParallel，避免单GPU环境问题）
    score_model = mutils.get_model(config.model.name)(config)
    score_model = score_model.to(device)
    score_model.train()

    ema = ExponentialMovingAverage(
        score_model.parameters(),
        decay=config.model.ema_rate
    )

    # 创建优化器
    optimizer = get_optimizer(config, score_model.parameters())
    optimize_fn = optimization_manager(config)

    # 创建SDE
    if config.training.sde == 'vesde':
        sde = VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales
        )
    elif config.training.sde == 'vpsde':
        sde = VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales
        )
    else:
        sde = subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales
        )

    # 损失函数
    reduce_mean = config.training.reduce_mean
    loss_fn = get_sde_loss_fn(sde, train=True, reduce_mean=reduce_mean)
    eval_loss_fn = get_sde_loss_fn(sde, train=False, reduce_mean=reduce_mean)

    # 数据加载
    train_loader, eval_loader = get_dataloaders(
        config.training.batch_size,
        eval_batch_size=config.training.batch_size
    )

    # 训练循环
    train_losses = []
    eval_losses = []
    log_freq = config.training.log_freq
    eval_freq = config.training.eval_freq

    start_time = time.time()

    step = 0
    while step < n_iters:
        for batch_data in train_loader:
            if step >= n_iters:
                break
            x, _ = batch_data
            x = x.to(device).float()

            # 均匀反量化
            if config.data.uniform_dequantization:
                x = x + (1. / 256.) * torch.rand_like(x)

            optimizer.zero_grad()
            loss = loss_fn(score_model, x)
            loss.backward()
            optimize_fn(optimizer, score_model.parameters(), step)

            ema.update(score_model.parameters())

            step += 1

            if step % log_freq == 0:
                train_losses.append(loss.item())
                elapsed = time.time() - start_time
                print(f"  Step {step}/{n_iters} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

            if step % eval_freq == 0:
                score_model.eval()
                eval_loss = 0.0
                eval_count = 0
                with torch.no_grad():
                    for eval_x, _ in eval_loader:
                        eval_x = eval_x.to(device).float()
                        if config.data.uniform_dequantization:
                            eval_x = eval_x + (1. / 256.) * torch.rand_like(eval_x)
                        eval_loss += eval_loss_fn(score_model, eval_x).item()
                        eval_count += 1
                        if eval_count >= 10:  # 限制评估步数以节省时间
                            break
                eval_loss /= eval_count
                eval_losses.append(eval_loss)
                score_model.train()
                print(f"  [Eval] Step {step} | Eval Loss: {eval_loss:.4f}")

    train_time = time.time() - start_time
    final_train_loss = np.mean(train_losses[-10:]) if train_losses else 0
    avg_eval_loss = np.mean(eval_losses) if eval_losses else 0
    final_eval_loss = eval_losses[-1] if eval_losses else 0

    print(f"\n  训练完成: {experiment_name}")
    print(f"  最终训练损失: {final_train_loss:.4f}")
    print(f"  平均评估损失: {avg_eval_loss:.4f}")
    print(f"  训练时间: {train_time:.1f}s\n")

    return {
        'experiment_name': experiment_name,
        'sde_type': config.training.sde,
        'lr': config.optim.lr,
        'model_nf': config.model.nf,
        'warmup': config.optim.warmup,
        'batch_size': config.training.batch_size,
        'ema_rate': config.model.ema_rate,
        'dropout': config.model.dropout,
        'beta1': config.optim.beta1,
        'n_iters': n_iters,
        'final_train_loss': float(final_train_loss),
        'avg_eval_loss': float(avg_eval_loss),
        'final_eval_loss': float(final_eval_loss),
        'train_time': float(train_time),
        'train_losses': train_losses,
        'eval_losses': eval_losses,
    }


# ============================================================
# 5. 网格搜索主函数
# ============================================================

def generate_grid(grid_dict):
    """生成参数网格的所有组合"""
    keys = list(grid_dict.keys())
    values = list(grid_dict.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_grid_search(phase=1, quick=False, custom_iters=None):
    """运行网格搜索"""

    # 确定训练步数
    if custom_iters:
        n_iters = custom_iters
    elif quick:
        n_iters = 500  # 快速模式
    else:
        n_iters = 2000 if phase == 1 else 3000

    # 确定网格
    if phase == 1:
        grid = GRID_PHASE1
    else:
        # 阶段二：基于阶段一结果动态填充
        results_file = os.path.join(SCRIPT_DIR, 'experiments/grid_search_phase1/grid_search_phase1_results.json')
        if not os.path.exists(results_file):
            print("错误: 请先运行阶段一 (python grid_search.py --phase 1)")
            return
        with open(results_file, 'r') as f:
            phase1_results = json.load(f)
        best = min(phase1_results, key=lambda x: x['avg_eval_loss'])
        grid = GRID_PHASE2.copy()
        # 在最优学习率附近搜索
        best_lr = best['lr']
        grid['lr'] = [best_lr * 0.5, best_lr, best_lr * 1.5]
        # 固定阶段一的最优参数
        fixed_sde = best['sde_type']
        fixed_nf = best['model_nf']
        fixed_warmup = best['warmup']

    # 生成所有组合
    combinations = list(generate_grid(grid))
    total = len(combinations)
    print(f"\n{'#'*60}")
    print(f"  网格搜索 - 阶段 {phase}")
    print(f"  总组合数: {total}")
    print(f"  每组训练步数: {n_iters}")
    print(f"  预计总步数: {total * n_iters}")
    print(f"{'#'*60}\n")

    # 创建实验目录
    exp_dir = f'experiments/grid_search_phase{phase}'
    os.makedirs(exp_dir, exist_ok=True)

    all_results = []
    best_loss = float('inf')
    best_config_name = None

    for i, combo in enumerate(combinations):
        exp_name = f"grid_p{phase}_{i+1:03d}"

        if phase == 1:
            # 阶段一：搜索sde_type, lr, nf, warmup
            # 根据NF调整batch_size
            nf = combo['model_nf']
            batch_size = 4 if nf >= 32 else 8

            config = create_config(
                sde_type=combo['sde_type'],
                lr=combo['lr'],
                model_nf=combo['model_nf'],
                warmup=combo['warmup'],
                batch_size=batch_size,
                n_iters=n_iters,
            )
        else:
            # 阶段二：搜索lr, ema_rate, dropout, beta1
            # 固定阶段一的最优参数
            batch_size = 4 if fixed_nf >= 32 else 8

            config = create_config(
                sde_type=fixed_sde,
                lr=combo['lr'],
                model_nf=fixed_nf,
                warmup=fixed_warmup,
                batch_size=batch_size,
                n_iters=n_iters,
                ema_rate=combo['ema_rate'],
                dropout=combo['dropout'],
                beta1=combo['beta1'],
            )

        try:
            result = train_single_config(config, exp_name, n_iters)
            all_results.append(result)

            if result['avg_eval_loss'] < best_loss:
                best_loss = result['avg_eval_loss']
                best_config_name = exp_name
                print(f"  ★ 新最优: {exp_name} (eval_loss={best_loss:.4f})")

            # 每次实验后保存中间结果
            results_file = f'{exp_dir}/grid_search_phase{phase}_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  ✗ 实验 {exp_name} 失败: {e}")
            traceback.print_exc()
            all_results.append({
                'experiment_name': exp_name,
                'error': str(e),
                **combo,
            })

    # 保存最终结果
    results_file = f'{exp_dir}/grid_search_phase{phase}_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  网格搜索阶段 {phase} 完成!")
    print(f"  总实验数: {len(all_results)}")
    print(f"  最优实验: {best_config_name}")
    print(f"  最优评估损失: {best_loss:.4f}")
    print(f"  结果保存至: {results_file}")
    print(f"{'='*60}\n")

    # 打印前5名
    valid_results = [r for r in all_results if 'avg_eval_loss' in r]
    valid_results.sort(key=lambda x: x['avg_eval_loss'])
    print("\n  排行榜 (Top 5):")
    print(f"  {'排名':<4} {'实验名':<20} {'SDE':<8} {'LR':<10} {'NF':<4} {'Warmup':<7} {'Eval Loss':<12}")
    print(f"  {'-'*70}")
    for rank, r in enumerate(valid_results[:5], 1):
        print(f"  {rank:<4} {r['experiment_name']:<20} {r['sde_type']:<8} "
              f"{r['lr']:<10} {r['model_nf']:<4} {r['warmup']:<7} {r['avg_eval_loss']:<12.4f}")

    return all_results


# ============================================================
# 6. 结果分析函数
# ============================================================

def analyze_results(phase=1):
    """分析网格搜索结果"""
    results_file = f'experiments/grid_search_phase{phase}/grid_search_phase{phase}_results.json'
    if not os.path.exists(results_file):
        print(f"结果文件不存在: {results_file}")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    valid_results = [r for r in results if 'avg_eval_loss' in r]
    if not valid_results:
        print("没有有效的结果")
        return

    valid_results.sort(key=lambda x: x['avg_eval_loss'])

    print(f"\n{'='*60}")
    print(f"  网格搜索阶段 {phase} 结果分析")
    print(f"{'='*60}")

    # 1. 参数重要性分析
    if phase == 1:
        params = ['sde_type', 'lr', 'model_nf', 'warmup']
    else:
        params = ['lr', 'ema_rate', 'dropout', 'beta1']

    print("\n  📊 各参数对评估损失的影响:")
    for param in params:
        groups = {}
        for r in valid_results:
            key = r[param]
            if key not in groups:
                groups[key] = []
            groups[key].append(r['avg_eval_loss'])

        print(f"\n    参数: {param}")
        for key, losses in sorted(groups.items(), key=lambda x: str(x[0])):
            avg = np.mean(losses)
            print(f"      {key}: 平均损失={avg:.4f} (±{np.std(losses):.4f}), 样本数={len(losses)}")

    # 2. Top 10
    print(f"\n  🏆 Top 10 实验:")
    print(f"  {'排名':<4} {'实验名':<20} {'SDE':<8} {'LR':<10} {'NF':<4} {'Warmup':<7} {'Eval Loss':<12} {'Train Time':<10}")
    print(f"  {'-'*80}")
    for rank, r in enumerate(valid_results[:10], 1):
        print(f"  {rank:<4} {r['experiment_name']:<20} {r['sde_type']:<8} "
              f"{r['lr']:<10} {r['model_nf']:<4} {r['warmup']:<7} "
              f"{r['avg_eval_loss']:<12.4f} {r['train_time']:<10.1f}s")

    # 3. 最优配置
    best = valid_results[0]
    print(f"\n  ✅ 最优配置:")
    print(f"    SDE类型: {best['sde_type']}")
    print(f"    学习率: {best['lr']}")
    print(f"    模型NF: {best['model_nf']}")
    print(f"    Warmup: {best['warmup']}")
    print(f"    评估损失: {best['avg_eval_loss']:.4f}")
    print(f"    训练时间: {best['train_time']:.1f}s")

    return valid_results


# ============================================================
# 7. 主入口
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDE网格搜索实验')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='搜索阶段: 1=粗搜索, 2=细搜索')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式（减少训练步数到500步）')
    parser.add_argument('--iters', type=int, default=None,
                        help='自定义每组训练步数')
    parser.add_argument('--analyze', action='store_true',
                        help='仅分析已有结果，不运行训练')
    args = parser.parse_args()

    if args.analyze:
        analyze_results(phase=args.phase)
    else:
        run_grid_search(
            phase=args.phase,
            quick=args.quick,
            custom_iters=args.iters
        )
