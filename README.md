# Score-SDE PyTorch 猫脸生成实验

> A Chinese project README for cat-face generation experiments based on Yang Song's score_sde_pytorch, comparing VE/VP/sub-VP SDEs on a consumer 4GB GPU.

本项目基于 [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) 二次开发，面向在消费级 GPU（4GB 显存）上学习并复现 *Score-Based Generative Modeling through Stochastic Differential Equations*（ICLR 2021）的核心思想，并在猫脸图像生成任务上系统对比 VE-SDE / VP-SDE / sub-VP-SDE 三类随机微分方程，配合学习率、模型容量、训练步数等超参数消融与网格搜索。上游原版英文 README 已保留为 [`UPSTREAM_README.md`](UPSTREAM_README.md)。

<p>
  <img src="assets/python-badge.svg" alt="Python 3.8+" />
  &nbsp;
  <img src="assets/pytorch-badge.svg" alt="PyTorch 1.7+" />
</p>

## 1. 项目背景与动机

Score-based 生成模型通过估计数据分布的得分函数（score），结合随机微分方程（SDE）将数据平滑地扰动为噪声，再通过逆向 SDE 还原出真实样本。该方法在 CIFAR-10 等数据集上取得了当时的 SOTA（FID 2.20）。然而原作者的实验依赖多卡大显存与超长训练，普通学习者难以复现。

本项目的动机有两点：

- **学原理**：通过亲手实现并训练 VE / VP / sub-VP 三类 SDE，直观理解前向加噪与逆向采样、得分匹配损失、Predictor-Corrector 采样器等核心机制。
- **做对比**：在同一台消费级 GPU 上，固定数据集（猫脸）与算力预算，横向对比三类 SDE 的收敛性与生成质量，并对学习率、模型容量（NF）、训练步数做消融，寻找 4GB 显存下的最优配置组合。

## 2. 硬件与环境

| 项目 | 配置 |
|:----|:----|
| GPU | 消费级显卡，约 4GB VRAM（与 DDPM / LDM 实验共用同一台机器） |
| Python | 3.8+ |
| PyTorch | 1.7.0+（CUDA） |
| 核心依赖 | `ml-collections`、`torchvision`、`ninja`、`absl-py`、`tensorflow`（仅用于 FID/IS 统计与数据集 IO，可按需精简） |

完整依赖见 [`requirements.txt`](requirements.txt)。在显存受限场景下，建议先用 `batch_size=2~4`、`nf=16~32` 的小配置跑通流程，再逐步加大。

## 3. 主要工作

- **三类 SDE 对比**：基于 `sde_lib.py` 中的 `VESDE` / `VPSDE` / `subVPSDE` 实现，在相同数据与算力下横向比较收敛速度与最终 loss。
- **超参数消融**：学习率（`1e-4 / 2e-4 / 5e-4`）、模型容量 NF（`16 / 32`）、训练步数（`5000 → 50000`）、warmup 与 EMA rate 等。
- **网格搜索**：[`grid_search.py`](grid_search.py) 以 `sde_type × lr × model_nf × warmup` 等维度自动枚举组合，批量训练并汇总结果。
- **结果汇总**：所有实验的 loss 曲线、超参与采样结果汇总到 [`experiments_summary.xlsx`](experiments_summary.xlsx)。
- **样本生成**：按工程约定，每组实验均生成可视觉检查的图像样本；结果统一进 Excel 汇总。
- **演示**：[`Score_SDE_demo_PyTorch.ipynb`](Score_SDE_demo_PyTorch.ipynb) 含已执行结果，可直接阅读流程与可视化。

## 4. 关键发现

在 4GB 显存、固定算力预算下的工程结论：

1. **VESDE 优于 VPSDE 与 sub-VPSDE**：收敛更稳定、最终 loss 更低，是消费级 GPU 上的首选。
2. **学习率 `5e-4` 优于 `2e-4 / 1e-4`**：收敛更快，在相同步数下 loss 更低。
3. **模型容量 `NF=32` 优于 `NF=16`**：尽管显存占用更高，但表达能力与生成质量更好。
4. **训练步数 `5000 → 50000`，loss 降低约 60%**：在算力允许范围内尽量延长训练可显著受益。

> 工程约定：每组实验都必须产出可视化样本，并把结果汇总到 Excel 文件，便于横向比较。

## 5. 目录结构

```text
score_sde_pytorch-main/
├── main.py                         # 上游统一入口（train/eval）
├── run_lib.py                     # 训练/评估循环封装
├── sde_lib.py                      # VE/VP/sub-VP 三类 SDE 定义
├── sampling.py                     # Predictor-Corrector 采样器
├── losses.py                       # 得分匹配损失
├── likelihood.py                   # 对数似然 / 概率流 ODE
├── evaluation.py                   # FID/IS/KID 评估
├── controllable_generation.py      # 可控生成（类别条件/补全/上色）
├── datasets.py                     # 数据集加载
├── utils.py
│
├── train_cat_sde.py                # 猫脸 SDE 基础训练
├── train_cat_sde_conditional.py    # 猫脸条件 SDE 训练
├── train_cat_generator.py          # 猫脸生成器训练
├── run_sde_train.py                # 训练运行入口（默认 VESDE, NF=32）
├── run_full_experiment.py          # 完整实验（sde_type/lr/nf/n_iters 可配）
├── run_optimized_full.py           # 优化版完整实验
├── run_opt.py                     # 参数搜索
├── grid_search.py                  # 网格搜索（sde_type×lr×nf×warmup）
├── generate_cat_from_cifar.py      # 从 CIFAR 生成猫
├── analyze_and_generate.py         # 分析 + 生成
│
├── models/                         # 模型库
│   ├── ncsnpp.py                   # NCSN++（主干，本项目主要使用）
│   ├── ncsnv2.py                   # NCSNv2
│   ├── ddpm.py                     # DDPM
│   ├── layers.py / layerspp.py     # 网络层
│   ├── normalization.py            # 归一化
│   ├── ema.py                      # 指数滑动平均
│   ├── up_or_down_sampling.py
│   ├── utils.py
│   └── __init__.py
│
├── configs/                        # ml_collections 配置
│   ├── default_cifar10_configs.py
│   ├── default_celeba_configs.py
│   ├── default_lsun_configs.py
│   ├── ve/                         # VE-SDE 配置
│   │   ├── cifar10_ncsnpp_deep_continuous.py
│   │   ├── cifar10_ncsnpp_continuous.py
│   │   ├── cifar10_ncsnpp.py
│   │   ├── cifar10_ddpm.py
│   │   ├── celeba_ncsnpp.py
│   │   ├── celebahq_256_ncsnpp_continuous.py
│   │   ├── ffhq_256_ncsnpp_continuous.py
│   │   ├── bedroom_ncsnpp_continuous.py
│   │   ├── church_ncsnpp_continuous.py
│   │   ├── ncsn/                    # 旧版 NCSN
│   │   └── ncsnv2/
│   ├── vp/                         # VP-SDE 配置
│   │   ├── cifar10_ncsnpp_deep_continuous.py
│   │   ├── cifar10_ncsnpp_continuous.py
│   │   ├── cifar10_ncsnpp.py
│   │   ├── cifar10_ddpmpp_deep_continuous.py
│   │   ├── cifar10_ddpmpp_continuous.py
│   │   ├── cifar10_ddpmpp.py
│   │   └── ddpm/                    # cifar10 / cifar10_continuous / cifar10_unconditional / celebahq / bedroom / church
│   └── subvp/                      # sub-VP SDE 配置
│       ├── cifar10_ncsnpp_deep_continuous.py
│       ├── cifar10_ncsnpp_continuous.py
│       ├── cifar10_ddpmpp_deep_continuous.py
│       ├── cifar10_ddpmpp_continuous.py
│       └── cifar10_ddpm_continuous.py
│
├── assets/                         # 示意图与徽章
├── experiments_summary.xlsx        # 实验结果汇总（工程约定）
├── Score_SDE_demo_PyTorch.ipynb    # 演示 notebook（含执行结果）
├── requirements.txt
├── LICENSE                         # Apache-2.0
├── UPSTREAM_README.md              # Yang Song 原版英文 README
└── README.md                       # 本文件
```

## 6. 快速开始

### 6.1 安装环境

```sh
pip install -r requirements.txt
```

### 6.2 选择配置

配置文件位于 `configs/` 下，按 SDE 类型分组，命名形如 `configs/<ve|vp|subvp>/<dataset>_<model>[_continuous][_deep].py`：

- VE-SDE：`configs/ve/cifar10_ncsnpp_deep_continuous.py`
- VP-SDE：`configs/vp/cifar10_ncsnpp_deep_continuous.py`
- sub-VP SDE：`configs/subvp/cifar10_ncsnpp_deep_continuous.py`

### 6.3 训练（上游入口，使用配置文件）

```sh
# 以 VE-SDE + NCSN++ 为例
python main.py --config configs/ve/cifar10_ncsnpp_deep_continuous.py \
               --mode train \
               --workdir results/ve_cifar10

# 改用 VP-SDE
python main.py --config configs/vp/cifar10_ncsnpp_deep_continuous.py \
               --mode train \
               --workdir results/vp_cifar10
```

> 4GB 显存提示：上游默认 `nf=128`、`num_res_blocks=8`，单卡难以容纳。建议改用本项目的轻量脚本（见 6.4），或手动将配置中的 `model.nf` 调小到 16/32、`batch_size` 调到 2~4。

### 6.4 训练（本项目脚本，内联轻量配置）

本项目脚本在内部以 `ml_collections` 内联配置，默认面向小显存优化：

```sh
# 单次完整实验（默认 VESDE, lr=2e-4, nf=16, n_iters=2000）
python run_full_experiment.py

# 基础训练入口（默认 VESDE, nf=32, batch_size=2）
python run_sde_train.py
```

如需切换 SDE / 学习率 / NF，直接修改脚本中的 `get_config(sde_type, lr, batch_size, n_iters, model_nf, ...)` 参数，或在 `run_full_experiment.py` / `grid_search.py` 中调整网格。

### 6.5 采样与评估

```sh
# 生成样本并计算 FID/IS（需先准备 assets/stats/ 下的统计文件）
python main.py --config configs/ve/cifar10_ncsnpp_deep_continuous.py \
               --mode eval \
               --workdir results/ve_cifar10 \
               --config.eval.enable_sampling
```

### 6.6 网格搜索

```sh
# 自动枚举 sde_type × lr × model_nf × warmup 组合
python grid_search.py
```

`grid_search.py` 默认网格覆盖 `sde_type ∈ {vesde, vpsde}`、`lr ∈ {1e-4, 2e-4, 5e-4}`、`model_nf ∈ {16, 32}`、`warmup ∈ {0, 500}`，结果汇总到 `experiments_summary.xlsx`。

## 7. 与上游项目的差异

本项目保留上游全部模型、配置与核心模块，在其之上做面向消费级 GPU 与猫脸生成任务的二次开发：

- **新增训练 / 实验脚本**：`train_cat_sde.py`、`train_cat_sde_conditional.py`、`train_cat_generator.py`、`run_sde_train.py`、`run_full_experiment.py`、`run_optimized_full.py`、`run_opt.py`、`grid_search.py`、`generate_cat_from_cifar.py`、`analyze_and_generate.py`。
- **轻量化配置**：内联配置将 `nf` 降至 16/32、`batch_size` 降至 2~4、`n_iters` 缩短，适配 4GB 显存；并按 NF 自动调整 `ch_mult` 与 `attn_resolutions`。
- **消融与汇总**：新增 `grid_search.py` 自动枚举超参组合，结果统一写入 `experiments_summary.xlsx`。
- **演示**：新增 `Score_SDE_demo_PyTorch.ipynb`（含已执行结果）。
- **保留上游**：`models/`、`configs/`、`sde_lib.py`、`sampling.py`、`losses.py` 等核心文件与上游一致；原版英文 README 保留为 `UPSTREAM_README.md`。

## 8. 关键文件速查表

| 文件 | 作用 |
|:----|:----|
| `sde_lib.py` | 定义 `VESDE` / `VPSDE` / `subVPSDE` 三类 SDE 及其前向/逆向过程 |
| `sampling.py` | Predictor-Corrector 采样器（`reverse_diffusion`、`langevin` 等） |
| `losses.py` | 得分匹配损失（DSM / SSM 等） |
| `likelihood.py` | 概率流 ODE 与对数似然计算 |
| `evaluation.py` | 样本质量评估（FID / IS / KID） |
| `controllable_generation.py` | 类别条件、补全、上色等可控生成 |
| `datasets.py` | 数据集加载与预处理 |
| `main.py` | 上游统一训练 / 评估入口 |
| `run_lib.py` | 训练 / 评估循环封装 |
| `models/ncsnpp.py` | NCSN++ 模型（本项目主干） |
| `models/ema.py` | 指数滑动平均（EMA） |
| `run_sde_train.py` | 训练运行入口（默认 VESDE, NF=32） |
| `run_full_experiment.py` | 完整实验（sde_type / lr / nf / n_iters 可配） |
| `grid_search.py` | 超参网格搜索 |
| `experiments_summary.xlsx` | 实验结果汇总 |
| `Score_SDE_demo_PyTorch.ipynb` | 演示 notebook（含执行结果） |

## 9. License

本项目沿用上游 [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) 的 **Apache-2.0** 许可证，详见 [`LICENSE`](LICENSE)。本仓库的二次开发部分（训练脚本、配置、实验汇总等）同样以 Apache-2.0 发布。

## 10. 致谢

感谢 [Yang Song](https://yang-song.github.io)、Jascha Sohl-Dickstein、Diederik P. Kingma、Abhishek Kumar、Stefano Ermon、Ben Poole 等作者提出 Score-based SDE 生成框架并开源 [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)。本项目在其代码基础上进行学习与二次开发，上游原版说明见 [`UPSTREAM_README.md`](UPSTREAM_README.md)。

如原作者论文对你的研究有帮助，请引用：

```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```
