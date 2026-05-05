# EL-MLFFs

基于论文 `EL-MLFF` 的机器学习力场仓库。当前仓库已经从早期的“读取外部基模型预测再做图网络融合”，扩展为一套更完整的 Torch-native 实现：

- 所有 `base model` 都用 PyTorch 重新实现
- 两种 `meta model` 都已实现
- `meta model` 主干改成了 MACE 风格的等变架构
- `base/meta` 的边都加入了平滑包络函数，保证 cutoff 处的平滑性
- 旧入口脚本仍然保留，但现在作为兼容层转发到新主线

仓库根目录下同时保留了论文和补充材料：

- [el-mlff-main-v2.pdf](/mnt/bn/bangchen/EL-MLFFs/el-mlff-main-v2.pdf)
- [el-mlff-v2-si.pdf](/mnt/bn/bangchen/EL-MLFFs/el-mlff-v2-si.pdf)

补充文档：

- [docs/training.md](/mnt/bn/bangchen/EL-MLFFs/docs/training.md)
  训练与多卡使用说明

## 仓库现状

当前推荐使用的是 `el-mlffs/` 目录下的新 Torch pipeline，而不是旧的 `GNNModel + Ensemble_Dataset` 老实现。

新主线解决了几个老版本问题：

- 不再写死 `54` 个原子
- 训练和评测统一使用 `extxyz` 数据
- `ConservativeEnergyMixer` 真正按 `E(R)` 建模，再由 `F = -dE/dR` 导出力
- checkpoint 现在是自描述格式，不再需要评测脚本猜模型配置

## 核心能力

### Base Models

当前 Torch 原生基模型包括：

- `dp`
- `nep`
- `mtp`
- `soap`
- `painn`
- `schnet`
- `mace`

对应实现文件：

- [torch_base_models.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/torch_base_models.py)

### Meta Models

当前元模型包括两种：

- `direct`
  直接拟合力
- `conservative`
  先预测总能量，再由自动微分得到力

对应实现文件：

- [torch_ensemble_models.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/torch_ensemble_models.py)

其中 `conservative` 架构的关键点是：

- base models 作为 meta model 的子模块参与计算图
- 总能量由 base energies 和原子级修正项共同组成
- 力通过 `autograd` 从总能量对坐标求导得到

## 平滑性与等变性

这版实现里，仓库已经明确采用两条设计原则：

1. `meta model` 使用 MACE 风格等变消息传递
2. 所有边的径向部分都乘平滑包络函数

目前默认使用的是紧支撑 `C∞` 的 `SmoothBumpEnvelope`，实现见：

- [torch_base_models.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/torch_base_models.py)

这意味着：

- cutoff 处函数值连续为 0
- cutoff 处各阶导数也连续为 0
- 更适合做保守力场和长时间动力学

## 代码结构

推荐重点阅读这些文件：

- [el-mlffs/torch_data.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/torch_data.py)
  `extxyz` 数据读取、PBC 图构建、能量到力的自动求导
- [el-mlffs/torch_base_models.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/torch_base_models.py)
  所有 Torch base models
- [el-mlffs/torch_ensemble_models.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/torch_ensemble_models.py)
  `direct` 和 `conservative` 两类 meta 架构
- [el-mlffs/torch_workflow.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/torch_workflow.py)
  数据划分、OOD split、原子类型收集
- [el-mlffs/train_torch_base.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/train_torch_base.py)
  训练单个 base model
- [el-mlffs/train_torch_ensemble.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/train_torch_ensemble.py)
  训练 ensemble / meta model
- [el-mlffs/eval_torch_ensemble.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/eval_torch_ensemble.py)
  评测和 uncertainty calibration 作图

## 环境

建议先创建环境：

```bash
conda env create -f environment.yml
```

新主线依赖的关键包包括：

- `torch`
- `torch-geometric`
- `ase`
- `e3nn`
- `scipy`
- `matplotlib`

多卡训练采用 PyTorch `DistributedDataParallel`，推荐用 `torchrun` 启动。训练脚本会自动根据环境变量 `WORLD_SIZE / RANK / LOCAL_RANK` 判断是否进入分布式模式。

## 推荐工作流

### 1. 训练单个 Torch base model

```bash
cd el-mlffs
python train_torch_base.py \
  --model-name painn \
  --data-file data/train.extxyz \
  --cutoff 5.0 \
  --save-path checkpoints/painn_torch.pth
```

批量训练并保存所有 base models：

```bash
cd el-mlffs
python train_all_torch_bases.py \
  --data-file data/train.extxyz \
  --cutoff 5.0 \
  --output-dir checkpoints/base_models
```

多卡训练单个 base model：

```bash
cd el-mlffs
torchrun --standalone --nproc_per_node=4 train_torch_base.py \
  --model-name painn \
  --data-file data/train.extxyz \
  --cutoff 5.0 \
  --batch-size 8 \
  --num-workers 4 \
  --save-path checkpoints/painn_torch.pth
```

### 2. 训练 Torch ensemble

随机划分训练：

```bash
cd el-mlffs
python train_torch_ensemble.py \
  --architecture direct \
  --data-file data/train.extxyz \
  --base-models dp nep painn schnet mace \
  --save-path checkpoints/direct_meta.pth
```

按力大小做 OOD 划分：

```bash
cd el-mlffs
python train_torch_ensemble.py \
  --architecture conservative \
  --data-file data/train.extxyz \
  --split-strategy ood \
  --base-models dp nep painn schnet mace \
  --save-path checkpoints/conservative_meta_ood.pth
```

多卡训练 ensemble：

```bash
cd el-mlffs
torchrun --standalone --nproc_per_node=4 train_torch_ensemble.py \
  --architecture conservative \
  --data-file data/train.extxyz \
  --split-strategy ood \
  --base-models dp nep painn schnet mace \
  --batch-size 8 \
  --num-workers 4 \
  --save-path checkpoints/conservative_meta_ood.pth
```

### 3. 评测单个 ensemble checkpoint

```bash
cd el-mlffs
python eval_torch_ensemble.py \
  --model-path checkpoints/conservative_meta_ood.pth
```

如果 checkpoint 已经带有元数据，评测时通常不需要再手工指定：

- `architecture`
- `base models`
- `cutoff`
- `split strategy`

### 4. 批量评测一个目录下的 checkpoint

```bash
cd el-mlffs
python test_all.py \
  --models-dir checkpoints \
  --output-csv output.csv
```

## 兼容入口

为了兼容旧的使用习惯，以下旧脚本名仍然可以使用：

- [main_stream.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/main_stream.py)
  随机划分训练的兼容入口
- [ood_train.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/ood_train.py)
  OOD 划分训练的兼容入口
- [eval.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/eval.py)
  单模型评测兼容入口
- [test_all.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/test_all.py)
  批量评测兼容入口

它们现在本质上只是对新 Torch pipeline 的一层包装，不再维护独立的旧训练逻辑。

## Checkpoint 格式

新的 checkpoint 不只是 `state_dict`，还额外保存：

- `metadata`
- `config`

典型元数据包括：

- ensemble 架构类型
- base-model 子集
- cutoff
- 数据文件路径
- split strategy
- 验证集指标

这使得：

- `eval_torch_ensemble.py` 可以自动恢复模型配置
- `test_all.py` 可以直接批量扫描和评测

## 老代码与新代码的关系

仓库里仍然保留了一些早期文件，例如：

- `dataset.py`
- `gnn_model.py`
- `training_utils.py`
- `prepare_data.py`

这些文件主要用于旧版本工作流、历史实验或论文复现对照。当前如果你要继续开发和补功能，应该优先在下面这些文件上工作：

- `torch_data.py`
- `torch_base_models.py`
- `torch_ensemble_models.py`
- `train_torch_base.py`
- `train_torch_ensemble.py`
- `eval_torch_ensemble.py`

## 当前建议的开发方向

如果继续完善仓库，优先级建议是：

1. 补 `conservation check` 和 NVE 评测主线
2. 补 reaction-path / NEB 类评测
3. 把更多论文 benchmark 接到当前 Torch pipeline
4. 继续提升 conservative mixer 的物理先验强度

## License

本项目采用 MIT License，见 [LICENSE](/mnt/bn/bangchen/EL-MLFFs/LICENSE)。
