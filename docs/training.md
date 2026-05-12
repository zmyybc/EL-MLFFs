# Training Guide

这份文档面向当前仓库的 Torch-native 主训练流程，覆盖：

- 数据格式
- 单个 `base model` 训练
- 批量训练全部 `base models`
- `ensemble / meta model` 训练
- 多卡训练
- 兼容入口
- 常见问题

## 1. 数据格式

当前主训练脚本统一读取 `extxyz` 数据，而不是旧版的 `pckl/csv` 组合。

默认脚本参数一般指向：

- `el-mlffs/data/train.extxyz`

也可以直接指定任意 `extxyz` 文件，例如：

- `el-mlffs/data_pbc.extxyz`

数据读取和图构建在：

- [torch_data.py](el-mlffs/torch_data.py)

当前训练流程默认期望每个构型至少包含：

- 原子种类
- 原子坐标
- 总能量 `energy`
- 原子力 `forces`

当前代码会自动兼容 `force` 和 `forces` 两种数组键，但如果两者都不存在会直接报错，而不是静默填 0。

如果是周期体系，还建议包含：

- `cell`
- PBC 信息

## 2. 训练单个 Base Model

单个 base model 的训练入口是：

- [train_torch_base.py](el-mlffs/train_torch_base.py)

支持的模型：

- `dp`
- `nep`
- `mtp`
- `soap`
- `painn`
- `schnet`
- `mace`

示例：

```bash
cd el-mlffs
python train_torch_base.py \
  --model-name painn \
  --data-file data/train.extxyz \
  --val-data-file data/test.extxyz \
  --cutoff 5.0 \
  --batch-size 16 \
  --epochs 50 \
  --save-path checkpoints/painn_torch.pth
```

常用参数：

- `--model-name`
  选择 base model 类型
- `--data-file`
  训练数据路径
- `--cutoff`
  邻接图 cutoff
- `--batch-size`
  单进程 batch size
- `--val-data-file`
  显式验证集路径；提供后不再从训练集随机切分
- `--epochs`
  训练轮数
- `--lr`
  学习率
- `--energy-weight`
  能量项损失权重
- `--force-weight`
  力项损失权重
- `--save-path`
  最优 checkpoint 保存路径
- `--train-ratio`
  训练/验证划分比例
- `--seed`
  划分随机种子
- `--num-workers`
  dataloader worker 数量

## 3. 批量训练全部 Base Models

批量训练入口：

- [train_all_torch_bases.py](el-mlffs/train_all_torch_bases.py)

默认会顺序训练：

- `dp`
- `nep`
- `mtp`
- `soap`
- `painn`
- `schnet`
- `mace`

示例：

```bash
cd el-mlffs
python train_all_torch_bases.py \
  --data-file data/train.extxyz \
  --val-data-file data/test.extxyz \
  --cutoff 5.0 \
  --batch-size 16 \
  --epochs 50 \
  --output-dir checkpoints/base_models
```

默认输出文件名是：

- `dp_torch.pth`
- `nep_torch.pth`
- `painn_torch.pth`
- `schnet_torch.pth`
- `mace_torch.pth`

如果只想训练其中几个：

```bash
python train_all_torch_bases.py \
  --models dp painn mace \
  --data-file data/train.extxyz \
  --output-dir checkpoints/base_models
```

## 4. 训练 Ensemble / Meta Model

主入口：

- [train_torch_ensemble.py](el-mlffs/train_torch_ensemble.py)

支持两种 meta 架构：

- `direct`
  直接拟合力
- `conservative`
  学习总能量，再由自动微分得到力

### 4.1 随机划分训练

```bash
cd el-mlffs
python train_torch_ensemble.py \
  --architecture direct \
  --data-file data/train.extxyz \
  --val-data-file data/test.extxyz \
  --base-models dp nep painn schnet mace \
  --batch-size 16 \
  --epochs 50 \
  --save-path checkpoints/direct_meta.pth
```

### 4.2 OOD 划分训练

```bash
cd el-mlffs
python train_torch_ensemble.py \
  --architecture conservative \
  --data-file data/train.extxyz \
  --val-data-file data/test.extxyz \
  --split-strategy ood \
  --base-models dp nep painn schnet mace \
  --batch-size 16 \
  --epochs 50 \
  --save-path checkpoints/conservative_meta_ood.pth
```

### 4.3 关键参数

- `--architecture`
  `direct` 或 `conservative`
- `--base-models`
  选择哪些 base models 参与 ensemble
- `--split-strategy`
  `random` 或 `ood`
- `--val-data-file`
  显式测试/验证集路径；提供后优先使用，不再应用随机或 OOD 划分
- `--train-base-models`
  默认冻结 base models；加上这个参数后会联合训练
- `--energy-weight`
  只对 `conservative` 架构有意义
- `--force-weight`
  力损失权重

## 5. 多卡训练

当前主训练流程已经支持 PyTorch `DistributedDataParallel`。

分布式工具在：

- [train_distributed.py](el-mlffs/train_distributed.py)

训练脚本会自动检测以下环境变量：

- `WORLD_SIZE`
- `RANK`
- `LOCAL_RANK`

因此推荐直接使用 `torchrun`。

### 5.1 多卡训练单个 Base Model

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

### 5.2 多卡训练 Ensemble

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

### 5.3 多卡训练时需要注意

- `batch-size` 是每个进程自己的 batch size，不是全局总 batch size
- 实际全局 batch size 约等于：
  `batch_size_per_rank * world_size`
- checkpoint 只在 `rank 0` 保存
- 日志也只在 `rank 0` 打印
- 验证指标已经做了跨进程聚合

## 6. 兼容入口

如果你还想保留旧命令名，可以使用这些包装脚本：

- [main_stream.py](el-mlffs/main_stream.py)
  随机划分训练入口
- [ood_train.py](el-mlffs/ood_train.py)
  OOD 划分训练入口

示例：

```bash
cd el-mlffs
python main_stream.py \
  --architecture direct \
  --data-file data/train.extxyz \
  --save-path models/main_stream_torch.pth
```

多卡方式同样可以直接用：

```bash
cd el-mlffs
torchrun --standalone --nproc_per_node=4 main_stream.py \
  --architecture direct \
  --data-file data/train.extxyz \
  --batch-size 8 \
  --num-workers 4 \
  --save-path models/main_stream_torch.pth
```

## 7. Checkpoint 说明

新的 checkpoint 不是纯 `state_dict`，而是一个包含三部分的字典：

- `state_dict`
- `metadata`
- `config`

这意味着训练后保存的模型会同时记录：

- base model 名称或 ensemble 架构
- 使用的数据路径
- cutoff
- split strategy
- base-model 子集
- 验证集指标

后续评测脚本可以直接根据这些信息恢复模型配置。

## 8. 评测相关

单模型评测入口：

- [eval_torch_ensemble.py](el-mlffs/eval_torch_ensemble.py)

示例：

```bash
cd el-mlffs
python eval_torch_ensemble.py \
  --model-path checkpoints/conservative_meta_ood.pth
```

批量扫描目录：

- [test_all.py](el-mlffs/test_all.py)

```bash
cd el-mlffs
python test_all.py \
  --models-dir checkpoints \
  --output-csv output.csv
```

## 9. 常见问题

### Q1. 现在脚本还读取旧版 `pckl` 吗？

主训练脚本默认不再读取旧版 `dataset_all_train.pckl`，而是统一读取 `extxyz`。

### Q2. 为什么多卡启动时 batch size 好像变小了？

因为当前参数里的 `--batch-size` 是每个 rank 的局部 batch size，不是全局 batch size。

### Q3. 为什么只看到一个进程保存 checkpoint？

这是预期行为。DDP 下只允许 `rank 0` 保存，避免多个进程同时写同一个文件。

### Q4. 旧版 checkpoint 能不能直接用于新评测脚本？

如果旧 checkpoint 没有 `metadata/config`，新评测脚本无法完全自动恢复配置。此时需要手动传：

- `architecture`
- `base models`
- `cutoff`

或者先把旧 checkpoint 迁移成新的保存格式。
