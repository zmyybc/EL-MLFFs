# Base Model Audit

这份审计面向 `el-mlffs/torch_base_models.py` 当前注册到主训练入口的五个 base models，重点区分：

- 是否可以视为“标准/接近标准”的实现
- 是否只是仓库内的轻量近似版
- 是否适合当论文级 baseline

## 当前结论

- `schnet`
  可以视为最接近标准 baseline 的实现。直接基于 `torch_geometric.nn.SchNet`，只额外加入了平滑包络。
- `painn`
  是一个可训练、可导出力的 Torch 重实现，但不是严格复刻原始 PaiNN 论文/官方实现的全细节版本。
- `dp`
  现在是更接近 DeepPot 范式的轻量重实现：使用平滑逆距离、邻居类型嵌入、环境矩阵聚合和不变量收缩；仍不是官方 DeepPot-SE 代码库。
- `nep`
  现在是更接近 NEP 范式的轻量重实现：包含径向项以及 `l=1/l=2` 角向矩不变量；仍不是官方 NEP 代码库。
- `mtp`
  是 Moment Tensor Potential 风格的描述符模型，基于径向矩和方向矩构造旋转不变量。
- `soap`
  是 SOAP 风格的描述符模型，基于类型分辨的球谐展开和 power spectrum 不变量。
- `mace`
  现在是 MACE-style 等变重实现，不再是原先那种明显偏 toy 的 lite 版；但依然不是官方 MACE 训练栈。

## 逐项说明

### `schnet`

实现位置：

- [torch_base_models.py](/mnt/bn/bangchen/EL-MLFFs/el-mlffs/torch_base_models.py)

判断依据：

- 继承自 `torch_geometric.nn.SchNet`
- 保留标准 SchNet 的 embedding / interaction / readout 主体
- 主要改动是把径向展开乘以 `SmoothBumpEnvelope`

建议：

- 可以作为当前仓库最可靠的标准基线之一优先训练

### `painn`

判断依据：

- 有标量/向量通道、消息传递与更新块，力通过能量自动求导得到
- 但实现是仓库内自写的 `SafePaiNNMessage / SafePaiNNUpdate`
- 没有证据表明其严格复刻官方 PaiNN 的全部结构细节与训练 recipe

建议：

- 可以继续训练和比较
- 但在写结论时应称作“PaiNN-style reimplementation”更准确

### `dp`

判断依据：

- 现在使用邻居类型条件化的 edge embedding
- 先构造环境矩阵，再做 DeepPot-style 不变量收缩
- 仍未接入官方 DeepMD/DeepPot 的完整工程栈与全部训练细节

建议：

- 可以作为 DeepPot-style baseline 使用
- 对外描述应写成“DeepPot-style reimplementation”而不是官方 DeepMD

### `nep`

判断依据：

- 当前实现包含径向项与角向矩不变量
- 角向部分不再只是简单方向范数，而是显式构造 `l=1/l=2` 旋转不变量
- 仍未完全覆盖官方 NEP 的全部不变量体系与训练细节

建议：

- 可以作为 NEP-style baseline 使用
- 对外描述应写成“NEP-style reimplementation”

### `mace`

判断依据：

- 当前主线已经切到更强的 MACE-style 等变实现
- 隐表示包含 `0e/1o/2e` 通道，并保留径向网络生成的张量积权重
- 仍不是官方 MACE 包，因此不能直接等同于官方 benchmark 配置

建议：

- 可以作为 MACE-style baseline 使用
- 如果要做严格论文复现，仍建议换成官方 MACE 训练栈
