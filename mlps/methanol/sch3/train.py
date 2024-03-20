import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np

import schnetpack.transform as trn

custom_data = spk.data.AtomsDataModule(
    'train-sch/dataset.db',
    val_path="test-sch/dataset.db",
    batch_size=512,
    val_batch_size=64,
    distance_unit='Ang',
    property_units={'energy':'eV', 'forces':'eV/Ang'},
    num_train=0.98,
    num_val=0.02,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=20,
    pin_memory=True, # set to false, when not using a GPU
)
custom_data.prepare_data()
custom_data.setup()



radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=5)
# 创建模型i
pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
schnet = spk.representation.SchNet(
    n_atom_basis=256,
    n_filters=256,
    n_interactions=3,
    radial_basis=radial_basis,
   cutoff_fn=spk.nn.CosineCutoff(5.0)
    # cutoff_network=spk.nn.cutoff.CosineCutoff
)

# 定义损失函数和优化器
pred_energy = spk.atomistic.Atomwise(n_in=256, output_key="energy")
pred_forces = spk.atomistic.Forces(energy_key="energy", force_key="forces")
nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets("energy", add_mean=True, add_atomrefs=False)
    ]
)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
output_energy = spk.task.ModelOutput(
    name="energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

output_forces = spk.task.ModelOutput(
    name="forces",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.99,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)
# 配置训练过程
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

# 初始化训练器i
forcetut="models"
logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)

callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(forcetut, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=forcetut,
    max_epochs=200, # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=custom_data)
