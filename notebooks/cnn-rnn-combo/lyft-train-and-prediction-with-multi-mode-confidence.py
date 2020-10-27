'''
# Lyft: Training with multi-mode confidence

![](http://www.l5kit.org/_images/av.jpg)
<cite>The image from L5Kit official document: <a href="http://www.l5kit.org/README.html">http://www.l5kit.org/README.html</a></cite>

Continued from the previous kernel:
 - [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition)
 - [Lyft: Deep into the l5kit library](https://www.kaggle.com/corochann/lyft-deep-into-the-l5kit-library)

In this kernel, I will run **pytorch CNN model training**. Especially, followings are new items to try:
 - Predict **multi-mode with confidence**: As written in [evaluation metric](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview/evaluation) page, we can predict **3 modes** of motion trajectory.
 - Training loss with **competition evaluation metric**
 - Use Training abstraction library **`pytorch-ignite` and `pytorch-pfn-extras`**.


[Update 2020/9/6]<br/>
Published prediction kernel: [Lyft: Prediction with multi-mode confidence](https://www.kaggle.com/corochann/lyft-prediction-with-multi-mode-confidence)<br/>
Try yourself how good score you can get using only single model without ensemble! :)
'''

'''
# Environment setup

 - Please add [pestipeti/lyft-l5kit-unofficial-fix](https://www.kaggle.com/pestipeti/lyft-l5kit-unofficial-fix) as utility script.
    - Official utility script "[philculliton/kaggle-l5kit](https://www.kaggle.com/mathurinache/kaggle-l5kit)" does not work with pytorch GPU.
 - Please add [lyft-config-files](https://www.kaggle.com/jpbremer/lyft-config-files) as dataset
 
See previous kernel [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition) for details.
'''

#>
# https://github.com/pfnet/pytorch-pfn-extras/releases/tag/v0.3.1
!pip install pytorch-pfn-extras==0.3.1

#>
import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import scipy as sp


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)

#>
import zarr

import l5kit
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='jshtml')
print("l5kit version:", l5kit.__version__)

#>
import torch
from pathlib import Path

import pytorch_pfn_extras as ppe
from math import ceil
from pytorch_pfn_extras.training import IgniteExtensionsManager
from pytorch_pfn_extras.training.triggers import MinValueTrigger

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import pytorch_pfn_extras.training.extensions as E

#>
# --- Dataset utils ---
from typing import Callable

from torch.utils.data.dataset import Dataset


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        batch = self.dataset[index]
        return self.transform(batch)

    def __len__(self):
        return len(self.dataset)

'''
## Function

To define loss function to calculate competition evaluation metric **in batch**.<br/>
It works with **pytorch tensor, so it is differentiable** and can be used for training Neural Network.
'''

#>
# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
import numpy as np

import torch
from torch import Tensor


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)

'''
## Model

pytorch model definition. Here model outputs both **multi-mode trajectory prediction & confidence of each trajectory**.
'''

#>
# --- Model utils ---
import torch
from torchvision.models import resnet18
from torch import nn
from typing import Dict


class LyftMultiModel(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        # TODO: support other than resnet18?
        backbone = resnet18(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 512

        # X, Y coords for the future positions (output shape: Bx50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)

        # pred (bs)x(modes)x(time)x(2D coords)
        # confidences (bs)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

#>
class LyftMultiRegressor(nn.Module):
    """Single mode prediction"""

    def __init__(self, predictor, lossfun=pytorch_neg_multi_log_likelihood_batch):
        super().__init__()
        self.predictor = predictor
        self.lossfun = lossfun

    def forward(self, image, targets, target_availabilities):
        pred, confidences = self.predictor(image)
        loss = self.lossfun(targets, pred, confidences, target_availabilities)
        metrics = {
            "loss": loss.item(),
            "nll": pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities).item()
        }
        ppe.reporting.report(metrics, self)
        return loss, metrics

'''
## Training with Ignite

I use [pytorch-ignite](https://github.com/pytorch/ignite) for training abstraction.<br/>
`Engine` defines the 1 iteration of training update.

 - [Official Document: IGNITE YOUR NETWORKS!](https://pytorch.org/ignite/)
'''

#>
# --- Training utils ---
from ignite.engine import Engine


def create_trainer(model, optimizer, device) -> Engine:
    model.to(device)

    def update_fn(engine, batch):
        model.train()
        optimizer.zero_grad()
        loss, metrics = model(*[elem.to(device) for elem in batch])
        loss.backward()
        optimizer.step()
        return metrics
    trainer = Engine(update_fn)
    return trainer

#>
# --- Utils ---
import yaml


def save_yaml(filepath, content, width=120):
    with open(filepath, 'w') as f:
        yaml.dump(content, f, width=width)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        content = yaml.safe_load(f)
    return content


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

'''
## Configs
'''

#>
# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },

    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 12,
        'shuffle': True,
        'num_workers': 4
    },

    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },

    'train_params': {
        'max_num_steps': 10000,
        'checkpoint_every_n_steps': 5000,

        # 'eval_every_n_steps': -1
    }
}

#>
flags_dict = {
    "debug": True,
    # --- Data configs ---
    "l5kit_data_folder": "/kaggle/input/lyft-motion-prediction-autonomous-vehicles",
    # --- Model configs ---
    "pred_mode": "multi",
    # --- Training configs ---
    "device": "cuda:0",
    "out_dir": "results/multi_train",
    "epoch": 2,
    "snapshot_freq": 50,
}

'''
# Main script

Now finished defining all the util codes. Let's start writing main script to train the model!
'''

'''
## Loading data

Here we will only use the first dataset from the sample set. (sample.zarr data is used for visualization, please use train.zarr / validate.zarr / test.zarr for actual model training/validation/prediction.)<br/>
We're building a `LocalDataManager` object. This will resolve relative paths from the config using the `L5KIT_DATA_FOLDER` env variable we have just set.
'''

#>
flags = DotDict(flags_dict)
out_dir = Path(flags.out_dir)
os.makedirs(str(out_dir), exist_ok=True)
print(f"flags: {flags_dict}")
save_yaml(out_dir / 'flags.yaml', flags_dict)
save_yaml(out_dir / 'cfg.yaml', cfg)
debug = flags.debug

#>
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = flags.l5kit_data_folder
dm = LocalDataManager(None)

print("Load dataset...")
train_cfg = cfg["train_data_loader"]
valid_cfg = cfg["valid_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

# Train dataset/dataloader
def transform(batch):
    return batch["image"], batch["target_positions"], batch["target_availabilities"]


train_path = "scenes/sample.zarr" if debug else train_cfg["key"]
train_zarr = ChunkedDataset(dm.require(train_path)).open()
print("train_zarr", type(train_zarr))
train_agent_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataset = TransformDataset(train_agent_dataset, transform)
if debug:
    # Only use 1000 dataset for fast check...
    train_dataset = Subset(train_dataset, np.arange(1000))
train_loader = DataLoader(train_dataset,
                          shuffle=train_cfg["shuffle"],
                          batch_size=train_cfg["batch_size"],
                          num_workers=train_cfg["num_workers"])
print(train_agent_dataset)

valid_path = "scenes/sample.zarr" if debug else valid_cfg["key"]
valid_zarr = ChunkedDataset(dm.require(valid_path)).open()
print("valid_zarr", type(train_zarr))
valid_agent_dataset = AgentDataset(cfg, valid_zarr, rasterizer)
valid_dataset = TransformDataset(valid_agent_dataset, transform)
if debug:
    # Only use 100 dataset for fast check...
    valid_dataset = Subset(valid_dataset, np.arange(100))
else:
    # Only use 1000 dataset for fast check...
    valid_dataset = Subset(valid_dataset, np.arange(1000))
valid_loader = DataLoader(
    valid_dataset,
    shuffle=valid_cfg["shuffle"],
    batch_size=valid_cfg["batch_size"],
    num_workers=valid_cfg["num_workers"]
)

print(valid_agent_dataset)
print("# AgentDataset train:", len(train_agent_dataset), "#valid", len(valid_agent_dataset))
print("# ActualDataset train:", len(train_dataset), "#valid", len(valid_dataset))
# AgentDataset train: 22496709 #valid 21624612
# ActualDataset train: 100 #valid 100

'''
## Prepare model & optimizer
'''

#>
device = torch.device(flags.device)

if flags.pred_mode == "multi":
    predictor = LyftMultiModel(cfg)
    model = LyftMultiRegressor(predictor)
else:
    raise ValueError(f"[ERROR] Unexpected value flags.pred_mode={flags.pred_mode}")

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

'''
## Write training code

pytorch-ignite & pytorch-pfn-extras are used here.

 - [pytorch/ignite](https://github.com/pytorch/ignite): It provides abstraction for writing training loop.
 - [pfnet/pytorch-pfn-extras](https://github.com/pfnet/pytorch-pfn-extras): It provides several "extensions" useful for training. Useful for **logging, printing, evaluating, saving the model, scheduling the learning rate** during training.
 
**[Note] Why training abstraction library is used?**

You may feel understanding training abstraction code below is a bit unintuitive compared to writing "raw" training loop.<br/>
The advantage of abstracting the code is that we can re-use implemented handler class for other training, other competition.<br/>
You don't need to write code for saving models, logging training loss/metric, show progressbar etc.
These are done by provided util classes in `pytorch-pfn-extras` library!

You may refer my other kernel in previous competition too: [Bengali: SEResNeXt training with pytorch](https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch)
'''

#>
# Train setup
trainer = create_trainer(model, optimizer, device)


def eval_func(*batch):
    loss, metrics = model(*[elem.to(device) for elem in batch])


valid_evaluator = E.Evaluator(
    valid_loader,
    model,
    progress_bar=False,
    eval_func=eval_func,
)

log_trigger = (10 if debug else 1000, "iteration")
log_report = E.LogReport(trigger=log_trigger)


extensions = [
    log_report,  # Save `log` to file
    valid_evaluator,  # Run evaluation for valid dataset in each epoch.
    # E.FailOnNonNumber()  # Stop training when nan is detected.
    E.ProgressBarNotebook(update_interval=10 if debug else 100),  # Show progress bar during training
    E.PrintReportNotebook(),  # Show "log" on jupyter notebook  
]

epoch = flags.epoch

models = {"main": model}
optimizers = {"main": optimizer}
manager = IgniteExtensionsManager(
    trainer,
    models,
    optimizers,
    epoch,
    extensions=extensions,
    out_dir=str(out_dir),
)
# Save predictor.pt every epoch
manager.extend(E.snapshot_object(predictor, "predictor.pt"),
               trigger=(flags.snapshot_freq, "iteration"))
# Check & Save best validation predictor.pt every epoch
# manager.extend(E.snapshot_object(predictor, "best_predictor.pt"),
#                trigger=MinValueTrigger("validation/main/nll", trigger=(flags.snapshot_freq, "iteration")))
# --- lr scheduler ---
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.99999)
manager.extend(lambda manager: scheduler.step(), trigger=(1, "iteration"))
# Show "lr" column in log
manager.extend(E.observe_lr(optimizer=optimizer), trigger=log_trigger)

# --- HACKING to fix ProgressBarNotebook bug ---
manager.iteration = 0
manager._iters_per_epoch = len(train_loader)

trainer.run(train_loader, max_epochs=epoch)

'''
So what is happening in above training abstraction? Let's understand what each extension did.

**Extensions** - Each role:
 - **`ProgressBar` (`ProgressBarNotebook`)**: Shows training progress in formatted style.
 - **`LogReport`**: Logging metrics reported by `ppe.reporter.report` (see `LyftMultiRegressor` for reporting point) method and save to **log** file. It automatically collects reported value in each iteration and saves the "mean" of reported value for regular frequency (for example every 1 epoch).
 - **`PrintReport` (`PrintReportNotebook`)**: Prints the value which `LogReport` collected in formatted style.
 - **`Evaluator`**: Evaluate on validation dataset.
 - **`snapshot_object`**: Saves the object. Here the `model` is saved in regular interval `flags.snapshot_freq`. Even you quit training using Ctrl+C without finishing all the epoch, the intermediate trained model is saved and you can use it for inference.
 - **lambda function with `scheduler.step()`**: You can invoke any function in regular interval specified by `trigger`. Here exponential decay of learning rate is applied by calling `scheduler.step()` every iteration.
 - **`observe_lr`**: `LogReport` will check optimizer's learning rate using this extension. So you can follow how the learning rate changed through the training.


Such many functionalities can be "added" easily using extensions!
'''

'''
You can obtrain training history results really easily by just accessing `LogReport` class, which is useful for managing a lot of experiments during kaggle competitions.
'''

#>
df = log_report.to_dataframe()
df.to_csv(out_dir/"log.csv", index=False)
df[["epoch", "iteration", "main/loss", "main/nll", "validation/main/loss", "validation/main/nll", "lr", "elapsed_time"]]

'''
The history log and model's weight are saved by "extensions" (`LogReport` and `E.snapshot_object` respectively) easily, which is a benefit of using training abstration.
'''

#>
# Let's see training results directory

!ls results/multi_train

'''
# Items to try

This kernel shows demonstration run of the training (`debug=True`). You can try these things to see how the score changes at first
 - set debug=False to train with actual training dataset
 - change training hyperparameters (training epoch, change optimizer, scheduler learning rate etc...)
   - Especially, just training much longer time may improve the score.
 
To go further, these items may be nice to try:
 - Change the cnn model (now simple resnet18 is used as baseline modeling)
 - Training the model using full dataset: [lyft-full-training-set](https://www.kaggle.com/philculliton/lyft-full-training-set)
 - Write your own rasterizer to prepare input image as motivation explained in previous kernel.
 - Consider much better scheme to predict multi-trajectory
    - The model just predicts multiple trajectory at the same time in this kernel, but it is possible to collapse "trivial" solution where all trajectory converges to same. How to avoid this?
'''

'''
# Next to go

[Update 2020/9/6]<br/>
Published prediction kernel: [Lyft: Prediction with multi-mode confidence](https://www.kaggle.com/corochann/lyft-prediction-with-multi-mode-confidence)<br/>
Try yourself how good score you can get using only single model without ensemble! :)

To understand the competition in more detail, please refer my other kernels too.
 - [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition)
 - [Lyft: Deep into the l5kit library](https://www.kaggle.com/corochann/lyft-deep-into-the-l5kit-library)
 - [Save your time, submit without kernel inference](https://www.kaggle.com/corochann/save-your-time-submit-without-kernel-inference)
 - [Lyft: pytorch implementation of evaluation metric](https://www.kaggle.com/corochann/lyft-pytorch-implementation-of-evaluation-metric)
'''

'''
# Further reference

 - Paper of this Lyft Level 5 prediction dataset: [One Thousand and One Hours: Self-driving Motion Prediction Dataset](https://arxiv.org/abs/2006.14480)
 - [jpbremer/lyft-scene-visualisations](https://www.kaggle.com/jpbremer/lyft-scene-visualisations)
'''

'''
<h3 style="color:red">If this kernel helps you, please upvote to keep me motivated :)<br>Thanks!</h3>
'''

'''
# Lyft: Prediction with multi-mode confidence

![](http://www.l5kit.org/_images/av.jpg)
<cite>The image from L5Kit official document: <a href="http://www.l5kit.org/README.html">http://www.l5kit.org/README.html</a></cite>

Continued from the previous kernel:
 - [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition)
 - [Lyft: Deep into the l5kit library](https://www.kaggle.com/corochann/lyft-deep-into-the-l5kit-library)
 - [Lyft: Training with multi-mode confidence](https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence)

In this kernel, I will run **prediction code for competition submission** where the model is trained in previous kernel.

<span style="color:red">I uploaded trained models as dataset <a href="https://www.kaggle.com/corochann/lyft-resnet18-baseline">lyft-resnet18-baseline</a><br/>
Please <b>upvote the dataset</b> as well if this kernel helps you :)</span>
'''

'''
# Environment setup

 - Please add [pestipeti/lyft-l5kit-unofficial-fix](https://www.kaggle.com/pestipeti/lyft-l5kit-unofficial-fix) as utility script.
    - Official utility script "[philculliton/kaggle-l5kit](https://www.kaggle.com/mathurinache/kaggle-l5kit)" does not work with pytorch GPU.
 - Please add dataset:
    - [lyft-config-files](https://www.kaggle.com/jpbremer/lyft-config-files)
    - [lyft-resnet18-baseline](https://www.kaggle.com/corochann/lyft-resnet18-baseline)
 
See previous kernel [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition) for details.
'''

#>
from l5kit.evaluation import write_pred_csv

'''
## Model

pytorch model definition. Here model outputs both **multi-mode trajectory prediction & confidence of each trajectory**.
'''

#>
# --- Utils ---
import yaml


def save_yaml(filepath, content, width=120):
    with open(filepath, 'w') as f:
        yaml.dump(content, f, width=width)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        content = yaml.safe_load(f)
    return content


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#>
# Referred https://www.kaggle.com/pestipeti/pytorch-baseline-inference
def run_prediction(predictor, data_loader):
    predictor.eval()

    pred_coords_list = []
    confidences_list = []
    timestamps_list = []
    track_id_list = []

    with torch.no_grad():
        dataiter = tqdm(data_loader)
        for data in dataiter:
            image = data["image"].to(device)
            # target_availabilities = data["target_availabilities"].to(device)
            # targets = data["target_positions"].to(device)
            pred, confidences = predictor(image)

            pred_coords_list.append(pred.cpu().numpy().copy())
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps_list.append(data["timestamp"].numpy().copy())
            track_id_list.append(data["track_id"].numpy().copy())
    timestamps = np.concatenate(timestamps_list)
    track_ids = np.concatenate(track_id_list)
    coords = np.concatenate(pred_coords_list)
    confs = np.concatenate(confidences_list)
    return timestamps, track_ids, coords, confs

'''
## Configs
'''

#>
flags_dict = {
    "debug": False,
    # --- Data configs ---
    "l5kit_data_folder": "/kaggle/input/lyft-motion-prediction-autonomous-vehicles",
    # --- Model configs ---
    "pred_mode": "multi",
    # --- Training configs ---
    "device": "cuda:0",
    "out_dir": "results/multi_train",
    "epoch": 2,
    "snapshot_freq": 50,
}

'''
# Main script

Now finished defining all the util codes. Let's start writing main script to train the model!
'''

'''
## Loading data

Here we will only use the first dataset from the sample set. (sample.zarr data is used for visualization, please use train.zarr / validate.zarr / test.zarr for actual model training/validation/prediction.)<br/>
We're building a `LocalDataManager` object. This will resolve relative paths from the config using the `L5KIT_DATA_FOLDER` env variable we have just set.
'''

#>
flags = DotDict(flags_dict)
out_dir = Path(flags.out_dir)
os.makedirs(str(out_dir), exist_ok=True)
print(f"flags: {flags_dict}")
save_yaml(out_dir / 'flags.yaml', flags_dict)
save_yaml(out_dir / 'cfg.yaml', cfg)
debug = flags.debug

#>
# set env variable for data
l5kit_data_folder = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder
dm = LocalDataManager(None)

print("Load dataset...")
default_test_cfg = {
    'key': 'scenes/test.zarr',
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 4
}
test_cfg = cfg.get("test_data_loader", default_test_cfg)

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

test_path = test_cfg["key"]
print(f"Loading from {test_path}")
test_zarr = ChunkedDataset(dm.require(test_path)).open()
print("test_zarr", type(test_zarr))
test_mask = np.load(f"{l5kit_data_folder}/scenes/mask.npz")["arr_0"]
test_agent_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
test_dataset = test_agent_dataset
if debug:
    # Only use 100 dataset for fast check...
    test_dataset = Subset(test_dataset, np.arange(100))
test_loader = DataLoader(
    test_dataset,
    shuffle=test_cfg["shuffle"],
    batch_size=test_cfg["batch_size"],
    num_workers=test_cfg["num_workers"],
    pin_memory=True,
)

print(test_agent_dataset)
print("# AgentDataset test:", len(test_agent_dataset))
print("# ActualDataset test:", len(test_dataset))

'''
## Prepare model & optimizer
'''

#>
device = torch.device(flags.device)

if flags.pred_mode == "multi":
    predictor = LyftMultiModel(cfg)
else:
    raise ValueError(f"[ERROR] Unexpected value flags.pred_mode={flags.pred_mode}")

pt_path = "/kaggle/input/lyft-resnet18-baseline/0918_predictor_full.pt"
print(f"Loading from {pt_path}")
predictor.load_state_dict(torch.load(pt_path))
predictor.to(device)

'''
# Inference!
'''

#>
# --- Inference ---
timestamps, track_ids, coords, confs = run_prediction(predictor, test_loader)

#>
csv_path = "submission.csv"
write_pred_csv(
    csv_path,
    timestamps=timestamps,
    track_ids=track_ids,
    coords=coords,
    confs=confs)
print(f"Saved to {csv_path}")

'''
# Next to go

To understand the competition in more detail, please refer my other kernels too.
 - [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition)
 - [Lyft: Deep into the l5kit library](https://www.kaggle.com/corochann/lyft-deep-into-the-l5kit-library)
 - [Save your time, submit without kernel inference](https://www.kaggle.com/corochann/save-your-time-submit-without-kernel-inference)
 - [Lyft: pytorch implementation of evaluation metric](https://www.kaggle.com/corochann/lyft-pytorch-implementation-of-evaluation-metric)
 - [Lyft: Training with multi-mode confidence](https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence)

Discussions:
 - [Is this the way how the autonomous car predicts motions?](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/179236)
 - [How to ensemble multi-trajectory predictions?](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/180931)
'''

'''
# Further reference

 - Paper of this Lyft Level 5 prediction dataset: [One Thousand and One Hours: Self-driving Motion Prediction Dataset](https://arxiv.org/abs/2006.14480)
 - [jpbremer/lyft-scene-visualisations](https://www.kaggle.com/jpbremer/lyft-scene-visualisations)
'''

'''
<h3 style="color:red">If this kernel helps you, please upvote to keep me motivated :)<br>Thanks!</h3>
'''
