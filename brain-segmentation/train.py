import argparse
import os
import shutil
from functools import partial
from typing import List, Tuple, Optional, Sequence

from omegaconf import OmegaConf
import torch
import torch.nn.parallel
from torch.optim.lr_scheduler import LinearLR
from dotenv import load_dotenv

from trainer import run_training
from utils.model_utils import get_model
from utils.data_utils import get_loader
from optimizers.lr_schedule import LinearWarmupCosineAnnealingLR

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction


def main(config: dict, debug: bool = False):
    load_dotenv()
    config = parse_congig(config, debug)
    run(config=config, debug=debug, **config)


def parse_congig(config: dict, debug: bool = False) -> dict:
    config['data_dir'] = os.path.expanduser(config['data_dir'])
    if debug:
        print('Using debug mode!!!')

        config['batch_size'] = 2
        config['num_workers'] = 0
        config['val_every'] = 1
        config['cache_num'] = 2

    return config


def run(
        log_dir: str,
        batch_size: int,
        model_name: str,
        inf_size: int,
        in_channels: int,
        out_channels: int,
        feature_size: int,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        pos_embed: str,
        norm_name: Tuple or str,
        dropout_rate: float,
        pretrained: bool,
        path_to_pretrain: Optional[str],
        path_to_checkpoint: Optional[str],
        smooth_nr: float,
        smooth_dr: float,
        sw_batch_size: int,
        infer_overlap: float,
        optim_name: str,
        optim_lr: float,
        optim_weight_decay: float,
        momentum: Optional[float],
        lrschedule_name: Optional[str],
        warmup_epochs: Optional[int],
        max_epochs: Optional[int],
        val_every: int,
        save_every: int,
        data_dir: str,
        spacing: Sequence[float],
        modality: int or Sequence,
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        RandFlipd_prob: float,
        RandRotate90d_prob: float,
        RandScaleIntensityd_prob: float,
        RandShiftIntensityd_prob: float,
        gauss_noise_prob: float,
        gauss_noise_std: float,
        gauss_smooth_prob: float,
        gauss_smooth_std: float or Tuple[float],
        n_workers: int,
        cache_num: int,
        device: str,
        config: dict,
        debug: bool,
        **kwargs,
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'device: {device.type}')

    loader = get_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        spacing=spacing,
        modality=modality,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
        roi_size=inf_size,
        RandFlipd_prob=RandFlipd_prob,
        RandRotate90d_prob=RandRotate90d_prob,
        RandScaleIntensityd_prob=RandScaleIntensityd_prob,
        RandShiftIntensityd_prob=RandShiftIntensityd_prob,
        gauss_noise_prob=gauss_noise_prob,
        gauss_noise_std=gauss_noise_std,
        gauss_smooth_prob=gauss_smooth_prob,
        gauss_smooth_std=gauss_smooth_std,
        n_workers=n_workers,
        cache_num=cache_num,
        device=device,
        debug=debug,
    )
    print(f"Batch size is: {batch_size}")

    model = get_model(
        model_name=model_name,
        in_channels=in_channels,
        out_channels=out_channels,
        inf_size=inf_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        pos_embed=pos_embed,
        norm_name=norm_name,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        path_to_pretrain=path_to_pretrain,
        path_to_checkpoint=path_to_checkpoint,
        device=device,
    )

    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=smooth_nr, smooth_dr=smooth_dr
    )
    post_label = AsDiscrete(to_onehot=out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.NONE, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=(inf_size, inf_size, inf_size),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr, weight_decay=optim_weight_decay)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=optim_lr, weight_decay=optim_weight_decay)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=optim_lr, momentum=momentum, nesterov=True,
            weight_decay=optim_weight_decay
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(optim_name))

    if lrschedule_name == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs
        )
    elif lrschedule_name == "warmup_linear":
        scheduler = LinearLR(optimizer, start_factor=1e-12, end_factor=1.0, total_iters=warmup_epochs)
    elif lrschedule_name == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        scheduler = None

    run_training(
        log_dir=log_dir,
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        val_every=val_every,
        save_every=save_every,
        device=device,
        config=config,
        model_inferer=model_inferer,
        scheduler=scheduler,
        post_label=post_label,
        post_pred=post_pred,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_unetr_config.yaml")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    config_name = args.config

    print(f'Using config {config_name}')

    config_folder = "configs"
    config_path = os.path.join(config_folder, config_name)

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["config_path"] = config_path
    main(cfg, debug=args.debug)
