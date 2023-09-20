import os
import shutil

import numpy as np
import torch

import wandb
from torch import nn
from omegaconf import OmegaConf


class WandBLogger:
    def __init__(
            self,
            config: dict,
            model: nn.Module,
    ):
        self.config = config
        self.run = wandb.init(
            project='coursework',
            name=self.config['run_name'],
            config=self.config,
            resume='allow',
        )

        log_dir = self.config['log_dir']
        if not wandb.run.offline:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            else:
                raise ValueError(f'Log dir {log_dir} already exists')

            shutil.copy2(config['config_path'], log_dir)

        wandb.watch(model, log='all', log_freq=100, log_graph=True)
        self.save_config()

    def __call__(self, metric_name: str, value: float):
        self.log({metric_name: value})

    def log(self, metric: dict):
        self.run.log(metric)

    def log_val_dice(self, val_metric: torch.Tensor):
        for i, val in enumerate(val_metric.cpu().numpy()):
            self.log({f'val/acc/class_{i}': val.item()})

    def log_image(self, image: np.ndarray, tag: str):
        self.run.log({tag: wandb.Image(image)})

    def save_config(self):
        config_name = 'train_config.yaml'
        path_to_save_config = os.path.join(self.run.dir, config_name)
        OmegaConf.save(self.config, path_to_save_config)
        wandb.save(config_name, policy='now')

    def finish(self):
        self.run.finish()
