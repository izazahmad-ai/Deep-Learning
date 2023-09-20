import os
from typing import Callable, Optional

from torch import nn
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from monai.data import decollate_batch

import visualize
from utils import utils
from logger import WandBLogger


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: Optimizer,
        epoch: int,
        loss_func: Callable,
        logger: WandBLogger,
        device: str or torch.device,
) -> float or torch.Tensor:
    model.train()
    epoch_loss = 0.

    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)

        utils.zero_grad(model)

        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        epoch_loss += loss_value / len(loader)
        print(
            "Epoch {} {}/{}".format(epoch, idx, len(loader)),
            "loss: {:.4f}".format(loss_value),
            )
        logger("train/loss", loss_value)

    utils.zero_grad(model)

    return epoch_loss


def val_epoch(
        model: nn.Module,
        loader: DataLoader,
        epoch: int,
        acc_func: Callable,
        logger: WandBLogger,
        device: str or torch.device,
        model_inferer=None,
        post_label=None,
        post_pred=None
):
    if model_inferer is None:
        model_inferer = lambda x: model(x)

    model.eval()
    with torch.no_grad():
        avg_acc = 0.
        images = []
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)

            if not logits.is_cuda:
                target = target.cpu()

            val_labels_list = decollate_batch(target)
            if post_label is not None:
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]

            val_outputs_list = decollate_batch(logits)
            if post_pred is not None:
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)

            acc_list = acc.detach().cpu().numpy()
            val_acc = np.mean([np.nanmean(l) for l in acc_list])
            avg_acc += val_acc / len(loader)

            if idx < 5:
                image = visualize.create_image_visual(
                    source=data.cpu().numpy()[0, 0],
                    target=target.cpu().numpy().squeeze(),
                    output=logits.cpu().numpy().argmax(1).squeeze(),
                )
                images.append(image)

        logger("val/acc", avg_acc)
        acc_by_class = acc_func.aggregate()[0].mean(0)
        logger.log_val_dice(acc_by_class)
        logger.log_image(np.vstack(images), "val")
        print(
            "validation {}".format(epoch),
            "acc/val",
            avg_acc,
        )
        print(acc_by_class)


def save_checkpoint(
        model: nn.Module,
        epoch: int,
        log_dir: str,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "state_dict": state_dict}

    snapshot_path = os.path.join(log_dir, 'snapshots')
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    filename = os.path.join(snapshot_path, f'chk_{epoch}.pt')
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        config: dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_func: Callable,
        acc_func: Callable,
        log_dir: str,
        device: str or torch.device,
        val_every: int,
        save_every: int,
        model_inferer=None,
        scheduler=None,
        start_epoch: int = 0,
        post_label=None,
        post_pred=None,
):
    logger = WandBLogger(
        config=config,
        model=model,
    )
    epoch = start_epoch
    while True:
        print("Epoch:", epoch)
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
            device=device,
            logger=logger,
        )

        print(
            "training {}".format(epoch),
            "loss: {:.4f}".format(train_loss),
        )

        if (epoch + 1) % val_every == 0:
            val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                device=device,
                logger=logger,
                model_inferer=model_inferer,
                post_label=post_label,
                post_pred=post_pred,
            )

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, epoch, log_dir, optimizer=optimizer, scheduler=scheduler
            )

        if scheduler is not None:
            scheduler.step()

        epoch += 1
