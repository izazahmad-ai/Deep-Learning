import os
from typing import Sequence, Tuple

import torch
from monai import data, transforms, apps
from torch.utils.data import Subset


def get_msd_vanilla_transforms(
        roi_size: int,
        spacing: Sequence[float],
        modality: int or Sequence = 0,
        RandFlipd_prob: float = 0.2,
        RandRotate90d_prob: float = 0.2,
        RandScaleIntensityd_prob: float = 0.1,
        RandShiftIntensityd_prob: float = 0.1,
        gauss_noise_prob: float = 0.0,
        gauss_noise_std: float = 0.1,
        gauss_smooth_prob: float = 0.0,
        gauss_smooth_std: float or Tuple[float] = (0.25, 1.5),
        device: torch.device = torch.device('cpu'),
        **kwargs,
):
    modality = [modality] if isinstance(modality, int) else modality    # modality = 0
    train_transforms_list = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.AsChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[modality]),
        transforms.AddChanneld(keys=["label"]),
        transforms.Spacingd(
            keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")
        ),
        transforms.ScaleIntensityd(keys=["image"]),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.FgBgToIndicesd(
            keys="label",
            fg_postfix="_fg",
            bg_postfix="_bg",
            image_key="image",
        ),
        transforms.EnsureTyped(keys=["image", "label"], device=torch.device('cpu'), track_meta=False),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(roi_size, roi_size, roi_size),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
            fg_indices_key='label_fg',
            bg_indices_key='label_bg',
        ),

        transforms.RandGaussianNoised(keys=["image"], prob=gauss_noise_prob, std=gauss_noise_std),
        transforms.RandGaussianSmoothd(keys=["image"], prob=gauss_smooth_prob,
                                       sigma_x=gauss_smooth_std, sigma_y=gauss_smooth_std, sigma_z=gauss_smooth_std),

        transforms.RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=RandRotate90d_prob, max_k=3),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=RandScaleIntensityd_prob),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=RandShiftIntensityd_prob),
    ]

    val_transforms_list = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.AsChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[modality]),
        transforms.AddChanneld(keys=["label"]),
        transforms.Spacingd(
            keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")
        ),
        transforms.ScaleIntensityd(keys=["image"]),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.RandGaussianNoised(keys=["image"], prob=gauss_noise_prob, std=gauss_noise_std),
        transforms.RandGaussianSmoothd(keys=["image"], prob=gauss_smooth_prob,
                                       sigma_x=gauss_smooth_std, sigma_y=gauss_smooth_std, sigma_z=gauss_smooth_std),

        transforms.EnsureTyped(keys=["image", "label"], device=torch.device('cpu'), track_meta=False)
    ]

    train_transform = transforms.Compose(train_transforms_list)
    val_transform = transforms.Compose(val_transforms_list)
    return train_transform, val_transform


def get_loader(
        data_dir: str,
        batch_size: int,
        spacing: Sequence[float],
        modality: int or Sequence,
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        roi_size: int,
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
        device: torch.device,
        debug: bool = False,
        **kwargs,
):
    train_transform, val_transform = get_msd_vanilla_transforms(
        roi_size=roi_size,
        spacing=spacing,
        modality=modality,
        RandFlipd_prob=RandFlipd_prob,
        RandRotate90d_prob=RandRotate90d_prob,
        RandScaleIntensityd_prob=RandScaleIntensityd_prob,
        RandShiftIntensityd_prob=RandShiftIntensityd_prob,
        gauss_noise_prob=gauss_noise_prob,
        gauss_noise_std=gauss_noise_std,
        gauss_smooth_prob=gauss_smooth_prob,
        gauss_smooth_std=gauss_smooth_std,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
        device=device,
    )

    train_ds = apps.DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        section="training",
        transform=train_transform,
        download=False,
        cache_num=cache_num,
    )
    if debug:
        train_ds = Subset(train_ds, indices=range(0, len(train_ds), 100))

    train_loader = data.ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers > 0,
    )

    val_ds = apps.DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        section="validation",
        transform=val_transform,
        download=False,
        cache_num=cache_num,
    )
    if debug:
        val_ds = Subset(val_ds, indices=range(0, len(val_ds), 40))

    val_loader = data.ThreadDataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers > 0,
        drop_last=True,
    )
    loader = [train_loader, val_loader]

    return loader
