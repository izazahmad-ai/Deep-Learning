import os.path

from monai import transforms


def test_wandb():
    import wandb
    # wandb.init(project="test")
    import os
    print(os.environ['WANDB_MODE'])


def test_transform():
    path_to_data = '/Users/vladimir/dev/data/msd/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz'
    transform1 = transforms.LoadImage()
    output = transform1(path_to_data)
    print(output[0].shape)
    transform2 = transforms.AsChannelFirst()
    output1 = transform2(output[0])
    print(output1.shape)
    modality = [0, 1, 2, 3]
    transform3 = transforms.Lambda(func=lambda x: x[modality])
    output2 = transform3(output1)
    print(output2.shape)
    # transform4 = transforms.AddChannel()
    # output3 = transform4(output2)
    # print(output3.shape)
    spacing = [1.0, 1.0, 1.0]
    transform5 = transforms.Spacing(pixdim=spacing, mode='bilinear')
    output4 = transform5(output2)
    print(output4.shape)
    transform6 = transforms.ScaleIntensity()
    output5 = transform6(output4)
    print(output5.shape)
    transform7 = transforms.CropForeground()
    output6 = transform7(output5)
    print(output6.shape)


def test_noise():
    import torch
    from torchvision.transforms import GaussianBlur
    import matplotlib.pyplot as plt

    def gauss_blur(image: torch.Tensor) -> torch.Tensor:
        blur = GaussianBlur(kernel_size=3, sigma=2.0)
        return blur(image)

    def gauss_noise(image: torch.Tensor) -> torch.Tensor:
        return image + torch.randn_like(image) * 0.2

    def show_hist(image: torch.Tensor, title: str):
        plt.hist(image.flatten().numpy(), bins=100)
        plt.title(title)
        plt.show()

    image = torch.randn(3, 224, 224)
    show_hist(image, 'original')
    show_hist(gauss_blur(image), 'gauss_blur')
    show_hist(gauss_noise(image), 'gauss_noise')


def test_msd_dataset():
    from utils.data_utils import get_loader
    import torch
    from pprint import pprint
    val_loader = get_loader(
        data_dir='/Users/vladimir/dev/data/msd/Task01_BrainTumour',
        batch_size=1,
        spacing=[1.0, 1.0, 1.0],
        modality=[0, 1, 2, 3],
        a_min=0,
        a_max=1,
        b_min=0,
        b_max=1,
        roi_size=64,
        RandFlipd_prob=0,
        RandRotate90d_prob=0,
        RandScaleIntensityd_prob=0,
        RandShiftIntensityd_prob=0,
        gauss_noise_prob=0,
        gauss_noise_std=0.2,
        gauss_smooth_prob=0,
        gauss_smooth_std=(0.25, 1.5),
        n_workers=0,
        cache_num=0,
        device=torch.device('cpu'),
    )[1]
    for sample in val_loader:
        path_to_sample = os.path.basename(sample['image_meta_dict']['filename_or_obj'][0])
        print(path_to_sample)
        isna = torch.isnan(sample['image']).sum()
        print(isna.item())


if __name__ == '__main__':
    test_msd_dataset()
    # test_noise()
    # test_mean()
    # test_transform()
    # test_wandb()
