import torch
import pytorch_model_summary
from monai.networks.nets import BasicUNet, UNet, ViT, ViTAutoEnc
from models.unetr import UNETR


def test_vit_autoenc():
    model = ViTAutoEnc(
        in_channels=4,
        out_channels=4,
        img_size=(64, 64, 64),
        patch_size=(16, 16, 16),
        spatial_dims=3,
    )
    batch = torch.randn(2, 4, 64, 64, 64)
    out = model(batch)
    print('VITAutoEnc')
    print(out[0].shape)
    pytorch_model_summary.summary(model, batch, show_input=True, print_summary=True)


def test_vit():
    model = ViT(
        in_channels=4,
        img_size=(64, 64, 64),
        patch_size=(16, 16, 16),
        num_classes=4,
        spatial_dims=3,
        classification=True,
    )

    batch = torch.randn(2, 4, 64, 64, 64)
    out = model(batch)
    print('VIT')
    print(out[0].shape)
    pytorch_model_summary.summary(model, batch, show_input=True, print_summary=True)



def test_unetr():
    if __name__ == '__main__':
        image_size = (64, 64, 64)

        model = UNETR(4, 4, image_size)
        batch = torch.randn(2, 4, *image_size)
        out = model(batch)
        print('UNETR')
        print(out.shape)

        pytorch_model_summary.summary(model, batch, print_summary=True)


def test_basic_unet():
    model = BasicUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        features=(16, 32, 64, 128, 128, 32),
    )
    batch = torch.randn(2, 4, 64, 64, 64)
    out = model(batch)
    print('BasicUnet')
    print(out.shape)
    pytorch_model_summary.summary(model, batch, show_input=True, print_summary=True)


def test_unet():
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    )
    batch = torch.randn(2, 4, 64, 64, 64)
    pytorch_model_summary.summary(model, batch, show_input=True, print_summary=True)



if __name__ == '__main__':
    test_vit_autoenc()
    test_vit()
    test_basic_unet()
    test_unetr()

