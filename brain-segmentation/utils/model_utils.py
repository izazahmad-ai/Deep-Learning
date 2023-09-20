from typing import Optional

import torch
from monai.networks.nets import UNet, BasicUNet
from models.vit_autoenc import ViTAutoEnc

from models.unetr import UNETR


def get_model(
        model_name: str = 'unetr',
        in_channels: int = 4,
        out_channels: int = 4,
        inf_size: int = 64,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = 'perceptron',
        norm_name: str = 'instance',
        dropout_rate: float = 0.,
        pretrained: bool = False,
        path_to_pretrain: Optional[str] = None,
        path_to_checkpoint: Optional[str] = None,
        device: torch.device = torch.device('cpu'),
        **kwargs,
):
    if model_name == 'unetr':
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=(inf_size, inf_size, inf_size),
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed=pos_embed,
            norm_name=norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=dropout_rate,
        )
    elif model_name == 'basic_unet':
        model = BasicUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            features=(16, 32, 64, 128, 128, 32),
        )
    elif model_name == 'vit':
        model = ViTAutoEnc(
            in_channels=in_channels,
            img_size=(inf_size, inf_size, inf_size),
            patch_size=(16, 16, 16),
            out_channels=out_channels,
            deconv_chns=16,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=12,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=3,
        )
    else:
        raise NotImplementedError(f'Model {model_name} is not implemented')

    if pretrained:
        model_dict = torch.load(path_to_pretrain)
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    if path_to_checkpoint is not None:
        checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("Use checkpoint weights")

    model.to(device)

    return model
