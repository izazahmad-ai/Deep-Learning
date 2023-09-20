from monai.networks.nets import ViTAutoEnc as MonaiViTAutoEnc


class ViTAutoEnc(MonaiViTAutoEnc):
    def forward(self, x):
        out = super().forward(x)
        return out[0]
