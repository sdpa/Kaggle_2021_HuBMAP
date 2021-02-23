import torch
import torch.nn as nn
from .unet_parts import *
import models.nn_ as nn_


class SegCaps(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SegCaps, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, padding=2, bias=False),
        )


        self.step_1 = nn.Sequential(  # 1/2
            CapsuleLayer(1, 16, "conv", k=5, s=2, t_1=2, z_1=16, routing=1, padding=2),
            CapsuleLayer(2, 16, "conv", k=5, s=1, t_1=4, z_1=16, routing=3, padding=2),
        )

        self.step_2 = nn.Sequential(  # 1/4
            CapsuleLayer(4, 16, "conv", k=5, s=2, t_1=4, z_1=32, routing=3, padding=2),
            CapsuleLayer(4, 32, "conv", k=5, s=1, t_1=8, z_1=32, routing=3, padding=2)
        )

        self.step_3 = nn.Sequential(  # 1/8
            CapsuleLayer(8, 32, "conv", k=5, s=2, t_1=8, z_1=64, routing=3, padding=2),
            CapsuleLayer(8, 64, "conv", k=5, s=1, t_1=8, z_1=32, routing=3, padding=2)
        )

        self.step_4 = CapsuleLayer(8, 32, "deconv", k=5, s=2, t_1=8, z_1=32, routing=3, padding=2)

        self.step_5 = CapsuleLayer(16, 32, "conv", k=5, s=1, t_1=4, z_1=32, routing=3, padding=2)

        self.step_6 = CapsuleLayer(4, 32, "deconv", k=5, s=2, t_1=4, z_1=16, routing=3, padding=2)

        self.step_7 = CapsuleLayer(8, 16, "conv", k=5, s=1, t_1=4, z_1=16, routing=3, padding=2)

        self.step_8 = CapsuleLayer(4, 16, "deconv", k=5, s=2, t_1=2, z_1=16, routing=3, padding=2)

        self.step_10 = CapsuleLayer(3, 16, "conv", k=5, s=1, t_1=1, z_1=16, routing=3, padding=2)

        self.outc = OutConv(16, n_classes)

        # self.conv_2 = nn.Sequential(
        #     nn.Conv2d(16, 1, 5, 1, padding=2),
        # )

    def forward(self, x):
        x = self.conv_1(x)
        x.unsqueeze_(1)

        skip_1 = x  # [N,1,16,H,W], [batch size, caps, features per cap, H, W]

        x = self.step_1(x)

        skip_2 = x  # [N,4,16,H/2,W/2]

        x = self.step_2(x)

        skip_3 = x  # [N,8,32,H/4,W/4]

        x = self.step_3(x)  # [N,8,32,H/8,W/8]
        x = self.step_4(x)  # [N,8,32,H/4,W/4]

        x = torch.cat((x, skip_3), 1)  # [N,16,32,H/4,W/4]

        x = self.step_5(x)  # [N,4,32,H/4,W/4]
        x = self.step_6(x)  # [N,4,16,H/2,W/2]

        x = torch.cat((x, skip_2), 1)   # [N,8,16,H/2,W/2]

        x = self.step_7(x)  # [N,4,16,H/2,W/2]
        x = self.step_8(x)  # [N,2,16,H,W]

        x = torch.cat((x, skip_1), 1)

        x = self.step_10(x)

        x.squeeze_(1)

        x = self.outc(x)

        # v_lens = self.compute_vector_length(x)
        # v_lens = v_lens.squeeze(1)
        # this was for the original segcaps implementation

        return x

    def compute_vector_length(self, x):
        out = (x.pow(2)).sum(1, True) + 1e-9
        out = out.sqrt()
        return out
