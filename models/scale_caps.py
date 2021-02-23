""" Full assembly of the UCap network parts """

from .unet_parts import *
from .layers import *
from . import resnet
from .transformer_routing import PrimaryCaps, ConvCaps
import torch.nn.functional as F

class ScaleCaps(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, args=[]):
        super(ScaleCaps, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.args = args

        # Encoder ops
        self.scale1 = resnet.__dict__[args.resnet_version](in_channels=args.img_c, planes=int(64/4))
        self.scale2 = resnet.__dict__[args.resnet_version](in_channels=args.img_c, planes=int(64/4))

        # Capsule ops
        self.primary1 = PrimaryCaps(A=64, B=16, K=3, P=4, stride=1, pad=1)
        self.primary2 = PrimaryCaps(A=64, B=16, K=3, P=4, stride=1, pad=1)

        self.convcaps1 = ConvCaps(A=16, B=16, K=3, P=4, stride=1, pad=1, args=self.args)
        self.convcaps2 = ConvCaps(A=16, B=16, K=3, P=4, stride=1, pad=1, args=self.args)

        # Upscale to match
        self.up_half = nn.Upsample(size=52, mode='bilinear', align_corners=True)

        # Capsule-wise Squeeze-Excitation
        self.pmean_excitation = nn.Sequential(
            nn.Linear(16, 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16, bias=False),
            nn.ReLU()
        )

        self.pmax_excitation = nn.Sequential(
            nn.Linear(16, 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16, bias=False),
            nn.ReLU()
        )

        self.pmin_excitation = nn.Sequential(
            nn.Linear(16, 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16, bias=False),
            nn.ReLU()
        )

        self.pvar_excitation = nn.Sequential(
            nn.Linear(16, 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16, bias=False),
            nn.ReLU()
        )

        self.pstat_excitation = nn.Sequential(
            nn.Linear(16, 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16, bias=False),
            nn.Sigmoid()
        )

        self.statsmat = nn.Parameter(torch.Tensor(4, 1))
        nn.init.kaiming_uniform_(self.statsmat.data)

        # Decoder ops
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.batch1 = nn.BatchNorm2d(256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.batch2 = nn.BatchNorm2d(128)

        self.outc = OutConv(128, n_classes)

    def forward(self, x):
        # Scaling
        x_full = x
        x_half = F.interpolate(x, size=int(x.shape[2]/2))

        # Encoding
        e1 = self.scale1(x_full)
        e2 = self.scale2(x_half)

        # Capsule ops
        c1 = self.primary1(e1)
        pav1 = torch.var(c1.reshape(c1.shape[0], int(c1.shape[1]/16), 16, c1.shape[2], c1.shape[3]), dim=2).reshape(
            c1.shape[0], int(c1.shape[1]/16), c1.shape[2]**2)
        sq1 = c1.reshape(c1.shape[0], int(c1.shape[1]/16), 16*c1.shape[2]**2)
        mean_ex1 = self.pmean_excitation(torch.mean(pav1, dim=-1))
        max_ex1 = self.pmax_excitation(torch.max(pav1, dim=-1).values)
        min_ex1 = self.pmin_excitation(torch.min(pav1, dim=-1).values)
        var_ex1 = self.pvar_excitation(torch.var(pav1, dim=-1))
        prestats_ex1 = torch.matmul(torch.stack([mean_ex1, max_ex1, min_ex1, var_ex1], dim=-1), self.statsmat)
        stats_ex1 = self.pstat_excitation(prestats_ex1.squeeze(-1))
        c1 = (sq1*stats_ex1.unsqueeze(-1)).reshape(c1.shape[0], c1.shape[1], c1.shape[2], c1.shape[3])

        c1c = self.convcaps1(c1)
        pav1 = torch.var(c1c.reshape(c1c.shape[0], int(c1c.shape[1]/16), 16, c1c.shape[2], c1c.shape[3]), dim=2).reshape(
            c1c.shape[0], int(c1c.shape[1]/16), c1c.shape[2]**2)
        sq1 = c1c.reshape(c1c.shape[0], int(c1c.shape[1]/16), 16*c1c.shape[2]**2)
        mean_ex1 = self.pmean_excitation(torch.mean(pav1, dim=-1))
        max_ex1 = self.pmax_excitation(torch.max(pav1, dim=-1).values)
        min_ex1 = self.pmin_excitation(torch.min(pav1, dim=-1).values)
        var_ex1 = self.pvar_excitation(torch.var(pav1, dim=-1))
        prestats_ex1 = torch.matmul(torch.stack([mean_ex1, max_ex1, min_ex1, var_ex1], dim=-1), self.statsmat)
        stats_ex1 = self.pstat_excitation(prestats_ex1.squeeze(-1))
        c1c = (sq1*stats_ex1.unsqueeze(-1)).reshape(c1c.shape[0], c1c.shape[1], c1c.shape[2], c1c.shape[3])

        c2 = self.primary2(e2)
        pav2 = torch.var(c2.reshape(c2.shape[0], int(c2.shape[1] / 16), 16, c2.shape[2], c2.shape[3]), dim=2).reshape(
            c2.shape[0], int(c2.shape[1] / 16), c2.shape[2] ** 2)
        sq2 = c2.reshape(c2.shape[0], int(c2.shape[1] / 16), 16 * c2.shape[2] ** 2)
        mean_ex2 = self.pmean_excitation(torch.mean(pav2, dim=-1))
        max_ex2 = self.pmax_excitation(torch.max(pav2, dim=-1).values)
        min_ex2 = self.pmin_excitation(torch.min(pav2, dim=-1).values)
        var_ex2 = self.pvar_excitation(torch.var(pav2, dim=-1))
        prestats_ex2 = torch.matmul(torch.stack([mean_ex2, max_ex2, min_ex2, var_ex2], dim=-1), self.statsmat)
        stats_ex2 = self.pstat_excitation(prestats_ex2.squeeze(-1))
        c2 = (sq2 * stats_ex2.unsqueeze(-1)).reshape(c2.shape[0], c2.shape[1], c2.shape[2], c2.shape[3])

        c2c = self.convcaps2(c2)
        pav2 = torch.var(c2c.reshape(c2c.shape[0], int(c2c.shape[1] / 16), 16, c2c.shape[2], c2c.shape[3]), dim=2).reshape(
            c2c.shape[0], int(c2c.shape[1] / 16), c2c.shape[2] ** 2)
        sq2 = c2c.reshape(c2c.shape[0], int(c2c.shape[1] / 16), 16 * c2c.shape[2] ** 2)
        mean_ex2 = self.pmean_excitation(torch.mean(pav2, dim=-1))
        max_ex2 = self.pmax_excitation(torch.max(pav2, dim=-1).values)
        min_ex2 = self.pmin_excitation(torch.min(pav2, dim=-1).values)
        var_ex2 = self.pvar_excitation(torch.var(pav2, dim=-1))
        prestats_ex2 = torch.matmul(torch.stack([mean_ex2, max_ex2, min_ex2, var_ex2], dim=-1), self.statsmat)
        stats_ex2 = self.pstat_excitation(prestats_ex2.squeeze(-1))
        c2c = (sq2 * stats_ex2.unsqueeze(-1)).reshape(c2c.shape[0], c2c.shape[1], c2c.shape[2], c2c.shape[3])

        # Alignment scaling
        c2c = self.up_half(c2c)

        # Concatinating our capsules
        c = torch.cat([c1c, c2c], dim=1)

        # Decoding
        d1 = self.up1(c)
        d1 = self.batch1(d1)
        d2 = self.up2(d1)
        d2 = self.batch2(d2)

        logits = self.outc(d2)

        return logits
