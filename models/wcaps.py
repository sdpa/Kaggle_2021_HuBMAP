""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .layers import *
from . import resnet
from .transformer_routing import PrimaryCaps, ConvCaps, UpConvCaps
import torch.nn.functional as F

class WCaps(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, args=[]):
        super(WCaps, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.args = args
        self.encoder_caps = nn.ModuleList()
        self.hencoder_caps = nn.ModuleList()
        self.qencoder_caps = nn.ModuleList()
        self.decoder_caps = nn.ModuleList()
        self.hdecoder_caps = nn.ModuleList()
        self.qdecoder_caps = nn.ModuleList()

        ##################################
        # Scaling ops, encoding
        ##################################
        if self.args.wscales:
            self.sdensehalf = DenseBlock(n_channels, 32, 6)
            self.pcaphalf = PrimaryCaps(A=192 + n_channels, B=1, K=1, P=self.args.P, stride=1, pad=0)

            self.sdensequarter = DenseBlock(n_channels, 32, 6)
            self.pcapquarter = PrimaryCaps(A=192 + n_channels, B=1, K=1, P=self.args.P, stride=1, pad=0)

        ##################################
        # Network body
        ##################################
        self.sdense = DenseBlock(n_channels, 32, 6)
        self.pcap = PrimaryCaps(A=192 + n_channels, B=1, K=1, P=self.args.P, stride=1, pad=0)

        for i in range(args.modules):
            for j in range(args.enc_ops):
                if (j == 0) and (i == 0):
                    self.encoder_caps.append(
                    ConvCaps(A=1, B=1, K=3, P=self.args.P, stride=2, pad=1, args=self.args))  # P=4*(2**(j+1))
                    self.decoder_caps.append(
                    UpConvCaps(A=1, B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args,
                    scale=2))  # P=4*(2**((self.args.enc_ops)-j-2))
                    if self.args.wscales:
                        self.hencoder_caps.append(
                        ConvCaps(A=1, B=1, K=3, P=self.args.P, stride=2, pad=1, args=self.args))  # P=4*(2**(j+1))
                        self.hdecoder_caps.append(
                        UpConvCaps(A=1, B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args,
                        scale=2))  # P=4*(2**((self.args.enc_ops)-j-2))
                        self.qencoder_caps.append(
                        ConvCaps(A=1, B=1, K=3, P=self.args.P, stride=2, pad=1, args=self.args))  # P=4*(2**(j+1))
                        self.qdecoder_caps.append(
                        UpConvCaps(A=1, B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args,
                        scale=2))  # P=4*(2**((self.args.enc_ops)-j-2))
                elif (j == 0) and (i != 0):
                    self.encoder_caps.append(
                    ConvCaps(A=i + 1, B=1, K=3, P=self.args.P, stride=2, pad=1, args=self.args))  # P=4*(2**(j+1))
                    self.decoder_caps.append(
                    UpConvCaps(A=1, B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args,
                    scale=2))  # P=4*(2**((self.args.enc_ops)-j-2))
                    if self.args.wscales:
                        self.hencoder_caps.append(
                        ConvCaps(A=i + 1, B=1, K=3, P=self.args.P, stride=2, pad=1, args=self.args))  # P=4*(2**(j+1))
                        self.hdecoder_caps.append(
                        UpConvCaps(A=1, B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args,
                        scale=2))  # P=4*(2**((self.args.enc_ops)-j-2))
                        self.qencoder_caps.append(
                        ConvCaps(A=i + 1, B=1, K=3, P=self.args.P, stride=2, pad=1, args=self.args))  # P=4*(2**(j+1))
                        self.qdecoder_caps.append(
                        UpConvCaps(A=1, B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args,
                        scale=2))  # P=4*(2**((self.args.enc_ops)-j-2))
                else:
                    self.encoder_caps.append(
                    ConvCaps(A=1, B=1, K=3, P=self.args.P, stride=2, pad=1, args=self.args))  # P=4*(2**(j+1))
                    self.decoder_caps.append(
                    UpConvCaps(A=i + 2, B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args,
                    scale=2))  # P=4*(2**((self.args.enc_ops)-j-2))
                    if self.args.wscales:
                        self.hencoder_caps.append(
                        ConvCaps(A=1, B=1, K=3, P=self.args.P, stride=2, pad=1, args=self.args))  # P=4*(2**(j+1))
                        self.hdecoder_caps.append(
                        UpConvCaps(A=i + 2, B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args,
                        scale=2))  # P=4*(2**((self.args.enc_ops)-j-2))
                        self.qencoder_caps.append(
                        ConvCaps(A=1, B=1, K=3, P=self.args.P, stride=2, pad=1, args=self.args))  # P=4*(2**(j+1))
                        self.qdecoder_caps.append(
                        UpConvCaps(A=i + 2, B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args,
                        scale=2))  # P=4*(2**((self.args.enc_ops)-j-2))

        if self.args.wscales:
            self.ocap = ConvCaps(A=3*(args.modules + 1), B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args)
        else:
            self.ocap = ConvCaps(A=args.modules + 1, B=1, K=3, P=self.args.P, stride=1, pad=1, args=self.args)
        self.outc = OutConv(int(self.args.P**2), n_classes)

    def forward(self, x):
        ##################################
        # Full scale
        ##################################
        x1 = self.sdense(x)
        xp = self.pcap(x1)
        enc_ops = [[] for _ in range(self.args.modules)]
        dec_ops = [[] for _ in range(self.args.modules)]
        cat_ops = [None] * self.args.enc_ops
        cat_ops[0] = xp

        ##################################
        # Lower scales
        ##################################
        if self.args.wscales:
            xh = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
            xh = self.sdensehalf(xh)
            xh = self.pcaphalf(xh)
            henc_ops = [[] for _ in range(self.args.modules)]
            hdec_ops = [[] for _ in range(self.args.modules)]
            hcat_ops = [None] * self.args.enc_ops
            hcat_ops[0] = xh

            xq = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
            xq = self.sdensequarter(xq)
            xq = self.pcapquarter(xq)
            qenc_ops = [[] for _ in range(self.args.modules)]
            qdec_ops = [[] for _ in range(self.args.modules)]
            qcat_ops = [None] * self.args.enc_ops
            qcat_ops[0] = xq

        dec_inv = list(range(self.args.enc_ops))
        dec_inv.reverse()

        for i in range(self.args.modules):
            for j in range(self.args.enc_ops):
                if i == 0:
                    enc_ops[i].append(self.encoder_caps[(i * self.args.enc_ops) + j](cat_ops[j]))
                    if self.args.wscales:
                        henc_ops[i].append(self.hencoder_caps[(i * self.args.enc_ops) + j](hcat_ops[j]))
                        qenc_ops[i].append(self.qencoder_caps[(i * self.args.enc_ops) + j](qcat_ops[j]))
                    if j != self.args.enc_ops-1:
                        cat_ops[j+1] = enc_ops[i][-1]
                        if self.args.wscales:
                            hcat_ops[j + 1] = henc_ops[i][-1]
                            qcat_ops[j + 1] = qenc_ops[i][-1]
                else:
                    if j == 0:
                        cat_ops[j] = torch.cat((cat_ops[j], dec_ops[i-1][-1]), dim=1)
                        enc_ops[i].append(self.encoder_caps[(i * self.args.enc_ops) + j](cat_ops[j]))
                        if self.args.wscales:
                            hcat_ops[j] = torch.cat((hcat_ops[j], hdec_ops[i - 1][-1]), dim=1)
                            henc_ops[i].append(self.hencoder_caps[(i * self.args.enc_ops) + j](hcat_ops[j]))
                            qcat_ops[j] = torch.cat((qcat_ops[j], qdec_ops[i - 1][-1]), dim=1)
                            qenc_ops[i].append(self.qencoder_caps[(i * self.args.enc_ops) + j](qcat_ops[j]))
                    else:
                        cat_ops[j] = torch.cat((cat_ops[j], enc_ops[i][-1]), dim=1)
                        enc_ops[i].append(self.encoder_caps[(i * self.args.enc_ops) + j](enc_ops[i][-1]))
                        if self.args.wscales:
                            hcat_ops[j] = torch.cat((hcat_ops[j], henc_ops[i][-1]), dim=1)
                            henc_ops[i].append(self.hencoder_caps[(i * self.args.enc_ops) + j](henc_ops[i][-1]))
                            qcat_ops[j] = torch.cat((qcat_ops[j], qenc_ops[i][-1]), dim=1)
                            qenc_ops[i].append(self.qencoder_caps[(i * self.args.enc_ops) + j](qenc_ops[i][-1]))
            for j in range(self.args.enc_ops):
                if (j == 0):
                    dec_ops[i].append(self.decoder_caps[(i*self.args.enc_ops)+j](enc_ops[i][-1]))
                    if self.args.wscales:
                        hdec_ops[i].append(self.hdecoder_caps[(i*self.args.enc_ops)+j](henc_ops[i][-1]))
                        qdec_ops[i].append(self.qdecoder_caps[(i * self.args.enc_ops) + j](qenc_ops[i][-1]))
                else:
                    dec_ops[i].append(self.decoder_caps[(i * self.args.enc_ops) + j](torch.cat((cat_ops[dec_inv[j-1]], dec_ops[i][-1]), dim=1)))
                    if self.args.wscales:
                        hdec_ops[i].append(self.hdecoder_caps[(i * self.args.enc_ops) + j](torch.cat((hcat_ops[dec_inv[j-1]], hdec_ops[i][-1]), dim=1)))
                        qdec_ops[i].append(self.qdecoder_caps[(i * self.args.enc_ops) + j](torch.cat((qcat_ops[dec_inv[j-1]], qdec_ops[i][-1]), dim=1)))

        if self.args.wscales:
            xh_out = torch.cat([hcat_ops[0], hdec_ops[-1][-1]], dim=1)
            xh_out = F.interpolate(xh_out, scale_factor=2, mode='bilinear', align_corners=True)
            xq_out = torch.cat([qcat_ops[0], qdec_ops[-1][-1]], dim=1)
            xq_out = F.interpolate(xq_out, scale_factor=4, mode='bilinear', align_corners=True)
            xo = self.ocap(torch.cat([cat_ops[0], dec_ops[-1][-1], xh_out, xq_out], dim=1))
        else:
            xo = self.ocap(torch.cat([cat_ops[0], dec_ops[-1][-1]], dim=1))

        logits = self.outc(xo)

        return logits
