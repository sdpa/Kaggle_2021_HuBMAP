""" Full assembly of the UCap network parts """

from .unet_parts import *
from .layers import *

class UCaps(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UCaps, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Down ops
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1),
        )
        self.conv1 = UDenseBlock(32, 16, 2, upsample=False)
        self.caps1 = UCapsuleLayer(4, 16, "conv", k=3, s=2, t_1=4, z_1=16, routing=3, padding=1)
        self.conv2 = UDenseBlock(64, 32, 2, upsample=False)
        self.caps2 = UCapsuleLayer(8, 16, "conv", k=3, s=2, t_1=8, z_1=16, routing=3, padding=1)
        self.conv3 = UDenseBlock(128, 64, 2, upsample=False)
        self.caps3 = UCapsuleLayer(8, 32, "conv", k=3, s=2, t_1=8, z_1=32, routing=3, padding=1)
        self.conv4 = UDenseBlock(256, 128, 2, upsample=False)
        self.caps4 = UCapsuleLayer(16, 32, "conv", k=3, s=2, t_1=16, z_1=32, routing=3, padding=1)
        self.conv5 = UDenseBlock(512, 256, 2, upsample=False)

        # Up ops
        self.upcaps_1 = UCapsuleLayer(32, 32, "deconv", k=3, s=2, t_1=16, z_1=32, routing=3, padding=1)
        self.upconv_1 = UDenseBlock(1024, 256, 2, upsample=True)
        self.upcaps_2 = UCapsuleLayer(16, 32, "deconv", k=3, s=2, t_1=8, z_1=32, routing=3, padding=1)
        self.upconv_2 = UDenseBlock(512, 128, 2, upsample=True)
        self.upcaps_3 = UCapsuleLayer(8, 32, "deconv", k=3, s=2, t_1=8, z_1=16, routing=3, padding=1)
        self.upconv_3 = UDenseBlock(256, 64, 2, upsample=True)
        self.upcaps_4 = UCapsuleLayer(8, 16, "deconv", k=3, s=2, t_1=4, z_1=16, routing=3, padding=1)
        self.upconv_4 = UDenseBlock(128, 32, 2, upsample=True)

        # Prediction mask
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Down ops
        xa = self.first_conv(x)
        x1 = self.conv1(xa)

        x1c = x1.clone()
        x1c = x1c.unsqueeze_(1)
        x1c = x1c.reshape([x1c.shape[0], 4, 16, x1c.shape[3], x1c.shape[4]]) # [batch size, caps, features per cap, H, W]
        x2 = self.caps1(x1c)
        x2 = x2.reshape([x2.shape[0], 1, 64, x2.shape[3], x2.shape[4]])
        x2.squeeze_(1)
        x3 = self.conv2(x2)

        x3c = x3.clone()
        x3c = x3c.unsqueeze_(1)
        x3c = x3c.reshape([x3c.shape[0], 8, 16, x3c.shape[3], x3c.shape[4]])
        x4 = self.caps2(x3c)
        x4 = x4.reshape([x4.shape[0], 1, 128, x4.shape[3], x4.shape[4]])
        x4.squeeze_(1)
        x5 = self.conv3(x4)

        x5c = x5.clone()
        x5c = x5c.unsqueeze_(1)
        x5c = x5c.reshape([x5c.shape[0], 8, 32, x5c.shape[3], x5c.shape[4]])
        x6 = self.caps3(x5c)
        x6 = x6.reshape([x6.shape[0], 1, 256, x6.shape[3], x6.shape[4]])
        x6.squeeze_(1)
        x7 = self.conv4(x6)

        x7c = x7.clone()
        x7c = x7c.unsqueeze_(1)
        x7c = x7c.reshape([x7c.shape[0], 16, 32, x7c.shape[3], x7c.shape[4]])
        x8 = self.caps4(x7c)
        x8 = x8.reshape([x8.shape[0], 1, 512, x8.shape[3], x8.shape[4]])
        x8.squeeze_(1)
        x9 = self.conv5(x8)

        # Up ops
        x9c = x9.clone()
        x9c = x9c.unsqueeze_(1)
        x9c = x9c.reshape([x9c.shape[0], 32, 32, x9c.shape[3], x9c.shape[4]])
        x10 = self.upcaps_1(x9c)
        x10 = x10.reshape([x10.shape[0], 1, 512, x10.shape[3], x10.shape[4]])
        x10.squeeze_(1)
        xcat1 = torch.cat((x10, x7), 1)
        x11 = self.upconv_1(xcat1)

        x11c = x11.clone()
        x11c = x11c.unsqueeze_(1)
        x11c = x11c.reshape([x11c.shape[0], 16, 32, x11c.shape[3], x11c.shape[4]])
        x12 = self.upcaps_2(x11c)
        x12 = x12.reshape([x12.shape[0], 1, 256, x12.shape[3], x12.shape[4]])
        x12.squeeze_(1)
        xcat2 = torch.cat((x12, x5), 1)
        x13 = self.upconv_2(xcat2)

        x13c = x13.clone()
        x13c = x13c.unsqueeze_(1)
        x13c = x13c.reshape([x13c.shape[0], 8, 32, x13c.shape[3], x13c.shape[4]])
        x14 = self.upcaps_3(x13c)
        x14 = x14.reshape([x14.shape[0], 1, 128, x14.shape[3], x14.shape[4]])
        x14.squeeze_(1)
        xcat3 = torch.cat((x14, x3), 1)
        x15 = self.upconv_3(xcat3)

        x15c = x15.clone()
        x15c = x15c.unsqueeze_(1)
        x15c = x15c.reshape([x15c.shape[0], 8, 16, x15c.shape[3], x15c.shape[4]])
        x16 = self.upcaps_4(x15c)
        x16 = x16.reshape([x16.shape[0], 1, 64, x16.shape[3], x16.shape[4]])
        x16.squeeze_(1)
        xcat4 = torch.cat((x16, x1), 1)
        x17 = self.upconv_4(xcat4)

        # Prediction mask
        logits = self.outc(x17)

        return logits
