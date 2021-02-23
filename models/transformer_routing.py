import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCaps(nn.Module):
    def __init__(self, A=32, B=32, K=1, P=4, stride=1, pad=1):
        super(PrimaryCaps, self).__init__()
        self.conv = nn.Conv2d(in_channels=A, out_channels=B * P * P,
                              kernel_size=K, stride=stride, padding=pad, bias=False)
        self.psize = P * P
        self.B = B
        self.nonlinear_act = nn.BatchNorm2d(self.psize*B)

    def forward(self, x_in):
        pose = self.nonlinear_act(self.conv(x_in))
        return pose


class ConvCaps(nn.Module):
    def __init__(self, A=32, B=32, K=3, P=4, stride=2, pad=0, args=[]):
        super(ConvCaps, self).__init__()
        self.A = A
        self.B = B
        self.k = K
        self.kk = K * K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.kkA = K * K * A
        self.pad = pad
        # self.W = nn.Parameter(torch.randn(self.kkA, B, self.P, self.P))

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.P, self.P))
        nn.init.kaiming_uniform_(self.W.data)

        num_heads = args.num_heads
        self.router = TransformerRouter(num_ind=B, num_heads=num_heads, dim=self.psize)

    def forward(self, x):
        b, c, h, w = x.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(x, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(b, l, self.kkA, self.P, self.P).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W) # Voting, affine transformation

        # [b*l, kkA, B, psize]
        v = pose_out.view(b * l, self.kkA, self.B, self.psize)

        # [b*l, B, psize]
        pose_out = self.router(v)

        # [b, l, B*psize]
        pose_out = pose_out.view(b, l, self.B*self.psize)

        oh = (h - self.k + (2 * self.pad)) / self.stride + 1
        ow = (w - self.k + (2 * self.pad)) / self.stride + 1

        pose_out = pose_out.view(b, int(oh), int(ow), self.B * self.psize)
        return pose_out.permute(0, 3, 1, 2)

class UpConvCaps(nn.Module):
    def __init__(self, A=32, B=32, K=3, P=4, stride=2, pad=0, args=[], scale=2):
        super(UpConvCaps, self).__init__()
        self.A = A
        self.B = B
        self.k = K
        self.kk = K * K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.kkA = K * K * A
        self.pad = pad
        # self.W = nn.Parameter(torch.randn(self.kkA, B, self.P, self.P))

        self.xup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.P, self.P))
        nn.init.kaiming_uniform_(self.W.data)

        num_heads = args.num_heads
        self.router = TransformerRouter(num_ind=B, num_heads=num_heads, dim=self.psize)

    def forward(self, x):
        # simple deconvolution
        x_up = self.xup(x)
        b, c, h, w = x_up.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(x_up, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(b, l, self.kkA, self.P, self.P).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W) # Voting, affine transformation

        # [b*l, kkA, B, psize]
        v = pose_out.view(b * l, self.kkA, self.B, self.psize)

        # [b*l, B, psize]
        pose_out = self.router(v)

        # [b, l, B*psize]
        pose_out = pose_out.view(b, l, self.B*self.psize)

        # if h != w:

        oh = (h - self.k + (2 * self.pad)) / self.stride + 1
        ow = (w - self.k + (2 * self.pad)) / self.stride + 1

        pose_out = pose_out.view(b, int(oh), int(ow), self.B * self.psize)
        return pose_out.permute(0, 3, 1, 2)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.M = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        # nn.init.xavier_uniform_(self.M.data)
        nn.init.kaiming_uniform_(self.M.data)

        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        if ln:
            self.ln0 = nn.BatchNorm1d(num_seeds*dim)
            self.ln1 = nn.BatchNorm1d(num_seeds*dim)
        self.fc_o = nn.Linear(dim, dim)

    def forward(self, X):
        dim_split = self.dim // self.num_heads

        M = self.M.repeat(X.size(0), 1, 1)
        K, V = self.fc_k(X), self.fc_v(X)

        M_ = torch.cat(M.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 3), 0)
        V_ = torch.cat(V.split(dim_split, 3), 0)
        S = torch.matmul(K_.transpose(1, 2), M_.unsqueeze(-1)).transpose(1, 2)
        A = torch.softmax(S / math.sqrt(self.dim), 1)
        O = torch.cat((M_ + torch.sum(A * V_, dim=1)).split(M.size(0), 0), 2)

        bs, a, dim = O.shape
        O = self.ln0(O.view(bs, a*dim)).view(bs, a, dim)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O.view(bs, a*dim)).view(bs, a, dim)

        return O

class TransformerRouter(nn.Module):
    def __init__(self, num_ind, num_heads, dim=16, ln=True):
        super(TransformerRouter, self).__init__()
        self.pma = PMA(dim, num_heads, num_ind, ln=ln)

    def forward(self, pose):
        pose = self.pma(pose)
        return pose

# # Debug
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# tmp = torch.randn((2, 128, 512, 512)).to(device)
# augmented_conv1 = ConvCaps(A=4, B=1, K=3, P=8, stride=2, pad=1, args=[]).to(device)
# conv_out1 = augmented_conv1(tmp)
# print(conv_out1.shape)
# #
# for name, param in augmented_conv1.named_parameters():
#     print('parameter name: ', name)
