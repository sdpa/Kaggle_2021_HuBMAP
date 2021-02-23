import torch
import torch.nn as nn
import torch.nn.functional as F

class FocusCaps(nn.Module):
    def __init__(self, in_capsules, in_channels, out_capsules, out_channels, op='conv', kernel_size=3, Nh=1, stride=1, padding=1,
                 c2p=False, mode='naive'):
        super(FocusCaps, self).__init__()
        self.in_capsules = in_capsules # Number of input capsules
        self.in_channels = in_channels # Number of channels per input capsule
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.Nh = Nh
        self.focus_caps = nn.ModuleList()
        self.ag_conv = nn.ModuleList()
        self.mode = mode  # To test focus caps against "naive" capsules, no routing sanity check
        self.stride = stride
        self.padding = padding
        self.op = op
        self.c2p = c2p

        assert (self.mode == 'naive') or (self.mode == 'dense') or (self.mode == 'att'), 'Please select the agreement mode ' \
                                                                                         '(att or dense) or set to naive ' \
                                                                                         'for no routing'
        if self.mode == 'dense':
            assert self.Nh != 0, "The number of agreement heads must be Nh >= 1"
            assert self.out_channels % 4 == 0, 'The number of output features per capsule must be divisible by 4'
            assert self.Nh <= int(0.25*(self.out_capsules * self.out_channels)), 'The number of agreement heads cannot ' \
                                                                                 'exceed the number of possible agreement ' \
                                                                                 'features'
            assert int(0.25*(self.out_capsules * self.out_channels)) % self.Nh == 0, 'The number of agreement heads must result ' \
                                                                                     'in even agreement feature distributions ' \
                                                                                     '(i.e. out_channels = 16, agreement features ' \
                                                                                     'sum to 4, Nh cant be 3)'


        if self.c2p:  # All children compete for parents
            if self.mode == 'naive':
                for i in range(self.in_capsules):
                    if self.op == 'conv':
                        self.focus_caps.append(
                            nn.Conv2d(self.in_channels, self.out_capsules * self.out_channels,
                                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False))
                    else:
                        self.focus_caps.append(
                            nn.ConvTranspose2d(self.in_channels, self.out_capsules * self.out_channels,
                                               kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                               output_padding=1, bias=False))

            elif self.mode == 'dense':
                for i in range(self.in_capsules):
                    if self.op == 'conv':
                        self.focus_caps.append(nn.Conv2d(self.in_channels, int(0.75*(self.out_capsules * self.out_channels)),
                                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False))
                    else:
                        self.focus_caps.append(nn.ConvTranspose2d(self.in_channels, int(0.75*(self.out_capsules * self.out_channels)),
                                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1, bias=False))

                for i in range(self.Nh):
                    self.ag_conv.append(nn.Conv2d(int((0.75*(self.out_capsules * self.out_channels)) +
                                                      (0.25*(self.out_capsules * self.out_channels)/self.Nh)*i),
                                                       int((0.25*(self.out_capsules * self.out_channels))/self.Nh),
                                                       kernel_size=3, stride=1, padding=1))

            elif self.mode == 'att':
                for i in range(self.in_capsules):
                    if self.op == 'conv':
                        self.focus_caps.append(nn.Conv2d(self.in_channels, int(2*(self.out_capsules * self.out_channels)),
                                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False))
                    else:
                        self.focus_caps.append(nn.ConvTranspose2d(self.in_channels, int(2*(self.out_capsules * self.out_channels)),
                                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1, bias=False))

                self.ag_conv.append(nn.Conv2d(int(self.out_capsules * self.out_channels), int(self.out_capsules * self.out_channels),
                                                    kernel_size=3, stride=1, padding=1))


        else:  # All parents compete for children
            if self.mode == 'naive':
                for i in range(self.out_capsules):
                    if self.op == 'conv':
                        self.focus_caps.append(
                            nn.Conv2d(self.in_channels * self.in_capsules, self.out_channels,
                                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False))
                    else:
                        self.focus_caps.append(
                            nn.ConvTranspose2d(self.in_channels * self.in_capsules, self.out_channels,
                                               kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                               output_padding=1, bias=False))

            elif self.mode == 'dense':
                for i in range(self.out_capsules):
                    if self.op == 'conv':
                        self.focus_caps.append(nn.Conv2d(self.in_channels * self.in_capsules, int(0.75*(self.out_channels)),
                                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False))
                    else:
                        self.focus_caps.append(nn.ConvTranspose2d(self.in_channels * self.in_capsules, int(0.75*(self.out_channels)),
                                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1, bias=False))

                for i in range(self.Nh):
                    self.ag_conv.append(nn.Conv2d(int((0.75*(self.out_capsules * self.out_channels)) +
                                                      (0.25*(self.out_capsules * self.out_channels)/self.Nh)*i),
                                                       int((0.25*(self.out_capsules * self.out_channels))/self.Nh),
                                                       kernel_size=3, stride=1, padding=1))
            elif self.mode == 'att':
                for i in range(self.out_capsules):
                    if self.op == 'conv':
                        self.focus_caps.append(nn.Conv2d(self.in_channels * self.in_capsules, 2*(self.out_channels),
                                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False))
                    else:
                        self.focus_caps.append(nn.ConvTranspose2d(self.in_channels * self.in_capsules, 2*(self.out_channels),
                                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1, bias=False))

                self.ag_conv.append(nn.Conv2d(int(self.out_capsules * self.out_channels), int(self.out_capsules * self.out_channels),
                                                    kernel_size=3, stride=1, padding=1))


    def forward(self, u):
        in_capsules = self.in_capsules
        in_channels = self.in_channels
        out_capsules = self.out_capsules
        out_channels = self.out_channels
        Nh = self.Nh
        N = u.shape[0]
        H_1 = u.shape[3]
        W_1 = u.shape[4]

        assert in_capsules == u.shape[1], 'Number of input capsules must match in_capsules. Check the shape of the input tensor.'

        # u.shape = [Batch, in caps, in caps feats, H, W]

        if self.c2p:
            u_t_list = [u_t.squeeze(1) for u_t in u.split(1, 1)]  # Split tensor into one view per child capsule
            u_hat_t_list = []

            for i, u_t in zip(range(in_capsules), u_t_list):  # u_t: [Batch, Features, H, W] (one input capsule)
                u_hat_t = self.focus_caps[i](u_t)
                H_1 = u_hat_t.shape[2]
                W_1 = u_hat_t.shape[3]

                if self.mode == 'naive':
                    u_hat_t = u_hat_t.reshape(N, out_capsules, out_channels, H_1, W_1).transpose_(1,3).transpose_(2, 4)
                elif self.mode == 'dense':
                    u_hat_t = u_hat_t.reshape(N, out_capsules, int(0.75 * out_channels), H_1, W_1).transpose_(1, 3).transpose_(2, 4)
                elif self.mode == 'att':
                    u_hat_t = u_hat_t.reshape(N, 2*out_capsules, out_channels, H_1, W_1).transpose_(1, 3).transpose_(2, 4)

                u_hat_t_list.append(u_hat_t)  # [Batch, H_1, W_1, Out Caps (or Out Caps and Att Caps), Out Caps Feats]

            if self.mode == 'naive':
                out_caps = self.naive_c2p(u_hat_t_list)
            elif self.mode == 'dense':
                out_caps = self.dense_c2p(u_hat_t_list, N, H_1, W_1, out_capsules, out_channels, Nh)
            elif self.mode == 'att':
                out_caps = self.att_c2p(u_hat_t_list, N, H_1, W_1, out_capsules, out_channels)

        else:
            u_hat_t_list = []
            u_t = u.reshape(N, in_capsules * in_channels, H_1, W_1)

            for i in range(out_capsules):
                u_hat_t = self.focus_caps[i](u_t)
                H_1 = u_hat_t.shape[2]
                W_1 = u_hat_t.shape[3]

                if self.mode == 'naive':
                    u_hat_t = u_hat_t.reshape(N, 1, out_channels, H_1, W_1).transpose_(1,3).transpose_(2, 4)
                elif self.mode == 'dense':
                    u_hat_t = u_hat_t.reshape(N, 1, int(0.75 * out_channels), H_1, W_1).transpose_(1, 3).transpose_(2, 4)
                elif self.mode == 'att':
                    u_hat_t = u_hat_t.reshape(N, 2, out_channels, H_1, W_1).transpose_(1, 3).transpose_(2, 4)

                u_hat_t_list.append(u_hat_t)  # [Batch, H_1, W_1, Out Cap (or Out Cap and Att Cap), Out Caps Feats]

            if self.mode == 'naive':
                out_caps = self.naive_p2c(u_hat_t_list)
            elif self.mode == 'dense':
                out_caps = self.dense_p2c(u_hat_t_list, N, H_1, W_1, out_capsules, out_channels, Nh)
            elif self.mode == 'att':
                out_caps = self.att_p2c(u_hat_t_list, N, H_1, W_1, out_capsules, out_channels)

        return out_caps

    def att_c2p(self, u_hat_t_list, N, H_1, W_1, out_capsules, out_channels):
        out_caps_list = []

        for i, u_hat_t in enumerate(u_hat_t_list):
            q, v = u_hat_t.split(out_capsules, dim=3)
            q_flat = torch.flatten(q.clone())
            q_flat -= q.clone().min()
            q_flat /= (q.clone().max() - q.clone().min())
            raw_caps = q_flat.reshape(N, H_1, W_1, out_capsules, out_channels) * v.clone()
            raw_caps = raw_caps.reshape(N, raw_caps.shape[3] * raw_caps.shape[4], H_1, W_1)
            ag_conv = self.ag_conv[0](raw_caps)
            ag_conv_flat = torch.flatten(ag_conv.clone())
            ag_conv_flat -= ag_conv.clone().min()
            ag_conv_flat /= (ag_conv.clone().max() - ag_conv.clone().min())
            u_hat_t_a = ag_conv_flat.reshape(N, H_1, W_1, out_capsules, out_channels) * raw_caps.reshape(N, H_1, W_1, out_capsules, out_channels)
            out_caps_list.append(u_hat_t_a)

        out_caps = sum(out_caps_list)
        out_caps = self.squash(out_caps)
        out_caps.transpose_(1, 3).transpose_(2, 4)

        return out_caps

    def att_p2c(self, u_hat_t_list, N, H_1, W_1, out_capsules, out_channels):
        raw_caps_list = []

        for i, u_hat_t in enumerate(u_hat_t_list):
            q, v = u_hat_t.split(1, dim=3)
            q_flat = torch.flatten(q.clone())
            q_flat -= q.clone().min()
            q_flat /= (q.clone().max() - q.clone().min())
            raw_caps = q_flat.reshape(N, H_1, W_1, 1, out_channels) * v.clone()
            raw_caps_list.append(raw_caps)

        out_caps = torch.cat(raw_caps_list, dim=3).reshape(N, out_capsules * out_channels, H_1, W_1)
        ag_conv = self.ag_conv[0](out_caps)
        ag_conv_flat = torch.flatten(ag_conv.clone())
        ag_conv_flat -= ag_conv.clone().min()
        ag_conv_flat /= (ag_conv.clone().max() - ag_conv.clone().min())
        out_caps = ag_conv_flat.reshape(N, H_1, W_1, out_capsules, out_channels) * out_caps.reshape(N, H_1, W_1, out_capsules, out_channels)
        out_caps = self.squash(out_caps)
        out_caps.transpose_(1, 3).transpose_(2, 4)

        return out_caps

    def naive_c2p(self, u_hat_t_list):
        out_caps_list = u_hat_t_list
        out_caps = sum(out_caps_list)
        out_caps = self.squash(out_caps)
        out_caps.transpose_(1, 3).transpose_(2, 4)

        return out_caps

    def naive_p2c(self, u_hat_t_list):
        out_caps = torch.cat(u_hat_t_list, dim=1)
        out_caps = self.squash(out_caps)
        out_caps.transpose_(2, 4)

        return out_caps

    def dense_c2p(self, u_hat_t_list, N, H_1, W_1, out_capsules, out_channels, Nh):
        out_caps_list = []
        ag_convs = []
        counter = 0
        for u_hat_t in u_hat_t_list:
            for i in range(Nh):
                u_hat_t = u_hat_t.reshape(N, u_hat_t.shape[3] * u_hat_t.shape[4], H_1, W_1)
                ag_convs.append(self.ag_conv[i](u_hat_t))
                u_hat_t = torch.cat((u_hat_t.reshape(N, H_1, W_1, out_capsules, int(0.75*(out_channels)) + i * int((0.25*(out_channels)/Nh))),
                                     ag_convs[i + counter].reshape(N, H_1, W_1, out_capsules, int((0.25*(out_channels))/Nh))), dim=4)
            counter += Nh
            out_caps_list.append(u_hat_t.copy())

        out_caps = sum(out_caps_list)
        out_caps = self.squash(out_caps)
        out_caps.transpose_(1, 3).transpose_(2, 4)

        return out_caps

    def dense_p2c(self, u_hat_t_list, N, H_1, W_1, out_capsules, out_channels, Nh):
        ag_convs = []
        out_caps = torch.cat(u_hat_t_list, dim=1)
        out_caps = out_caps.reshape((N, H_1, W_1, out_caps.shape[1], out_caps.shape[2]))

        for i in range(Nh):
            out_caps = out_caps.reshape(N, out_caps.shape[3] * out_caps.shape[4], H_1, W_1)
            ag_convs.append(self.ag_conv[i](out_caps))
            out_caps = torch.cat((out_caps.reshape(N, H_1, W_1, out_capsules, int(0.75*(out_channels)) + i * int((0.25*(out_channels)/Nh))),
                                 ag_convs[i].reshape(N, H_1, W_1, out_capsules, int((0.25*(out_channels))/Nh))), dim=4)

        out_caps = self.squash(out_caps)
        out_caps.transpose_(1, 3).transpose_(2, 4)

        return out_caps

    def squash(self, p):
        p_norm_sq = (p * p).sum(-1, True)
        p_norm = (p_norm_sq + 1e-9).sqrt()
        v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
        return v


## Debugging tool
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# tmp = torch.randn((1, 16, 8, 64, 64)).to(device)  # [Batch, in caps, in caps feats, H, W]
# augmented_caps1 = FocusCaps(in_capsules=16, in_channels=8, out_capsules=10, out_channels=16, op='conv', kernel_size=6,
#                             Nh=2, stride=4, padding=2, c2p=False, mode='att').to(device)
# conv_out1 = augmented_caps1(tmp)
# print(conv_out1.shape)
#
# for name, param in augmented_caps1.named_parameters():
#     print('parameter name: ', name)

# augmented_caps2 = FocusCaps(in_capsules=2, in_channels=16, out_capsules=4, out_channels=32, op='deconv', kernel_size=3,
#                             Nh=2, naive_caps=True, stride=2, padding=1).to(device)
# conv_out2 = augmented_caps2(tmp)
# print(conv_out2.shape)